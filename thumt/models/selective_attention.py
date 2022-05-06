# part of this code is from fariseq-mmt(https://github.com/libeineu/fairseq_mmt)

import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads
from .transformer import Transformer

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class SelectiveAttention(nn.Module):
    def __init__(self, qdim, kdim, vdim, attn_dim, intermediate_dim, output_dim, num_heads=1, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim

        self.qkhead_dim = attn_dim // num_heads
        self.vhead_dim = intermediate_dim // num_heads               
        self.scale = self.qkhead_dim ** -0.5

        self.q_proj = Linear(qdim, attn_dim, bias=qkv_bias)
        self.k_proj = Linear(kdim, attn_dim, bias=qkv_bias)
        self.v_proj = Linear(vdim, intermediate_dim, bias=qkv_bias)   
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(intermediate_dim, output_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        Tq, Bq, Cq = query.shape
        Tk, Bk, Ck = key.shape
        Tv, Bv, Cv = value.shape
        assert Bq == Bk == Bv
        assert Tk == Tv
        assert Cq == self.qdim
        assert Ck == self.kdim
        assert Cv == self.vdim
        bsz = Bq
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
       
        q *= self.scale
        
        q = q.contiguous().view(Tq, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.vhead_dim).transpose(0, 1)
        # B*H, T, C//H

        attn = (q @ k.transpose(-2, -1)) 
        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, Tq, Tk)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn = attn.view(bsz * self.num_heads, Tq, Tk)

        attn = attn.softmax(dim=-1)
        attn_after_drop = self.attn_drop(attn)

        x = (attn_after_drop @ v)
        assert list(x.size()) == [bsz * self.num_heads, Tq, self.vhead_dim]
        x = x.transpose(0, 1).contiguous().view(Tq, bsz, self.intermediate_dim)
        x = self.proj(x)
        return x, attn

class SelectiveAttnTransformer(modules.Module):
    def __init__(self, params, name="selective_attn_transformer"):
        super(SelectiveAttnTransformer, self).__init__(name=name)
        self.params = params

        if params.prefix_only:
            self._model = [Transformer(params)]
        else:
            self._model = Transformer(params)

        self.hidden_size = params.hidden_size
        self.image_feat_dim = params.image_feat_dim

        self.image_dropout = nn.Dropout(params.image_dropout)
        self.text_dropout = nn.Dropout(params.text_dropout)

        self.selective_attn = SelectiveAttention(qdim=self.hidden_size, 
                                                 kdim=self.image_feat_dim,
                                                 vdim=self.image_feat_dim,
                                                 attn_dim=self.hidden_size,
                                                 intermediate_dim=self.hidden_size,
                                                 output_dim=self.hidden_size,
                                                 num_heads=1,
                                                 attn_drop=params.sa_dropout)
        self.gate_dense = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)

    @property
    def transformer_model(self):
        return self._model[0] if self.params.prefix_only else self._model

    def eval(self):
        self.transformer_model.eval()
        return super().eval()

    def train(self, mode=True):
        self.transformer_model.train(mode)
        return super().train(mode)

    def load_model(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])

    def fuse_image_feature(self, text, image):
        text = self.text_dropout(text)
        image = self.image_dropout(image)

        output, _map = self.selective_attn(query=text, key=image, value=image)

        merge = torch.cat([output, text], dim=-1)
        gate = torch.sigmoid(self.gate_dense(merge))

        return (1 - gate)*text + gate*output

    def encode(self, features, state):
        state = self.transformer_model.encode(features, state)

        fused_feat = self.fuse_image_feature()

        return state

    def decode(self, features, state, mode="infer"):
        batch_size = features['source'].shape[0]
        if mode == 'infer':
            past_key_values = tuple([None] * self.params.num_decoder_layers)
        else:
            past_key_values = self.get_prefix('decode', batch_size)

        logits, state = self.transformer_model.decode(features, state, mode, past_key_values)

        return logits, state

    def forward(self, features, labels):
        state = self.transformer_model.empty_state(features["target"].shape[0],
                                                   labels.device)
        mask = features["target_mask"]
        mask = mask.to(torch.float32)

        state = self.encode(features, state)
        logits, _ = self.decode(features, state, 'train')

        loss = self.criterion(logits, labels)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    def empty_state(self, batch_size, device):
        prefix_key_values = self.get_prefix('decode', batch_size, split_heads=False)
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": prefix_key_values[i][0].to(device),
                    "v": prefix_key_values[i][1].to(device)
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def default_params(name=None):
        params = Transformer.default_params(name)

        params.add_hparam('prefix_only', True)
        params.add_hparam('image_feat_dim', 768)
        params.add_hparam('sa_dropout', 0.1)
        params.add_hparam('image_dropout', 0.1)
        params.add_hparam('text_dropout', 0)

        return params