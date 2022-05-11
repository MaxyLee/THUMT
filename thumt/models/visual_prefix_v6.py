'''
visual prefix version 6
all vit feature cross-attention with prefix
'''

import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads, PrefixNet
from .transformer import Transformer
from thumt.modules import MultiHeadAttention


class VisualPrefixNet(nn.Module):
    def __init__(self, visual_prefix_length, visual_prefix_size, hidden_size, num_heads, dropout=0.0):
        super(VisualPrefixNet, self).__init__()

        self.emb = nn.Parameter(torch.empty([visual_prefix_length, hidden_size]))
        self.visual_dropout = nn.Dropout(dropout)
        self.visual_mapping = nn.Linear(visual_prefix_size, hidden_size)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads)
        self.mlp1 = nn.Linear(hidden_size, hidden_size*4)
        self.mlp2 = nn.Linear(hidden_size*4, hidden_size)
        # self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x, return_weights=False):
        if return_weights:
            x, weights = self.cross_attn(self.emb.unsqueeze(0), bias=None, memory=self.visual_mapping(x), return_weights=return_weights)
            return self.dropout(self.mlp2(torch.tanh(self.mlp1(x)))), weights
        x = self.cross_attn(self.emb.unsqueeze(0), bias=None, memory=self.visual_mapping(self.visual_dropout(x)))
        return self.mlp2(torch.tanh(self.mlp1(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.xavier_uniform_(self.mlp2.weight)

class VisualPrefixModel(nn.Module):
    def __init__(self, model, params):
        super(VisualPrefixModel, self).__init__()

        self._num_encoder_layers = model.num_encoder_layers
        self._prefix_length = params.visual_prefix_length
        self._prefix_size = params.visual_prefix_size
        self._hidden_size = params.hidden_size
        self._num_heads = params.num_heads

        self.encoder_key_net = nn.ModuleList([
            VisualPrefixNet(self._prefix_length, self._prefix_size, self._hidden_size, self._num_heads)
            for _ in range(self._num_encoder_layers)
        ])
        self.encoder_value_net = nn.ModuleList([
            VisualPrefixNet(self._prefix_length, self._prefix_size, self._hidden_size, self._num_heads)
            for _ in range(self._num_encoder_layers)
        ])

    def forward(self, x):
        past = []

        for i in range(self._num_encoder_layers):
            key = self.encoder_key_net[i].forward(x)
            value = self.encoder_value_net[i].forward(x)

            past.append((key, value))

        return tuple(past)

class VisualPrefixModelv2(nn.Module):
    def __init__(self, model, params):
        super(VisualPrefixModelv2, self).__init__()

        self._num_encoder_layers = model.num_encoder_layers
        self._num_decoder_layers = model.num_decoder_layers
        self._num_visual_layers = params.num_visual_layers
        self._prefix_length = params.visual_prefix_length
        self._prefix_size = params.visual_prefix_size
        self._hidden_size = params.hidden_size
        self._num_heads = params.num_heads
        self._dropout = params.prefix_dropout

        self.encoder_key_net = nn.ModuleList([
            VisualPrefixNet(self._prefix_length, self._prefix_size, self._hidden_size, self._num_heads, self._dropout)
            for _ in range(self._num_visual_layers)
        ] + [
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size, self._dropout)
            for _ in range(self._num_encoder_layers - self._num_visual_layers)
        ])
        self.encoder_value_net = nn.ModuleList([
            VisualPrefixNet(self._prefix_length, self._prefix_size, self._hidden_size, self._num_heads, self._dropout)
            for _ in range(self._num_visual_layers)
        ] + [
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size, self._dropout)
            for _ in range(self._num_encoder_layers - self._num_visual_layers)
        ])

        self.decoder_key_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size, self._dropout)
            for _ in range(self._num_decoder_layers)
        ])
        self.decoder_value_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size, self._dropout)
            for _ in range(self._num_decoder_layers)
        ])

class VisualPrefixTransformer(modules.Module):
    def __init__(self, params, name="visual_prefix_transformer"):
        super(VisualPrefixTransformer, self).__init__(name=name)
        self.params = params

        if params.prefix_only:
            self._model = [Transformer(params)]
        else:
            self._model = Transformer(params)

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_encoder_layers = self.transformer_model.num_encoder_layers
        self.num_decoder_layers = self.transformer_model.num_decoder_layers
        self.visual_prefix_size = params.visual_prefix_size
        self.visual_prefix_length = params.visual_prefix_length        
        self.num_visual_layers = params.num_visual_layers

        self.visual_prefix_net = VisualPrefixModelv2(self.transformer_model, params)

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

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])

    def get_prefix(self, layer, batch_size, split_heads=True, img_feature=None, return_weights=False):
        prefix_state = {
            'prefix': [],
            'attn': []
        }

        if layer == 'encode':
            for i in range(self.num_encoder_layers):
                if i in range(self.num_visual_layers):
                    if return_weights:
                        key, weights_k = self.visual_prefix_net.encoder_key_net[i].forward(img_feature, return_weights=return_weights)
                        value, weights_v = self.visual_prefix_net.encoder_value_net[i].forward(img_feature, return_weights=return_weights)
                        prefix_state['attn'].append({
                            'k': weights_k,
                            'v': weights_v
                        })
                    else:
                        key = self.visual_prefix_net.encoder_key_net[i].forward(img_feature)
                        value = self.visual_prefix_net.encoder_value_net[i].forward(img_feature)
                else:
                    # [prefix_length, hidden_size]
                    key = self.visual_prefix_net.encoder_key_net[i].forward()
                    value = self.visual_prefix_net.encoder_value_net[i].forward()
                    # [batch_size, prefix_length, hidden_size]
                    key = key.unsqueeze(0).repeat([batch_size, 1, 1])
                    value = value.unsqueeze(0).repeat([batch_size, 1, 1])
                if split_heads:
                    # [batch_size, num_heads, prefix_length, head_dim]
                    key = _split_heads(key, self.num_heads, self.head_dim)
                    value = _split_heads(value, self.num_heads, self.head_dim)

                prefix_state['prefix'].append((key, value))
        elif layer == 'decode':
            for i in range(self.num_decoder_layers):
                # [prefix_length, hidden_size]
                key = self.visual_prefix_net.decoder_key_net[i].forward()
                value = self.visual_prefix_net.decoder_value_net[i].forward()
                # [batch_size, prefix_length, hidden_size]
                key = key.unsqueeze(0).repeat([batch_size, 1, 1])
                value = value.unsqueeze(0).repeat([batch_size, 1, 1])
                if split_heads:
                    # [batch_size, num_heads, prefix_length, head_dim]
                    key = _split_heads(key, self.num_heads, self.head_dim)
                    value = _split_heads(value, self.num_heads, self.head_dim)

                prefix_state['prefix'].append((key, value))
        else:
            raise LookupError("Unknown prefix layer: %s" % layer)

        return prefix_state

    def encode(self, features, state, return_weights=False):
        batch_size = features['source'].shape[0]

        prefix_state = self.get_prefix('encode', batch_size, img_feature=features['img_feature'], return_weights=return_weights)
        
        state = self.transformer_model.encode(features, state, past_key_values=prefix_state['prefix'], return_weights=return_weights)

        if return_weights:
            state.update({
                'prefix-attn': prefix_state['attn']
            })

        return state

    def decode(self, features, state, mode="infer", return_weights=False):
        batch_size = features['source'].shape[0]
        if mode == 'infer':
            past_key_values = tuple([None] * self.params.num_decoder_layers)
        else:
            past_key_values = self.get_prefix('decode', batch_size)['prefix']

        logits, state = self.transformer_model.decode(features, state, mode, past_key_values, return_weights)

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
        prefix_key_values = self.get_prefix('decode', batch_size, split_heads=False)['prefix']
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

        params.train_steps = 20000
        params.warmup_steps = 800
        params.learning_rate = 7e-4

        params.add_hparam('prefix_only', True)
        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('prefix_length', 10)
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        params.add_hparam('num_visual_layers', 1)
        params.add_hparam('prefix_dropout', 0.0)
        # params.add_hparam('clip_length', 10)
        params.add_hparam('clip_transformer_num_heads', 8)
        params.add_hparam('clip_transformer_num_layers', 8)

        return params