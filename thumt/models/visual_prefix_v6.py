'''
visual prefix version 6
all vit feature cross-attention with prefix
'''

from turtle import forward
import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads
from .transformer import Transformer
from thumt.modules import MultiHeadAttention

class VisualPrefixNet(nn.Module):
    def __init__(self, visual_prefix_length, visual_prefix_size, hidden_size, num_heads):
        super(VisualPrefixNet, self).__init__()

        self.emb = nn.Parameter(torch.empty([visual_prefix_length, hidden_size]))
        self.visual_mapping = nn.Linear(visual_prefix_size, hidden_size)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads)

        self.reset_parameters()

    def forward(self, x):
        return self.cross_attn(self.emb, bias=None, memory=self.visual_mapping(x))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)

class VisualPrefixModel(nn.Module):
    def __init__(self, model, params):
        super(VisualPrefixModel, self).__init__()

        self._num_encoder_layers = model.num_encoder_layers
        self._num_decoder_layers = model.num_decoder_layers
        self._prefix_length = params.visual_prefix_length
        self._hidden_size = params.hidden_size
        self._num_heads = params.num_heads

        self.encoder_key_net = nn.ModuleList([
            VisualPrefixNet(params)
            for _ in range(self._num_encoder_layers)
        ])
        self.encoder_value_net = nn.ModuleList([
            VisualPrefixNet(params)
            for _ in range(self._num_encoder_layers)
        ])

        # self.decoder_key_net = nn.ModuleList([
        #     PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
        #     for _ in range(self._num_decoder_layers)
        # ])
        # self.decoder_value_net = nn.ModuleList([
        #     PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
        #     for _ in range(self._num_decoder_layers)
        # ])

    def forward(self, x):
        past = []

        for i in range(self._num_encoder_layers):
            key = self.encoder_key_net[i].forward(x)
            value = self.encoder_value_net[i].forward(x)

            past.append((key, value))

        return tuple(past)


class VisualPrefixTransformer(modules.Module):
    def __init__(self, model, params, name="visual_prefix_transformer"):
        super(VisualPrefixTransformer, self).__init__(name=name)
        self.params = params
        self._model = [model]

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_encoder_layers = model.num_encoder_layers
        self.num_decoder_layers = model.num_decoder_layers
        self.visual_prefix_size = params.visual_prefix_size
        self.visual_prefix_length = params.visual_prefix_length

        # init visual mapping network
        # if params.mapping_type == 'mlp':
        #     self.visual_mapping = MLP((self.visual_prefix_size, 
        #                                  (self.hidden_size * self.visual_prefix_length) // 2,
        #                                   self.hidden_size * self.visual_prefix_length))
        # elif params.mapping_type == 'transformer':
        #     self.visual_mapping = TransformerMapper(params)
        # else:
        #     raise LookupError("Unknown visual prefix model %s" % params.mapping_type)
        

        self.visual_prefix_net = VisualPrefixModel(model, params)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)

    @property
    def transformer_model(self):
        return self._model[0]

    def eval(self):
        self.transformer_model.eval()
        return super().eval()

    def train(self, mode=True):
        self.transformer_model.train(mode)
        return super().train(mode)

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])

    def get_prefix(self, img_feature, split_heads=True):
        past_key_values = []

        for i in range(self.num_encoder_layers):
            key = self.visual_prefix_net.encoder_key_net[i].forward(img_feature)
            value = self.visual_prefix_net.encoder_value_net[i].forward(img_feature)
            if split_heads:
                # [batch_size, num_heads, prefix_length, head_dim]
                key = _split_heads(key, self.num_heads, self.head_dim)
                value = _split_heads(value, self.num_heads, self.head_dim)

            past_key_values.append((key, value))

        return past_key_values

    def encode(self, features, state):
        past_key_values = self.get_prefix(features['img_feature'])
        
        state = self.transformer_model.encode(features, state, past_key_values=past_key_values)

        return state

    def decode(self, features, state, mode="infer"):
        return self.transformer_model.decode(features, state, mode)

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
        return self.transformer_model.empty_state(batch_size, device)

    @staticmethod
    def default_params(name=None):
        params = Transformer.default_params(name)

        params.train_steps = 20000
        params.warmup_steps = 800
        params.learning_rate = 7e-4

        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('prefix_length', 10)
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        # params.add_hparam('clip_length', 10)
        params.add_hparam('clip_transformer_num_heads', 8)
        params.add_hparam('clip_transformer_num_layers', 8)

        return params