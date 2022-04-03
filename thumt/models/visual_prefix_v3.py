'''
visual prefix version 3
prefix-tuning + visual-prefix
'''

import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads
from .transformer import Transformer
from torch.nn import functional as nnf

from thumt.models.prefix import PrefixModel
from thumt.models.visual_prefix import MLP, TransformerMapper


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

        self.prefix_net = PrefixModel(model, params.prefix_length, params.hidden_size)

        # init visual prefix net
        if params.mapping_type == 'mlp':
            self.visual_prefix_net = MLP((self.visual_prefix_size, 
                                         (self.hidden_size * self.visual_prefix_length) // 2,
                                          self.hidden_size * self.visual_prefix_length))
        elif params.mapping_type == 'transformer':
            self.visual_prefix_net = TransformerMapper(params)
        else:
            raise LookupError("Unknown visual prefix model %s" % params.mapping_type)

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

    def get_prefix(self, layer, batch_size, split_heads=True):
        past_key_values = []

        if layer == 'encode':
            for i in range(self.num_encoder_layers):
                # [prefix_length, hidden_size]
                key = self.prefix_net.encoder_key_net[i].forward()
                value = self.prefix_net.encoder_value_net[i].forward()
                # [batch_size, prefix_length, hidden_size]
                key = key.unsqueeze(0).repeat([batch_size, 1, 1])
                value = value.unsqueeze(0).repeat([batch_size, 1, 1])
                if split_heads:
                    # [batch_size, num_heads, prefix_length, head_dim]
                    key = _split_heads(key, self.num_heads, self.head_dim)
                    value = _split_heads(value, self.num_heads, self.head_dim)

                past_key_values.append((key, value))
        elif layer == 'decode':
            for i in range(self.num_decoder_layers):
                # [prefix_length, hidden_size]
                key = self.prefix_net.decoder_key_net[i].forward()
                value = self.prefix_net.decoder_value_net[i].forward()
                # [batch_size, prefix_length, hidden_size]
                key = key.unsqueeze(0).repeat([batch_size, 1, 1])
                value = value.unsqueeze(0).repeat([batch_size, 1, 1])
                if split_heads:
                    # [batch_size, num_heads, prefix_length, head_dim]
                    key = _split_heads(key, self.num_heads, self.head_dim)
                    value = _split_heads(value, self.num_heads, self.head_dim)

                past_key_values.append((key, value))
        else:
            raise LookupError("Unknown prefix layer: %s" % layer)

        return past_key_values

    def encode(self, features, state):
        batch_size = features['source'].shape[0]
        past_key_values = self.get_prefix('encode', batch_size)
        visual_prefix = self.visual_prefix_net(features['img_feature']).view(-1, self.visual_prefix_length, self.hidden_size)

        state = self.transformer_model.encode(features, state, past_key_values=past_key_values, prefix=visual_prefix)

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

        params.train_steps = 10000
        params.warmup_steps = 400
        params.learning_rate = 7e-4

        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('prefix_length', 64)
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        # params.add_hparam('clip_length', 10)
        params.add_hparam('clip_transformer_num_heads', 8)
        params.add_hparam('clip_transformer_num_layers', 8)

        return params