'''
visual prefix version 1
part of the codes are from ClipCap: https://github.com/rmokady/CLIP_prefix_caption
'''

import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads
from .transformer import Transformer

class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                nn.Linear(sizes[i], sizes[i + 1], bias=bias)
            )
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class TransformerMapper(nn.Module):
    def __init__(self, params):
        super(TransformerMapper, self).__init__()
        
        self.clip_length = params.clip_length
        self.transformer = Transformer(params.transformer_mapper)
        self.linear = nn.Linear(params.visual_prefix_size, params.clip_length * params.hidden_size)
        self.prefix_const = nn.Parameter(torch.randn(params.visual_prefix_length, params.hidden_size), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x).view(batch_size, self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(batch_size, *self.prefix_const.shape)
        x = torch.cat([x, prefix], dim=1)
        return self.transformer(x)[:, self.clip_length:]

class VisualPrefixTransformer(modules.Module):
    def __init__(self, model, params, name="visual_prefix_transformer"):
        super(VisualPrefixTransformer, self).__init__(name=name)
        self.params = params
        self._model = [model]

        self.hidden_size = params.hidden_size
        self.visual_prefix_size = params.visual_prefix_size
        self.visual_prefix_length = params.visual_prefix_length

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

    def encode(self, features, state):
        visual_prefix = self.visual_prefix_net(features['img_feature']).view(-1, self.visual_prefix_length, self.hidden_size)

        state = self.transformer_model.encode(features, state, prefix=visual_prefix)

        return state

    def decode(self, features, state, mode="infer"):
        logits, state = self.transformer_model.decode(features, state, mode)

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
        return self.transformer_model.empty_state(batch_size, device)

    @staticmethod
    def default_params(name=None):
        params = Transformer.default_params(name)

        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        params.add_hparam('clip_length', 10)

        return params