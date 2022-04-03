'''
visual prefix version 2
vision encoder is tunable
'''

import clip
import torch
import torch.nn as nn
import thumt.modules as modules

from .prefix import _split_heads
from .transformer import Transformer
from torch.nn import functional as nnf

from thumt.models.visual_prefix import MLP, TransformerMapper

def get_vision_encoder(name='ViT-B/32', device='cpu'):
    model, preprocess = clip.load(name, device=device, jit=False)
    return model.visual, preprocess, model.dtype

class VisualPrefixTransformer(modules.Module):
    def __init__(self, model, params, name="visual_prefix_transformer"):
        super(VisualPrefixTransformer, self).__init__(name=name)
        self.params = params
        self._model = [model]

        self.hidden_size = params.hidden_size
        self.visual_prefix_size = params.visual_prefix_size
        self.visual_prefix_length = params.visual_prefix_length

        self.vision_encoder, self.preprocess, self.vision_dtype = get_vision_encoder(device=params.device)

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
        img_features = self.vision_encoder(features['image']).float()
        visual_prefix = self.visual_prefix_net(img_features).view(-1, self.visual_prefix_length, self.hidden_size)

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

        params.train_steps = 10000
        params.warmup_steps = 400
        params.learning_rate = 7e-4

        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        # params.add_hparam('clip_length', 10)
        params.add_hparam('clip_transformer_num_heads', 8)
        params.add_hparam('clip_transformer_num_layers', 8)

        return params