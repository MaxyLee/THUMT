'''
visual prefix version 4
all vit features
'''

import torch
import torch.nn as nn
import thumt.modules as modules

from .vision_transformer import VisionTransformer, _transform
from .transformer import Transformer

from PIL import Image
from thumt.models.visual_prefix import ClipTransformer


class TransformerMapper(nn.Module):
    def __init__(self, params):
        super(TransformerMapper, self).__init__()
        
        self.transformer = ClipTransformer(params.hidden_size, params.clip_transformer_num_heads, params.clip_transformer_num_layers)
        self.linear = nn.Linear(params.visual_prefix_size, params.hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(x)
        return x

class VisualPrefixTransformer(modules.Module):
    def __init__(self, model, params, name="visual_prefix_transformer"):
        super(VisualPrefixTransformer, self).__init__(name=name)
        self.params = params
        self._model = [model]

        self.hidden_size = params.hidden_size
        self.visual_prefix_size = params.visual_prefix_size

        self.vision_encoder = self.get_vision_encoder()
        self.preprocess = _transform(params.image_resolution)
        self.vision_dtype = torch.float32

        # init visual prefix net
        self.visual_prefix_net = TransformerMapper(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)

    def get_vision_encoder(self):
        vit = VisionTransformer(self.params.image_resolution, self.params.vision_patch_size, self.params.vision_width, self.params.vision_layers, self.params.vision_heads, self.visual_prefix_size)
        if self.params.vit_checkpoint is not None:
            vit.load_state_dict(torch.load(self.params.vit_checkpoint))

        return vit

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
        img_features = self.vision_encoder(features['image'])
        visual_prefix = self.visual_prefix_net(img_features)

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

        params.train_steps = 20000
        params.warmup_steps = 800
        params.learning_rate = 7e-4

        params.add_hparam('mapping_type', 'mlp')
        params.add_hparam('visual_prefix_length', 10)
        params.add_hparam('visual_prefix_size', 512)
        # params.add_hparam('clip_length', 10)
        params.add_hparam('clip_transformer_num_heads', 8)
        params.add_hparam('clip_transformer_num_layers', 8)

        #vision transformer
        params.add_hparam('vision_width', 768)
        params.add_hparam('vision_layers', 12)
        params.add_hparam('vision_patch_size', 32)
        params.add_hparam('vision_grid_size', 7)
        params.add_hparam('vision_heads', 12)
        params.add_hparam('image_resolution', 224)

        return params