import torch
import torch.nn as nn
import thumt.modules as modules

from .transformer import Transformer

def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(*new_shape)
    return tensor.permute(0, 2, 1, 3)

class PrefixNet(nn.Module):
    def __init__(self, length, emb_size, hidden_size, dropout=0.0):
        super(PrefixNet, self).__init__()

        self.emb = nn.Parameter(torch.empty([length, emb_size]))
        self.mlp1 = nn.Linear(emb_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, emb_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self):
        return self.dropout(self.mlp2(torch.tanh(self.mlp1(self.emb))))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.xavier_uniform_(self.mlp2.weight)

class PrefixModel(nn.Module):
    def __init__(self, model, prefix_length, hidden_size):
        super(PrefixModel, self).__init__()

        self._num_encoder_layers = model.num_encoder_layers
        self._num_decoder_layers = model.num_decoder_layers
        self._prefix_length = prefix_length
        self._hidden_size = hidden_size

        self.encoder_key_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
            for _ in range(self._num_encoder_layers)
        ])
        self.encoder_value_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
            for _ in range(self._num_encoder_layers)
        ])

        self.decoder_key_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
            for _ in range(self._num_decoder_layers)
        ])
        self.decoder_value_net = nn.ModuleList([
            PrefixNet(self._prefix_length, self._hidden_size, 4 * self._hidden_size)
            for _ in range(self._num_decoder_layers)
        ])

    def forward(self, x):
        past = []

        for i in range(self._num_encoder_layers):
            key = self.encoder_key_net[i].forward(x)
            value = self.encoder_value_net[i].forward(x)

            past.append((key, value))

        return tuple(past)

class PrefixTransformer(modules.Module):
    def __init__(self, model, params, name="prefix_transformer"):
        super(PrefixTransformer, self).__init__(name=name)
        self.params = params
        self._model = [model]

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_encoder_layers = model.num_encoder_layers
        self.num_decoder_layers = model.num_decoder_layers

        self.prefix_net = PrefixModel(model, params.prefix_length, params.hidden_size)

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

        state = self.transformer_model.encode(features, state, past_key_values)

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
        params.add_hparam('prefix_length', 64)
        return params