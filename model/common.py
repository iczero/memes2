import dataclasses
import enum
import json
import struct
from typing import Self

import torch
from torch import nn

@enum.verify(enum.CONTINUOUS)
class ControlTokens(enum.IntEnum):
    # 0 to 255 are for byte values
    PAD = 256
    "Padding token used for input"
    MASK = 257
    "Mask token used for input"
    START_OF_TEXT = 258
    "Start of text token"
    END_OF_TEXT = 259
    "End of text token"

    UNUSED_4 = 260
    UNUSED_5 = 261
    UNUSED_6 = 262
    UNUSED_7 = 263

@dataclasses.dataclass
class ModelConfig:
    d_hidden_latent: int
    "Hidden dimension size (aka model dimension) of latent layers"
    d_intermediate_latent: int
    "Size of feedforward inner dimension (usually 4 * d_hidden_latent) of latent layers"
    d_hidden_bytelevel: int
    "Hidden dimension size of byte-level layers"
    d_intermediate_bytelevel: int
    "Intermediate dimension size of byte-level layers"
    bytes_per_latent: int
    "How many bytes per latent (integer)"
    n_bytelevel_layers: int
    "How many byte-level layers to have per encode and decode"
    n_latent_layers: int
    "How many latent layers to have in the model"
    n_attention_heads: int
    "Number of attention heads"
    activation: str
    "Activation function"
    dtype: str
    "Data type of model"
    qkv_bias: bool
    "Whether Q/K/V linear layers in attention should have bias"
    vocab_size: int = 256 + len(ControlTokens)
    "Vocabulary size, not read from config"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            d_hidden_latent=int(obj['d_hidden_latent']),
            d_intermediate_latent=int(obj['d_intermediate_latent']),
            d_hidden_bytelevel=int(obj['d_hidden_bytelevel']),
            d_intermediate_bytelevel=int(obj['d_intermediate_bytelevel']),
            bytes_per_latent=int(obj['bytes_per_latent']),
            n_bytelevel_layers=int(obj['n_bytelevel_layers']),
            n_latent_layers=int(obj['n_latent_layers']),
            n_attention_heads=int(obj['n_attention_heads']),
            activation=str(obj['activation']),
            dtype=str(obj['dtype']),
            qkv_bias=bool(obj['qkv_bias']),
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def get_dtype(self):
        return {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
        }[self.dtype]

    def get_activation(self):
        if self.activation == 'gelu':
            return nn.GELU()
        if self.activation == 'silu':
            return nn.SiLU()
        if self.activation == 'relu':
            return nn.ReLU()
        if self.activation == 'leakyrelu':
            return nn.LeakyReLU()

        raise RuntimeError('unknown activation')

@dataclasses.dataclass
class TrainConfig:
    lr: float
    "Learning rate"
    weight_decay: float
    "Weight decay"
    batch_size: int
    "Batch size"
    accumulate_gradients: int
    "How many batches to run before running the optimizer step"
    optimizer: str
    "Optimizer to use"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls(
            lr=float(obj['lr']),
            weight_decay=float(obj['weight_decay']),
            batch_size=int(obj['batch_size']),
            optimizer=str(obj['optimizer']),
            accumulate_gradients=int(obj['accumulate_gradients']),
        )

    def to_dict(self):
        return dataclasses.asdict(self)

    def make_param_groups(self, named_parameters):
        exclude_wd = []
        default = []
        for name, param in named_parameters:
            if len(param.shape) < 2 \
                    or name.endswith('.bias') \
                    or name.endswith('.positional_encodings'):
                exclude_wd.append(param)
            else:
                default.append(param)

        return [
            { 'params': exclude_wd, 'weight_decay': 0.0 },
            { 'params': default },
        ]

    def make_optimizer(self, named_parameters, allow_fused=False):
        groups = self.make_param_groups(named_parameters)
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(
                groups,
                self.lr,
                weight_decay=self.weight_decay,
                fused=allow_fused,
            )

        if self.optimizer == 'SGD':
            return torch.optim.SGD(
                groups,
                self.lr,
                weight_decay=self.weight_decay,
            )

        raise RuntimeError('unknown optimizer ' + self.optimizer)
