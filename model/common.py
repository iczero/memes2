import dataclasses
import enum
import json
from typing import Self
import typing

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

def dataclass_from_dict(cls, obj: dict[str, typing.Any]):
    annotations = cls.__annotations__
    validated = {}
    for k, v in obj.items():
        if k not in annotations:
            raise ValueError(f'unexpected key {repr(k)}')
        if k.endswith('_'):
            raise ValueError(f'encountered non-configurable key {repr(k)}')

        wanted_type = annotations[k]
        if wanted_type in (int, float, str, bool):
            validated[k] = wanted_type(v)
        else:
            validated[k] = dataclass_from_dict(wanted_type, v)

    return cls(**validated)

@dataclasses.dataclass
class ModelConfig:
    d_hidden_latent: int
    "Hidden dimension size (aka model dimension) of latent layers"
    d_hidden_bytelevel: int
    "Hidden dimension size of byte-level layers"
    bytes_per_latent: int
    "How many bytes per latent (integer)"
    n_bytelevel_encode_layers: int
    "How many byte-level layers for encode"
    n_bytelevel_decode_layers: int
    "How many byte-level layers for decode"
    n_latent_layers: int
    "How many latent layers to have in the model"
    n_attention_heads: int
    "Number of attention heads"
    hc_expansion: int
    "Hyperconnections expansion rate. Set to 1 to disable hyperconnections"
    activation: str
    "Activation function"
    dtype: str
    "Data type of model"
    d_intermediate_latent: int = 0
    "Size of feedforward inner dimension (usually 4 * d_hidden_latent) of latent layers"
    d_qkv_latent: int = 0
    "Dimension of q/k/v vectors for attention in latent layers"
    d_intermediate_bytelevel: int = 0
    "Intermediate dimension size of byte-level layers"
    d_qkv_bytelevel: int = 0
    "Dimension of q/k/v vectors in byte-level layers"
    qkv_bias: bool = True
    "Whether Q/K/V linear layers in attention should have bias"
    max_seq_len: int = 16384
    "Maximum sequence length, in bytes (used by rope)"
    hc_gating_init: float = 0.01
    "Initialization for static gating parameters for hyperconnections"
    hc_sk_iters: int = 16
    "Sinkhorn-Knopp iterations for hyperconnections residual mixing"
    vocab_size_: int = 256 + len(ControlTokens)
    "Vocabulary size, not read from config"

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return dataclass_from_dict(cls, obj)

    # defaults
    def __post_init__(self):
        assert self.n_bytelevel_encode_layers >= 1
        assert self.n_bytelevel_decode_layers >= 1

        if self.d_intermediate_latent == 0:
            self.d_intermediate_latent = self.d_hidden_latent * 4
        if self.d_qkv_latent == 0:
            self.d_qkv_latent = self.d_hidden_latent // self.n_attention_heads * 2
        if self.d_intermediate_bytelevel == 0:
            self.d_intermediate_bytelevel = self.d_hidden_bytelevel * 4
        if self.d_qkv_bytelevel == 0:
            self.d_qkv_bytelevel = self.d_hidden_bytelevel // self.n_attention_heads * 2

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
        return dataclass_from_dict(cls, obj)

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

@dataclasses.dataclass
class CombinedConfig:
    model_config: ModelConfig
    train_config: TrainConfig

    @classmethod
    def from_dict(cls, obj) -> Self:
        return dataclass_from_dict(cls, obj)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def load_config(path: str):
    with open(path, 'r') as f:
        return CombinedConfig.from_dict(json.load(f))
