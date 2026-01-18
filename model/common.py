from pathlib import Path
from collections.abc import Sequence
import dataclasses
import enum
import json
import subprocess
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

    BEGIN_SECTION = 260
    END_SECTION = 261
    KEY = 262
    VALUE = 263

def dataclass_from_dict(cls, obj: dict[str, typing.Any], ignore_nonconfigurable = False):
    annotations = cls.__annotations__
    validated = {}
    for k, v in obj.items():
        if k not in annotations:
            raise ValueError(f'unexpected key {repr(k)}')
        if k.endswith('_'):
            if not ignore_nonconfigurable:
                raise ValueError(f'encountered non-configurable key {repr(k)}')

            continue

        wanted_type = annotations[k]
        if wanted_type in (int, float, str, bool):
            validated[k] = wanted_type(v)
        else:
            validated[k] = dataclass_from_dict(wanted_type, v, ignore_nonconfigurable=ignore_nonconfigurable)

    return cls(**validated)

@dataclasses.dataclass
class ModelConfig:
    d_hidden_latent: int
    "Hidden dimension size (aka model dimension) of latent layers"
    d_hidden_bytelevel: int
    "Hidden dimension size of byte-level layers"
    bytes_per_latent: int
    "How many bytes per latent (integer)"
    bytelevel_attn_window: int
    "How many surrounding bytes each byte should attend to in bytelevel layers"
    # note: layer counts do not include the bytelevel-to-latent and latent-to-bytelevel layer.
    # hence, the total "layer count" is n_bytelevel_encode_layers + 1 + n_latent_layers + 1 + n_bytelevel_decode_layers.
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
    rope_max_seq_len: int
    "Maximum sequence length, in bytes (used by rope)"
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
    hc_gating_init: float = 0.01
    "Initialization for learnable static gating parameters for hyperconnections"
    hc_sk_iters: int = 16
    "Sinkhorn-Knopp iterations for hyperconnections residual mixing"
    vocab_size_: int = 256 + len(ControlTokens)
    "Vocabulary size, not read from config"

    @classmethod
    def from_dict(cls, obj: dict, ignore_nonconfigurable = False) -> Self:
        return dataclass_from_dict(cls, obj, ignore_nonconfigurable=ignore_nonconfigurable)

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
    clip_grad_norm: float
    "Clip gradient norms"
    accumulate_gradients: int
    "Gradient accumulation steps, set to 1 to disable accumulation"
    optimizer: str
    "Optimizer to use"
    full_seq_len: int
    """
    Full length of the model input sequence. This may include multiple sequences
    packed together. Changing this will require re-compiling the model forward
    and backward passes.
    """
    train_max_seq_len: int
    "Max sequence length during training. Longer sequences will be truncated"
    mask_ratio_lower: float
    "Masking ratio lower bound"
    mask_ratio_upper: float
    "Masking ratio upper bound"

    @classmethod
    def from_dict(cls, obj: dict, ignore_nonconfigurable = False) -> Self:
        return dataclass_from_dict(cls, obj, ignore_nonconfigurable=ignore_nonconfigurable)

    def __post_init__(self):
        if self.train_max_seq_len > self.full_seq_len:
            raise ValueError('train_max_seq_len must not be longer than full_seq_len')

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclasses.dataclass
class CombinedConfig:
    model_config: ModelConfig
    train_config: TrainConfig

    @classmethod
    def from_dict(cls, obj: dict, ignore_nonconfigurable = False) -> Self:
        return dataclass_from_dict(cls, obj, ignore_nonconfigurable=ignore_nonconfigurable)

    def __post_init__(self):
        if self.train_config.full_seq_len % self.model_config.bytes_per_latent != 0:
            raise ValueError('full_seq_len must be a multiple of bytes_per_latent')

        if self.train_config.full_seq_len > self.model_config.rope_max_seq_len:
            raise ValueError('rope max_seq_len must be >= full_seq_len')

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def load_config(path: str):
    with open(path, 'r') as f:
        return CombinedConfig.from_dict(json.load(f))

def make_tokens(*parts: bytes | int | ControlTokens | Sequence[bytes | int | ControlTokens]):
    out: list[int] = []
    for part in parts:
        match part:
            case ControlTokens() as token:
                out.append(token.value)
            case int() as token:
                out.append(token)
            case bytes() as bytestr:
                out.extend(bytestr)
            case str():
                raise ValueError('please use bytes')
            case Sequence() as inner_list:
                # there's probably a better way to do this, but whatever
                for part in inner_list:
                    match part:
                        case ControlTokens() as token:
                            out.append(token.value)
                        case int() as token:
                            out.append(token)
                        case bytes() as bytestr:
                            out.extend(bytestr)
                        case str():
                            raise ValueError('please use bytes')

    return torch.tensor(out, device='cpu', dtype=torch.int32)


def tokens_repr(tokens: torch.Tensor | list[int]) -> bytes:
    out = []
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    for value in tokens:
        match value:
            case b if b < 256:
                out.append(bytes([value]))
            case ControlTokens.PAD:
                out.append(b'<|pad|>')
            case ControlTokens.MASK:
                out.append(b'<|mask|>')
            case ControlTokens.START_OF_TEXT:
                out.append(b'<|startoftext|>')
            case ControlTokens.END_OF_TEXT:
                out.append(b'<|endoftext|>')
            case ControlTokens.BEGIN_SECTION:
                out.append(b'<|beginsection|>')
            case ControlTokens.END_SECTION:
                out.append(b'<|endsection|>')
            case ControlTokens.KEY:
                out.append(b'<|key|>')
            case ControlTokens.VALUE:
                out.append(b'<|value|>')
            case other:
                out.append(b'<|?' + str(other).encode('ascii') + b'?|>')

    return b''.join(out)

def padding_needed(current_length: int, chunk_length: int) -> int:
    if current_length % chunk_length == 0:
        return 0

    return chunk_length - (current_length % chunk_length)

def current_git_commit() -> str | None:
    proc = subprocess.run(
        args=['git', 'rev-parse', 'HEAD'],
        cwd=Path(__name__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0 or len(proc.stdout) == 0:
        return None

    return proc.stdout.strip()
