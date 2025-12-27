import einops
import torch
import torch.nn.functional as F
from torch import nn

from model.rotary_encoding import RotaryPositionalEncoding

# batch, seq, d_hidden

ROPE_SEQ_LEN = 16384

class FeedforwardGLU(nn.Module):
    def __init__(self, d_hidden: int, d_intermediate: int, activation, bias=True):
        super().__init__()
        self.d_intermediate = d_intermediate
        self.activation = activation
        self.w1 = nn.Linear(d_hidden, d_intermediate * 2, bias=bias)
        self.w2 = nn.Linear(d_intermediate, d_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hidden -> intermediate up-proj, split apart branches
        w_a, w_b = self.w1(x).split(self.d_intermediate, dim=-1)
        # run activation function, then multiply back
        intermediate = w_a * self.activation(w_b)
        # intermediate -> hidden down-proj, then add bypass
        return x + self.w2(intermediate)

class SelfAttention(nn.Module):
    rope: nn.Module | None

    def __init__(self, d_hidden: int, n_attention_heads: int, use_rope: bool | nn.Module = True, bias=True):
        self.d_hidden = d_hidden
        self.n_attention_heads = n_attention_heads

        self.pre_norm = nn.RMSNorm([d_hidden], elementwise_affine=True)
        # W_Q, W_K, W_V combined
        self.w_qkv_linear = nn.Linear(
            d_hidden,
            3 * n_attention_heads * d_hidden,
            bias=bias
        )

        if isinstance(use_rope, nn.Module):
            self.rope = use_rope
        elif use_rope:
            self.rope = RotaryPositionalEncoding(d_hidden, ROPE_SEQ_LEN)
        else:
            self.rope = None

        self.out_linear = nn.Linear(
            n_attention_heads * d_hidden,
            d_hidden,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_merged = einops.rearrange(
            self.w_qkv_linear(x),
            # transpose because SDPA wants heads before seq
            '... seq (qkv heads hidden) -> ... heads seq qkv hidden',
            hidden=self.d_hidden, heads=self.n_attention_heads, qkv=3,
        )
        # unpack from merged
        q = qkv_merged[..., 0, :]
        k = qkv_merged[..., 1, :]
        v = qkv_merged[..., 2, :]
        # q/k/v shape: (batch, heads, seq, d_hidden)

        if self.rope is not None:
            # apply RoPE
            q = self.rope(q)
            k = self.rope(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        # shape: (batch, heads, seq, d_hidden)

        # transpose and concat
        attn_concat = einops.rearrange(attn_out, '... heads seq hidden -> ... seq (heads hidden)')
        merged = self.out_linear(attn_concat)

        # add bypass
        return x + merged

# byte-level to latent attention layer
class ByteToLatentAttention(nn.Module):
    def __init__(self, d_hidden_latent: int, d_hidden_bytelevel: int, bytes_per_latent: int):
        # TODO: both of these need n_attention_heads
        # query: byte -> merge -> w_q -> sdpa (with the larger latent dim)
        self.w_q = nn.Linear(d_hidden_latent, d_hidden_latent)
        # key: up-project byte to latent to match query
        # value: up-project byte to latent for merged out
        self.w_kv_merged = nn.Linear(d_hidden_bytelevel, d_hidden_latent * 2)
        # residual is transformed with linear before being added
        # byte -> merge -> resid_proj -> latent
        # (worst case it becomes zero)
        self.resid_linear = nn.Linear(d_hidden_latent, d_hidden_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# latent to byte-level attention layer
class LatentToByte(nn.Module):
    def __init__(self, d_hidden_latent: int, d_hidden_bytelevel: int, bytes_per_latent: int):
        # TODO: should we do q/k at latent dim or bytelevel dim?
        #       if we want to keep relative compute cost the same, it should be bytelevel dim
        # query: generate bytes_per_latent queries per latent input
        # key: keep dimensions
        self.w_qk_merged = nn.Linear(d_hidden_latent, d_hidden_latent)
        # value: down-project latent to byte
        # residual is transformed with linear before being added
        # latent -> resid_proj -> split -> byte
        self.resid_linear = nn.Linear(d_hidden_latent, d_hidden_latent)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
