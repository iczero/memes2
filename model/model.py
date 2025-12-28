import einops
import torch
import torch.nn.functional as F
from torch import nn

from model.rotary_encoding import RotaryPositionalEncoding

# batch, seq, d_hidden

ROPE_SEQ_LEN = 16384

class FeedforwardGLU(nn.Module):
    "Feedforward layer using GLU"
    def __init__(self, d_hidden: int, d_intermediate: int, activation, bias=True):
        super().__init__()
        self.d_intermediate = d_intermediate
        self.activation = activation
        self.pre_norm = nn.RMSNorm([d_hidden], elementwise_affine=True)
        self.w1 = nn.Linear(d_hidden, d_intermediate * 2, bias=bias)
        self.w2 = nn.Linear(d_intermediate, d_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do pre-norm
        normalized = self.pre_norm(x)
        # hidden -> intermediate up-proj, split apart branches
        w_a, w_b = self.w1(normalized).split(self.d_intermediate, dim=-1)
        # run activation function, then multiply back
        intermediate = w_a * self.activation(w_b)
        # intermediate -> hidden down-proj, then add bypass
        return x + self.w2(intermediate)

class SelfAttention(nn.Module):
    "Self attention layer"
    rope: RotaryPositionalEncoding | None

    def __init__(
        self,
        d_hidden: int,
        d_qkv: int,
        n_attention_heads: int,
        use_rope: bool | RotaryPositionalEncoding = True,
        bias=True
    ):
        self.d_hidden = d_hidden
        self.d_qkv = d_qkv
        self.n_attention_heads = n_attention_heads

        self.pre_norm = nn.RMSNorm([d_hidden], elementwise_affine=True)
        # W_Q, W_K, W_V combined
        self.w_qkv_linear = nn.Linear(
            d_hidden,
            3 * n_attention_heads * d_qkv,
            bias=bias
        )

        if isinstance(use_rope, nn.Module):
            self.rope = use_rope
            assert self.rope.d_hidden() == d_qkv, "RoPE d_hidden mismatch"
        elif use_rope:
            self.rope = RotaryPositionalEncoding(d_qkv, ROPE_SEQ_LEN)
        else:
            self.rope = None

        self.w_out = nn.Linear(
            n_attention_heads * d_qkv,
            d_hidden,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do pre-norm
        normalized = self.pre_norm(x)
        qkv_merged = einops.rearrange(
            self.w_qkv_linear(normalized),
            # transpose because SDPA wants heads before seq
            '... seq (split heads d_qkv) -> ... heads seq split d_qkv',
            split=3, heads=self.n_attention_heads,
        )
        # unpack from merged
        q = qkv_merged[..., 0, :]
        k = qkv_merged[..., 1, :]
        v = qkv_merged[..., 2, :]
        # q/k/v shape: (batch, heads, seq, d_qkv)

        if self.rope is not None:
            # apply RoPE
            q = self.rope(q)
            k = self.rope(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        # shape: (batch, heads, seq, d_qkv)

        # transpose and concat
        attn_concat = einops.rearrange(
            attn_out,
            '... heads seq d_qkv -> ... seq (heads d_qkv)'
        )
        out = self.w_out(attn_concat)

        # add bypass
        return x + out

def reshape_last(x: torch.Tensor, new_dim: int) -> torch.Tensor:
    "Reshape the last dimension of a tensor"
    return x.flatten(-2, -1).unflatten(-1, (-1, new_dim))

# byte-level to latent attention layer
class ByteToLatentAttention(nn.Module):
    def __init__(
        self,
        d_hidden_latent: int,
        d_hidden_bytelevel: int,
        d_qkv_bytelevel: int,
        bytes_per_latent: int,
        n_attention_heads: int,
        rope_bytelevel: RotaryPositionalEncoding,
        bias=True,
    ):
        self.d_hidden_latent = d_hidden_latent
        self.d_hidden_bytelevel = d_hidden_bytelevel
        self.d_qkv_bytelevel = d_qkv_bytelevel
        self.bytes_per_latent = bytes_per_latent
        self.n_attention_heads = n_attention_heads

        self.pre_norm = nn.RMSNorm([d_hidden_bytelevel], elementwise_affine=True)

        # query: byte -> concat -> w_q -> sdpa
        self.w_q = nn.Linear(
            bytes_per_latent * d_hidden_bytelevel,
            n_attention_heads * d_qkv_bytelevel,
            bias=bias,
        )
        # key/value: input to d_qkv
        self.w_kv_merged = nn.Linear(
            d_hidden_bytelevel,
            2 * n_attention_heads * d_qkv_bytelevel,
            bias=bias,
        )

        self.rope_bytelevel = rope_bytelevel
        assert self.rope_bytelevel.d_hidden == d_qkv_bytelevel

        self.w_out = nn.Linear(n_attention_heads * d_qkv_bytelevel, d_hidden_latent)

        # bypass is transformed with linear before being added
        # byte -> merge -> bypass_proj -> latent
        # (worst case it becomes zero)
        self.bypass_linear = nn.Linear(bytes_per_latent * d_hidden_bytelevel, d_hidden_latent, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(x)

        # form byte groups
        bytelevel_merged = normalized.flatten(-2, -1) \
            .unflatten(-1, (-1, self.d_hidden_bytelevel * self.bytes_per_latent))

        q = einops.rearrange(
            self.w_q(bytelevel_merged),
            '... seq (heads d_qkv) -> ... heads seq d_qkv',
            heads=self.n_attention_heads
        )

        kv_merged = einops.rearrange(
            self.w_kv_merged(normalized),
            '... seq (split heads d_qkv) -> ... heads seq split d_qkv',
            split=2, heads=self.n_attention_heads,
        )
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]

        q = self.rope_bytelevel(q, stride=self.bytes_per_latent)
        k = self.rope_bytelevel(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_concat = einops.rearrange(
            attn_out,
            '... heads seq d_qkv -> ... seq (heads d_qkv)',
        )
        out = self.w_out(attn_concat)

        bytelevel_merged_bypass = x.flatten(-2, -1) \
            .unflatten(-1, (-1, self.d_hidden_bytelevel * self.bytes_per_latent))
        bypass = self.bypass_linear(bytelevel_merged_bypass)

        return out + bypass

class ResamplingAttention(nn.Module):
    def __init__(
        self,
        d_hidden_in: int,
        d_hidden_out: int,
        d_qkv: int,
        d_hidden_in_reshape: int,
        n_attention_heads: int,
        rope: RotaryPositionalEncoding,
        rope_stride_q: int,
        rope_stride_k: int,
        bias=True,
    ):
        self.d_hidden_in = d_hidden_in
        self.d_hidden_out = d_hidden_out
        self.d_qkv = d_qkv
        self.d_hidden_in_reshape = d_hidden_in_reshape
        self.n_attention_heads = n_attention_heads
        self.rope_stride_q = rope_stride_q
        self.rope_stride_k = rope_stride_k

        self.pre_norm = nn.RMSNorm([d_hidden_in], elementwise_affine=True)

        # query: byte -> concat -> w_q -> sdpa
        self.w_q = nn.Linear(
            d_hidden_in_reshape,
            n_attention_heads * d_qkv,
            bias=bias,
        )
        # key/value: input to d_qkv
        self.w_kv_merged = nn.Linear(
            d_hidden_in,
            2 * n_attention_heads * d_qkv,
            bias=bias,
        )

        self.rope = rope
        assert self.rope.d_hidden == d_qkv

        self.w_out = nn.Linear(n_attention_heads * d_qkv, d_hidden_out)

        # bypass is transformed with linear before being added
        # byte -> merge -> bypass_proj -> latent
        # (worst case it becomes zero)
        self.bypass_linear = nn.Linear(d_hidden_in_reshape, d_hidden_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(x)

        # form byte groups
        hidden_reshaped = reshape_last(normalized, self.d_hidden_in_reshape)

        q = einops.rearrange(
            self.w_q(hidden_reshaped),
            '... seq (heads d_qkv) -> ... heads seq d_qkv',
            heads=self.n_attention_heads
        )

        kv_merged = einops.rearrange(
            self.w_kv_merged(normalized),
            '... seq (split heads d_qkv) -> ... heads seq split d_qkv',
            split=2, heads=self.n_attention_heads,
        )
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]

        q = self.rope(q, stride=self.rope_stride_q)
        k = self.rope(k, stride=self.rope_stride_k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_concat = einops.rearrange(
            attn_out,
            '... heads seq d_qkv -> ... seq (heads d_qkv)',
        )
        out = self.w_out(attn_concat)

        bypass = self.bypass_linear(reshape_last(x, self.d_hidden_in_reshape))

        return out + bypass

# latent to byte-level attention layer
class LatentToByteAttention(nn.Module):
    def __init__(
        self,
        d_hidden_latent: int,
        d_hidden_bytelevel: int,
        bytes_per_latent: int,
        n_attention_heads: int,
    ):
        # TODO: should we do q/k at latent dim or bytelevel dim?
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
