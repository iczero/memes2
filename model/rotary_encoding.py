import torch
from torch import Tensor, nn

# m = q, k index
# d = q, k dimension
# i = arange(1, d // 2)
# t = base ** (-2 * i / d)

# (batch, n_heads, seq, qkv_dim)

class RotaryPositionalEncoding(nn.Module):
    "Implements rotary positional encoding (RoPE)"
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, hidden_dim: int, seq_len: int, base = 10000):
        super().__init__()
        assert hidden_dim % 2 == 0
        # compute coefficients at full precision
        dim_idx = torch.arange(0, hidden_dim, 2, dtype=torch.float64)
        theta = base ** (-dim_idx / hidden_dim)
        base = theta.unsqueeze(0) * torch.arange(seq_len).type_as(theta).unsqueeze(-1)
        self.register_buffer('sin_cached', base.sin(), persistent=False)
        self.register_buffer('cos_cached', base.cos(), persistent=False)

    def forward(self, x, stride = 1):
        # [[ cos(m*t), -sin(m*t) ],  @  [[a],
        #  [ sin(m*t),  cos(m*t) ]]      [b]]
        # a2: cos(m*t) * a + sin(m*t) * -b
        # b2: sin(m*t) * a + cos(m*t) *  b

        # strided rope is an experiment
        # byte-level: 0 1 2 3   4 5 6 7
        #     latent:     2         6
        stride_start = stride // 2
        cos_cached = self.cos_cached[stride_start::stride, :]
        sin_cached = self.sin_cached[stride_start::stride, :]

        # in: (batch..., seq, n_embed)
        a = x[..., 0::2]
        b = x[..., 1::2]
        cos_resized = cos_cached.narrow(-2, 0, a.shape[-2])
        sin_resized = sin_cached.narrow(-2, 0, a.shape[-2])
        a2 = cos_resized * a + sin_resized * -b
        b2 = sin_resized * a + cos_resized * b
        return torch.stack((a2, b2), dim=-1).flatten(-2, -1)
