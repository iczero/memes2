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

    def __init__(self, d_hidden: int, seq_len: int, base = 50000, device: torch.device | None = None):
        super().__init__()
        assert d_hidden % 2 == 0
        # compute coefficients at full precision
        dim_idx = torch.arange(0, d_hidden, 2, dtype=torch.float64, device=device)
        theta = base ** (-dim_idx / d_hidden)
        base = theta.unsqueeze(0) * torch.arange(seq_len, device=device) \
            .type_as(theta).unsqueeze(-1)
        self.register_buffer('sin_cached', base.sin(), persistent=False)
        self.register_buffer('cos_cached', base.cos(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        # [[ cos(m*t), -sin(m*t) ],  @  [[a],
        #  [ sin(m*t),  cos(m*t) ]]      [b]]
        # a2: cos(m*t) * a + sin(m*t) * -b
        # b2: sin(m*t) * a + cos(m*t) *  b

        dtype = x.dtype
        # in: (batch..., seq, n_embed)
        a = x[..., 0::2]
        b = x[..., 1::2]
        if positions is not None:
            cos_resized = self.cos_cached[positions]
            sin_resized = self.sin_cached[positions]
        else:
            cos_resized = self.cos_cached.narrow(-2, 0, a.shape[-2])
            sin_resized = self.sin_cached.narrow(-2, 0, a.shape[-2])
        a2 = (cos_resized * a + sin_resized * -b).to(dtype=dtype)
        b2 = (sin_resized * a + cos_resized * b).to(dtype=dtype)
        return torch.stack((a2, b2), dim=-1).flatten(-2, -1)

    def d_hidden(self):
        return self.cos_cached.shape[-1] * 2
