import einops
import torch
import torch.nn.attention.flex_attention as fa
import torch.nn.functional as F
from torch import nn

from model.common import ModelConfig
from model.masking import create_fa_doc_mask, lengths_to_bytelevel, lengths_to_doc_ids, lengths_to_positions
from model.rotary_encoding import RotaryPositionalEncoding

class FeedforwardGLU(nn.Module):
    "Feedforward layer using GLU"
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
        # intermediate -> hidden down-proj
        return self.w2(intermediate)

class SelfAttention(nn.Module):
    "Self attention layer"
    rope: RotaryPositionalEncoding | None

    def __init__(
        self,
        d_hidden: int,
        d_qkv: int,
        n_attention_heads: int,
        rope: RotaryPositionalEncoding | None = None,
        bias=True
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_qkv = d_qkv
        self.n_attention_heads = n_attention_heads

        # W_Q, W_K, W_V combined
        self.w_qkv_linear = nn.Linear(
            d_hidden,
            3 * n_attention_heads * d_qkv,
            bias=bias
        )

        self.rope = rope
        if rope is not None:
            assert rope.d_hidden() == d_qkv, "RoPE d_hidden mismatch"

        self.w_out = nn.Linear(
            n_attention_heads * d_qkv,
            d_hidden,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor | None = None,
        attn_block_mask = None,
    ) -> torch.Tensor:
        # do pre-norm
        qkv_merged = einops.rearrange(
            self.w_qkv_linear(x),
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
            q = self.rope(q, rope_pos)
            k = self.rope(k, rope_pos)

        attn_out = fa.flex_attention(
            q, k, v,
            block_mask=attn_block_mask,
        )
        # shape: (batch, heads, seq, d_qkv)

        # transpose and concat
        attn_concat = einops.rearrange(
            attn_out,
            '... heads seq d_qkv -> ... seq (heads d_qkv)'
        )
        return self.w_out(attn_concat)

class SinkhornKnopp(nn.Module):
    "Sinkhorn-Knopp operator as defined in mHC paper"
    def __init__(self, n_iter: int):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.exp()
        for _i in range(0, self.n_iter):
            x = F.normalize(x, p=1, dim=-1)
            x = F.normalize(x, p=1, dim=-2)

        return x

def reshape_last(x: torch.Tensor, new_dim: int) -> torch.Tensor:
    "Reshape the last dimension of a tensor"
    return x.flatten(-2, -1).unflatten(-1, (-1, new_dim))

class ByteToLatentAttention(nn.Module):
    def __init__(
        self,
        d_hidden_bytelevel: int,
        d_hidden_latent: int,
        d_qkv_bytelevel: int,
        bytes_per_latent: int,
        n_attention_heads: int,
        rope_bytelevel: RotaryPositionalEncoding,
        bias=True,
    ):
        super().__init__()
        self.inner = ResamplingAttention(
            d_hidden_bytelevel,
            d_hidden_latent,
            d_qkv_bytelevel,
            bytes_per_latent * d_hidden_bytelevel,
            n_attention_heads,
            rope_bytelevel,
            bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_pos_bytelevel: torch.Tensor,
        rope_pos_latent: torch.Tensor,
        attn_block_mask,
    ):
        return self.inner(x, rope_pos_latent, rope_pos_bytelevel, attn_block_mask)

class LatentToByteAttention(nn.Module):
    def __init__(
        self,
        d_hidden_latent: int,
        d_hidden_bytelevel: int,
        d_qkv_latent: int,
        bytes_per_latent: int,
        n_attention_heads: int,
        rope_latent: RotaryPositionalEncoding,
        bias=True,
    ):
        super().__init__()
        self.inner = ResamplingAttention(
            d_hidden_latent,
            d_hidden_bytelevel,
            d_qkv_latent,
            d_hidden_latent // bytes_per_latent,
            n_attention_heads,
            rope_latent,
            bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_pos_latent: torch.Tensor,
        rope_pos_bytelevel: torch.Tensor,
        attn_block_mask,
    ):
        return self.inner(x, rope_pos_bytelevel, rope_pos_latent, attn_block_mask)

class ResamplingAttention(nn.Module):
    def __init__(
        self,
        d_hidden_in: int,
        d_hidden_out: int,
        d_qkv: int,
        d_hidden_in_reshape: int,
        n_attention_heads: int,
        rope: RotaryPositionalEncoding,
        bias=True,
    ):
        super().__init__()
        self.d_hidden_in = d_hidden_in
        self.d_hidden_out = d_hidden_out
        self.d_qkv = d_qkv
        self.d_hidden_in_reshape = d_hidden_in_reshape
        self.n_attention_heads = n_attention_heads

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
        assert self.rope.d_hidden() == d_qkv

        self.w_out = nn.Linear(n_attention_heads * d_qkv, d_hidden_out)

        # bypass is transformed with linear before being added
        # byte -> merge -> bypass_proj -> latent
        # (worst case it becomes zero)
        self.bypass_linear = nn.Linear(d_hidden_in_reshape, d_hidden_out, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_pos_q: torch.Tensor,
        rope_pos_k: torch.Tensor,
        attn_block_mask,
    ) -> torch.Tensor:
        normalized = self.pre_norm(x)

        # group for latent, or split when going back to bytelevel
        hidden_reshaped = reshape_last(normalized, self.d_hidden_in_reshape)

        # query from hidden_reshaped
        q = einops.rearrange(
            self.w_q(hidden_reshaped),
            '... seq (heads d_qkv) -> ... heads seq d_qkv',
            heads=self.n_attention_heads
        )

        # key/value from previous sequence without concat/split
        kv_merged = einops.rearrange(
            self.w_kv_merged(normalized),
            '... seq (split heads d_qkv) -> ... heads seq split d_qkv',
            split=2, heads=self.n_attention_heads,
        )
        k = kv_merged[..., 0, :]
        v = kv_merged[..., 1, :]

        q = self.rope(q, positions=rope_pos_q)
        k = self.rope(k, positions=rope_pos_k)

        attn_out = fa.flex_attention(
            q, k, v,
            block_mask=attn_block_mask,
        )
        attn_concat = einops.rearrange(
            attn_out,
            '... heads seq d_qkv -> ... seq (heads d_qkv)',
        )
        out = self.w_out(attn_concat)

        # linear transform for skip connection
        bypass = self.bypass_linear(reshape_last(x, self.d_hidden_in_reshape))

        return out + bypass

class HyperconnectionParams(nn.Module):
    def __init__(self, d_hidden: int, hc_expansion: int, hc_gating_init: float, sk_iters: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.hc_expansion = hc_expansion

        self.norm = nn.RMSNorm([self.hc_expansion * self.d_hidden])
        self.dynamic_proj = nn.Linear(
            d_hidden * hc_expansion,
            hc_expansion * (2 + hc_expansion),
            bias=False,
        )
        self.gating_pre_post = nn.Parameter(torch.tensor([hc_gating_init] * 2))
        self.gating_res = nn.Parameter(torch.tensor([hc_gating_init]))
        self.static_mapping = nn.Parameter(torch.zeros((2 + hc_expansion, hc_expansion)))
        self.sk = SinkhornKnopp(sk_iters)

    def _combine_gating(self) -> torch.Tensor:
        # combine gating into the form [[pre_gate], [post_gate], [res_gate], [res_gate], ...]
        return torch.concat([
            self.gating_pre_post,
            self.gating_res.expand(self.hc_expansion),
        ], dim=-1).unsqueeze(-1)

    # returns: H_pre, H_post, H_res
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        # compupte dynamic mappings for H_pre, H_post, H_res
        dynamic_mapping = self.dynamic_proj(x_norm).unflatten(-1, (2 + self.hc_expansion, -1))
        # apply learned gate to dynamic mappings
        dyn_gated = dynamic_mapping * self._combine_gating()
        # add static mappings
        combined = dyn_gated + self.static_mapping
        # split into H_pre, H_post, H_res
        h_pre, h_post, h_res = combined.split([1, 1, self.hc_expansion], dim=-2)
        h_pre = F.sigmoid(h_pre)
        h_post = 2 * F.sigmoid(h_post)
        h_res = self.sk(h_res)
        return h_pre, h_post, h_res

def hc_apply_pre(x_wide: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    # reshape residual stream from flat to 2d form
    x_split = x_wide.unflatten(-1, (h_pre.shape[-1], -1))
    # reweight by h_pre
    x_reweight = x_split * h_pre.unsqueeze(-1)
    # sum to obtain layer input
    return x_reweight.sum(-2)

def hc_apply_post(x_wide: torch.Tensor, layer_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    # expand layer output and reweight by h_post
    layer_out_expanded = layer_out.unsqueeze(-2) * h_post.unsqueeze(-1)
    # convert skip from concated to 2d
    x_split = x_wide.unflatten(-1, (h_res.shape[-1], -1))
    # apply h_res to skip connection
    x_reweight = h_res @ x_split
    # merge layer out back into residual stream
    merged = layer_out_expanded + x_reweight
    # re-flatten residual stream
    return merged.flatten(-2, -1)

class HcExpand(nn.Module):
    def __init__(self, hc_expansion: int):
        self.hc_expansion = hc_expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # repeat input by hc_expansion times in last dimension
        repeats = [1] * len(x.shape)
        repeats[-1] = self.hc_expansion
        return x.repeat(repeats)

class HcMerge(nn.Module):
    def __init__(self, hc_expansion: int):
        self.hc_expansion = hc_expansion
        self.merge_weights = nn.Parameter(torch.ones(hc_expansion))

    def forward(self, x_wide: torch.Tensor) -> torch.Tensor:
        # reshape to 2d form
        x_split = x_wide.unflatten(-1, (self.hc_expansion, -1))
        # apply weights
        x_reweight = x_split * self.merge_weights.unsqueeze(-1)
        # sum
        return x_reweight.sum(-2)

class ByteLevelBlock(nn.Module):
    def __init__(self, config: ModelConfig, rope_bytelevel: RotaryPositionalEncoding):
        super().__init__()
        self.attn_norm = nn.RMSNorm([config.d_hidden_bytelevel], elementwise_affine=True)
        self.attention = SelfAttention(
            config.d_hidden_bytelevel,
            config.d_qkv_bytelevel,
            config.n_attention_heads,
            rope_bytelevel,
            config.qkv_bias,
        )
        self.ff_norm = nn.RMSNorm([config.d_hidden_bytelevel], elementwise_affine=True)
        self.feedforward = FeedforwardGLU(
            config.d_hidden_bytelevel,
            config.d_intermediate_bytelevel,
            config.get_activation(),
        )

    def forward(self, x: torch.Tensor, rope_pos_bytelevel: torch.Tensor, attn_mask: fa.BlockMask):
        x2 = self.attn_norm(x)
        x2 = self.attention(x2, rope_pos_bytelevel, attn_mask)
        x = x2 + x
        x2 = self.ff_norm(x)
        x2 = self.feedforward(x2)
        x = x2 + x
        return x

class QuestionableTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rope_bytelevel = RotaryPositionalEncoding(
            config.d_qkv_bytelevel,
            config.max_seq_len,
        )
        self.rope_latent = RotaryPositionalEncoding(
            config.d_qkv_latent,
            config.max_seq_len,
        )

        self.embedding = nn.Embedding(config.vocab_size_, config.d_hidden_bytelevel)

        self.encode_layers = nn.ModuleList()
        for _ in range(0, config.n_bytelevel_encode_layers):
            self.encode_layers.append(SelfAttention(
                config.d_hidden_bytelevel,
                config.d_qkv_bytelevel,
                config.n_attention_heads,
                self.rope_bytelevel,
                config.qkv_bias,
            ))
            self.encode_layers.append(FeedforwardGLU(
                config.d_hidden_bytelevel,
                config.d_intermediate_bytelevel,
                config.get_activation()
            ))

        self.intermediate_layers = nn.ModuleList()
        for i in range(0, config.n_latent_layers):
            if i == 0:
                self.intermediate_layers.append(ByteToLatentAttention(
                    config.d_hidden_bytelevel,
                    config.d_hidden_latent,
                    config.d_qkv_bytelevel,
                    config.bytes_per_latent,
                    config.n_attention_heads,
                    self.rope_bytelevel,
                    config.qkv_bias
                ))
            else:
                self.intermediate_layers.append(SelfAttention(
                    config.d_hidden_latent,
                    config.d_qkv_latent,
                    config.n_attention_heads,
                    self.rope_latent,
                    config.qkv_bias
                ))

            self.intermediate_layers.append(FeedforwardGLU(
                config.d_hidden_latent,
                config.d_intermediate_latent,
                config.get_activation(),
            ))

        self.decode_layers = nn.ModuleList()
        for i in range(0, config.n_bytelevel_decode_layers):
            if i == 0:
                self.decode_layers.append(LatentToByteAttention(
                    config.d_hidden_latent,
                    config.d_hidden_bytelevel,
                    config.d_qkv_latent,
                    config.bytes_per_latent,
                    config.n_attention_heads,
                    self.rope_latent,
                    config.qkv_bias,
                ))
            else:
                self.decode_layers.append(SelfAttention(
                    config.d_hidden_bytelevel,
                    config.d_qkv_bytelevel,
                    config.n_attention_heads,
                    self.rope_bytelevel,
                    config.qkv_bias,
                ))

            self.decode_layers.append(FeedforwardGLU(
                config.d_hidden_bytelevel,
                config.d_intermediate_bytelevel,
                config.get_activation(),
            ))

        self.lm_head = nn.Linear(config.d_hidden_bytelevel, config.vocab_size_, bias=False)

    # note: seq_latent_lengths is latent length (i.e. byte_length // bytes_per_latent)
    def forward(self, x: torch.Tensor, seq_lengths_latent: torch.Tensor) -> torch.Tensor:
        bytes_per_latent = self.config.bytes_per_latent
        seq_lengths_bytelevel = lengths_to_bytelevel(seq_lengths_latent, bytes_per_latent)
        seq_positions_latent = lengths_to_positions(seq_lengths_latent)
        seq_positions_bytelevel = lengths_to_positions(seq_lengths_bytelevel)
        transition_stride_start = (bytes_per_latent - 1) // 2
        seq_positions_bytelevel_transition = seq_positions_bytelevel[transition_stride_start::bytes_per_latent]
        doc_ids_latent = lengths_to_doc_ids(seq_lengths_latent)
        attn_mask_latent = create_fa_doc_mask(doc_ids_latent)
        attn_mask_bytelevel = create_fa_doc_mask(
            doc_ids_latent,
            bytes_per_latent,
            q_is_bytes=True,
            kv_is_bytes=True,
            # limit attention to +/- (bytelevel_attn_window / 2) surrounding bytes
            additional_mask=lambda q_idx, kv_idx: (q_idx - kv_idx).abs() <= self.config.bytelevel_attn_window // 2,
        )
        attn_mask_bytelevel_to_latent = create_fa_doc_mask(
            doc_ids_latent,
            bytes_per_latent,
            q_is_bytes=False,
            kv_is_bytes=True,
            # latent queries should attend to surrounding bytelevel_attn_window bytes (as groups of bytes_per_latent)
            additional_mask=lambda q_idx, kv_idx: \
                (q_idx - kv_idx // bytes_per_latent).abs() \
                    <= self.config.bytelevel_attn_window // (2 * bytes_per_latent),
        )
        attn_mask_latent_to_bytelevel = create_fa_doc_mask(
            doc_ids_latent,
            bytes_per_latent,
            q_is_bytes=True,
            kv_is_bytes=False,
            # bytelevel queries should attend to surrounding (bytelevel_attn_window / 4) latents
            additional_mask=lambda q_idx, kv_idx: \
                (q_idx // bytes_per_latent - kv_idx).abs() \
                    <= self.config.bytelevel_attn_window // (2 * bytes_per_latent),
        )
        x = self.embedding(x)
        for layer in self.encode_layers:
            x = layer(x)
        for layer in self.intermediate_layers:
            x = layer(x)
        for layer in self.decode_layers:
            x = layer(x)
        x = self.lm_head(x)
        return x
