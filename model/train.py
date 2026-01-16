import argparse
import sys
import typing
import torch
import torch.nn.functional as F
from torch import nn
from model.common import CombinedConfig, ControlTokens, ModelConfig, TrainConfig, load_config, make_tokens, padding_needed
from model.model import QuestionableTransformer, SeqInfo

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5

torch.set_float32_matmul_precision('high')

ENABLE_ASSERTIONS = True

class Trainer:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    enable_autocast: bool
    enable_compile = True

    model: QuestionableTransformer
    optimizer: torch.optim.AdamW | torch.optim.SGD

    def __init__(
        self,
        config: CombinedConfig,
        device: torch.device,
        model_dtype: torch.dtype | None = None,
    ):
        self.model_config = config.model_config
        self.train_config = config.train_config
        self.device = device

        if model_dtype is not None:
            self.model_dtype = model_dtype

        self.model = QuestionableTransformer(self.model_config) \
            .to(device=device, dtype=self.model_dtype)

        self.optimizer = self.make_optimizer(self.train_config.optimizer)

        self._cached_forward = None

    def make_optim_param_groups(self):
        exclude_wd = []
        default = []
        for name, param in self.model.named_parameters():
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

    def make_optimizer(self, optim_type: str):
        groups = self.make_optim_param_groups()
        if optim_type == 'AdamW':
            return torch.optim.AdamW(
                groups,
                self.train_config.lr,
                weight_decay=self.train_config.weight_decay,
            )

        if optim_type == 'SGD':
            return torch.optim.SGD(
                groups,
                self.train_config.lr,
                weight_decay=self.train_config.weight_decay,
            )

        raise RuntimeError('unknown optimizer ' + optim_type)

    def _make_forward(self, refresh = False):
        if self._cached_forward is not None and not refresh:
            return self._cached_forward

        enable_autocast = self.enable_autocast

        @torch.compile(fullgraph=True, disable=not self.enable_compile)
        def inner_forward(
            seq_input: torch.Tensor,
            expected_output: torch.Tensor,
            output_mask: torch.Tensor,
            seq_info: SeqInfo,
        ):
            with torch.autocast(self.device.type, torch.bfloat16, enabled=enable_autocast):
                model_out = self.model(seq_input, seq_info)

            loss_all = F.cross_entropy(
                # transpose from (seq, C) to (C, seq), then push singleton batch dim
                model_out.transpose(-2, -1).unsqueeze(0),
                # F.cross_entropy requires long instead of int32
                expected_output.to(dtype=torch.long).unsqueeze(0),
                reduction='none',
            ).squeeze(0)

            loss_masked = torch.where(output_mask, loss_all, 0)
            loss = loss_masked.sum() / output_mask.sum()

            out_sample = torch.argmax(model_out.detach(), dim=-1)
            return loss, out_sample

        self._cached_forward = inner_forward
        return inner_forward

    def forward_batch(
        self,
        seq_input: torch.Tensor,
        expected_output: torch.Tensor,
        output_mask: torch.Tensor,
        seq_info: SeqInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ENABLE_ASSERTIONS:
            assert seq_input.shape == expected_output.shape
            assert seq_input.shape == output_mask.shape
            assert output_mask.dtype == torch.bool

        return self._make_forward()(seq_input, expected_output, output_mask, seq_info)

    def optimizer_step(self):
        return torch.compile(disable=not self.enable_compile)(self.optimizer.step)()

    def zero_grad(self, set_to_none = True):
        self.optimizer.zero_grad(set_to_none)

    @typing.overload
    def make_input(self, seqs: list[torch.Tensor], seq_output_masks: None = None, override_length: int | None = None) -> tuple[int, torch.Tensor, torch.Tensor]: ...
    @typing.overload
    def make_input(self, seqs: list[torch.Tensor], seq_output_masks: list[torch.Tensor], override_length: int | None = None) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def make_input(
        self,
        seqs: list[torch.Tensor],
        seq_output_masks: list[torch.Tensor] | None = None,
        override_length: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor] | tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - `total_seqs`: how many sequences we managed to fit (starting from index 0 in `seqs`)
            - `packed_seq`: the packed input sequence
            - `seq_lengths`: length of each sequence in `packed_seq` in units of latents
              NOTE: `seq_lengths` will end with a lot of ones if end padding was needed.
            - `output_mask`: (optional, only if `seq_output_masks provided`) Combined output mask.
        """
        bpl = self.model_config.bytes_per_latent
        target_length = self.train_config.full_seq_len
        if override_length is not None:
            if override_length % self.model_config.bytes_per_latent != 0:
                raise ValueError('length must be a multiple of bytes_per_latent')
            target_length = override_length

        total_seqs = 0
        total_length = 0
        padded_seqs = []
        seq_lengths_latent = []
        output_masks = []
        for i, seq in enumerate(seqs):
            seq_output_mask = None
            if seq_output_masks is not None:
                seq_output_mask = seq_output_masks[i]

            seq_padding = padding_needed(seq.shape[0], bpl)
            if seq_padding > 0:
                seq = torch.cat([
                    seq,
                    make_tokens([ControlTokens.PAD] * seq_padding).to(device='cpu'),
                ], dim=0)
                if seq_output_mask is not None:
                    seq_output_mask = torch.cat([
                        seq_output_mask,
                        torch.tensor([False] * seq_padding, device='cpu', dtype=torch.bool),
                    ], dim=0)

            seq_length = seq.shape[0]
            if total_length + seq_length > target_length:
                break

            padded_seqs.append(seq)
            total_length += seq_length
            seq_lengths_latent.append(seq_length // bpl)
            total_seqs += 1
            if seq_output_mask is not None:
                output_masks.append(seq_output_mask)

        end_padding_length = target_length - total_length
        end_padding_seqlens = [1] * (end_padding_length // 4)

        if end_padding_length > 0:
            end_padding = make_tokens([ControlTokens.PAD] * end_padding_length)
            padded_seqs.append(end_padding)

        packed_seq = torch.cat(padded_seqs, dim=0)
        seq_lengths = torch.tensor(
            seq_lengths_latent + end_padding_seqlens,
            device='cpu',
            dtype=torch.int32,
        )

        if seq_output_masks is not None:
            if end_padding_length > 0:
                output_masks.append(
                    torch.tensor([False] * end_padding_length, device='cpu', dtype=torch.bool),
                )

            output_mask = torch.cat(output_masks, dim=0)
            return total_seqs, packed_seq, seq_lengths, output_mask
        else:
            return total_seqs, packed_seq, seq_lengths
