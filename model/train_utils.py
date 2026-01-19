import time
import typing
import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from typing import Self, Iterable
from model.common import CombinedConfig, ControlTokens, ModelConfig, TrainConfig, make_tokens, padding_needed, current_git_commit
from model.model import QuestionableTransformer, SeqInfo
from pathlib import Path

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5

torch.set_float32_matmul_precision('high')

ENABLE_ASSERTIONS = True
USE_MLFLOW = True

class Trainer:
    model_config: ModelConfig
    train_config: TrainConfig
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    enable_autocast = True
    enable_compile = True

    model: QuestionableTransformer
    optimizer: torch.optim.AdamW | torch.optim.SGD

    use_mlflow: bool
    "True if we should log to mlflow"

    step: int = 0
    checkpoint_dir: Path

    def __init__(
        self,
        config: CombinedConfig,
        device: torch.device,
        checkpoint_dir: Path,
        model_dtype: torch.dtype | None = None,
        _restoring: bool = False,
    ):
        self.model_config = config.model_config
        self.train_config = config.train_config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if model_dtype is not None:
            self.model_dtype = model_dtype

        self.model = QuestionableTransformer(self.model_config) \
            .to(device=device, dtype=self.model_dtype)

        self.optimizer = self.make_optimizer(self.train_config.optimizer)

        self._cached_forward = None

        if not _restoring:
            self.log_config()

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

    def log_config(self):
        if not USE_MLFLOW:
            return

        # we log both model and train config into the same namespace.
        # there should not be any conflicts.
        mlflow.log_params(self.model_config.to_dict())
        mlflow.log_params(self.train_config.to_dict())

        commit = current_git_commit()
        if commit is not None:
            mlflow.log_param('revision', commit)

    def _make_forward(self, refresh = False):
        if self._cached_forward is not None and not refresh:
            return self._cached_forward

        enable_autocast = self.enable_autocast

        # warning: do NOT use cudagraphs. there appears to be a memory leak
        # in the cudagraphs implementation that is not resolved by
        # `cudagraphs_mark_step_begin`. additionally, the performance improvement
        # is not significant enough to justify the headache
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

    def clip_grad_norm(self):
        return nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.clip_grad_norm,
            error_if_nonfinite=True,
        )

    @typing.overload
    def make_packed(self, seqs: Iterable[torch.Tensor], override_length: int | None = None) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Construct a packed sequence from an iterable of sequences.

        `seqs` may have two dimensions, in which case sequences are processed as batches.
        This is useful to specify for example the masked and expected output simultaneously.

        Returns:
            - `total_seqs`: how many sequences we managed to fit (starting from index 0 in `seqs`)
            - `packed_seq`: the packed input sequence
            - `seq_lengths`: length of each sequence in `packed_seq` in units of latents
              NOTE: `seq_lengths` will end with a lot of ones if end padding was needed.
        """
        ...
    @typing.overload
    def make_packed(
        self,
        seqs: Iterable[tuple[torch.Tensor, torch.Tensor]],
        override_length: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct a packed sequence from an iterable of tuples of sequences and their
        corresponding output masks.

        `seqs` may have two dimensions, in which case sequences are processed as batches.
        This is useful to specify for example the masked and expected output simultaneously.

        Returns:
            - `total_seqs`: how many sequences we managed to fit (starting from index 0 in `seqs`)
            - `packed_seq`: the packed input sequence
            - `seq_lengths`: length of each sequence in `packed_seq` in units of latents
              NOTE: `seq_lengths` will end with a lot of ones if end padding was needed.
            - `output_mask`: Combined output mask.
        """
        ...
    def make_packed(
        self,
        seqs: Iterable[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        override_length: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor] | tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        bpl = self.model_config.bytes_per_latent
        target_length = self.train_config.full_seq_len

        if override_length is not None:
            if override_length % bpl != 0:
                raise ValueError('length must be a multiple of bytes_per_latent')
            target_length = override_length

        def expand_padding(seq_shape: torch.Size, padding: torch.Tensor):
            out_shape = list(seq_shape)
            out_shape[-1] = padding.shape[0]
            return padding.expand(out_shape)

        total_seqs = 0
        total_length = 0
        padded_seqs = []
        seq_lengths_latent = []
        output_masks = []
        seq_shape_for_pad: torch.Size | None = None
        has_output_masks: bool | None = None
        seq_iter = seqs.__iter__()

        while True:
            # bubble StopIteration
            seq = next(seq_iter)

            seq_output_mask = None
            if not isinstance(seq, torch.Tensor):
                seq, seq_output_mask = seq
                if has_output_masks is None:
                    has_output_masks = True
                elif not has_output_masks:
                    raise RuntimeError('expected item to have an output mask (previous iteration returned one)')
            else:
                if has_output_masks is None:
                    has_output_masks = False
                elif has_output_masks:
                    raise RuntimeError('expected item to not have an output mask (previous iteration did not return one)')

            if seq_shape_for_pad is None:
                seq_shape_for_pad = seq.shape

            seq_padding = padding_needed(seq.shape[-1], bpl)
            if seq_padding > 0:
                seq = torch.cat([
                    seq,
                    expand_padding(
                        seq.shape,
                        make_tokens([ControlTokens.PAD] * seq_padding),
                    ),
                ], dim=-1)
                if seq_output_mask is not None:
                    seq_output_mask = torch.cat([
                        seq_output_mask,
                        torch.tensor([False] * seq_padding, device='cpu', dtype=torch.bool),
                    ], dim=0)

            seq_length = seq.shape[-1]
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
            assert seq_shape_for_pad is not None
            padded_seqs.append(expand_padding(seq_shape_for_pad, end_padding))

        packed_seq = torch.cat(padded_seqs, dim=-1)
        seq_lengths = torch.tensor(
            seq_lengths_latent + end_padding_seqlens,
            device='cpu',
            dtype=torch.int32,
        )

        if has_output_masks:
            if end_padding_length > 0:
                output_masks.append(
                    torch.tensor([False] * end_padding_length, device='cpu', dtype=torch.bool),
                )

            output_mask = torch.cat(output_masks, dim=0)
            return total_seqs, packed_seq, seq_lengths, output_mask
        else:
            return total_seqs, packed_seq, seq_lengths

    def state_dict(self):
        out = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': CombinedConfig(self.model_config, self.train_config).to_dict(),
            'step': self.step,
        }
        if USE_MLFLOW:
            out['run_id'] = mlflow.active_run().info.run_id
        return out

    @classmethod
    def load_state_dict(
            cls,
            state,
            device: torch.device,
            checkpoint_dir: Path,
            replace_config: CombinedConfig | None = None,
    ) -> Self:
        if replace_config is not None:
            combined_config = replace_config
        else:
            combined_config = CombinedConfig.from_dict(state['config'], ignore_nonconfigurable=True)
        inst = cls(combined_config, device=device, checkpoint_dir=checkpoint_dir, _restoring=True)
        inst.step = state['step']
        inst.model.load_state_dict(state['model'])
        inst.optimizer.load_state_dict(state['optimizer'])
        return inst

    def save_checkpoint(self, additional: dict | None = None):
        state = self.state_dict()
        if additional is not None:
            for k, v in additional.items():
                state[k] = v

        out_name = f'step{self.step:07d}-{int(time.time())}.pt'
        self.checkpoint_dir.mkdir(exist_ok=True)
        with open(self.checkpoint_dir / out_name, 'wb') as f:
            torch.save(state, f)

        print(f'saved checkpoint {out_name}')

    @classmethod
    def load_checkpoint(
            cls,
            checkpoint_path: Path,
            device: torch.device,
            replace_config: CombinedConfig | None = None,
    ) -> tuple[Self, dict]:
        # TODO: data loader state, somehow
        checkpoint_dir = checkpoint_path.absolute().parent
        with open(checkpoint_path, 'rb') as f:
            state = torch.load(f, map_location='cpu')

        inst = cls.load_state_dict(state, device, checkpoint_dir, replace_config=replace_config)
        # better way to do this?
        del state['config']
        del state['step']
        del state['model']
        del state['optimizer']
        return inst, state
