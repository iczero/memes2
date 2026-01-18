from typing import Iterator
import contextlib
import typing
import argparse
import mlflow
import signal
import sys
import torch
import torch.nn.functional as F
from torch import nn
from model.common import ControlTokens, load_config, make_tokens, tokens_repr
from model.pile_loader import filter_text, load_dataset
from model.train_utils import Trainer
from pathlib import Path

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5
torch.set_float32_matmul_precision('high')

mlflow.config.enable_async_logging()
mlflow.set_tracking_uri('http://127.0.0.1:5000')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--config', type=str, help='path to config', default=None)
arg_parser.add_argument('--data', type=str, help='path to dataset', required=True)
arg_parser.add_argument('--no-compile', action='store_true', help='disable torch.compile')
arg_parser.add_argument('--use-cpu', action='store_true', help='use cpu instead of cuda')
start_or_load = arg_parser.add_mutually_exclusive_group(required=True)
start_or_load.add_argument('--new', type=str, help='start new run in provided checkpoint directory', default=None)
start_or_load.add_argument('--load', type=str, help='load checkpoint file and continue', default=None)

# thanks gemini
def span_mask(
    input_tensor: torch.Tensor,
    mask_range: tuple = (0.2, 0.35),
    mean_len: float = 10.0,
    std_len: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies span-based masking to a 1D input tensor.

    Args:
        input_tensor: 1D tensor of tokens.
        mask_range: (lower, upper) percentage of tokens to mask.
        mean_len: Average length of a masked span (island).
        std_len: Standard deviation of the span length.
    """
    seq_len = input_tensor.size(0)
    device = input_tensor.device

    # Initialize the boolean mask as False
    mask_bool = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # Determine the target number of tokens to mask
    lower, upper = mask_range
    target_pct = torch.empty(1, device=device).uniform_(lower, upper).item()
    target_count = int(target_pct * seq_len)

    current_masked_count = 0

    # Keep adding "islands" until the budget is met
    while current_masked_count < target_count:
        # 1. Sample span length from normal distribution
        # Use absolute to avoid negative lengths; clamp to at least 1
        span_len = torch.normal(torch.tensor(mean_len), torch.tensor(std_len)).round().int().item()
        span_len = max(1, span_len)

        # 2. Pick a random starting position
        start_pos = 1
        end_pos = seq_len - 1
        start_idx = torch.randint(start_pos, end_pos, (1,)).item()

        # 3. Calculate end index (clamped to seq length)
        end_idx = min(start_idx + span_len, seq_len)

        # 4. Apply to mask_bool
        mask_bool[start_idx:end_idx] = True

        # 5. Update the unique count (handles overlapping spans)
        current_masked_count = mask_bool.sum().item()

    # Create the output tensor by cloning and filling masked positions
    masked_input = input_tensor.clone()
    masked_input[mask_bool] = ControlTokens.MASK

    return masked_input, mask_bool

def main():
    args = arg_parser.parse_args()

    device = torch.device('cuda')
    if args.use_cpu:
        device = torch.device('cpu')

    if args.new is not None:
        if args.config is None:
            print('error: config required when starting new run')
            sys.exit(1)

        checkpoint_dir = Path(args.new)

        combined_config = load_config(args.config)
        mlflow_ctxmgr = mlflow.start_run(run_name=checkpoint_dir.name)
        trainer = Trainer(combined_config, device=device, checkpoint_dir=checkpoint_dir)
    elif args.load is not None:
        replace_config = None
        if args.config is not None:
            print('info: config provided with load, replacing config!')
            replace_config = load_config(args.config)
        checkpoint_path = Path(args.load)

        trainer, run_id = Trainer.load_checkpoint(checkpoint_path, device, replace_config=replace_config)
        mlflow_ctxmgr = mlflow.start_run(run_id=run_id)
    else:
        raise RuntimeError('required option missing')

    if args.no_compile:
        print('disabling torch.compile')
        trainer.enable_compile = False

    model_config = trainer.model_config
    train_config = trainer.train_config

    print(f'model parameter count: {sum(p.numel() for p in trainer.model.parameters()):n}')

    data_iter = filter_text(load_dataset(open(args.data, 'rb')))

    def seq_iter_fn() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for text in data_iter:
            # TODO: make this less garbage
            max_len = train_config.train_max_seq_len - 2
            out_seq = make_tokens(
                ControlTokens.START_OF_TEXT,
                text.encode()[:max_len],
                ControlTokens.END_OF_TEXT,
            )

            in_masked, out_mask = span_mask(
                out_seq,
                mask_range=(train_config.mask_ratio_lower, train_config.mask_ratio_upper),
            )

            stacked = torch.stack([in_masked, out_seq], dim=0)

            yield stacked, out_mask

        print('data exhausted!')

    seq_iter = seq_iter_fn()

    @contextlib.contextmanager
    def save_on_exit():
        try:
            yield
            print('train loop exited, saving checkpoint')
            trainer.save_checkpoint()
        except Exception:
            if trainer.step < 10:
                print('error occurred but not saving checkpoint (too early)')
            else:
                print('error occurred, saving checkpoint')
                trainer.save_checkpoint()
            raise
        # don't save on KeyboardInterrupt, the signal handler already handles that

    go_away = typing.cast(bool, False) # needed for ty for some reason
    save_now = typing.cast(bool, False)

    def signal_handler(sig, _frame):
        nonlocal go_away, save_now
        if sig == signal.SIGINT:
            if go_away:
                print('exiting now!')
                raise KeyboardInterrupt

            go_away = True
            print('exiting soon')
        elif sig == signal.SIGUSR1:
            print('queued checkpoint')
            save_now = True
        elif sig == signal.SIGQUIT:
            # ^\ (ctrl-backslash) sends this usually
            print('SIGQUIT received, exiting immediately')
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    with mlflow_ctxmgr, save_on_exit():
        while not go_away:
            seq_count, packed, seq_lengths, out_mask = trainer.make_packed(seq_iter)
            if seq_count == 0:
                # this should not happen
                print('error: seq_count is zero! skippping')
                continue
            packed = packed.to(device=trainer.device)
            seq_lengths = seq_lengths.to(device=trainer.device)
            out_mask = out_mask.to(trainer.device)

            in_masked = packed[0]
            out_seq = packed[1]

            trainer.zero_grad()
            seq_info = trainer.model.make_seq_info(seq_lengths, compile=trainer.enable_compile)
            loss, sample = trainer.forward_batch(in_masked, out_seq, out_mask, seq_info)
            loss.backward()
            grad_norm = trainer.clip_grad_norm()
            trainer.optimizer_step()
            accuracy = (sample[out_mask] == out_seq[out_mask]).to(dtype=torch.float).mean().item()

            mlflow.log_metrics({
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
                'accuracy': accuracy,
                'seq_count': seq_count,
            }, step=trainer.step)

            print(f'step {trainer.step}: loss {loss.item():.6f}, accuracy {accuracy * 100:.3f}%, grad norm {grad_norm.item()}')
            if trainer.step % 64 == 0 and trainer.step > 0:
                print('sample expected:', tokens_repr(out_seq[out_mask]))
                print('sample output:  ', tokens_repr(sample[out_mask]))

            trainer.step += 1

            if save_now:
                save_now = False
                trainer.save_checkpoint()

        trainer.save_checkpoint()

if __name__ == '__main__' and not hasattr(__builtins__, '__IPYTHON__'):
    main()
