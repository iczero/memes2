import argparse
import mlflow
import signal
import torch
import torch.nn.functional as F
from torch import nn
from model.common import ControlTokens, load_config, make_tokens, tokens_repr
from model.model import QuestionableTransformer, SeqInfo
from model.pile_loader import filter_text, load_dataset
from model.train import Trainer

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5
torch.set_float32_matmul_precision('high')
device = torch.device('cuda')

mlflow.config.enable_async_logging()
mlflow.set_tracking_uri('http://127.0.0.1:5000')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config', help='path to config')
arg_parser.add_argument('data', help='path to dataset')

# thanks gemini
def span_mask(
    input_tensor: torch.Tensor,
    mask_range: tuple = (0.1, 0.2),
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
    go_away = False

    def go_away_handler(sig, frame):
        nonlocal go_away
        if sig == signal.SIGINT:
            if go_away:
                print('exiting now!')
                raise KeyboardInterrupt

            go_away = True
            print('exiting soon')

    signal.signal(signal.SIGINT, go_away_handler)

    with mlflow.start_run():
        combined_config = load_config(args.config)
        trainer = Trainer(combined_config, device=device)
        model_config = trainer.model_config
        train_config = trainer.train_config

        data_iter = filter_text(load_dataset(open(args.data, 'rb')))

        print('parameters:', sum(p.numel() for p in trainer.model.parameters()))

        def make_one_seq():
            text = next(data_iter, None)
            if text is None:
                print('data exhausted!')
                return None

            # TODO: make this less garbage
            max_len = train_config.full_seq_len - 2
            out_seq = make_tokens(
                ControlTokens.START_OF_TEXT,
                text[:max_len].encode(),
                ControlTokens.END_OF_TEXT,
            )

            in_masked, out_mask = span_mask(out_seq)

            stacked = torch.stack([in_masked, out_seq], dim=0)

            return stacked, out_mask

        step = 0
        while not go_away:
            raw_len = 0
            seqs = []
            masks = []
            while raw_len < train_config.full_seq_len:
                seq, mask = make_one_seq()
                seqs.append(seq)
                masks.append(mask)
                raw_len += len(mask)

            seq_count, packed, seq_lengths, out_mask = trainer.make_packed(seqs, masks)
            #print(f'packed {_seq_count} sequences')
            packed = packed.to(device=trainer.device)
            seq_lengths = seq_lengths.to(device=trainer.device)
            out_mask = out_mask.to(trainer.device)

            in_masked = packed[0]
            out_seq = packed[1]
            #print('in_masked', tokens_repr(in_masked), in_masked.shape)
            #print('out_seq', tokens_repr(out_seq), out_seq.shape)
            seq_info = trainer.model.make_seq_info(seq_lengths, compile=trainer.enable_compile)

            trainer.zero_grad()
            loss, sample = trainer.forward_batch(in_masked, out_seq, out_mask, seq_info)
            loss.backward()
            grad_norm = trainer.clip_grad_norm()
            trainer.optimizer_step()

            mlflow.log_metrics({
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
                'seq_count': seq_count,
            }, step=step)

            print(f'step {step}: loss {loss.item()}, grad norm {grad_norm.item()}')
            if step % 64 == 0:
                print('sample output:', tokens_repr(sample))

            step += 1

if __name__ == '__main__' and not hasattr(__builtins__, '__IPYTHON__'):
    main()
