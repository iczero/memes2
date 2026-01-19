import IPython
import argparse
import torch
from model.common import ControlTokens, make_tokens, tokens_repr, padding_needed
from model.train_utils import Trainer
from pathlib import Path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('checkpoint', type=str, help='checkpoint to load')
arg_parser.add_argument('--device', type=str, help='device to use', default='cpu')

def main():
    args = arg_parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    trainer, additional = Trainer.load_checkpoint(checkpoint_path, device)
    model = trainer.model
    model_config = trainer.model_config
    del trainer, additional

    def infer(s: str):
        with torch.inference_mode():
            tokens = make_tokens(
                ControlTokens.START_OF_TEXT,
                s.encode('utf-8'),
                ControlTokens.END_OF_TEXT,
            )
            tokens = torch.where(tokens == ord(b'_'), ControlTokens.MASK.value, tokens)
            padding_len = padding_needed(len(tokens), model_config.bytes_per_latent)
            if padding_len > 0:
                tokens = torch.cat([
                    tokens,
                    make_tokens([ControlTokens.PAD] * padding_len),
                ], dim=0)

            print('input:', tokens_repr(tokens))
            seq_info = model.make_seq_info(
                torch.tensor([len(tokens) // model_config.bytes_per_latent], device='cpu', dtype=torch.int32),
                compile=False,
            )
            out = model(tokens, seq_info).argmax(dim=-1)
            print('output:', tokens_repr(out))

    IPython.embed(colors='Linux')

if __name__ == '__main__':
    main()
