import argparse
import torch
import torch.nn.functional as F
from torch import nn
from model.common import ControlTokens, load_config
from model.model import QuestionableTransformer, SeqInfo
from model.train import Trainer

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5

torch.set_float32_matmul_precision('high')

ENABLE_ASSERTIONS = True
DISABLE_COMPILE = False

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config', help='path to config')
args = arg_parser.parse_args()

combined_config = load_config(args.config)
device = torch.device('cuda')

trainer = Trainer(combined_config, device=device)

print('parameters:', sum(p.numel() for p in trainer.model.parameters()))

def main():
    pass # TODO:

if __name__ == '__main__' and not hasattr(__builtins__, '__IPYTHON__'):
    main()
