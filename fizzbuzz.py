# %%
import torch
from model.common import load_config
from model.model import QuestionableTransformer

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5

FIZZ = 3
BUZZ = 5

combined_config = load_config('configs/test.json')
model_config = combined_config.model_config
train_config = combined_config.train_config

device = torch.device('cuda')
torch.set_float32_matmul_precision('high')
model_dtype = model_config.get_dtype()

def make_fizzbuzz(x_min: int, x_max: int):
    pass

model_config = load_config('configs/test.json').model_config
model = QuestionableTransformer(model_config)
model = model.to(device=device, dtype=model_config.get_dtype())

print('parameters:', sum(p.numel() for p in model.parameters()))
