# %%
import torch
from model.common import load_config
from model.model import QuestionableTransformer

model_config = load_config('configs/test.json').model_config
model = QuestionableTransformer(model_config)
model = model.to(dtype=model_config.get_dtype())

print(model)
print('parameters:', sum(p.numel() for p in model.parameters()))
