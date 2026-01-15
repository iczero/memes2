# %%
import torch
import torch.nn.functional as F
from model.common import ControlTokens, load_config
from model.model import QuestionableTransformer, SeqInfo

import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.5

ENABLE_ASSERTIONS = True
DISABLE_COMPILE = False

combined_config = load_config('configs/test.json')
model_config = combined_config.model_config
train_config = combined_config.train_config

device = torch.device('cuda')
torch.set_float32_matmul_precision('high')
model_dtype = torch.float32

model_config = load_config('configs/test.json').model_config
model = QuestionableTransformer(model_config)
model = model.to(device=device, dtype=model_dtype)

print('parameters:', sum(p.numel() for p in model.parameters()))

optimizer = train_config.make_optimizer(model.named_parameters())

@torch.compile(fullgraph=True, disable=DISABLE_COMPILE)
def compiled_forward(
    seq_input: torch.Tensor,
    expected_output: torch.Tensor,
    output_mask: torch.Tensor,
    seq_info: SeqInfo,
):
    with torch.autocast(device.type, torch.bfloat16, enabled=True):
        model_out = model(seq_input, seq_info)

    loss_all = F.cross_entropy(
        # transpose from (seq, C) to (C, seq), then push singleton batch dim
        model_out.transpose(-2, -1).unsqueeze(0),
        expected_output.unsqueeze(0),
        reduction='none',
    )

    loss_masked = torch.where(output_mask, loss_all, 0)
    loss = loss_masked.sum() / output_mask.sum()

    out_sample = torch.argmax(model_out.detach(), dim=-1)
    return loss, out_sample

def train_step(
    # input sequence
    seq_input: torch.Tensor,
    # expected output sequence, shape should match seq_input
    expected_output: torch.Tensor,
    # boolean tensor, true at bytelevel positions to include in loss
    output_mask: torch.Tensor,
    # length of sequences in units of latent
    seq_latent_lens: torch.Tensor,
):
    if ENABLE_ASSERTIONS:
        assert seq_input.shape == expected_output.shape
        assert seq_input.shape == output_mask.shape
        assert output_mask.dtype == torch.bool

    seq_info = model.make_seq_info(seq_latent_lens, compile=not DISABLE_COMPILE)
    loss, sampled_out = compiled_forward(
        seq_input,
        expected_output,
        output_mask,
        seq_info,
    )
    loss_detached = loss.detach().clone()

    loss.backward()
    torch.compile(disable=DISABLE_COMPILE)(optimizer.step)()
    optimizer.zero_grad(set_to_none=True)

    return loss_detached, sampled_out

def bytes_to_tokens(bstr: bytes, prepend_sot = True) -> torch.Tensor:
    if prepend_sot:
        sot = [ControlTokens.START_OF_TEXT]
    else:
        sot = []
    return torch.tensor(sot + list(bstr), dtype=torch.int32, device='cpu')

def tokens_to_printable_bytes(tokens: torch.Tensor) -> bytes:
    out = []
    for value in tokens.tolist():
        if value <= 255:
            out.append(bytes([value]))
        elif value == ControlTokens.PAD:
            out.append(b'<|pad|>')
        elif value == ControlTokens.MASK:
            out.append(b'<|mask|>')
        elif value == ControlTokens.START_OF_TEXT:
            out.append(b'<|startoftext|>')
        elif value == ControlTokens.END_OF_TEXT:
            out.append(b'<|endoftext|>')
        else:
            out.append(b'<|unused|>')

    return b''.join(out)

# begin memes

import random

FIZZ = 3
BUZZ = 5

DIGITS = {
    '0': b'zero', '1': b'one', '2': b'two', '3': b'three', '4': b'four',
    '5': b'five', '6': b'six', '7': b'seven', '8': b'eight', '9': b'nine',
}
def make_fizzbuzz(target_seq_len: int):
    # 8-digit fizzbuzz generator
    MIN = 10_000_000
    MAX = 99_999_999

    total_len = 0
    in_seqs = []
    out_seqs = []
    out_masks = []
    while total_len < target_seq_len:
        num = random.randint(MIN, MAX)
        num_str = str(num)
        num_exploded = b' '.join(DIGITS[d] for d in num_str)

        if num % (FIZZ * BUZZ) == 0:
            fizzbuzz_out = b'fizzbuzz'
        elif num % FIZZ == 0:
            fizzbuzz_out = b'    fizz'
        elif num % BUZZ == 0:
            fizzbuzz_out = b'    buzz'
        else:
            fizzbuzz_out = num_str.encode('ascii')

        if ENABLE_ASSERTIONS:
            assert len(fizzbuzz_out) == 8

        common = bytes_to_tokens(num_exploded + b' -> ', True)
        seq_len = common.shape[0] + 8
        padding_len = model_config.bytes_per_latent - (seq_len % model_config.bytes_per_latent)
        padding_tokens = torch.tensor([ControlTokens.PAD] * padding_len, device='cpu', dtype=common.dtype)
        in_seq = torch.concat([
            common,
            torch.tensor([ControlTokens.MASK] * 8, device='cpu', dtype=common.dtype),
            padding_tokens,
        ])
        out_seq = torch.concat([
            common,
            bytes_to_tokens(fizzbuzz_out, False),
            padding_tokens,
        ])
        out_mask = torch.tensor(
            [False] * common.shape[0] + [True] * 8 + [False] * padding_len,
            device='cpu',
        )

        if ENABLE_ASSERTIONS:
            assert in_seq.shape == out_seq.shape
            assert in_seq.shape == out_mask.shape
            assert in_seq.shape[0] % model_config.bytes_per_latent == 0

        total_len += in_seq.shape[0]
        in_seqs.append(in_seq)
        out_seqs.append(out_seq)
        out_masks.append(out_mask)

    end_padding_len = target_seq_len - total_len
    end_padding = torch.tensor([ControlTokens.PAD] * end_padding_len, dtype=torch.int32, device='cpu')
    end_padding_mask = torch.tensor([False] * end_padding_len, dtype=torch.bool, device='cpu')

    in_seq = torch.concat(in_seqs + [end_padding])
    out_seq = torch.concat(out_seqs + [end_padding])
    out_mask = torch.concat(out_masks + [end_padding_mask])

    seq_lens = [len(seq) // 4 for seq in in_seqs]
    if end_padding_len > 0:
        seq_lens.append(end_padding_len // 4)

    seq_lens = torch.tensor(seq_lens, device='cpu', dtype=torch.int32)

    return in_seq, out_seq, out_mask, seq_lens

def train_for_iter(n: int, sample_interval = 16, seq_len = 4096):
    for i in range(n):
        in_seq, out_seq, out_mask, seq_lens = make_fizzbuzz(seq_len)
        loss, sample = train_step(
            in_seq.to(device),
            out_seq.to(device),
            out_mask.to(device),
            seq_lens.to(device),
        )
        print(f'iter {i}\n  loss', loss.item())
        if i > 0 and i % sample_interval == 0:
            sample_expected = out_seq[out_mask]
            sample_predicted = sample[out_mask]
            print('  expected:  ' + repr(tokens_to_printable_bytes(sample_expected)))
            print('  predicted: ' + repr(tokens_to_printable_bytes(sample_predicted)))
