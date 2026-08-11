"""
Correctness check for PA1 Part C (see radix_cache.py): radix-shared generation must
produce byte-identical greedy output to independent (no-reuse) generation, for both the
non-flash (manual attention) and flash (SDPA) code paths, and across chunked-prefill shapes
(prompts of different lengths sharing a common prefix, plus a prompt with no shared prefix
at all).

Uses a small randomly-initialized model (no need to download GPT-2 weights) since this only
tests the tree-splitting / cache-reuse / masking logic, not model quality.
"""
import torch
from model import GPT, GPTConfig
from radix_cache import generate_radix_shared, generate_independent

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig(block_size=128, vocab_size=200, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)

prompts = [
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 6, 7],   # shares [1,2,3,4] with prompt 0, then diverges
    [1, 2, 3, 9],         # shares [1,2,3] with prompts 0/1
    [1, 2, 8],             # shares [1,2]
    [42, 43, 44],          # no shared prefix at all
    [1, 2, 3, 4, 5],       # exact duplicate of prompt 0
]

max_new_tokens = 6

for flash in (True, False):
    for greedy_mode in (True,):
        model = GPT(config)
        model.eval()
        model.to(device)
        for block in model.transformer.h:
            block.attn.flash = flash
            if not flash and not hasattr(block.attn, 'bias'):
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to(device),
                )

        with torch.no_grad():
            shared = generate_radix_shared(model, prompts, max_new_tokens, device, greedy=True)
            for block in model.transformer.h:
                block.attn.reset_cache()
                block.attn.use_cache = False
            indep = generate_independent(model, prompts, max_new_tokens, device, greedy=True)

        ok = shared == indep
        print(f"flash={flash}: radix-shared == independent -> {ok}")
        if not ok:
            for i, (a, b) in enumerate(zip(shared, indep)):
                if a != b:
                    print(f"  MISMATCH prompt {i}: shared={a} indep={b}")
            raise SystemExit(1)

print("All correctness checks passed.")
