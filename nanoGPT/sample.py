"""
Naive sequential sampling (NO cache, NO radix, NO tricks)
Works with the naive model.py you posted
"""

import os
import torch
import tiktoken
from model import GPT, GPTConfig
import time
# -----------------------------------------------------------------------------
init_from = "gpt2"   # gpt2 / gpt2-medium / gpt2-large / gpt2-xl
device = "cuda" if torch.cuda.is_available() else "cpu"

max_new_tokens = 150
temperature = 0.8
top_k = 200
seed = 1337

prompts = [
    "Django is a framework in python\n",
    "Django is a framework in python used for\n",
    "Django supports\n",
]

# -----------------------------------------------------------------------------

torch.manual_seed(seed)

# tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda t: enc.decode(t)

# model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)

# -----------------------------------------------------------------------------
# SEQUENTIAL PROMPT PROCESSING (NAIVE)
# -----------------------------------------------------------------------------

with torch.no_grad():
    start = time.time()
    for i, prompt in enumerate(prompts, 1):
        print(f"\n================ PROMPT {i} ================")
        print(prompt)

        # encode full prompt
        idx = torch.tensor(
            encode(prompt),
            dtype=torch.long,
            device=device
        )[None, :]   # (1, T)

        # generate (NAIVE: full forward every step)
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        print("\nOUTPUT:")
        print(decode(out[0].tolist()))
        print("===========================================\n")
    end = time.time()

    print(f"Time required For Without Caching: {end - start:.3f} sec")
