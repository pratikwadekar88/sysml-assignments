"""
PA1 Part C worker: benchmarks ONE (device, strategy) configuration -- strategy
is "radix-shared" (radix_cache.generate_radix_shared) or "independent"
(radix_cache.generate_independent) -- over a fixed synthetic batch of prompts
that all share a common prefix, and prints one JSON line of results (mean/std
wall time over several repeats, plus peak memory).

Synthetic batch: num_prompts prompts, each = [shared prefix of prefix_len
random tokens] + [unique suffix of suffix_len random tokens]. The suffix's
first token is forced distinct per prompt so the radix tree branches
immediately after the shared prefix (matches the assignment's example: a
common prefix followed by N independent leaves), giving deterministic
prefix-sharing regardless of the random draw.
"""
import sys
import time
import json
import argparse
import statistics

import torch

from model import GPT, GPTConfig
from radix_cache import generate_radix_shared, generate_independent


def make_prompts(num_prompts, prefix_len, suffix_len, vocab_size, generator):
    prefix = torch.randint(0, vocab_size, (prefix_len,), generator=generator).tolist()
    prompts = []
    for i in range(num_prompts):
        suffix = torch.randint(0, vocab_size, (suffix_len,), generator=generator).tolist()
        suffix[0] = (i * 37 + 1) % vocab_size  # force branch right after the shared prefix
        prompts.append(prefix + suffix)
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--strategy", choices=["radix-shared", "independent"], required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--suffix-len", type=int, default=16)
    p.add_argument("--output-len", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    block_size = max(1024, args.prefix_len + args.suffix_len + args.output_len)
    config = GPTConfig(block_size=block_size, vocab_size=50304, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=0.0, bias=True)
    model = GPT(config)
    model.eval()
    model.to(device)

    gen = torch.Generator().manual_seed(1234)
    prompts = make_prompts(args.num_prompts, args.prefix_len, args.suffix_len, config.vocab_size, gen)

    fn = generate_radix_shared if args.strategy == "radix-shared" else generate_independent

    def run_once():
        for block in model.transformer.h:
            block.attn.reset_cache()
            block.attn.use_cache = False
        with torch.no_grad():
            fn(model, prompts, args.output_len, device, greedy=True)

    for _ in range(args.warmup):
        run_once()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(args.repeats):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    result = {
        "device": args.device,
        "strategy": args.strategy,
        "num_prompts": args.num_prompts,
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
    if is_cuda:
        result["cuda_peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
