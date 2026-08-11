"""
PA1 Part B worker: benchmarks ONE (device, cache on/off, input_len, output_len)
configuration of full-model autoregressive generation and prints a single JSON
line of results (mean/std wall time over several repeats, plus peak memory) to
stdout. Run in its own process for the same isolation reasons as
bench_attention_worker.py (clean CUDA peak-memory counters, and CPU RSS
measured per-config by the driver via `/usr/bin/time -v`).

Uses a randomly-initialized GPT-2-small-shaped model (12 layer / 12 head / 768
embd) rather than downloading pretrained weights: parameter *values* don't
affect the time/memory cost of a forward pass, only the shapes do, and this
keeps the benchmark network-free and fast to set up.
"""
import sys
import time
import json
import argparse
import statistics

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    block_size = max(1024, args.input_len + args.output_len)
    config = GPTConfig(block_size=block_size, vocab_size=50304, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=0.0, bias=True)
    model = GPT(config)
    model.eval()
    model.to(device)

    idx = torch.randint(0, config.vocab_size, (1, args.input_len), dtype=torch.long, device=device)

    def run_once():
        for block in model.transformer.h:
            block.attn.reset_cache()
            block.attn.use_cache = False
        with torch.no_grad():
            model.generate(idx, args.output_len, use_cache=args.use_cache, greedy=True)

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
        "use_cache": args.use_cache,
        "input_len": args.input_len,
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
