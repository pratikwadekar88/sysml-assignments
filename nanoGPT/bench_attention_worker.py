"""
Worker: benchmarks ONE (device, flash/standard, forward-only/forward+backward,
batch_size, seq_len) configuration of nanoGPT's CausalSelfAttention and prints
a single JSON line of results to stdout.

Run in its own process so that:
  - CPU peak RSS (measured by the caller via `/usr/bin/time -v`) reflects only
    this configuration, not leftover allocations from prior runs.
  - CUDA peak memory stats can't leak across configs even if the allocator
    doesn't fully release cached blocks.
"""
import sys
import time
import json
import argparse
from dataclasses import dataclass

import torch

from model import CausalSelfAttention


@dataclass
class Config:
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 4096
    bias: bool = True
    dropout: float = 0.0


def make_attn(device, flash: bool, config: Config):
    attn = CausalSelfAttention(config)
    attn.flash = flash
    if not flash:
        attn.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )
    attn.to(device)
    return attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--mode", choices=["flash", "standard"], required=True)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--n-head", type=int, default=12)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    config = Config(n_embd=args.n_embd, n_head=args.n_head, block_size=max(4096, args.seq_len))
    flash = args.mode == "flash"

    attn = make_attn(device, flash, config)
    x = torch.randn(args.batch_size, args.seq_len, config.n_embd, device=device,
                     requires_grad=args.backward)

    def step():
        if args.backward:
            x.grad = None
            for param in attn.parameters():
                param.grad = None
            out = attn(x)
            out.sum().backward()
        else:
            with torch.no_grad():
                attn(x)

    is_cuda = device.type == "cuda"

    for _ in range(args.warmup):
        step()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) / args.iters * 1000.0
    result = {
        "device": args.device,
        "mode": args.mode,
        "backward": args.backward,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "wall_ms": wall_ms,
    }
    if is_cuda:
        result["cuda_peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
