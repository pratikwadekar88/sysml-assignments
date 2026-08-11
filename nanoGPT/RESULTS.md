# PA1 Benchmark Results: KV Caching (Part B) and Shared-Prefix Reuse (Part C)

This branch (`pa1-kv-cache-bench`) covers PA1 Parts B and C: measuring the
performance impact of KV caching, and of reusing cached KV state across a
batch of prompts that share a prefix. Part A (baseline inference time/memory
vs. prompt/output size, no caching) and Part D (CUDA FlashAttention) live in
`nanoGPT/bench_attention_*` and on the `pa4-flash-attention-bench` branch
respectively, and are not repeated here.

## What changed in the code

- **`model.py`**: fixed two chunked-prefill bugs in `CausalSelfAttention.forward`
  that only manifest when `1 < T < T_ctx` (i.e. prefilling a suffix chunk on
  top of an already-cached prefix -- exactly what Part C needs and Part B's
  single-token decode never exercises):
  - Non-flash path: `self.bias[:,:,:T,:T]` (shape `(T,T)`) does not broadcast
    against the actual attention matrix (shape `(T,T_ctx)`), so this branch
    would crash.
  - Flash (SDPA) path: `is_causal = (T==T_ctx)` evaluated to `False` for a
    suffix chunk, which made SDPA apply *no* mask at all -- suffix tokens
    could silently attend to future suffix tokens. This doesn't crash, it
    just produces wrong logits.
  - Both are replaced with one formula: query row `i` may see key columns
    `0..P+i` where `P = T_ctx - T` is the already-cached prefix length. This
    covers full prefill (`P=0`), single-token decode (`T=1`), and chunked
    prefill (`1<T<T_ctx`) uniformly.
- **`model.py`**: added `GPT.get_kv_cache()` / `GPT.set_kv_cache()` to
  snapshot/restore the per-block `(cache_k, cache_v)` tensors, and a
  `pos_start` / `greedy` argument to `GPT.generate()`, needed to resume
  generation from a pre-loaded shared-prefix cache and to get deterministic
  output for correctness checking.
- **`radix_cache.py`** (new): builds a radix tree over a batch of tokenized
  prompts (with correct node-splitting, not just the assignment's
  no-splitting example) and provides `generate_radix_shared` (prefill each
  distinct prefix segment exactly once, reuse cached KV for every prompt
  below it) and `generate_independent` (same KV-cache machinery, but each
  prompt is prefilled and decoded from scratch -- the baseline Part C is
  meant to beat).

## Correctness

`verify_radix_cache.py` runs both the flash (SDPA) and non-flash (manual)
attention paths against a batch with overlapping/duplicate/no-shared-prefix
prompts (including one prompt that is a strict prefix of another), and
asserts `generate_radix_shared(..., greedy=True) == generate_independent(...,
greedy=True)` token-for-token. Passes on both CPU and CUDA:

```
$ .venv/bin/python verify_radix_cache.py
flash=True: radix-shared == independent -> True
flash=False: radix-shared == independent -> True
All correctness checks passed.
```

## Config

- Model: randomly-initialized GPT-2-small-shaped model (12 layer, 12 head,
  768 embd, 123.7M params) via `GPTConfig`/`GPT` -- **not** pretrained GPT-2
  weights. Parameter *values* don't affect the time/memory cost of a forward
  pass, only the shapes do, so this keeps the benchmark network-free.
  `dtype=float32` (no autocast) on both devices.
- GPU: NVIDIA RTX 3050 6GB Laptop GPU (sm_86). CPU: WSL2 host.
- Each config: 1 warmup run (untimed) + 5 timed repeats. `torch.cuda.synchronize()`
  brackets every timed call. Mean/std computed over the 5 repeats.
- GPU memory: `cuda.reset_peak_memory_stats()` after warmup, `max_memory_allocated()`
  read at the end of the repeats. CPU memory: peak RSS of the whole (isolated,
  one-config-per-process) worker subprocess, via `/usr/bin/time -v`.
- Generation is greedy (`greedy=True`) throughout, so timings aren't affected
  by `torch.multinomial` sampling variance.
- Scripts: `bench_kv_cache_worker.py` / `bench_kv_cache_driver.py` (Part B),
  `bench_radix_worker.py` / `bench_radix_driver.py` (Part C). Reproduce with
  `python bench_kv_cache_driver.py` / `python bench_radix_driver.py`.

## Part B: KV caching, with vs. without

Batch size 1. `input_len` = prompt length, `output_len` = tokens generated.
Without caching, every generation step reprocesses the entire sequence seen
so far; with caching, every step after the first only processes the newest
token. All combinations of `{cpu, cuda} x {no-cache, cache} x input_len x output_len`:

| device | cache | input | output | mean ms | std ms | speedup (no-cache/cache) | peak mem |
|---|---|---|---|---|---|---|---|
| cpu | off | 32 | 32 | 1059.7 | 22.9 | -- | 985.6 MB RSS |
| cpu | off | 32 | 64 | 2604.1 | 27.8 | -- | 985.2 MB RSS |
| cpu | off | 128 | 32 | 2631.9 | 43.3 | -- | 985.6 MB RSS |
| cpu | off | 128 | 64 | 5581.5 | 101.5 | -- | 985.6 MB RSS |
| cpu | off | 256 | 32 | 4779.6 | 62.5 | -- | 985.7 MB RSS |
| cpu | off | 256 | 64 | 9922.3 | 37.6 | -- | 985.7 MB RSS |
| cpu | **on** | 32 | 32 | 435.1 | 18.3 | **2.44x** | 985.6 MB RSS |
| cpu | **on** | 32 | 64 | 826.8 | 9.6 | **3.15x** | 985.6 MB RSS |
| cpu | **on** | 128 | 32 | 485.6 | 16.5 | **5.42x** | 985.8 MB RSS |
| cpu | **on** | 128 | 64 | 905.3 | 13.9 | **6.17x** | 985.7 MB RSS |
| cpu | **on** | 256 | 32 | 565.9 | 11.4 | **8.45x** | 985.6 MB RSS |
| cpu | **on** | 256 | 64 | 1009.4 | 11.2 | **9.83x** | 985.7 MB RSS |
| cuda | off | 32 | 32 | 197.2 | 3.2 | -- | 487.3 MB alloc |
| cuda | off | 32 | 64 | 431.4 | 2.8 | -- | 489.1 MB alloc |
| cuda | off | 128 | 32 | 340.4 | 6.8 | -- | 490.7 MB alloc |
| cuda | off | 128 | 64 | 669.0 | 0.6 | -- | 491.7 MB alloc |
| cuda | off | 256 | 32 | 523.4 | 1.4 | -- | 494.7 MB alloc |
| cuda | off | 256 | 64 | 1080.8 | 1.3 | -- | 495.9 MB alloc |
| cuda | **on** | 32 | 32 | 174.4 | 7.3 | **1.13x** | 489.9 MB alloc |
| cuda | **on** | 32 | 64 | 303.0 | 10.6 | **1.42x** | 492.1 MB alloc |
| cuda | **on** | 128 | 32 | 173.9 | 4.8 | **1.96x** | 513.3 MB alloc |
| cuda | **on** | 128 | 64 | 325.3 | 18.3 | **2.06x** | 513.3 MB alloc |
| cuda | **on** | 256 | 32 | 186.0 | 15.4 | **2.81x** | 521.0 MB alloc |
| cuda | **on** | 256 | 64 | 340.4 | 12.1 | **3.18x** | 521.0 MB alloc |

**Time**: caching wins everywhere, and the speedup grows with sequence
length on both devices, exactly as expected -- without caching, generation
step `t` redoes `O(t)` work it already did at step `t-1`, so total work is
`O(T^2)` in the final sequence length; with caching each step is `O(1)`
incremental work, `O(T)` total. CPU shows a bigger multiplier (up to 9.8x)
than GPU (up to 3.2x) at these sizes: the GPU is nowhere near compute-bound at
prompt/output lengths of a few hundred tokens on a 12-layer model, so a good
chunk of the no-cache GPU time is fixed kernel-launch/Python overhead that
caching doesn't remove, whereas the CPU actually feels the quadratic-vs-linear
compute difference.

**Memory**: CPU peak RSS barely moves (~985-986MB throughout) because it's
whole-process RSS dominated by the fixed ~124M-parameter model weights (~495MB
in fp32) plus the PyTorch/CUDA runtime import overhead; the KV cache itself is
a few hundred KB to a few MB at these lengths, far below the noise floor of
process-level RSS. GPU peak *allocated* memory is a cleaner signal and shows
something worth calling out: **caching uses slightly *more* peak memory than
no caching** at matched input/output lengths (e.g. 521.0MB vs 495.9MB at
256/64). This is the cost of the cache implementation noted in the code
comment at `model.py`'s `use_cache` branch: `torch.cat` builds a brand-new,
longer tensor on every step, so for one step both the old cache and the new
concatenated cache are simultaneously resident before the old one is freed.
At the short sequence lengths tested here, that transient doubling outweighs
the savings from not re-deriving old K/V, and net peak memory goes up, not
down. Since it is only a proof of concept, the code deliberately keeps the
simpler `torch.cat` version; a production cache would pre-allocate a
fixed-size buffer and write into it in place (also noted in that comment).

## Part C: radix-tree shared-prefix reuse vs. independent processing

`{cpu, cuda} x {radix-shared, independent}` -- all 4 combinations, plus a
`num_prompts` and a `prefix_len` sweep to show *why* the speedup behaves the
way it does. Every batch is `num_prompts` prompts sharing one common prefix of
`prefix_len` tokens, each with a distinct `suffix_len`-token suffix (so the
radix tree is one shared root segment branching into `num_prompts` leaves),
followed by `output_len` generated tokens per prompt. Greedy decoding, 1
warmup + 5 timed repeats per config, same as Part B.

### Sweep 1: number of prompts (prefix=64, suffix=16, output=16)

| device | strategy | num_prompts | mean ms | std ms | speedup (indep/radix) |
|---|---|---|---|---|---|
| cpu | independent | 2 | 530.2 | 10.9 | -- |
| cpu | radix-shared | 2 | 462.8 | 18.9 | **1.15x** |
| cpu | independent | 4 | 1011.9 | 48.9 | -- |
| cpu | radix-shared | 4 | 1008.4 | 18.5 | **1.00x** |
| cpu | independent | 8 | 1927.1 | 14.0 | -- |
| cpu | radix-shared | 8 | 1771.6 | 32.7 | **1.09x** |
| cuda | independent | 2 | 158.0 | 6.0 | -- |
| cuda | radix-shared | 2 | 162.8 | 5.3 | 0.97x |
| cuda | independent | 4 | 329.9 | 21.5 | -- |
| cuda | radix-shared | 4 | 324.1 | 22.2 | 1.02x |
| cuda | independent | 8 | 684.1 | 48.1 | -- |
| cuda | radix-shared | 8 | 665.2 | 59.1 | 1.03x |

At `output_len=16` the speedup is small (0.97x-1.15x, mostly noise). That's
expected once you separate the two phases: **decode** (autoregressive
single-token generation) costs exactly the same total work in both
strategies -- there is no cross-prompt sharing during decode, each of the
`num_prompts x output_len` generated tokens needs its own forward pass either
way -- while **prefill** is the only phase where sharing helps (the common
prefix is forwarded once instead of `num_prompts` times). With
`prefix_len=64` and `output_len=16`, decode (`num_prompts x 16` steps) is a
large fraction of total work, so the prefill saving is diluted. Sweep 2
isolates the prefill saving directly.

### Sweep 2: shared prefix length (num_prompts=8, suffix=16, output=4 -- output kept small so prefill dominates)

| device | strategy | prefix_len | mean ms | std ms | speedup (indep/radix) |
|---|---|---|---|---|---|
| cpu | independent | 32 | 572.5 | 11.2 | -- |
| cpu | radix-shared | 32 | 494.2 | 9.2 | **1.16x** |
| cpu | independent | 128 | 1032.7 | 25.5 | -- |
| cpu | radix-shared | 128 | 576.4 | 20.2 | **1.79x** |
| cpu | independent | 256 | 1524.5 | 28.3 | -- |
| cpu | radix-shared | 256 | 752.3 | 30.5 | **2.03x** |
| cuda | independent | 32 | 181.1 | 18.0 | -- |
| cuda | radix-shared | 32 | 167.5 | 9.8 | **1.08x** |
| cuda | independent | 128 | 207.5 | 7.6 | -- |
| cuda | radix-shared | 128 | 175.8 | 3.1 | **1.18x** |
| cuda | independent | 256 | 257.4 | 7.2 | -- |
| cuda | radix-shared | 256 | 206.3 | 12.5 | **1.25x** |

This is the expected story: as the shared prefix grows relative to the
per-prompt suffix, `independent`'s cost grows roughly linearly with
`num_prompts x prefix_len` (it redoes the whole prefix for every prompt) while
`radix-shared`'s prefix cost stays roughly flat (~`prefix_len`, computed
once). Speedup climbs from ~1.1x at `prefix_len=32` to ~2.0x (CPU) / ~1.25x
(GPU) at `prefix_len=256`, and would keep climbing with a longer shared prefix
or more prompts sharing it -- e.g. the "large common system prompt + many user
prompts" scenario the assignment calls out. CPU shows a larger effect than GPU
for the same reason as Part B: GPU time at these tiny per-call sizes is more
overhead-bound, so it under-represents the FLOPs actually saved.

### Memory: radix-shared costs more peak memory than independent

| device | strategy | num_prompts | peak mem |
|---|---|---|---|
| cuda | independent | 2 / 4 / 8 | 496.0 MB alloc (flat) |
| cuda | radix-shared | 2 | 510.5 MB alloc |
| cuda | radix-shared | 4 | 521.7 MB alloc |
| cuda | radix-shared | 8 | 544.2 MB alloc |

`independent` processes one prompt fully (prefill + decode) before moving to
the next and resets its cache each time, so its peak memory is that of a
*single* prompt's cache, independent of `num_prompts`. `radix-shared` keeps
every tree node's KV-cache snapshot alive for the whole traversal (so it's
available when a later sibling branch needs it), so peak memory grows with
`num_prompts` (more branch caches alive at once). **This is the real
trade-off of Part C's design**: trading compute (fewer redundant forward
passes) for memory (more cached tensors held simultaneously) -- the opposite
of Part B's cache, which trades memory for compute. CPU RSS doesn't show this
(same ~986MB-dominated-by-weights issue as Part B).

## Conclusions

- KV caching (Part B) turns per-step generation cost from `O(t)` into `O(1)`,
  giving up to ~9.8x (CPU) / ~3.2x (GPU) wall-time speedup at the sizes
  tested, growing with sequence length as theory predicts. Its naive
  `torch.cat`-based cache implementation, however, has a peak-memory cost
  that's actually *higher* than not caching at all at short lengths, because
  of the allocate-before-free churn on every step.
- Shared-prefix reuse (Part C) only pays off in the *prefill* phase; at fixed
  small prefixes and larger output lengths the benefit is swamped by decode
  cost that's identical in both strategies. Isolating prefill (small output,
  growing prefix) shows the expected trend clearly: up to ~2x (CPU) / ~1.25x
  (GPU) speedup at `prefix_len=256`, growing with the shared prefix length --
  and by the same logic, would grow further with more sharing prompts or
  longer shared prefixes (e.g. a large system prompt).
- Part C's speedup comes at the cost of *more* peak memory, not less --
  the opposite trade-off from Part B -- because reuse requires keeping
  multiple branch caches alive simultaneously.
- In both parts, GPU speedups are consistently smaller than CPU speedups at
  matched configurations, because the GPU is overhead-bound rather than
  compute-bound at this model size (123.7M params) and these sequence
  lengths; the relative benefit of both optimizations should grow on GPU too
  as prompt/output/batch sizes scale up and compute starts to dominate over
  fixed per-call overhead.
