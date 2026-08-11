# PA4 Benchmark Results: FlashAttention vs. Standard Attention (CUDA)

This branch (`pa4-flash-attention-bench`) covers PA4 Part D: a custom CUDA
FlashAttention forward-pass kernel (`flash.cu`), benchmarked against a naive,
fully-parallel standard attention CUDA kernel that materializes the full
`N x N` score matrix (`standard_attention.cu`), via `attention_bench.cu`
(both kernels run in the same process on identical random Q/K/V, so timing,
memory, and the correctness check all come from one execution). Per
instruction, the PyTorch reference/tiled-loop parts of the PA4 spec (Parts
A-C) are **not** implemented here -- this branch is CUDA-only.

## What was already there vs. what changed

`flash.cu`, `standard_attention.cu`, and `attention_bench.cu` were already
implemented and working before this branch (two bugs -- a `__syncthreads()`
typo and a missing `-inf` init for the running max -- had already been fixed).
This branch adds:
- `bench_repeat.py`: wraps `attention_bench` (which times one cudaEvent-timed
  sample per process invocation) to run each `(N, B, NH)` config 10 times and
  report mean/std, since the binary itself doesn't loop internally.
- `occupancy_check.cu`: gets flash's actual achieved occupancy from CUDA's own
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API.
- This file.

## Config

- GPU: NVIDIA RTX 3050 6GB Laptop GPU, sm_86, 20 SMs, 1536 max threads/SM,
  102400 B max shared mem/SM (49152 B without opting into the larger dynamic
  shared-memory limit).
- Both kernels: `float32`, `Bc=Br=32`, `d=64` (head dim), causal-equivalent
  full attention (no masking -- both kernels compute the same unmasked
  scaled-dot-product attention).
- Each `(N, B, NH)` config: 10 repeats, each a fresh process invocation of
  `attention_bench N seed B NH` (seed incremented per repeat so each run's
  random Q/K/V differ slightly). `cudaEventRecord`/`cudaEventElapsedTime`
  brackets the timed kernel launch, `cudaDeviceSynchronize()` beforehand.
- Reproduce: `nvcc -O3 -arch=sm_86 -o flash flash.cu && nvcc -O3 -arch=sm_86 -o
  standard_attention standard_attention.cu && nvcc -O3 -arch=sm_86 -o
  attention_bench attention_bench.cu && python3 bench_repeat.py --repeats 10`

## Profiling: `ncu`/`nvprof` are not usable in this environment

`nvprof` refuses to profile compute capability >= 7.5 devices at all ("Use
NVIDIA Nsight Compute"). Nsight Compute (`ncu`) is installed
(`/opt/nvidia/nsight-compute/2025.4.0/ncu`) but fails with
`ERR_NVGPUCTRPERM`: WSL2's GPU passthrough doesn't expose the performance-
counter access `ncu` needs, and there's no way to grant it from inside WSL2
(it requires a host-side driver setting). So the occupancy/shared-memory/
throughput numbers below are **not** profiler-measured; occupancy comes from
CUDA's own runtime occupancy API (a real number computed by the CUDA driver
from the kernel's actual resource usage, just not a *profiled* one), and
memory throughput is analytically derived from the kernel's source and
labeled as such.

### Occupancy (via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`)

```
$ nvcc -O3 -arch=sm_86 -o occupancy_check occupancy_check.cu && ./occupancy_check
flash_attention_kernel: block=32 threads, dynamic smem=28672 bytes
  -> 3 resident blocks/SM, 96/1536 active threads/SM -> 6.25% occupancy

compute_scores (standard attention): block=256 threads, no dynamic smem
  -> 6 resident blocks/SM, 1536/1536 active threads/SM -> 100% occupancy
```

flash's kernel launches only `Br=32` threads per block (one thread per
Q-tile row), and each block needs 28,672 B of dynamic shared memory
(`Qi: Br*d + Kj: Bc*d + Vj: Bc*d + S: Br*Bc`, all `float`). At 102,400 B
shared memory per SM, that caps residency at 3 blocks/SM regardless of how
many total blocks (`B*NH`) the grid launches -- only `3 x 32 = 96` of the
SM's 1536 thread slots are ever filled, 6.25% occupancy. Standard attention's
`compute_scores` kernel uses a plain 256-thread block with no shared memory,
so it reaches full occupancy.

### Global memory traffic (analytically derived from `flash.cu`'s loop structure, not profiled)

The assignment's loop order is outer-over-K/V-tiles (`j`), inner-over-Q-tiles
(`i`) -- see `flash.cu`'s two nested loops. This lets each K/V tile be loaded
into shared memory *once* per `j` and reused for every Q-tile inside that
iteration (cheap: `K`, `V` are each read exactly `N*d` elements total, no
redundancy). But because a fixed Q-tile is revisited on every outer
iteration, its running statistics (`m`, `l`) and output accumulator `O`
cannot stay resident in registers across the whole K/V sweep -- they must
round-trip through global memory at *every* `(i,j)` tile pair, and `Q` itself
is reloaded from global memory at every `(i,j)` pair too (`flash.cu` lines
~60-64, ~104-116). So per `(b,h)` slice:

- `Q` reads: `Tc * N * d` (redundant -- reloaded once per outer iteration)
- `K`, `V` reads: `N * d` each (no redundancy)
- `O` reads + writes: `Tc * N * d` each (redundant round-trip every tile pair)
- `m`, `l` reads + writes: `Tc * N` each

At `N=1024, B=8, NH=12` (`Bc=Br=32` -> `Tc=Tr=32`): total traffic works out
to ≈2.52 GB. Measured `flash_mean_ms=479.2` at this config gives **≈5.25
GB/s achieved** -- nowhere near a GDDR6 mobile GPU's actual bandwidth (order
100s of GB/s), confirming the kernel is nowhere near memory-bandwidth-bound.
It's occupancy-bound: with only 96 of 1536 thread slots per SM filled, there
aren't nearly enough concurrent warps to hide memory latency, which is the
real reason flash is slower than standard attention at every tested shape
(see below) despite standard doing asymptotically more work
(`O(N^2)` score matrix vs. flash's `O(N)` auxiliary state).

## Timing, memory, and speedup: flash vs. standard

`speedup = standard_ms / flash_ms` (>1 means flash is faster; **this custom
kernel is a teaching implementation of the FlashAttention tiling algorithm,
not a competitively optimized one** -- it is slower than standard at every
shape tested, sometimes dramatically so).

### Sweep 1: `B=1, NH=1` (flash.cu's own default shape -- one thread block total)

| N | flash mean±std ms | standard mean±std ms | speedup | max\|diff\| |
|---|---|---|---|---|
| 128 | 3.19 ± 0.06 | 0.12 ± 0.06 | 0.036x | 8.4e-09 |
| 256 | 12.04 ± 0.09 | 0.16 ± 0.03 | 0.013x | 6.5e-09 |
| 512 | 47.27 ± 0.05 | 0.44 ± 0.04 | 0.009x | 7.9e-09 |
| 1024 | 168.22 ± 7.04 | 1.35 ± 0.07 | 0.008x | 1.1e-08 |
| 2048 | 662.77 ± 0.20 | 5.04 ± 0.01 | 0.008x | 1.1e-08 |

With `B*NH=1`, the grid launches exactly **one** thread block total (32
threads), regardless of `N` -- the other 19 SMs sit idle. This is the extreme
case of the occupancy problem above: flash is 27x-127x slower than standard
here, entirely a parallelism-starvation artifact, not a property of the
FlashAttention algorithm itself.

### Sweep 2: `B=8, NH=12` (matches GPT-2-small shape: 96 blocks -- one per (batch, head))

| N | flash mean±std ms | standard mean±std ms | speedup | max\|diff\| |
|---|---|---|---|---|
| 128 | 7.97 ± 0.51 | 2.21 ± 0.12 | 0.277x | 1.7e-08 |
| 256 | 33.90 ± 0.17 | 8.91 ± 0.01 | 0.263x | 1.3e-08 |
| 512 | 121.10 ± 4.93 | 30.57 ± 1.24 | 0.252x | 1.3e-08 |
| 1024 | 479.24 ± 0.18 | 119.81 ± 0.03 | 0.250x | 1.3e-08 |
| 2048 | 1920.10 ± 1.47 | 484.46 ± 0.56 | 0.252x | 1.5e-08 |

Giving flash a real grid (96 blocks vs. 1) closes most of the gap seen in
Sweep 1 (127x at N=1024 -> 4x here), but it then **plateaus at a stable ~4x
slower** and stays there as N grows -- consistent with the occupancy
ceiling above: 96 blocks is enough to give all 20 SMs work (96/20 ≈ 4.8
blocks/SM to schedule against a 3-blocks/SM residency cap), but each
resident block still only occupies 6.25% of its SM's thread slots, so
throughput never approaches standard's fully-occupied kernels regardless of
how many total blocks are queued.

### Sweep 3: memory-pressure / OOM boundary (`B=8, NH=12`, large N)

| N | flash mean±std ms | standard mean±std ms | speedup | max\|diff\| | standard score matrix | gpu free (post-run) |
|---|---|---|---|---|---|---|
| 3072 | 4332.2 ± 4.5 | 1108.2 ± 1.5 | 0.256x | 1.6e-08 | 3.62 GB | -- |
| 4096 | 7704.6 ± 4.1 | 3219.1 ± 6.5 | 0.418x | 1.6e-08 | 6.44 GB | 0 B |
| 4608 | 9757.9 ± 4.2 | 6561.5 ± 4.9 | 0.672x | 1.4e-08 | 8.15 GB | 0 B |
| 5120 | 12051.2 ± 7.8 | 10504.9 ± 14.3 | 0.872x | 1.6e-08 | 10.07 GB | 0 B |

Standard attention's `O(N^2)` score matrix (`B*NH*N*N*4` bytes) crosses the
6 GB card's *physical* VRAM between N=3072 (3.62 GB, comfortably fits) and
N=4096 (6.44 GB, exceeds the 6,441,926,656-byte total). It does **not**
hard-fail here, though: WSL2 quietly oversubscribes CUDA unified memory into
host RAM rather than returning `cudaErrorMemoryAllocation`, so `standard_ms`
degrades sharply instead of erroring (30ms/1024 -> 10,505ms/5120, a much
steeper-than-quadratic blowup once it's paging through the PCIe-attached
host memory) -- `gpu_free_bytes` reads 0 for N>=4096, confirming the card's
own 6 GB is fully consumed and the rest is coming from system RAM. Flash's
auxiliary state (`flash_aux_bytes` = `2*B*NH*N*4`, a few MB even at N=5120)
never approaches this cliff, so its "slowdown" from 3072->5120 is smooth and
roughly proportional to N (as expected: flash's compute is `O(N)` per Q-row,
`O(N^2)` total same as standard, but its memory footprint is `O(N)` not
`O(N^2)`). By N=5120, standard's oversubscription penalty has eaten enough of
its advantage that the speedup gap has almost closed (0.87x) -- extrapolating
the trend, standard would likely become the *slower* kernel a bit past this
point, purely because of the memory-oversubscription penalty, not because
flash's compute became competitive.

## Correctness

`max_abs_diff` between the two kernels' outputs stays in the `1e-8` to
`1e-9` range across every config above (float32 rounding only), comfortably
under the spec's `1e-3` cross-implementation tolerance -- flash's tiled
online-softmax recurrence is numerically converging to the same answer as
the direct softmax, as it should for a numerically-stable-but-equivalent
reformulation.

## Conclusions

- This kernel implements the FlashAttention *algorithm* correctly
  (max abs diff ~1e-8 vs. standard attention at every tested shape and
  scale), but is **not** faster than standard attention on this GPU at any
  tested `N` -- it's 4x-127x slower depending on grid shape, closing only to
  ~0.87x at the largest N tested, where standard's advantage is being eaten
  by host-RAM oversubscription rather than flash actually catching up on
  compute.
- The reason is occupancy, not algorithm: `Br=32` threads/block and 28 KB
  shared memory/block caps residency at 3 blocks/SM (6.25% of the SM's
  thread slots) regardless of grid size, while standard's plain 256-thread,
  no-shared-memory kernel reaches 100% occupancy. A production FlashAttention
  kernel (e.g. the one PyTorch's SDPA dispatches to) uses far more threads
  per block, warp-level primitives, and tensor cores to fix exactly this; this
  is a teaching-oriented implementation of the tiling/online-softmax
  *recurrence*, not a competitively tuned kernel.
- Where flash *does* win, categorically, is memory: `O(N)` auxiliary state
  vs. standard's `O(N^2)` score matrix means flash never hits the ~6 GB
  VRAM ceiling that standard hits at N≈4096 on this card, and its runtime
  degrades smoothly with N instead of falling off a cliff into host-RAM
  oversubscription.
