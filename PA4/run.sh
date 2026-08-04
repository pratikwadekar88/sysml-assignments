#!/usr/bin/env bash
# Reproduces the Flash Attention vs Standard Attention CUDA kernel benchmark.
#
# Builds flash.cu (the fixed, working version -- see the two bug fixes noted
# at the top of that file), standard_attention.cu (naive fully-parallel
# baseline), and attention_bench.cu (runs both on identical random Q/K/V in
# one process: correctness check + timing + memory).
#
# Then runs four sweeps that build on each other:
#   1. B=1, NH=1  -- flash.cu's own default problem shape. Reproduces the
#      ~127x slowdown caused by flash's kernel launching a single thread
#      block (32 threads) regardless of sequence length.
#   2. B=8, NH=12 -- same shape as the nanoGPT/PyTorch benchmark (GPT-2-small:
#      d_head=64, 12 heads, batch 8). Giving flash a real grid (B*NH=96
#      blocks) closes most of the gap (~127x -> ~4x).
#   3. Higher B*NH (32x16=512, 64x16=1024) -- tests whether more grid
#      parallelism closes the remaining gap. It plateaus at ~3-4x: flash's
#      28KB shared-mem-per-block footprint caps it at 3 resident blocks/SM
#      (6.25% occupancy) vs standard's ~100% occupancy, regardless of how
#      many total blocks are launched.
#   4. Memory-pressure / OOM boundary at B=8, NH=12 -- standard attention's
#      O(N^2) score matrix crosses the 6GB card's physical VRAM around
#      N=4096 (WSL2 quietly oversubscribes into system RAM rather than
#      hard-failing), degrading wall-clock time until it hard OOMs around
#      N=6144 (score matrix ~14.5GB, past the whole system's RAM+VRAM pool).
#      Flash's KB-scale auxiliary state never approaches this cliff.
#
# Usage: ./run.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${CUDA_ARCH:-sm_86}"   # override with CUDA_ARCH=sm_XX for a different GPU
OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/attention_bench_run.log"
: > "$LOG"

echo "==> Compiling (arch=$ARCH)" | tee -a "$LOG"
nvcc -O3 -arch="$ARCH" -o flash flash.cu
nvcc -O3 -arch="$ARCH" -o standard_attention standard_attention.cu
nvcc -O3 -arch="$ARCH" -o attention_bench attention_bench.cu

# Runs one config; tolerates a non-zero exit (expected for the OOM boundary
# test in sweep 4) without aborting the rest of the script.
run() {
    ./attention_bench "$@" 2>&1 | tee -a "$LOG"
    local status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        echo "(exited with status $status)" | tee -a "$LOG"
    fi
}

{
echo
echo "=== [1/4] B=1, NH=1 -- flash.cu's own default problem shape ==="
} | tee -a "$LOG"
for N in 128 256 512 1024 2048 4096 8192; do run "$N" 42 1 1; done

{
echo
echo "=== [2/4] B=8, NH=12 -- matches the nanoGPT/PyTorch benchmark shape ==="
} | tee -a "$LOG"
for N in 128 256 512 1024 2048; do run "$N" 42 8 12; done

{
echo
echo "=== [3/4] Higher batch*heads -- does more grid parallelism close the gap? ==="
} | tee -a "$LOG"
run 256 42 32 16
run 512 42 32 16
run 128 42 64 16
run 256 42 64 16

{
echo
echo "=== [4/4] Memory-pressure / OOM boundary at B=8, NH=12 ==="
} | tee -a "$LOG"
for N in 3072 3584 3900 4096 4608 5120 6144; do run "$N" 42 8 12; done

echo | tee -a "$LOG"
echo "Done. Full log: $LOG" | tee -a "$LOG"
