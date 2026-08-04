#!/usr/bin/env bash
# Reproduces the Flash Attention vs Standard Attention benchmark for nanoGPT's
# CausalSelfAttention (PyTorch's scaled_dot_product_attention vs the manual
# QK^T -> mask -> softmax -> V path), on CPU and GPU, forward-only and
# forward+backward, across sequence lengths.
#
# Each configuration runs in its own subprocess (via bench_attention_driver.py
# -> bench_attention_worker.py) so CPU peak RSS and CUDA peak memory stats
# can't leak between configs.
#
# Usage: ./run.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

.venv/bin/python bench_attention_driver.py \
  --batch-size 8 \
  --cpu-seq-lens 128 256 512 1024 \
  --gpu-seq-lens 128 256 512 1024 2048 4096 \
  --warmup 3 --iters 10 \
  --out "$OUT_DIR/bench_attention_results.json" \
  2>&1 | tee "$OUT_DIR/bench_attention_run.log"

echo
echo "Done."
echo "Raw results: $OUT_DIR/bench_attention_results.json"
echo "Full log:    $OUT_DIR/bench_attention_run.log"
