"""
Wraps PA4/attention_bench (flash.cu vs standard_attention.cu, run in-process
against identical Q/K/V) to get mean/std timing: the binary reports one
cudaEvent-timed sample per invocation, so this calls it `--repeats` times per
(N, B, NH) config and aggregates. Also carries through memory footprint,
speedup, and the correctness (max abs diff) fields the binary already prints.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys

LINE_RE = re.compile(
    r"N=(?P<N>\d+) B=(?P<B>\d+) NH=(?P<NH>\d+) flash_ms=(?P<flash_ms>[\d.]+) "
    r"standard_ms=(?P<standard_ms>[\d.]+) speedup=(?P<speedup>[\d.eE+-]+) "
    r"max_abs_diff=(?P<max_abs_diff>[\d.eE+-]+) flash_err=(?P<flash_err>.+?) "
    r"std_err=(?P<std_err>.+?) flash_aux_bytes=(?P<flash_aux_bytes>\d+) "
    r"standard_score_matrix_bytes=(?P<standard_score_matrix_bytes>\d+) "
    r"gpu_free_bytes=(?P<gpu_free_bytes>\d+) gpu_total_bytes=(?P<gpu_total_bytes>\d+)"
)
OOM_RE = re.compile(r"standard_alloc_error=(?P<err>.+)")


def run_config(binary, N, B, NH, repeats, seed=42):
    flash_times, std_times = [], []
    max_diff = None
    aux_bytes = score_bytes = None
    oom_error = None
    for r in range(repeats):
        proc = subprocess.run([binary, str(N), str(seed + r), str(B), str(NH)],
                               capture_output=True, text=True, timeout=120)
        out = proc.stdout.strip()
        m = LINE_RE.search(out)
        if m:
            d = m.groupdict()
            flash_times.append(float(d["flash_ms"]))
            std_times.append(float(d["standard_ms"]))
            max_diff = float(d["max_abs_diff"])
            aux_bytes = int(d["flash_aux_bytes"])
            score_bytes = int(d["standard_score_matrix_bytes"])
        else:
            m2 = OOM_RE.search(out)
            oom_error = m2.group("err") if m2 else (proc.stderr or out)[-500:]
            break  # OOM is deterministic for this shape; no point repeating

    result = {"N": N, "B": B, "NH": NH, "repeats": len(flash_times)}
    if flash_times:
        result.update({
            "flash_mean_ms": statistics.mean(flash_times),
            "flash_std_ms": statistics.stdev(flash_times) if len(flash_times) > 1 else 0.0,
            "standard_mean_ms": statistics.mean(std_times),
            "standard_std_ms": statistics.stdev(std_times) if len(std_times) > 1 else 0.0,
            "speedup_standard_over_flash": statistics.mean(std_times) / statistics.mean(flash_times),
            "max_abs_diff": max_diff,
            "flash_aux_bytes": aux_bytes,
            "standard_score_matrix_bytes": score_bytes,
        })
    if oom_error:
        result["standard_oom"] = True
        result["error"] = oom_error
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="./attention_bench")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--out", default="attention_bench_repeat_results.json")
    p.add_argument("--sweep", choices=["default", "custom"], default="default")
    args = p.parse_args()

    configs = []
    # [1] flash.cu's own default shape (B=1, NH=1): shows the effect of grid parallelism.
    for N in (128, 256, 512, 1024, 2048):
        configs.append((N, 1, 1))
    # [2] matches nanoGPT/PyTorch benchmark shape (GPT-2-small: 12 heads, batch 8).
    for N in (128, 256, 512, 1024, 2048):
        configs.append((N, 8, 12))
    # [3] memory-pressure / OOM boundary for standard attention's O(N^2) score matrix.
    for N in (3072, 4096, 4608, 5120):
        configs.append((N, 8, 12))

    all_results = []
    for N, B, NH in configs:
        print(f"Running N={N} B={B} NH={NH} x{args.repeats} ...", file=sys.stderr, flush=True)
        r = run_config(args.binary, N, B, NH, args.repeats)
        all_results.append(r)
        print(json.dumps(r), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
