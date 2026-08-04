"""
Driver: runs bench_attention_worker.py once per configuration in its own
subprocess and collects the results. For CPU runs, wraps the subprocess with
`/usr/bin/time -v` to capture true peak RSS (Maximum resident set size) of
that isolated process. For GPU runs, uses the CUDA allocator's own peak-memory
counters (printed by the worker), which are precise without needing an
external wrapper.
"""
import re
import json
import argparse
import subprocess
import sys

PYTHON = sys.executable
WORKER = "bench_attention_worker.py"

MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def run_one(device, mode, backward, batch_size, seq_len, warmup, iters, n_embd, n_head):
    worker_cmd = [
        PYTHON, WORKER,
        "--device", device,
        "--mode", mode,
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--n-embd", str(n_embd),
        "--n-head", str(n_head),
    ]
    if backward:
        worker_cmd.append("--backward")

    if device == "cpu":
        cmd = ["/usr/bin/time", "-v"] + worker_cmd
    else:
        cmd = worker_cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    if result is None:
        return {
            "device": device, "mode": mode, "backward": backward,
            "batch_size": batch_size, "seq_len": seq_len,
            "error": (proc.stderr or proc.stdout)[-800:],
        }

    if device == "cpu":
        m = MAXRSS_RE.search(proc.stderr)
        if m:
            result["cpu_peak_rss_mb"] = int(m.group(1)) / 1024.0

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu-seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--gpu-seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--out", type=str, default="bench_attention_results.json")
    args = p.parse_args()

    all_results = []

    plans = [("cpu", args.cpu_seq_lens)]
    if not args.skip_gpu:
        plans.append(("cuda", args.gpu_seq_lens))

    for device, seq_lens in plans:
        for backward in (False, True):
            for seq_len in seq_lens:
                for mode in ("flash", "standard"):
                    print(f"Running device={device} backward={backward} seq_len={seq_len} mode={mode} ...",
                          file=sys.stderr, flush=True)
                    r = run_one(device, mode, backward, args.batch_size, seq_len,
                                args.warmup, args.iters, args.n_embd, args.n_head)
                    all_results.append(r)
                    print(json.dumps(r), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
