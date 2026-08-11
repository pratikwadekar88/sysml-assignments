"""
Driver for PA1 Part C: sweeps {device} x {radix-shared, independent} x
{num_prompts} -- all combinations -- via bench_radix_worker.py, one isolated
subprocess per configuration.
"""
import re
import json
import argparse
import subprocess
import sys

PYTHON = sys.executable
WORKER = "bench_radix_worker.py"

MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def run_one(device, strategy, num_prompts, prefix_len, suffix_len, output_len, repeats, warmup):
    worker_cmd = [
        PYTHON, WORKER,
        "--device", device,
        "--strategy", strategy,
        "--num-prompts", str(num_prompts),
        "--prefix-len", str(prefix_len),
        "--suffix-len", str(suffix_len),
        "--output-len", str(output_len),
        "--repeats", str(repeats),
        "--warmup", str(warmup),
    ]
    cmd = ["/usr/bin/time", "-v"] + worker_cmd if device == "cpu" else worker_cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    if result is None:
        return {
            "device": device, "strategy": strategy, "num_prompts": num_prompts,
            "error": (proc.stderr or proc.stdout)[-800:],
        }

    if device == "cpu":
        m = MAXRSS_RE.search(proc.stderr)
        if m:
            result["cpu_peak_rss_mb"] = int(m.group(1)) / 1024.0

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-prompts", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--suffix-len", type=int, default=16)
    p.add_argument("--output-len", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--out", type=str, default="bench_radix_results.json")
    args = p.parse_args()

    all_results = []
    devices = ["cpu"] + ([] if args.skip_gpu else ["cuda"])

    for device in devices:
        for strategy in ("radix-shared", "independent"):
            for num_prompts in args.num_prompts:
                print(f"Running device={device} strategy={strategy} num_prompts={num_prompts} ...",
                      file=sys.stderr, flush=True)
                r = run_one(device, strategy, num_prompts, args.prefix_len, args.suffix_len,
                            args.output_len, args.repeats, args.warmup)
                all_results.append(r)
                print(json.dumps(r), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
