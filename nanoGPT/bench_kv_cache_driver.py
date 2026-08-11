"""
Driver for PA1 Part B: sweeps {device} x {cache on/off} x {input_len, output_len}
via bench_kv_cache_worker.py, one isolated subprocess per configuration. See
bench_attention_driver.py for the rationale (clean CUDA peak-memory counters
per config; `/usr/bin/time -v` for true CPU peak RSS).
"""
import re
import json
import argparse
import subprocess
import sys

PYTHON = sys.executable
WORKER = "bench_kv_cache_worker.py"

MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def run_one(device, use_cache, input_len, output_len, repeats, warmup):
    worker_cmd = [
        PYTHON, WORKER,
        "--device", device,
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--repeats", str(repeats),
        "--warmup", str(warmup),
    ]
    if use_cache:
        worker_cmd.append("--use-cache")

    cmd = ["/usr/bin/time", "-v"] + worker_cmd if device == "cpu" else worker_cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    if result is None:
        return {
            "device": device, "use_cache": use_cache,
            "input_len": input_len, "output_len": output_len,
            "error": (proc.stderr or proc.stdout)[-800:],
        }

    if device == "cpu":
        m = MAXRSS_RE.search(proc.stderr)
        if m:
            result["cpu_peak_rss_mb"] = int(m.group(1)) / 1024.0

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-lens", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--output-lens", type=int, nargs="+", default=[32, 128])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--out", type=str, default="bench_kv_cache_results.json")
    args = p.parse_args()

    all_results = []
    devices = ["cpu"] + ([] if args.skip_gpu else ["cuda"])

    for device in devices:
        for use_cache in (False, True):
            for input_len in args.input_lens:
                for output_len in args.output_lens:
                    print(f"Running device={device} use_cache={use_cache} "
                          f"input_len={input_len} output_len={output_len} ...",
                          file=sys.stderr, flush=True)
                    r = run_one(device, use_cache, input_len, output_len, args.repeats, args.warmup)
                    all_results.append(r)
                    print(json.dumps(r), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
