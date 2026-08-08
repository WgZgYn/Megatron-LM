#!/usr/bin/env python3
"""Parse half-layer PP benchmark logs and print a clean comparison table.

Usage: python parse_pp_bench.py [/tmp/pp_bench_pp2_baseline.log ...]
       or just: python parse_pp_bench.py  (reads all /tmp/pp_bench_*.log)
"""
import re, sys, glob, os

def parse_log(path):
    """Extract key metrics from a benchmark log file."""
    if not os.path.exists(path):
        return None

    with open(path) as f:
        text = f.read()

    # per-rank params: "> number of parameters on (tensor, pipeline) model parallel rank (0, 0): 85.2M"
    params = {}
    for m in re.finditer(r'> number of parameters on.*rank \((\d+), (\d+)\): ([\d.]+)([KMB])',
                          text):
        tp, pp, val, unit = m.groups()
        val = float(val)
        if unit == 'K': val /= 1000
        elif unit == 'B': val *= 1000
        key = f"tp{tp}_pp{pp}"
        if key not in params:
            params[key] = val

    # per-rank memory: "[STEP ...] RANK=0 tp=0 pp=0 dp=0 | mem_used=1234MiB"
    mems = {}
    for m in re.finditer(r'\[STEP\s+\d+\] RANK=(\d+) tp=(\d+) pp=(\d+).*?mem_used=([\d.]+)MiB',
                          text):
        rank, tp, pp, mb = m.groups()
        key = f"r{rank}_tp{tp}_pp{pp}"
        mems[key] = float(mb)

    # iteration time: "elapsed_time_per_iteration_ms: 4949.5"
    times = []
    for m in re.finditer(r'elapsed_time_per_iteration_ms:\s*([\d.]+)', text):
        times.append(float(m.group(1)))

    # PP TIMING (latest few): "warmup=19ms | 1f1b=79ms | cooldown=109ms | total=207ms"
    pp_timings = []
    for m in re.finditer(r'\[PP TIMING\].*?warmup=([\d.]+)ms.*?1f1b=([\d.]+)ms.*?cooldown=([\d.]+)ms.*?total=([\d.]+)ms', text):
        pp_timings.append(tuple(map(float, m.groups())))

    # config: "split-all-layers" or "pipeline-split-layers"
    split_mode = "baseline"
    if '--split-all-layers' in text:
        split_mode = "split-all"
    elif 'pipeline_split_layers' in text:
        split_mode = "split-selective"

    # layer distribution from [MODEL] log
    layers_info = []
    for m in re.finditer(r'\[MODEL\] RANK=\s*\d+.*?\[chunk0: (\d+) layers\]', text):
        layers_info.append(int(m.group(1)))

    return {
        'params': params,
        'mem': mems,
        'times': times,
        'timings': pp_timings,
        'mode': split_mode,
        'layers': layers_info,
    }


def summarize(name, data):
    if data is None:
        return f"{name:25s}  MISSING"
    t = data['times']
    tt = data['timings']
    avg_time = sum(t) / len(t) if t else 0
    avg_total = sum(x[3] for x in tt) / len(tt) if tt else 0
    mem_vals = list(data['mem'].values())
    avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else 0
    params_vals = list(data['params'].values())
    avg_params = sum(params_vals) / len(params_vals) if params_vals else 0
    layers = data['layers']
    layers_str = f"{layers[0]}+{layers[-1]}" if len(layers) >= 2 else str(layers) if layers else "?"
    return (f"{name:25s} {avg_params:6.0f}M  {avg_mem:6.0f}MiB  "
            f"{avg_time:8.0f}ms  {avg_total:5.0f}ms  {data['mode']:15s}  "
            f"layers/rank: {layers_str}")


def main():
    logs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("/tmp/pp_bench_*.log"))
    if not logs:
        print("No log files found. Usage: python parse_pp_bench.py <log1> <log2> ...")
        sys.exit(1)

    print(f"{'Config':25s} {'Params':>6s}  {'Memory':>6s}  {'Time/iter':>8s}  "
          f"{'PP total':>5s}  {'Mode':15s}  Extra")
    print("-" * 110)

    for path in logs:
        name = os.path.basename(path).replace(".log", "").replace("pp_bench_", "")
        data = parse_log(path)
        print(summarize(name, data))


if __name__ == "__main__":
    main()
