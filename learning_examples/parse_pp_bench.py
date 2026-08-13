#!/usr/bin/env python3
"""Parse half-layer PP benchmark logs into a clean comparison table.

Usage: python parse_pp_bench.py [/tmp/pp_bench_*.log]
       or just: python parse_pp_bench.py  (reads all /tmp/pp_bench_*.log)
"""
import re, sys, glob, os, statistics


def parse_int_list(value):
    """Parse either a shell-style list ("11 13") or a Python list ("[11, 13]")."""
    if not value or value.strip() == "None":
        return None
    return [int(item) for item in re.findall(r'\d+', value)] or None


def find_distribution(text, cli_option, config_key, debug_key=None):
    """Read a stage distribution from CLI, Megatron config, or debug output."""
    patterns = [
        rf'--{re.escape(cli_option)}\s+([\d\s]+?)(?:\s+--|\s*\n|\s*$)',
        rf'\b{re.escape(config_key)}\s+\.*\s+(\[[^\n]*\]|None)',
    ]
    if debug_key:
        patterns.append(rf'\b{re.escape(debug_key)}=(\[[^\n]*?\]|None)')

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            distribution = parse_int_list(match.group(1))
            if distribution is not None:
                return distribution
    return None


def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    # --- Config from filename ---
    # e.g. pp2_baseline, pp4_split_uniform, pp2_half_layer_11_13
    name = os.path.basename(path).replace(".log", "").replace("pp_bench_", "")
    pp_size = 2 if name.startswith("pp2") else 4 if name.startswith("pp4") else None

    # --- Config from log content ---
    mode = "baseline"
    decoder_num_layers = find_distribution(
        text,
        "decoder-num-layers-per-pipeline-stage",
        "decoder_num_layers_per_pipeline_stage",
        "decoder_dist",
    )
    decoder_num_half_layers = find_distribution(
        text,
        "decoder-num-half-layers-per-pipeline-stage",
        "decoder_num_half_layers_per_pipeline_stage",
        "half_dist",
    )
    pipeline_split_layers = find_distribution(
        text,
        "pipeline-split-layers",
        "pipeline_split_layers",
    )

    split_all_enabled = (
        '--split-all-layers' in text
        or re.search(r'\bsplit_all=True\b', text)
        or re.search(r'\bsplit_all_layers\s+\.*\s+True\b', text)
    )
    if split_all_enabled:
        mode = "split-all"

    if decoder_num_half_layers:
        mode = "split-all(half)"

    # If half-layer distribution was auto-derived, detect from log
    m = re.search(r'auto-derived pipeline_split_layers=\[([\d,\s]*)\]', text)
    if m and m.group(1).strip():
        pipeline_split_layers = [int(x.strip()) for x in m.group(1).split(',')]

    # --- Per-rank memory ---
    # "[Rank 0] (after 1 iterations) memory (MB) | allocated: 956.1 | max allocated: 956.1"
    rank_mems = {}
    for m in re.finditer(
        r'\[Rank (\d+)\].*?memory \(MB\)\s*\|\s*allocated:\s*([\d.]+)\s*\|\s*max allocated:\s*([\d.]+)',
        text
    ):
        rank = int(m.group(1))
        rank_mems[rank] = (float(m.group(2)), float(m.group(3)))

    # --- Iteration times (skip iter 1 = compilation / CUDA graph capture) ---
    # "elapsed time per iteration (ms): 641.8"
    iter_times = []
    for m in re.finditer(r'elapsed time per iteration \(ms\):\s*([\d.]+)', text):
        t = float(m.group(1))
        if t > 1000:  # skip compilation outlier
            continue
        iter_times.append(t)

    # --- Total params ---
    total_params = None
    m = re.search(r'Total number of parameters in billions:\s*([\d.]+)', text)
    if m:
        total_params = float(m.group(1)) * 1000

    # --- PP SCHEDULE bubble ---
    bubble_est = None
    m = re.search(r'bubble_est=([\d.]+)%', text)
    if m:
        bubble_est = float(m.group(1))

    return {
        'name': name, 'pp_size': pp_size, 'mode': mode,
        'decoder_num_layers': decoder_num_layers,
        'decoder_num_half_layers': decoder_num_half_layers,
        'pipeline_split_layers': pipeline_split_layers,
        'rank_mems': rank_mems,
        'iter_times': iter_times,
        'total_params': total_params,
        'bubble_est': bubble_est,
    }


def summarize(data):
    if data is None:
        return f"{'MISSING':28s}"

    # Config string
    mode = data['mode']
    cfg = ""
    if data['decoder_num_half_layers']:
        cfg = f"half=[{','.join(map(str, data['decoder_num_half_layers']))}]"
    elif data['decoder_num_layers']:
        cfg = f"full=[{','.join(map(str, data['decoder_num_layers']))}]"
    if mode == 'baseline':
        cfg = "uniform"

    # Iteration time (median, skip first warmup iter)
    t = data['iter_times']
    time_str = f"{statistics.median(t):.0f}ms" if t else "N/A"

    # Memory: per-rank max_allocated, show each rank
    mems = data['rank_mems']
    if mems:
        mem_str = "  ".join(f"R{r}={int(v[1])}MiB" for r, v in sorted(mems.items()))
    else:
        mem_str = "N/A"

    pp = data.get('pp_size', '?')

    return (f"{data['name']:28s}  {mem_str:>50s}  {time_str:>8s}  "
            f"PP={pp}  {mode:12s}  {cfg}")


def main():
    logs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("/tmp/pp_bench_*.log"))
    if not logs:
        print("No log files. Usage: python parse_pp_bench.py <log1> [log2 ...]")
        sys.exit(1)

    header = (f"{'Config':28s}  {'Memory (max allocated per rank)':>50s}  {'Time(median)':>8s}  "
              f"{'PP':>4s}  {'Mode':12s}  {'Distribution'}")
    print(header)
    print("-" * len(header))

    results = []
    for path in sorted(logs):
        data = parse_log(path)
        if data:
            results.append(data)
            print(summarize(data))

    if not results:
        print("No valid log files found.")


if __name__ == "__main__":
    main()
