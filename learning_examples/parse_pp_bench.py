#!/usr/bin/env python3
"""Parse half-layer PP benchmark logs into a clean comparison table.

Usage: python parse_pp_bench.py [/tmp/pp_bench_*.log]
       or just: python parse_pp_bench.py  (reads all /tmp/pp_bench_*.log)
"""
import re, sys, glob, os

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    # --- Total params ---
    total_params = None
    m = re.search(r'Total number of parameters in billions:\s*([\d.]+)', text)
    if m:
        total_params = float(m.group(1)) * 1000  # convert to millions

    # --- Per-rank params (Megatron's own log) ---
    # "Number of parameters in most loaded shard in billions: 0.0352"
    most_loaded = None
    m = re.search(r'Number of parameters in most loaded shard in billions:\s*([\d.]+)', text)
    if m:
        most_loaded = float(m.group(1)) * 1000

    # --- Per-rank memory (Megatron's "[Rank X]" log, after iter 1) ---
    # "[Rank 0] (after 1 iterations) memory (MB) | allocated: 426.9 | max allocated: 426.9 | reserved: 532.0 | max reserved: 532.0"
    rank_mems = {}
    for m in re.finditer(
        r'\[Rank (\d+)\].*?memory \(MB\).*?allocated:\s*([\d.]+).*?max allocated:\s*([\d.]+)',
        text
    ):
        rank = int(m.group(1))
        rank_mems[rank] = (float(m.group(2)), float(m.group(3)))

    # --- Iteration time ---
    # "[2026-08-07 16:58:11] iteration 2/30 | ... | elapsed time per iteration (ms): 289.2 |"
    iter_times = []
    for m in re.finditer(r'elapsed time per iteration \(ms\):\s*([\d.]+)', text):
        t = float(m.group(1))
        if t > 1000:
            # iter 1 includes CUDA graph capture / compilation, skip as outlier
            continue
        iter_times.append(t)

    # --- PP phase timing ---
    pp_phases = {}
    for m in re.finditer(
        r'\[PP TIMING\] step=\d+ RANK=(\d+) pp_rank=(\d+) \| '
        r'warmup=([\d.]+)ms \| 1f1b=([\d.]+)ms \| '
        r'cooldown=([\d.]+)ms \| total=([\d.]+)ms', text
    ):
        rank = int(m.group(1))
        pp_phases[rank] = {
            'warmup': float(m.group(3)),
            'f1b': float(m.group(4)),
            'cooldown': float(m.group(5)),
            'total': float(m.group(6)),
        }

    # --- Split mode ---
    # Check per-rank layer inventory: if [MODEL] log shows 2x layers vs
    # baseline (e.g. 6 vs 3 for PP=4), it's split_all_layers.
    layer_counts_per_rank = []
    for m in re.finditer(r'\[MODEL\].*?\[chunk0: (\d+) layers\]', text):
        layer_counts_per_rank.append(int(m.group(1)))
    # Also check model log for AttentionSubLayer mention (split_all produces these)
    has_attn_half = 'AttentionSubLayer' in text or 'FFNSubLayer' in text

    if has_attn_half:
        mode = 'split-all'
    elif '--split-all-layers' in text or 'pipeline_split_layers' in text:
        mode = 'split-selective' if 'pipeline_split_layers' in text else 'split-all'
    else:
        mode = 'baseline'

    # --- Layer count from [MODEL] ---
    layer_counts = set()
    for m in re.finditer(r'\[MODEL\].*?\[chunk0: (\d+) layers\]', text):
        layer_counts.add(int(m.group(1)))

    # --- [PP SCHEDULE] bubble estimate ---
    bubble_est = None
    m = re.search(r'bubble_est=([\d.]+)%', text)
    if m:
        bubble_est = float(m.group(1))

    return {
        'total_params': total_params,
        'most_loaded': most_loaded,
        'rank_mems': rank_mems,
        'iter_times': iter_times,
        'pp_phases': pp_phases,
        'mode': mode,
        'layers': sorted(layer_counts),
        'bubble_est': bubble_est,
    }


def summarize(name, data):
    if data is None:
        return f"{name:28s}  MISSING"

    # Avg iteration time (excluding iter 1)
    t = data['iter_times']
    avg_t = sum(t) / len(t) if t else 0
    t_str = f"{avg_t:7.0f}ms" if t else "    N/A"

    # Avg memory across all ranks
    mems = data['rank_mems']
    if mems:
        avg_alloc = sum(v[0] for v in mems.values()) / len(mems)
        avg_max = sum(v[1] for v in mems.values()) / len(mems)
        mem_str = f"{avg_max:6.0f}MiB"
        # Show per-rank spread if asymmetric
        vals = sorted(set(int(v[1]) for v in mems.values()))
        if len(vals) > 1:
            mem_str = f"{vals[0]}-{vals[-1]}MiB"
    else:
        mem_str = "   N/A"

    # PP total time (from timing log)
    phases = data['pp_phases']
    pp_total = sum(p['total'] for p in phases.values()) / len(phases) if phases else 0

    # Layers
    lc = data['layers']
    layers_str = "+".join(str(x) for x in lc) if lc else "?"

    # Params
    params_str = f"{data['most_loaded']:5.0f}M" if data['most_loaded'] else "  N/A"

    bubble_str = f"bubble={data['bubble_est']:.0f}%" if data['bubble_est'] else ""

    return (f"{name:28s} {params_str:>6s}  {mem_str:>10s}  {t_str:>8s}  "
            f"{pp_total:6.0f}ms  {data['mode']:15s}  "
            f"layers/rank:{layers_str:>6s}  {bubble_str}")


def main():
    logs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("/tmp/pp_bench_*.log"))
    if not logs:
        print("No log files. Usage: python parse_pp_bench.py <log1> [log2 ...]")
        sys.exit(1)

    header = (f"{'Config':28s} {'Params':>6s}  {'Memory':>10s}  {'Time/iter':>8s}  "
              f"{'PP total':>6s}  {'Mode':15s}  {'Layers':>10s}  Bubble")
    print(header)
    print("-" * len(header))

    for path in logs:
        name = os.path.basename(path).replace(".log", "").replace("pp_bench_", "")
        data = parse_log(path)
        print(summarize(name, data))


if __name__ == "__main__":
    main()
