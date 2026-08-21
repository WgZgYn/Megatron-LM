"""Analyse profiling outputs: aggregate communication metrics and compare DP/TP/PP.

Inputs (produced by ``run.py`` in each ``outputs/<tag>_<mode>_pN/`` dir):
  * ``comm_rank*.jsonl``  — one record per collective (op, bytes, ms, gbps).
  * ``steps_rank*.jsonl`` — wall-clock step time per rank.
  * ``meta.json``         — run configuration.

Outputs:
  * a per-mode console table (per-collective count / latency / bandwidth),
  * ``summary.csv`` per mode + a combined ``summary_all.csv``,
  * ``comm_comparison.png`` — 3-panel comparison (comm ms/step, effective
    bandwidth, comm share of step time) across modes.

The matplotlib charts apply the dataviz rules: fixed categorical color order
(DP=blue, TP=orange, PP=aqua — the validated slots 1/2/3), one axis per panel,
direct labels, recessive grid. The interactive timeline lives in TensorBoard
(``tensorboard --logdir outputs/<mode>/trace``); these charts are the summary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics

# Validated categorical slots 1/2/3 (see dataviz palette): fixed, never re-ranked.
MODE_COLOR = {
    "dp": "#2a78d6",
    "tp": "#eb6834",
    "pp": "#1baf7a",
}
MODE_ORDER = ["dp", "tp", "pp"]

# Chart chrome (dataviz light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[idx]


def _load_jsonl(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_mode(out_dir):
    """Load all artifacts of one run directory into an analysis dict."""
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    warmup = int(meta.get("warmup", 0))
    comm = _load_jsonl(sorted(glob.glob(os.path.join(out_dir, "comm_rank*.jsonl"))))
    steps = _load_jsonl(sorted(glob.glob(os.path.join(out_dir, "steps_rank*.jsonl"))))

    comm = [r for r in comm if r.get("step", 0) >= warmup]
    steps = [r for r in steps if r.get("step", 0) >= warmup]

    # -- per-collective aggregates ----------------------------------------
    per_op = {}
    op_rows = {}
    for r in comm:
        op = r["op"]
        op_rows.setdefault(op, []).append(r)

    for op, rows in op_rows.items():
        ms = [r["ms"] for r in rows]
        per_op[op] = {
            "op": op,
            "count": len(rows),
            "avg_ms": statistics.fmean(ms),
            "p95_ms": _percentile(ms, 0.95),
            "total_ms": sum(ms),
            "total_bytes": sum(r["bytes"] for r in rows),
            "avg_gbps": statistics.fmean(r["gbps"] for r in rows),
        }

    # -- strategy-level aggregates ----------------------------------------
    rank_step_ms = {}
    rank_step_syncs = {}
    for r in comm:
        key = (r["rank"], r["step"])
        rank_step_ms[key] = rank_step_ms.get(key, 0.0) + r["ms"]
        rank_step_syncs[key] = rank_step_syncs.get(key, 0) + 1

    comm_ms_per_step = statistics.fmean(rank_step_ms.values()) if rank_step_ms else 0.0
    syncs_per_step = statistics.fmean(rank_step_syncs.values()) if rank_step_syncs else 0.0
    step_ms = statistics.fmean(s["ms"] for s in steps) if steps else 0.0

    total_bytes = sum(r["bytes"] for r in comm)
    total_ms = sum(r["ms"] for r in comm)
    effective_gbps = (total_bytes / 1e9) / (total_ms / 1e3) if total_ms > 0 else 0.0

    return {
        "mode": meta["mode"],
        "world_size": meta["world_size"],
        "out_dir": out_dir,
        "meta": meta,
        "per_op": per_op,
        "comm_ms_per_step": comm_ms_per_step,
        "syncs_per_step": syncs_per_step,
        "step_ms": step_ms,
        "comm_fraction_pct": 100.0 * comm_ms_per_step / step_ms if step_ms > 0 else 0.0,
        "effective_gbps": effective_gbps,
        "avg_call_ms": statistics.fmean(r["ms"] for r in comm) if comm else 0.0,
    }


def print_tables(results):
    for res in results:
        mode = res["mode"]
        print(f"\n=== {mode} (world={res['world_size']}) — {res['out_dir']} ===")
        print(f"  step time {res['step_ms']:.2f} ms | comm/step {res['comm_ms_per_step']:.2f} ms "
              f"({res['comm_fraction_pct']:.1f}%) | {res['syncs_per_step']:.1f} syncs/step | "
              f"effective {res['effective_gbps']:.2f} GB/s")
        print(f"  {'op':24} {'count':>7} {'avg_ms':>9} {'p95_ms':>9} {'avg_GB/s':>10} {'total_MB':>9}")
        print("  " + "-" * 78)
        for op in sorted(res["per_op"]):
            d = res["per_op"][op]
            print(f"  {op:24} {d['count']:7d} {d['avg_ms']:9.3f} {d['p95_ms']:9.3f} "
                  f"{d['avg_gbps']:10.2f} {d['total_bytes']/1e6:9.2f}")


def write_csvs(results, base_dir):
    per_op_fields = ["op", "count", "avg_ms", "p95_ms", "total_ms", "total_bytes", "avg_gbps"]
    strat_fields = ["mode", "world_size", "step_ms", "comm_ms_per_step", "comm_fraction_pct",
                    "syncs_per_step", "effective_gbps", "avg_call_ms"]

    strat_rows = []
    for res in results:
        with open(os.path.join(res["out_dir"], "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=per_op_fields)
            w.writeheader()
            for op in sorted(res["per_op"]):
                w.writerow({k: res["per_op"][op][k] for k in per_op_fields})
        strat_rows.append({k: res[k] for k in strat_fields})

    path = os.path.join(base_dir, "summary_all.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=strat_fields)
        w.writeheader()
        w.writerows(strat_rows)
    return path


def plot_comparison(results, base_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed; skipping plot (tables/CSVs written)")
        return None

    data = {r["mode"]: r for r in results if r["mode"] in MODE_ORDER}
    modes = [m for m in MODE_ORDER if m in data]
    if len(modes) < 2:
        print("[analyze] need >=2 modes for the comparison plot; skipping")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=SURFACE)
    panels = [
        ("comm_ms_per_step", "Comm time / step (ms)", "ms"),
        ("effective_gbps", "Effective bandwidth (GB/s)", "GB/s"),
        ("comm_fraction_pct", "Comm share of step time", "%"),
    ]

    for ax, (key, title, unit) in zip(axes, panels):
        labels = modes
        values = [data[m][key] for m in modes]
        colors = [MODE_COLOR[m] for m in modes]

        # Horizontal bars, strategies on the y-axis (identity is textual).
        y = list(range(len(modes)))
        ax.barh(y, values, color=colors, height=0.6, edgecolor="none", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, color=INK)
        ax.invert_yaxis()  # dp on top
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.set_xlabel(unit, fontsize=9, color=MUTED)

        for yi, v in zip(y, values):
            ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=9, color=INK)

        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_facecolor(SURFACE)

    fig.suptitle("Distributed-training communication: DP vs TP vs PP",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = os.path.join(base_dir, "comm_comparison.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Analyse DP/TP/PP profiling outputs")
    p.add_argument("dirs", nargs="*", help="output dirs; defaults to every outputs/* with meta.json")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(args.out_dir, "*"))
                      if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")))

    if not dirs:
        print("[analyze] no output dirs found; run `run.py` first")
        return

    results = [r for r in (load_mode(d) for d in dirs) if r]
    print_tables(results)
    csv_path = write_csvs(results, args.out_dir)
    print(f"\n[analyze] wrote per-mode summary.csv + {csv_path}")

    png = plot_comparison(results, args.out_dir)
    if png:
        print(f"[analyze] wrote {png}")
    print("[analyze] open timelines: tensorboard --logdir <mode_dir>/trace")


if __name__ == "__main__":
    main()
