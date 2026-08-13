#!/usr/bin/env python3
"""Parse structured PP benchmark logs and compare controlled experiments."""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys


META_PATTERN = re.compile(r'^\[BENCH-META\]\s+(\{.*\})$', re.MULTILINE)
ITERATION_PATTERN = re.compile(
    r'iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)',
    re.IGNORECASE,
)


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _legacy_metadata(path, text):
    """Best-effort compatibility for logs created before BENCH-META."""
    name = os.path.basename(path).replace('.log', '').replace('pp_bench_', '')
    pp_match = re.match(r'pp(\d+)', name)

    def distribution(option):
        match = re.search(rf'--{option}\s+([\d\s]+?)(?:\s+--|\s*\n|\s*$)', text)
        return [int(item) for item in match.group(1).split()] if match else None

    return {
        'experiment': name,
        'configuration': name,
        'repeat': None,
        'mode': 'legacy',
        'pp': int(pp_match.group(1)) if pp_match else None,
        'dp': None,
        'tp': None,
        'full_distribution': distribution('decoder-num-layers-per-pipeline-stage'),
        'half_distribution': distribution('decoder-num-half-layers-per-pipeline-stage'),
        'warmup_iters': 1,
    }


def parse_log(path):
    with open(path, encoding='utf-8', errors='replace') as log_file:
        text = log_file.read()

    meta_match = META_PATTERN.search(text)
    metadata = json.loads(meta_match.group(1)) if meta_match else _legacy_metadata(path, text)
    warmup_iters = int(metadata.get('warmup_iters', 0))

    samples = [
        float(elapsed)
        for iteration, elapsed in ITERATION_PATTERN.findall(text)
        if int(iteration) > warmup_iters
    ]
    if not samples:
        all_times = [
            float(value)
            for value in re.findall(r'elapsed time per iteration \(ms\):\s*([\d.]+)', text)
        ]
        samples = all_times[warmup_iters:]

    rank_memory = {}
    for match in re.finditer(
        r'\[Rank (\d+)\].*?memory \(MB\).*?max allocated:\s*([\d.]+)', text
    ):
        rank = int(match.group(1))
        rank_memory[rank] = max(rank_memory.get(rank, 0.0), float(match.group(2)))

    rank_params = {}
    for match in re.finditer(r'\[MODEL\] RANK=\s*(\d+).*?params=([\d.]+)M', text):
        rank_params[int(match.group(1))] = float(match.group(2))

    phase_totals = {}
    for match in re.finditer(r'\[PP TIMING\].*?RANK=(\d+).*?total=([\d.]+)ms', text):
        phase_totals.setdefault(int(match.group(1)), []).append(float(match.group(2)))
    rank_phase_medians = {
        rank: statistics.median(values[warmup_iters:])
        for rank, values in phase_totals.items()
        if values[warmup_iters:]
    }

    result = dict(metadata)
    result.update(
        path=path,
        complete='[before the start of training step]' in text and bool(samples),
        samples=len(samples),
        median_ms=statistics.median(samples) if samples else None,
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples) if samples else None,
        max_ms=max(samples) if samples else None,
        max_memory_mib=max(rank_memory.values()) if rank_memory else None,
        memory_imbalance_mib=(
            max(rank_memory.values()) - min(rank_memory.values()) if rank_memory else None
        ),
        parameter_imbalance_m=(
            max(rank_params.values()) - min(rank_params.values()) if rank_params else None
        ),
        stage_time_ratio=(
            max(rank_phase_medians.values()) / min(rank_phase_medians.values())
            if rank_phase_medians and min(rank_phase_medians.values()) > 0
            else None
        ),
    )
    return result


def _format_distribution(result):
    if result.get('half_distribution'):
        return 'half=' + str(result['half_distribution']).replace(' ', '')
    if result.get('full_distribution'):
        return 'full=' + str(result['full_distribution']).replace(' ', '')
    return 'default'


def print_table(results):
    header = (
        f"{'Experiment':30} {'Mode':28} {'Distribution':22} "
        f"{'N':>4} {'Median':>9} {'P95':>9} {'MaxMem':>10} {'StageRatio':>10}"
    )
    print(header)
    print('-' * len(header))
    for result in results:
        median = f"{result['median_ms']:.1f}ms" if result['median_ms'] is not None else 'N/A'
        p95 = f"{result['p95_ms']:.1f}ms" if result['p95_ms'] is not None else 'N/A'
        memory = (
            f"{result['max_memory_mib']:.0f}MiB"
            if result['max_memory_mib'] is not None else 'N/A'
        )
        ratio = (
            f"{result['stage_time_ratio']:.3f}"
            if result['stage_time_ratio'] is not None else 'N/A'
        )
        print(
            f"{result['experiment']:30} {result.get('mode', '?'):28} "
            f"{_format_distribution(result):22} {result['samples']:4d} "
            f"{median:>9} {p95:>9} {memory:>10} {ratio:>10}"
        )

    groups = {}
    for result in results:
        groups.setdefault(result.get('configuration', result['experiment']), []).append(result)
    if any(len(group) > 1 for group in groups.values()):
        print('\nConfiguration aggregates (median of run medians):')
        print(f"{'Configuration':30} {'Runs':>4} {'Median':>9} {'Range':>20}")
        print('-' * 67)
        for configuration, group in groups.items():
            medians = [item['median_ms'] for item in group if item['median_ms'] is not None]
            if not medians:
                continue
            print(
                f"{configuration:30} {len(medians):4d} "
                f"{statistics.median(medians):8.1f}ms "
                f"[{min(medians):.1f}, {max(medians):.1f}]ms"
            )


def write_csv(path, results):
    fields = [
        'experiment', 'configuration', 'repeat', 'mode', 'pp', 'dp', 'tp', 'full_distribution',
        'half_distribution', 'samples', 'median_ms', 'p95_ms', 'min_ms',
        'max_ms', 'max_memory_mib', 'memory_imbalance_mib',
        'parameter_imbalance_m', 'stage_time_ratio', 'complete', 'path',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='*')
    parser.add_argument('--csv', dest='csv_path')
    args = parser.parse_args()

    paths = args.logs or sorted(glob.glob('/tmp/pp4_partition_bench/*.log'))
    if not paths:
        parser.error('no benchmark logs found')

    results = [parse_log(path) for path in sorted(paths)]
    print_table(results)
    if args.csv_path:
        write_csv(args.csv_path, results)

    if any(result['samples'] == 0 for result in results):
        sys.exit(2)


if __name__ == '__main__':
    main()
