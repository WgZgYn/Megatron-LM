#!/usr/bin/env bash
# Nsight Systems wrapper for one profiling run (Linux + NCCL).
#
# Usage:
#   bash launchers/nsys_wrap.sh <mode> <ngpus> [output-prefix]
#
# Captures CUDA kernels + the NVTX ranges emitted by run.py (forward /
# backward / optimizer / step phases) plus cudnn/cublas. torch.profiler is
# disabled here (--no-profile) so the two profilers do not interfere; the
# CommTimer (--measure-comm) can be kept on for per-collective latency/bandwidth
# CSV alongside the nsys timeline.

set -euo pipefail

MODE=${1:-dp}
NGPUS=${2:-4}
OUT=${3:-nsys_${MODE}_p${NGPUS}}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODEL=(--hidden 1024 --layers 8 --heads 16 --ffn 4096 --seq 512 --vocab 32000)
COMMON=(--global-batch 8 --steps 6 --warmup 2 --seed 1234)

case "$MODE" in
  dp) EXTRA=(--mode dp) ;;
  tp) EXTRA=(--mode tp) ;;
  pp) EXTRA=(--mode pp --num-microbatches "$NGPUS") ;;
  *) echo "unknown mode: $MODE (expected dp|tp|pp)" >&2; exit 2 ;;
esac

# nsys profiles the whole torchrun process tree (all ranks). Use
# --trace-fork-before-exec / --capture-range if you want to narrow the window.
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --output="$OUT" \
  --force-overwrite=true \
  torchrun --nproc_per_node="$NGPUS" --nnodes=1 run.py \
    --backend nccl --device cuda --tag "nsys_${MODE}" --no-profile \
    "${MODEL[@]}" "${COMMON[@]}" "${EXTRA[@]}"

echo "[nsys] report: ${OUT}.nsys-rep  (open with nsys-ui, or export with 'nsys stats')"
