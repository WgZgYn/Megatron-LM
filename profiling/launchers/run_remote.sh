#!/usr/bin/env bash
# Remote multi-GPU profiling launcher (Linux + NCCL).
#
# Usage:
#   bash launchers/run_remote.sh <mode> <ngpus> [tag]
#     mode  : dp | tp | pp
#     ngpus : number of GPUs (one parallelism dimension = world size)
#     tag   : experiment label (default: timestamp)
#
# The same "classic transformer" model config is used across modes so the
# per-mode communication profile is directly comparable.

set -euo pipefail

MODE=${1:-dp}
NGPUS=${2:-4}
TAG=${3:-$(date +%Y%m%d_%H%M%S)}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# "Classic transformer", moderate size: ~180M params, fits 24GB GPUs with a
# small micro-batch and fp32. Adjust for the target hardware.
MODEL=(--hidden 1024 --layers 8 --heads 16 --ffn 4096 --seq 512 --vocab 32000)
COMMON=(--global-batch 8 --steps 10 --warmup 2 --seed 1234)

case "$MODE" in
  dp) EXTRA=(--mode dp) ;;
  tp) EXTRA=(--mode tp) ;;
  pp) EXTRA=(--mode pp --num-microbatches "$NGPUS") ;;
  *) echo "unknown mode: $MODE (expected dp|tp|pp)" >&2; exit 2 ;;
esac

echo "[launch] mode=$MODE ngpus=$NGPUS tag=$TAG"
torchrun --nproc_per_node="$NGPUS" --nnodes=1 run.py \
  --backend nccl --device cuda --tag "$TAG" \
  "${MODEL[@]}" "${COMMON[@]}" "${EXTRA[@]}"

echo "[launch] done. outputs in outputs/${TAG}_${MODE}_p${NGPUS}/"
