#!/bin/bash

# Profile a 124M GPT-3 (GPT-2 small) model across DP / TP / PP using
# Megatron-LM's built-in profiling, not hand-rolled timers.
#
# Usage:
#   bash examples/gpt3/train_gpt3_124m_profile.sh MODE NGPUS VOCAB_FILE MERGE_FILE
#
#     MODE : dp | tp | pp   (one parallelism dimension, uses all NGPUS)
#     NGPUS: number of GPUs on this node (torchrun --nproc_per_node)
#     VOCAB_FILE / MERGE_FILE: gpt2 vocab/merge (needed for --mock-data)
#
# Env (optional):
#   PROFILE_MODE : timers | torch | nsys   (default timers)
#       timers -> --timing-log-level 2 : per-phase min/max timing to stdout
#                 (forward/backward-compute, forward/backward-send/recv,
#                  all-grads-sync, params-all-gather, ...)
#       torch  -> --profile --use-pytorch-profiler : trace to $LOG_DIR/tb
#                 (open with `tensorboard --logdir $LOG_DIR/tb`)
#       nsys   -> --profile + cudaProfilerApi : report to $LOG_DIR/nsys_$MODE.nsys-rep
#   SEQ_LENGTH   : sequence length (default 2048)
#   PRECISION    : fp16 | bf16 (default fp16; V100 -> fp16, Ampere+ -> bf16)
#   TRAIN_ITERS  : total iterations (default 20)
#   PROFILE_STEP_START / PROFILE_STEP_END : steps captured by torch/nsys (default 15/18)
#   MASTER_ADDR / MASTER_PORT
#   LOG_DIR      : output dir (default /tmp/gpt3_124m_profile)
#
# Examples:
#   bash examples/gpt3/train_gpt3_124m_profile.sh dp 4 vocab.json merges.txt
#   PROFILE_MODE=torch bash examples/gpt3/train_gpt3_124m_profile.sh pp 4 vocab.json merges.txt
#   PROFILE_MODE=nsys bash examples/gpt3/train_gpt3_124m_profile.sh tp 4 vocab.json merges.txt

set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODE=${1:-dp}
NGPUS=${2:-4}
VOCAB_FILE=${3:-}
MERGE_FILE=${4:-}
if [[ -z "$VOCAB_FILE" || -z "$MERGE_FILE" ]]; then
    echo "Usage: $0 MODE NGPUS VOCAB_FILE MERGE_FILE" >&2
    exit 1
fi

PROFILE_MODE=${PROFILE_MODE:-timers}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
PRECISION=${PRECISION:-fp16}
TRAIN_ITERS=${TRAIN_ITERS:-20}
PROFILE_STEP_START=${PROFILE_STEP_START:-15}
PROFILE_STEP_END=${PROFILE_STEP_END:-18}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
LOG_DIR=${LOG_DIR:-/tmp/gpt3_124m_profile}
mkdir -p "$LOG_DIR"

# ---- 124M GPT-3 (GPT-2 small) config -------------------------------------
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
FFN_HIDDEN_SIZE=3072          # 4 * hidden_size
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NGPUS))   # num_microbatches == NGPUS for tp/pp

DISTRIBUTED_ARGS=(
    --nproc_per_node "$NGPUS"
    --nnodes 1
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --attention-backend auto
)

# ---- one parallelism dimension, DP implied = NGPUS / (tp * pp) ------------
case "$MODE" in
  dp) PARALLEL_ARGS=(--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1) ;;
  tp) PARALLEL_ARGS=(--tensor-model-parallel-size "$NGPUS" --pipeline-model-parallel-size 1) ;;
  pp) PARALLEL_ARGS=(--tensor-model-parallel-size 1 --pipeline-model-parallel-size "$NGPUS") ;;
  *) echo "unknown mode: $MODE (expected dp|tp|pp)" >&2; exit 2 ;;
esac

TRAIN_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --lr 6.0e-5
    --min-lr 6.0e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.001
    --lr-decay-iters 1000
)
if [[ "$PRECISION" == "fp16" ]]; then
    TRAIN_ARGS+=(--fp16)
elif [[ "$PRECISION" == "bf16" ]]; then
    TRAIN_ARGS+=(--bf16)
else
    echo "PRECISION must be fp16 or bf16, got: $PRECISION" >&2
    exit 1
fi

DATA_ARGS=(
    --mock-data
    --vocab-file "$VOCAB_FILE"
    --merge-file "$MERGE_FILE"
    --split 949,50,1
)

# ---- profiling args -------------------------------------------------------
PROFILE_ARGS=()
case "$PROFILE_MODE" in
  timers)
    PROFILE_ARGS=(
        --timing-log-level 2
        --timing-log-option minmax
        --log-interval 1
    )
    ;;
  torch)
    PROFILE_ARGS=(
        --profile
        --use-pytorch-profiler
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks 0
        --tensorboard-dir "$LOG_DIR/tb"
        --log-interval 1
    )
    ;;
  nsys)
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks 0
        --log-interval 1
    )
    ;;
  *)
    echo "PROFILE_MODE must be timers|torch|nsys, got: $PROFILE_MODE" >&2
    exit 2
    ;;
esac

LOG_ARGS=(--eval-interval 1000 --eval-iters 0)

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}")

echo "[profile] mode=$MODE ngpus=$NGPUS profile=$PROFILE_MODE precision=$PRECISION seq=$SEQ_LENGTH"

if [[ "$PROFILE_MODE" == "nsys" ]]; then
    nsys profile -s none -t nvtx,cuda \
        -o "$LOG_DIR/nsys_${MODE}" --force-overwrite true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        "${CMD[@]}"
    echo "[profile] nsys report: $LOG_DIR/nsys_${MODE}.nsys-rep"
else
    "${CMD[@]}"
fi

echo "[profile] timers in stdout; torch trace in $LOG_DIR/tb (if PROFILE_MODE=torch)"
