#!/bin/bash

# Controlled PP=4 comparison for default, uneven full-layer, and half-layer plans.
# Usage: bash examples/gpt3/train_gpt3_half_layer_bench.sh VOCAB_FILE MERGE_FILE
# Optional: ONLY_EXPERIMENT=pp4_manual_uniform bash ...

set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

VOCAB_FILE=${1:-}
MERGE_FILE=${2:-}
if [[ -z "$VOCAB_FILE" || -z "$MERGE_FILE" ]]; then
    echo "Usage: $0 VOCAB_FILE MERGE_FILE"
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
REPEATS=${REPEATS:-3}
BENCH_PROFILE=${BENCH_PROFILE:-small}
LOG_DIR=${LOG_DIR:-/tmp/pp4_partition_bench}
mkdir -p "$LOG_DIR"

if [[ "$BENCH_PROFILE" == "small" ]]; then
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTENTION_HEADS=12
    SEQUENCE_LENGTH=128
    MAX_POSITION_EMBEDDINGS=128
    MICRO_BATCH_SIZE=2
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
    TRAIN_ITERS=${TRAIN_ITERS:-40}
    WARMUP_ITERS=${WARMUP_ITERS:-10}
    LEARNING_RATE=6.0e-5
    MIN_LEARNING_RATE=6.0e-6
    LR_WARMUP_FRACTION=.001
    MODEL_EXTRA_ARGS=()
    TRAINING_EXTRA_ARGS=()
elif [[ "$BENCH_PROFILE" == "large" ]]; then
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTENTION_HEADS=12
    SEQUENCE_LENGTH=1024
    MAX_POSITION_EMBEDDINGS=1024
    MICRO_BATCH_SIZE=8
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
    TRAIN_ITERS=${TRAIN_ITERS:-1500}
    WARMUP_ITERS=${WARMUP_ITERS:-100}
    LEARNING_RATE=3.0e-4
    MIN_LEARNING_RATE=1.0e-5
    LR_WARMUP_FRACTION=0.01
    MODEL_EXTRA_ARGS=(--attention-backend auto)
    TRAINING_EXTRA_ARGS=(
        --lr-decay-iters 1000
    )
else
    echo "BENCH_PROFILE must be small or large, got: $BENCH_PROFILE" >&2
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node 4
    --nnodes 1
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --seq-length "$SEQUENCE_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    "${MODEL_EXTRA_ARGS[@]}"
)

TRAIN_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr "$LEARNING_RATE"
    --lr-decay-style cosine
    --min-lr "$MIN_LEARNING_RATE"
    --lr-warmup-fraction "$LR_WARMUP_FRACTION"
    "${TRAINING_EXTRA_ARGS[@]}"
)

DATA_ARGS=(
    --mock-data
    --vocab-file "$VOCAB_FILE"
    --merge-file "$MERGE_FILE"
    --split 949,50,1
)

LOG_ARGS=(--log-interval 1 --eval-interval 1000 --eval-iters 0)
PARALLEL_ARGS=(--tensor-model-parallel-size 1 --pipeline-model-parallel-size 4)

run_exp() {
    local label=$1
    local mode=$2
    local full_dist=$3
    local half_dist=$4
    shift 4

    if [[ -n "${ONLY_EXPERIMENT:-}" && "$ONLY_EXPERIMENT" != "$label" ]]; then
        return
    fi

    local repeat
    for repeat in $(seq 1 "$REPEATS"); do
        local experiment="${label}_r${repeat}"
        local log_path="$LOG_DIR/pp_bench_${experiment}.log"
        {
            printf '[BENCH-META] {"experiment":"%s","configuration":"%s","repeat":%s,' \
                "$experiment" "$label" "$repeat"
            printf '"profile":"%s","mode":"%s","pp":4,"dp":1,"tp":1,' \
                "$BENCH_PROFILE" "$mode"
            printf '"num_layers":%s,"full_distribution":%s,"half_distribution":%s,' \
                "$NUM_LAYERS" \
                "$full_dist" "$half_dist"
            printf '"micro_batch_size":%s,"global_batch_size":%s,"sequence_length":%s,' \
                "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$SEQUENCE_LENGTH"
            printf '"train_iters":%s,"warmup_iters":%s}\n' "$TRAIN_ITERS" "$WARMUP_ITERS"

            torchrun "${DISTRIBUTED_ARGS[@]}" \
                pretrain_gpt.py "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" \
                "${PARALLEL_ARGS[@]}" "$@" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}"
        } 2>&1 | tee "$log_path"
    done
}

if [[ "$BENCH_PROFILE" == "small" ]]; then
    # E0/E1 isolate the overhead of supplying an explicit but identical plan.
    run_exp "pp4_default_uniform" "default_uniform" "[3,3,3,3]" "null"
    run_exp "pp4_manual_uniform" "manual_uniform" "[3,3,3,3]" "null" \
        --decoder-num-layers-per-pipeline-stage 3 3 3 3

    # E2/E3 compare native first/last controls with the equivalent explicit plan.
    run_exp "pp4_first_last_2_2" "native_first_last" "[2,4,4,2]" "null" \
        --decoder-first-pipeline-num-layers 2 --decoder-last-pipeline-num-layers 2
    run_exp "pp4_explicit_2_4_4_2" "manual_first_last_equivalent" "[2,4,4,2]" "null" \
        --decoder-num-layers-per-pipeline-stage 2 4 4 2

    # E4/E5 cover moderate and deliberately severe full-layer imbalance.
    run_exp "pp4_uneven_2_3_4_3" "moderate_uneven" "[2,3,4,3]" "null" \
        --decoder-num-layers-per-pipeline-stage 2 3 4 3
    run_exp "pp4_uneven_1_2_6_3" "severe_uneven" "[1,2,6,3]" "null" \
        --decoder-num-layers-per-pipeline-stage 1 2 6 3

    # E6/E7/E8 are the half-layer candidates.
    run_exp "pp4_half_5_7_6_6" "half_layer" "null" "[5,7,6,6]" \
        --decoder-num-half-layers-per-pipeline-stage 5 7 6 6
    run_exp "pp4_half_4_8_9_3" "half_near_best_move_to_stage2" "null" "[4,8,9,3]" \
        --decoder-num-half-layers-per-pipeline-stage 4 8 9 3
    run_exp "pp4_half_5_8_8_3" "half_near_best_reduce_edges" "null" "[5,8,8,3]" \
        --decoder-num-half-layers-per-pipeline-stage 5 8 8 3
else
    # The large profile confirms only the small-profile winner and its two
    # nearby half-layer candidates. Keep this matrix separate from small logs.
    run_exp "pp4_large_default_uniform" "large_default_uniform" "[3,3,3,3]" "null"
    run_exp "pp4_large_explicit_2_4_4_2" "large_manual_first_last" "[2,4,4,2]" "null" \
        --decoder-num-layers-per-pipeline-stage 2 4 4 2
    run_exp "pp4_large_half_4_8_9_3" "large_half_near_best_stage2" "null" "[4,8,9,3]" \
        --decoder-num-half-layers-per-pipeline-stage 4 8 9 3
    run_exp "pp4_large_half_5_8_8_3" "large_half_near_best_first" "null" "[5,8,8,3]" \
        --decoder-num-half-layers-per-pipeline-stage 5 8 8 3
fi

echo "Logs: $LOG_DIR"
echo "Parse: python learning_examples/parse_pp_bench.py $LOG_DIR/*.log"
