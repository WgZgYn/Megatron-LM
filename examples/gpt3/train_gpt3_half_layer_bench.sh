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
TRAIN_ITERS=${TRAIN_ITERS:-60}
WARMUP_ITERS=${WARMUP_ITERS:-10}
REPEATS=${REPEATS:-3}
LOG_DIR=${LOG_DIR:-/tmp/pp4_partition_bench}
mkdir -p "$LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node 4
    --nnodes 1
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 128
    --max-position-embeddings 128
)

TRAIN_ARGS=(
    --micro-batch-size 2
    --global-batch-size 32
    --train-iters "$TRAIN_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
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
            printf '"mode":"%s","pp":4,"dp":1,"tp":1,' "$mode"
            printf '"num_layers":12,"full_distribution":%s,"half_distribution":%s,' \
                "$full_dist" "$half_dist"
            printf '"micro_batch_size":2,"global_batch_size":32,"sequence_length":128,'
            printf '"train_iters":%s,"warmup_iters":%s}\n' "$TRAIN_ITERS" "$WARMUP_ITERS"

            MEGATRON_DEBUG_LOG=1 PP_TIMING_INTERVAL=1 torchrun "${DISTRIBUTED_ARGS[@]}" \
                pretrain_gpt.py "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" \
                "${PARALLEL_ARGS[@]}" "$@" "${DATA_ARGS[@]}" "${LOG_ARGS[@]}"
        } 2>&1 | tee "$log_path"
    done
}

# E0/E1 isolate the overhead of supplying an explicit but identical plan.
run_exp "pp4_default_uniform" "default_uniform" "[3,3,3,3]" "null"
run_exp "pp4_manual_uniform" "manual_uniform" "[3,3,3,3]" "null" \
    --decoder-num-layers-per-pipeline-stage 3 3 3 3

# E2/E3 compare Megatron's first/last-stage interface with the equivalent plan.
run_exp "pp4_first_last_2_2" "native_first_last" "[2,4,4,2]" "null" \
    --decoder-first-pipeline-num-layers 2 --decoder-last-pipeline-num-layers 2
run_exp "pp4_explicit_2_4_4_2" "manual_first_last_equivalent" "[2,4,4,2]" "null" \
    --decoder-num-layers-per-pipeline-stage 2 4 4 2

# E4 is a moderate perturbation; E5 deliberately creates a severe bottleneck.
run_exp "pp4_uneven_2_3_4_3" "moderate_uneven" "[2,3,4,3]" "null" \
    --decoder-num-layers-per-pipeline-stage 2 3 4 3
run_exp "pp4_uneven_1_2_6_3" "severe_uneven" "[1,2,6,3]" "null" \
    --decoder-num-layers-per-pipeline-stage 1 2 6 3

# E6 moves half a layer from stage 0 to stage 1. Only layer 3 crosses a stage.
run_exp "pp4_half_5_7_6_6" "half_layer" "null" "[5,7,6,6]" \
    --decoder-num-half-layers-per-pipeline-stage 5 7 6 6

echo "Logs: $LOG_DIR"
echo "Parse: python learning_examples/parse_pp_bench.py $LOG_DIR/*.log"
