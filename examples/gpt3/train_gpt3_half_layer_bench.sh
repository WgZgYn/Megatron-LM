#!/bin/bash

# Half-layer PP benchmark: baseline vs split_all_layers.
# 4xV100, PP=2 or PP=4, TP=1, DP=1.
# Usage: bash train_gpt3_half_layer_bench.sh <VOCAB_FILE> <MERGE_FILE>

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Fixed: 2 GPUs for PP=2, 4 GPUs for PP=4.  DP=1 in all tests to isolate PP effect.
# TP=1 throughout (no tensor parallelism).
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0

VOCAB_FILE=$1; MERGE_FILE=$2
if [ -z "$VOCAB_FILE" ] || [ -z "$MERGE_FILE" ]; then
    echo "Usage: bash train_gpt3_half_layer_bench.sh <VOCAB_FILE> <MERGE_FILE>"
    exit 1
fi

DISTRIBUTED_ARGS=(--nnodes $NUM_NODES
                  --master_addr $MASTER_ADDR --master_port $MASTER_PORT)

# 12 layers, hidden=768, ~124M params, mock data, fp16
GPT_MODEL_ARGS=(--num-layers 12 --hidden-size 768 --num-attention-heads 12
                --seq-length 128 --max-position-embeddings 128)

TRAINING_ARGS=(--micro-batch-size 2 --global-batch-size 16 --train-iters 30
               --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95
               --init-method-std 0.006 --clip-grad 1.0 --fp16
               --lr 6.0e-5 --lr-decay-style cosine --min-lr 6.0e-6
               --lr-warmup-fraction .001)

DATA_ARGS=(--mock-data --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --split 949,50,1)
EVAL_ARGS=(--log-interval 1 --eval-interval 100 --eval-iters 0)

run_exp() {
    local gpus="$1"; local label="$2"; shift 2
    echo "==== $label (${gpus} GPUs, PP=$PP_SIZE, DP=1, TP=1) ===="
    torchrun --nproc_per_node $gpus ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} ${TRAINING_ARGS[@]} "$@" ${DATA_ARGS[@]} ${EVAL_ARGS[@]} \
        2>&1 | tee "/tmp/pp_bench_${label}.log"
    echo ""
}

# ═══ PP=2 experiments ═══

echo "=== PP=2: 2 GPUs, DP=1, TP=1 (6+6 full layers per stage) ==="
echo ""

# A1) Baseline: PP=2 uniform whole layers, 2 GPUs
run_exp 2 "pp2_baseline" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2

# A2) Split-all: same model, pure overhead test
run_exp 2 "pp2_split_uniform" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --split-all-layers

# A3) Uneven (mild): 4+8 full → 8+16 half-layers
run_exp 2 "pp2_split_uneven_4_8" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --split-all-layers --decoder-num-layers-per-pipeline-stage 4 8

# A4) Uneven (extreme): 3+9 full → 6+18 half-layers
run_exp 2 "pp2_split_uneven_3_9" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --split-all-layers --decoder-num-layers-per-pipeline-stage 3 9

# A5) Half-layer granularity: 11+13 half-layers, auto-split at layer 6
run_exp 2 "pp2_half_layer_11_13" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --split-all-layers --decoder-num-half-layers-per-pipeline-stage 11 13

# ═══ PP=4 experiments ═══

echo "=== PP=4: 4 GPUs, DP=1, TP=1 (3+3+3+3 full layers per stage) ==="
echo ""

# B1) Baseline: PP=4 uniform whole layers
run_exp 4 "pp4_baseline" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4

# B2) Split-all: same model, pure overhead test
run_exp 4 "pp4_split_uniform" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 \
    --split-all-layers

# B3) Half-layer granularity: [5,7,6,6] half-layers → auto split at layer 3
#     GPU0=5 halves, GPU1=7, GPU2=6, GPU3=6 (24 total)
run_exp 4 "pp4_half_layer_5_7_6_6" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 \
    --split-all-layers --decoder-num-half-layers-per-pipeline-stage 5 7 6 6

# B4) Uneven: [1,2,5,4] full layers → [2,4,10,8] half-layers
#     stage0 light, stage2 heavy
run_exp 4 "pp4_split_uneven" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 \
    --split-all-layers --decoder-num-layers-per-pipeline-stage 1 2 5 4

echo ""
echo "=== SUMMARY ==="
echo "PP=2 (2 GPUs, DP=1, TP=1) logs:"
echo "  A1 baseline:     /tmp/pp_bench_pp2_baseline.log"
echo "  A2 split uniform: /tmp/pp_bench_pp2_split_uniform.log"
echo "  A3 split 4+8:    /tmp/pp_bench_pp2_split_uneven_4_8.log"
echo "  A4 split 3+9:    /tmp/pp_bench_pp2_split_uneven_3_9.log"
echo "  A5 half-layer 11+13: /tmp/pp_bench_pp2_half_layer_11_13.log"
echo "PP=4 (4 GPUs, DP=1, TP=1) logs:"
echo "  B1 baseline:     /tmp/pp_bench_pp4_baseline.log"
echo "  B2 split uniform: /tmp/pp_bench_pp4_split_uniform.log"
echo "  B3 half-layer 5+7+6+6: /tmp/pp_bench_pp4_half_layer_5_7_6_6.log"
echo "  B4 split uneven:  /tmp/pp_bench_pp4_split_uneven.log"
echo ""
echo "Key numbers to compare (grep from logs):"
echo "  'number of parameters'  → per-rank params"
echo "  'mem_used'              → per-rank memory (MiB)"
echo "  'elapsed'               → per-iteration time (ms)"
