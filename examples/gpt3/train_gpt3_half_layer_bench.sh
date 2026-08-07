#!/bin/bash

# Half-layer PP benchmark: compare baseline (whole-layer) vs split_all_layers.
# Usage: bash train_gpt3_half_layer_bench.sh <VOCAB_FILE> <MERGE_FILE>
#   2 GPUs, PP=2, TP=1, DP=1 — isolates PP + split overhead.
#
# Runs three configurations:
#   A) Baseline:   PP=2, uniform 6+6 whole layers
#   B) Split-all:  PP=2, uniform 6+6 layers → 12+12 half-layers (same result!)
#   C) Split-all + uneven (optional): compare balanced vs imbalanced half-layer counts

set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

VOCAB_FILE=$1
MERGE_FILE=$2
if [ -z "$VOCAB_FILE" ] || [ -z "$MERGE_FILE" ]; then
    echo "Usage: bash train_gpt3_half_layer_bench.sh <VOCAB_FILE> <MERGE_FILE>"
    echo "  e.g. VOCAB_FILE=gpt2-vocab.json MERGE_FILE=gpt2-merges.txt"
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Small model: 12 layers, hidden=512, mock data, bf16
GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --seq-length 128
    --max-position-embeddings 128
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 8
    --train-iters 30
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
)

DATA_ARGS=(
    --mock-data
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 100
    --eval-iters 0
)

echo "============================================================"
echo " A) BASELINE: PP=2, uniform 6+6 whole layers"
echo "============================================================"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee /tmp/pp_bench_baseline.log
echo ""
echo "Baseline done. Extracting memory/time from log..."
grep -E "mem_used|number of parameters|iteration|elapsed" /tmp/pp_bench_baseline.log | tail -20

echo ""
echo "============================================================"
echo " B) SPLIT-ALL: PP=2, 6+6 whole → 12+12 half-layers"
echo "============================================================"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --split-all-layers \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee /tmp/pp_bench_split.log
echo ""
echo "Split-all done. Extracting memory/time from log..."
grep -E "mem_used|number of parameters|iteration|elapsed" /tmp/pp_bench_split.log | tail -20

echo ""
echo "============================================================"
echo " C) UNEVEN SPLIT (optional): PP=2, [4,8] half-layers"
echo "    GPU0 gets 2 full layers (4 halves), GPU1 gets 4 full layers (8 halves)"
echo "============================================================"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --split-all-layers \
    --decoder-num-layers-per-pipeline-stage 2 4 \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee /tmp/pp_bench_uneven.log
echo ""
echo "Uneven split done. Extracting memory/time from log..."
grep -E "mem_used|number of parameters|iteration|elapsed" /tmp/pp_bench_uneven.log | tail -20

echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo "A) Baseline (PP=2, uniform whole):   /tmp/pp_bench_baseline.log"
echo "B) Split-all (PP=2, 12+12 halves):   /tmp/pp_bench_split.log"
echo "C) Uneven   (PP=2, 4+8 halves):      /tmp/pp_bench_uneven.log"
