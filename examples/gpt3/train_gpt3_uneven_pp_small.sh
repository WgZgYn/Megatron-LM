#!/bin/bash

# Small GPT with uneven per-pipeline-stage layer distribution.
# Usage: bash train_gpt3_uneven_pp_small.sh <VOCAB_FILE> <MERGE_FILE>
#   e.g. VOCAB_FILE=gpt2-vocab.json MERGE_FILE=gpt2-merges.txt
# Runs on a single node, 4 GPUs, with mock data.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

VOCAB_FILE=$1 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$2 #<Specify path to file>/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

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
    --train-iters 50
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

# Uneven pipeline: one layer on stage 0, five on stage 1, two on stage 2, four on stage 3.
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --decoder-num-layers-per-pipeline-stage 1 5 2 4
)

DATA_ARGS=(
    --mock-data
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 25
    --eval-iters 2
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
