# Pipeline Partition Runtime Design

This note describes the runtime path used by uneven full-layer allocation and
attention/FFN half-layer allocation. It is the implementation contract; older
experiment notes may describe intermediate prototypes.

## End-to-end construction path

1. `megatron/training/arguments.py` parses CLI options and validates world-size
   constraints. Full-layer and half-layer distributions use separate options.
2. `TransformerConfig` validates model-level invariants. It does not mutate one
   distribution into another or synthesize communication boundaries.
3. `pipeline_partition.py` converts the chosen distribution into canonical,
   half-open half-layer intervals `[half_start, half_end)` for every PP rank.
4. `get_gpt_decoder_block_spec()` maps each interval element to an
   `AttentionSubLayer` or `FFNSubLayer` `ModuleSpec` while retaining the original
   1-based global Transformer layer number.
5. `TransformerBlock` treats full layers and half-layer fragments uniformly as
   logical layers with a `Tensor -> (Tensor, context)` contract. Attention packs
   its normalized output and residual into one `[2, S, B, H]` tensor; FFN
   unpacks it.
6. `get_tensor_shapes()` reads the same partition plan. A stage whose interval
   ends on attention advertises one `[2, S, B, H]` P2P tensor; every other
   decoder boundary advertises one `[S, B, H]` tensor. The schedule always
   executes exactly one P2P operation per GPT stage boundary.
7. `backward_step()`, pseudo-deallocation, and custom autograd remain on the
   original Megatron single-output path.

## Configuration semantics

- `--decoder-num-layers-per-pipeline-stage`: full Transformer layer counts.
  The entries sum to `num_layers`. With `--split-all-layers`, each full layer is
  represented internally as an attention/FFN pair, but the CLI unit stays a
  full layer.
- `--decoder-num-half-layers-per-pipeline-stage`: attention/FFN fragment counts.
  It requires `--split-all-layers`; entries sum to `num_layers * 2`.
- `--pipeline-split-layers`: selective splitting at existing full-layer stage
  boundaries. It is separate from split-all mode.

Example for four layers and PP=2:

```text
half-layer indices: 0:A1 1:F1 2:A2 | 3:F2 4:A3 5:F3 6:A4 7:F4
distribution:       [3, 5]
stage 0 sends:      stack(pre_mlp_layernorm_output, residual) -> [2,S,B,H]
stage 1 starts:     FFNSubLayer(global_layer_number=2)
```

## Runtime invariants

- Half-layer intervals cover exactly `[0, num_layers * 2)` with no gaps or
  overlaps.
- Model construction and P2P shape planning consume the same interval object.
- Every GPT stage boundary has exactly one forward send/receive and one backward
  send/receive. Boundary state is encoded inside the tensor shape.
- Global layer numbering is stable across full and split representations. This
  is required for initialization, MoE routing metadata, logging, and checkpoint
  prefixes.
- Split-all mode currently excludes VPP, full recompute, and standalone
  embedding/loss stages. These should be added only with dedicated scheduling
  and checkpoint tests.

## Suggested reading order in this repository

1. `megatron/training/arguments.py`: topology and CLI validation.
2. `megatron/core/parallel_state.py`: rank groups and pipeline rank mapping.
3. `megatron/core/models/gpt/gpt_model.py` and `gpt_layer_specs.py`: model and
   `ModuleSpec` construction.
4. `megatron/core/transformer/transformer_block.py`: layer instantiation,
   execution, and checkpoint key layout.
5. `megatron/core/pipeline_parallel/schedules.py`: warmup, 1F1B, cooldown, and
   tensor lifetime.
6. `megatron/core/pipeline_parallel/p2p_communication.py`: NCCL send/receive
   ordering and batching.
7. `megatron/core/distributed/finalize_model_grads.py`: DP gradient completion
   after pipeline backward.

For future experiments, record `world_size`, TP, PP, CP, EP, and the derived DP
for every run. A PP=2 run on four GPUs with TP=CP=EP=1 has DP=2 and is not a
pure PP=2 versus PP=4 comparison.
