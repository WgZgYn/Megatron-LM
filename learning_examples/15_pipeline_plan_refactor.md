# Pipeline partition refactor and experiment plan

## Objective

The implementation supports two user-visible capabilities:

1. Assign an uneven number of full Transformer layers to PP stages.
2. Place a PP boundary between the attention and FFN fragments of a layer.

The implementation uses half-layer coordinates only as a planning unit. It does
not split every Transformer layer into two Python modules. Attention and FFN are
materialized separately only when they belong to different PP stages.

## Architecture

`PipelinePlan` is the only resolved representation after configuration:

```text
CLI / TransformerConfig
        |
        v
PipelinePlan
  - per-stage fragments
  - input boundary kind
  - output boundary kind
        |                     |
        v                     v
stage ModuleSpec list     schedule tensor shape
```

The model builder and schedule must not independently recalculate stage
boundaries. This invariant prevents send/receive shape disagreement.

`--split-all-layers` remains accepted for compatibility but is a no-op. New
half-layer experiments use `--decoder-num-half-layers-per-pipeline-stage`.

## PP=4 experiment matrix

All experiments fix PP=4, TP=1, DP=1, model shape, precision, sequence length,
micro batch size, global batch size, and number of microbatches.

| Case | Distribution | Purpose |
|---|---|---|
| Default uniform | Megatron default | Reference |
| Manual uniform | full `[3,3,3,3]` | Measure planner/materializer overhead |
| Native first/last | first=2, last=2 | Existing Megatron interface |
| Equivalent explicit | full `[2,4,4,2]` | Check semantic and performance equivalence |
| Moderate uneven | full `[2,3,4,3]` | Plausible load adjustment |
| Severe uneven | full `[1,2,6,3]` | Deliberate stage bottleneck |
| Half-layer | half `[5,7,6,6]` | Finer boundary at layer 3 |

Each configuration runs in three independent processes by default. The first
ten training iterations are excluded. Report the median and P95 within each run,
then the median and range of the independent run medians.

## Validation gates

1. Planner unit tests: intervals, fragments, boundary kinds, invalid configs.
2. Materializer tests: global dense/MoE specs become the expected local modules.
3. Single-process numerical tests: split and unsplit outputs and gradients agree.
4. Distributed tests: PP4 forward/backward and 1F1B complete for all boundary kinds.
5. Formal experiment: run only after the four validation gates pass.

## Future compatibility

### 1F1B

The current non-interleaved 1F1B schedule is supported through one packed
boundary Tensor. Add timeout-based distributed tests for one, two, and sixteen
microbatches so warmup, steady state, and cooldown are all exercised.

### VPP

VPP requires a plan for `(physical_rank, virtual_chunk)` rather than only a
physical rank. Do not add VPP branches to the current planner. Extend `StagePlan`
with chunk ownership, then derive both chunk construction and communication from
that plan. Half-layer boundaries remain rejected until this representation and
distributed tests exist.

### Recompute and CUDA graphs

Full activation recompute currently assumes complete Transformer layers.
Supporting a cross-stage layer requires fragment-aware checkpoint ownership.
CUDA graph support additionally requires stable packed boundary shapes. Keep
these modes rejected until output and gradient equivalence tests are available.

### Operator-level partitioning

Do not encode operator cuts as more boolean flags. Generalize `LayerFragment`
from `{FULL, ATTENTION, FFN}` to a sequence of named operations with an explicit
boundary schema. The plan must describe transferred values, dtype, and shape;
the schedule must continue to consume this schema without knowing operator names.
