# Next-phase research plan

## Current checkpoint

The controlled PP=4 results establish three working facts:

1. Default uniform and explicit uniform allocation have no material runtime
   difference in this setup.
2. `[2,4,4,2]` is faster than `[3,3,3,3]`, so layer-count uniformity is not
   compute uniformity when the first and last pipeline stages have extra work.
3. `[1,2,6,3]` is slower, confirming that a bad uneven plan exposes a
   pipeline bottleneck. The half-layer baseline is stable but is not yet a
   performance win.

The two new candidates in `train_gpt3_half_layer_bench.sh` are the final
near-best half-layer comparison for this phase:

| Candidate | Half-layer distribution | Question |
|---|---:|---|
| Near-best A | `[4,8,9,3]` | Does moving half a layer from the final stage help? |
| Near-best B | `[5,8,8,3]` | Does moving one half-layer from the final to first stage help? |

Keep PP=4, TP=1, DP=1, `micro_batch_size=2`, `global_batch_size=16`, model
shape, sequence length, and repeat policy fixed. Treat these experiments as a
completed allocation matrix, not as an open-ended search.

After the small-profile conclusion is recorded, run the script with
`BENCH_PROFILE=large`. It keeps the 12-layer model unchanged and uses
`seq_length=1024`, `micro_batch_size=8`, `global_batch_size=64`, 1500 training
iterations, `attention-backend=auto`, and `lr_decay_iters=1000`. The resulting
eight microbatches per iteration preserve the small experiment's pipeline
schedule. Run only the default, `[2,4,4,2]`, `[4,8,9,3]`, and `[5,8,8,3]`
cases. First verify the default case fits in GPU memory before launching the
full matrix.

## Phase 1: finish the controlled experiment

Run the existing script three times per configuration, parse the logs, and
record median, P95, all rank memory values (`R0` through `R3`), and rank-level timing. A candidate is only
interesting if its independent-run median improves without a worse P95 or a
new communication error. Do not claim a half-layer benefit from one run.

Keep the parser as a measurement tool, not a source of interpretation. Use the
structured `BENCH-META` record for configuration identity and keep raw logs
with the CSV output.

## Phase 2: local Nsight learning

Use the local single-GPU environment first (`F:\PycharmProjects\llm\.venv\Scripts`).
The goal is to learn the tools and establish a repeatable capture workflow,
not to profile the distributed benchmark immediately.

### Nsight Systems (`nsys`)

Learn to identify end-to-end step time, CPU launch gaps, CUDA kernel order,
H2D/D2H copies, synchronization points, NCCL ranges, compute/communication
overlap, and forward/backward/optimizer/data-loader phase boundaries. Start
with a short warmup plus one or two measured steps. Add NVTX ranges for these
phases before capturing, and record the command, commit, and model config.

### Nsight Compute (`ncu`)

Profile selected kernels rather than the whole run. Prioritize achieved
occupancy, SM and tensor-core utilization, memory and arithmetic throughput,
eligible warps, registers, shared memory, launch dimensions, kernel duration,
and memory/synchronization/issue-slot stalls. Compare identical kernel shapes;
occupancy alone is not evidence of faster training.

## Phase 3: remote profiling and local visualization

After the local workflow is repeatable, capture short representative windows
on the remote server. Start with one baseline and one near-best plan. Download
`.nsys-rep`, `.qdrep`, or exported CSV artifacts locally and record tool and
driver versions. The first distributed question is whether `[2,4,4,2]` wins
because of lower first/last-rank compute, less pipeline waiting, or better
communication overlap.

## Phase 4: VPP compatibility feasibility

The first VPP milestone is correctness and construction feasibility, not
performance. Extend the plan conceptually from `physical_rank -> fragments`
to `(physical_rank, virtual_chunk) -> fragments`. Do not add independent split
calculations in VPP code.

Validation order:

1. Reject unsupported combinations with a clear error where necessary.
2. Build a PP=4, VPP=2 full-layer plan and verify chunk-local modules/offsets.
3. Test one half-layer boundary in a virtual chunk with two microbatches.
4. Run 1F1B warmup, steady-state, and cooldown; assert one matching receive
   for every send with the same boundary schema.
5. Compare loss and gradients against a deterministic non-VPP reference.

Only after these pass should half-layer plus VPP performance be measured.

## Phase 5: simple offline allocation algorithm

Implement a deterministic allocator outside the runtime first. Model each
full layer as ordered attention and MLP tasks. For device `d`, estimate
`cost(d, task) = calibrated_rate(task, d) + boundary_comm_cost(task)`.

Given ordered tasks, assign contiguous prefixes to stages while minimizing
the maximum estimated stage cost, with an optional cross-stage-boundary
penalty. Start with dynamic programming over prefix length and stage count;
it is easier to validate than a greedy heuristic and supports full- and
half-layer cuts.

Required outputs are the distribution, predicted stage costs, bottleneck ratio,
and boundary locations. Runtime construction consumes the validated plan and
does not rerun the optimizer.

Evaluation order:

1. Synthetic rates: known optima and invalid-input handling.
2. Single-GPU calibration: attention and MLP forward/backward rates by shape.
3. Two-device simulation: predicted versus measured stage ordering.
4. PP=4 remote test: predicted bottleneck versus rank timing and wait time.

## Research questions

The useful question is not merely whether a layer can be split. It is how to
place ordered attention/MLP fragments when stage-local compute, boundary
communication, first/last-stage overhead, and VPP constraints interact.
Potential contributions are a reproducible fragment-level cost model, a plan
schema shared by construction and scheduling, and an allocator for
heterogeneous devices that preserves transformer order. These remain
hypotheses until profiler and calibration data support them.
