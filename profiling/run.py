"""Unified profiling harness for DP / TP / PP on a classic transformer.

Run under ``torchrun`` for multi-GPU (NCCL), or directly with ``python`` for a
single-process smoke test. Two capture mechanisms run side by side:

* ``torch.profiler`` — exports per-rank traces (TensorBoard timeline) plus a
  ``key_averages`` table, the source of truth for the full CPU/CUDA timeline.
* ``CommTimer`` — logs every collective (DDP gradient all-reduce, TP
  all-reduce, PP send/recv) as a JSONL record with byte count, latency and
  effective bandwidth, consumed by ``analyze.py``.

Example (remote, 4 GPUs, DP):
    torchrun --nproc_per_node=4 run.py --mode dp --steps 8 --tag baseline

See PLAN.md / README.md for the full matrix and analysis workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

from comm import CommTimer
from model import MiniGPT, ModelConfig
from parallelism import dp, pp, tp


@contextmanager
def nvtx_range(name: str):
    """NVTX range for nsys / torch.profiler phase labels (CUDA only)."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def parse_args():
    p = argparse.ArgumentParser(description="DP/TP/PP profiling harness")
    p.add_argument("--mode", choices=["dp", "tp", "pp"], default="dp")
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl",
                   help="nccl for remote multi-GPU; gloo for local CPU smoke")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # model
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--vocab", type=int, default=50304)

    # data / run
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--num-microbatches", type=int, default=None,
                   help="PP microbatches; defaults to world size")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)

    # capture
    p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--measure-comm", action=argparse.BooleanOptionalAction, default=True)

    # output
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def make_tokens(cfg, batch, seed, device, dtype):
    """Deterministic synthetic data, identical on every rank."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), generator=g)
    ids = ids.to(device)
    labels = ids.clone()
    return ids, labels


def _train_step(model, tokens, labels, optimizer, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward"):
        logits = model(tokens)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    with nvtx_range("backward"):
        loss.backward()
    with nvtx_range("optimizer"):
        optimizer.step()
    return loss


def build(args, cfg, rank, world, device, timer):
    loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)

    if args.mode == "dp":
        model = MiniGPT(cfg).to(device)
        if world > 1:
            device_ids = [int(os.environ.get("LOCAL_RANK", 0))] if device == "cuda" else None
            model = dp.wrap_ddp(model, device_ids, measure_comm=args.measure_comm, timer=timer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        return model, optimizer, loss_fn, "dp", None

    if args.mode == "tp":
        col = partial(tp.ColumnParallelLinear, tp_size=world, rank=rank, group=None, timer=timer)
        row = partial(tp.RowParallelLinear, tp_size=world, rank=rank, group=None, timer=timer)
        model = MiniGPT(cfg, tp_size=world, col_factory=col, row_factory=row).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        return model, optimizer, loss_fn, "tp", None

    if args.mode == "pp":
        full = MiniGPT(cfg).to(device)  # build full, then slice into stages
        stage = pp.build_stage(full, cfg, rank, world, device)
        optimizer = torch.optim.AdamW(stage.parameters(), lr=3e-4)
        return stage, optimizer, loss_fn, "pp", stage

    raise ValueError(args.mode)


def export_key_averages(key_averages, path):
    """Dump torch.profiler key_averages as CSV (one row per aggregated event)."""
    rows = []
    for evt in key_averages:
        rows.append(
            {
                "name": evt.key,
                "count": getattr(evt, "count", 0),
                "cpu_time_total_us": getattr(evt, "cpu_time_total", 0),
                "cuda_time_total_us": getattr(evt, "cuda_time_total", 0),
                "self_cpu_us": getattr(evt, "self_device_time_total", 0),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("name,count,cpu_time_total_us,cuda_time_total_us,self_device_us\n")
        for r in rows:
            f.write(
                f"{r['name']},{r['count']},{r['cpu_time_total_us']:.3f},"
                f"{r['cuda_time_total_us']:.3f},{r['self_cpu_us']:.3f}\n"
            )


def main():
    args = parse_args()

    using_torchrun = "RANK" in os.environ
    if using_torchrun:
        dist.init_process_group(backend=args.backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank, world = 0, 1

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[run] CUDA unavailable, falling back to CPU")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    torch.manual_seed(args.seed)

    cfg = ModelConfig(
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_hidden=args.ffn,
        seq_len=args.seq,
        vocab_size=args.vocab,
    )
    dtype = torch.float32

    mode_label = f"{args.mode}_p{world}"
    out_dir = os.path.join(args.out_dir, f"{args.tag + '_' if args.tag else ''}{mode_label}")
    os.makedirs(out_dir, exist_ok=True)

    timer = CommTimer(
        rank, world, device_type=device, enabled=args.measure_comm,
        log_path=os.path.join(out_dir, f"comm_rank{rank}.jsonl"),
    )

    model, optimizer, loss_fn, mode, stage = build(args, cfg, rank, world, device, timer)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[run] mode={mode} world={world} backend={args.backend} device={device}")
        print(f"[run] model params={n_params/1e6:.2f}M  steps={args.steps} warmup={args.warmup}")

    # ---- data ------------------------------------------------------------
    if mode == "dp":
        local_batch = args.global_batch // world
        tokens, labels = make_tokens(cfg, local_batch, args.seed, device, dtype)
        microbatches = None
    elif mode == "tp":
        tokens, labels = make_tokens(cfg, args.global_batch, args.seed, device, dtype)
        microbatches = None
    else:  # pp
        M = args.num_microbatches or max(1, world)
        assert args.global_batch % M == 0, (args.global_batch, M)
        micro = args.global_batch // M
        microbatches = [make_tokens(cfg, micro, args.seed, device, dtype) for _ in range(M)]
        tokens = labels = None

    # ---- profiler --------------------------------------------------------
    prof = None
    if args.profile:
        from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

        logdir = os.path.join(out_dir, "trace")
        os.makedirs(logdir, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=args.warmup, active=args.steps - args.warmup, repeat=1),
            on_trace_ready=tensorboard_trace_handler(logdir, worker_name=f"rank{rank}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    # ---- training loop ---------------------------------------------------
    step_ms = []
    for step in range(args.steps):
        timer.set_step(step)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "pp":
            loss = pp.pp_train_step(
                stage, microbatches, timer, rank, world, cfg.hidden_size,
                dtype, device, optimizer, loss_fn,
            )
        else:
            loss = _train_step(model, tokens, labels, optimizer, loss_fn)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        step_ms.append(dt)

        loss_val = loss.item() if loss is not None else float("nan")
        print(f"[rank{rank}] step {step}  loss={loss_val:.4f}  time={dt:.2f} ms")

        if prof is not None:
            prof.step()

    # ---- teardown / export ----------------------------------------------
    if prof is not None:
        prof.stop()
        with open(os.path.join(out_dir, f"key_averages_rank{rank}.txt"), "w", encoding="utf-8") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total"))
        export_key_averages(prof.key_averages(), os.path.join(out_dir, f"key_averages_rank{rank}.csv"))

    timer.flush()
    with open(os.path.join(out_dir, f"steps_rank{rank}.jsonl"), "w", encoding="utf-8") as f:
        for step, ms in enumerate(step_ms):
            f.write(json.dumps({"rank": rank, "step": step, "ms": round(ms, 6)}) + "\n")

    if rank == 0:
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": mode,
                    "world_size": world,
                    "backend": args.backend,
                    "device": device,
                    "model": {
                        "hidden": cfg.hidden_size,
                        "layers": cfg.num_layers,
                        "heads": cfg.num_heads,
                        "ffn": cfg.ffn_hidden,
                        "seq": cfg.seq_len,
                        "vocab": cfg.vocab_size,
                    },
                    "global_batch": args.global_batch,
                    "num_microbatches": args.num_microbatches or max(1, world),
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "seed": args.seed,
                    "measure_comm": args.measure_comm,
                    "profile": args.profile,
                },
                f,
                indent=2,
            )

    if using_torchrun:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
