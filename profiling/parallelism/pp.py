"""Gpipe-style pipeline parallelism over transformer blocks.

The model's transformer blocks are split *contiguously* across ``pp_size``
stages: stage 0 owns the embeddings, the last stage owns the final layernorm
+ lm_head + loss. Activations (forward) and gradients (backward) flow between
neighbouring ranks with *blocking* point-to-point ``send``/``recv`` — the
explicit synchronous communication this profiling harness is meant to capture.

The schedule is the simplest Gpipe (``F`` then ``B``, no 1F1B overlap): forward
all microbatches in order, then backward in reverse order. This makes every
send/recv a clear, separate synchronization point on the timeline.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn


@contextmanager
def _nvtx(name):
    """NVTX range for nsys / torch.profiler phase labels (CUDA only)."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def stage_rank_to_ids(rank: int, pp_size: int, num_layers: int):
    """Contiguous split of layer ids across stages; returns ``(start, end)``."""
    per = num_layers // pp_size
    rem = num_layers % pp_size
    counts = [per + (1 if r < rem else 0) for r in range(pp_size)]
    start = sum(counts[:rank])
    return start, start + counts[rank]


class PPStage(nn.Module):
    """The subset of the model owned by one pipeline rank."""

    def __init__(self, tok_emb, pos_emb, blocks, ln_f, lm_head, is_first, is_last):
        super().__init__()
        self.tok_emb = tok_emb
        self.pos_emb = pos_emb
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.is_first = is_first
        self.is_last = is_last

    def embed(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        return self.tok_emb(ids) + self.pos_emb(pos)

    def body(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def head(self, x):
        return self.lm_head(self.ln_f(x))


def build_stage(model, cfg, rank, pp_size, device):
    start, end = stage_rank_to_ids(rank, pp_size, cfg.num_layers)
    is_first = rank == 0
    is_last = rank == pp_size - 1
    return PPStage(
        tok_emb=model.tok_emb if is_first else None,
        pos_emb=model.pos_emb if is_first else None,
        blocks=list(model.blocks[start:end]),
        ln_f=model.ln_f if is_last else None,
        lm_head=model.lm_head if is_last else None,
        is_first=is_first,
        is_last=is_last,
    ).to(device)


def pp_train_step(
    stage,
    microbatches,
    timer,
    rank,
    pp_size,
    hidden,
    dtype,
    device,
    optimizer,
    loss_fn,
    autocast_ctx=None,
    scaler=None,
):
    """One Gpipe-style step: forward all microbatches, then backward in reverse.

    ``microbatches`` is a list of ``(ids, labels)`` tuples. Every rank holds the
    same microbatches (identical seed); only stage 0 consumes ``ids`` and only
    the last stage consumes ``labels``. Returns the loss on the last stage (or
    ``None`` elsewhere).
    """
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    M = len(microbatches)
    mb, seq = microbatches[0][0].shape
    act_shape = (mb, seq, hidden)
    next_rank = rank + 1 if rank < pp_size - 1 else None
    prev_rank = rank - 1 if rank > 0 else None
    loss = None

    saved = []

    # ---- forward: recv -> body -> send (last stage also head + loss) ------
    with _nvtx("pp_forward"):
        for m in range(M):
            ids, labels = microbatches[m]
            if rank == 0:
                x = stage.embed(ids)
            else:
                x = torch.empty(*act_shape, device=device, dtype=dtype)
                timer.timed_recv(x, prev_rank, tag=m, name="pp_recv_act")
            x = x.requires_grad_(True)
            with autocast_ctx:
                x_out = stage.body(x)

            if rank == pp_size - 1:
                with autocast_ctx:
                    logits = stage.head(x_out)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                saved.append((x, x_out, loss))
            else:
                timer.timed_send(x_out, next_rank, tag=m, name="pp_send_act")
                saved.append((x, x_out, None))

    # ---- backward: recv grad -> backward -> send grad (reverse order) -----
    with _nvtx("pp_backward"):
        for m in reversed(range(M)):
            x, x_out, mb_loss = saved[m]
            if rank == pp_size - 1:
                if scaler is not None:
                    scaler.scale(mb_loss).backward()
                else:
                    mb_loss.backward()
            else:
                grad = torch.empty(*act_shape, device=device, dtype=dtype)
                timer.timed_recv(grad, next_rank, tag=m + M, name="pp_recv_grad")
                x_out.backward(grad)

            if rank > 0:
                grad_x = x.grad
                if grad_x is None:  # defensive: no trainable path into x
                    grad_x = torch.zeros_like(x)
                timer.timed_send(grad_x, prev_rank, tag=m + M, name="pp_send_grad")

    with _nvtx("optimizer"):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss
