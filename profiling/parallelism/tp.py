"""Megatron-style tensor parallelism for a standalone transformer.

Convention (matches Megatron-LM):
- ``ColumnParallelLinear``: weight sharded along the *output* rows; the input
  is replicated and the output stays sharded (no forward all-reduce). The
  *backward* all-reduces the input gradient because the input is replicated.
- ``RowParallelLinear``: weight sharded along the *input* columns; the input
  is sharded and the output partial sum is all-reduced in the *forward* to a
  replicated tensor.

Per transformer block this yields 2 forward + 2 backward all-reduces:
  forward:  attention output projection (row) + MLP down projection (row)
  backward: attention QKV projection (col) + MLP up projection (col)

The two ``torch.autograd.Function`` subclasses insert the gradient all-reduce
where PyTorch's plain autograd cannot (it has no notion of a distributed
sum), so both the forward *and* backward communication show up in the profile.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _AllReduce(torch.autograd.Function):
    """Forward: all-reduce (on a clone). Backward: identity (pass grad through)."""

    @staticmethod
    def forward(ctx, x, timer, group, name):
        ctx.timer, ctx.group, ctx.name = timer, group, name
        y = x.clone()
        if timer is not None:
            timer.timed_all_reduce(y, group=group, name=name)
        else:
            dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


class _AllReduceBackward(torch.autograd.Function):
    """Forward: identity. Backward: all-reduce the gradient (in place)."""

    @staticmethod
    def forward(ctx, x, timer, group, name):
        ctx.timer, ctx.group, ctx.name = timer, group, name
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.timer is not None:
            ctx.timer.timed_all_reduce(grad, group=ctx.group, name=ctx.name)
        else:
            dist.all_reduce(grad, group=ctx.group)
        return grad, None, None, None


class ColumnParallelLinear(nn.Module):
    """``Y = X @ W^T`` with ``W`` sharded along output rows.

    Input replicated, output sharded. Backward all-reduces the input grad.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        tp_size=1,
        rank=0,
        group=None,
        timer=None,
    ):
        super().__init__()
        assert out_features % tp_size == 0, (out_features, tp_size)
        self.tp_size = tp_size
        self.rank = rank
        self.group = group
        self.timer = timer
        self.out_per_rank = out_features // tp_size

        # Init the FULL weight then slice, so a TP model is numerically equal
        # to the unsharded model (same seed -> identical full weight everywhere).
        full = torch.empty(out_features, in_features)
        nn.init.normal_(full, mean=0.0, std=0.02)
        start = rank * self.out_per_rank
        self.weight = nn.Parameter(full[start : start + self.out_per_rank].contiguous())

        if bias:
            full_b = torch.zeros(out_features)
            self.bias = nn.Parameter(full_b[start : start + self.out_per_rank].contiguous())
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.tp_size > 1:
            x = _AllReduceBackward.apply(x, self.timer, self.group, "tp_col_grad_allreduce")
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """``Y = X @ W^T`` with ``W`` sharded along input columns.

    Input sharded, output partial sum all-reduced in the forward (replicated).
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        tp_size=1,
        rank=0,
        group=None,
        timer=None,
    ):
        super().__init__()
        assert in_features % tp_size == 0, (in_features, tp_size)
        self.tp_size = tp_size
        self.rank = rank
        self.group = group
        self.timer = timer
        self.in_per_rank = in_features // tp_size

        full = torch.empty(out_features, in_features)
        nn.init.normal_(full, mean=0.0, std=0.02)
        start = rank * self.in_per_rank
        self.weight = nn.Parameter(full[:, start : start + self.in_per_rank].contiguous())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)  # partial sum over in_per_rank
        if self.tp_size > 1:
            y = _AllReduce.apply(y, self.timer, self.group, "tp_row_allreduce")
        return y
