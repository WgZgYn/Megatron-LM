"""Data parallelism via DDP, with an optional timed gradient all-reduce hook.

The default DDP gradient hook already exposes one all-reduce per gradient
bucket. ``timed_allreduce_hook`` mirrors it exactly (``tensor.div_(world)``
then all-reduce — see ``torch.distributed.algorithms.ddp_comm_hooks._allreduce_fut``)
but performs a *synchronous* all-reduce so ``CommTimer`` can record the
per-bucket latency and effective bandwidth. When ``measure_comm`` is False the
default asynchronous hook is kept, so the torch.profiler trace shows realistic
gradient/backward overlap.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def timed_allreduce_hook(timer):
    """DDP comm hook: divide by world size, time the all-reduce, return a future."""

    def hook(state, bucket):
        process_group = state  # `state` IS the process group passed to register_comm_hook
        group = process_group if process_group is not None else dist.group.WORLD
        world_size = group.size()

        tensor = bucket.buffer()
        tensor.div_(world_size)
        timer.timed_all_reduce(tensor, group=process_group, name="dp_grad_allreduce")

        # The hook contract: return a Future resolving to a *single* tensor
        # (the default hook does `allreduce(...).get_future().then(lambda f: f.value()[0])`).
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    return hook


def wrap_ddp(model, device_ids, measure_comm=False, timer=None):
    ddp = DDP(model, device_ids=device_ids, find_unused_parameters=False)
    if measure_comm and timer is not None:
        ddp.register_comm_hook(None, timed_allreduce_hook(timer))
    return ddp
