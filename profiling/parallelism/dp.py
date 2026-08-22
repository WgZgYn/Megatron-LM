"""Data parallelism via DDP, with an optional timed gradient all-reduce hook.

``timed_allreduce_hook`` mirrors the default DDP hook exactly (``tensor.div_(world)``
then an async all-reduce — see ``torch.distributed.algorithms.ddp_comm_hooks._allreduce_fut``)
and additionally times the all-reduce. The all-reduce stays **non-blocking**, so
DDP keeps its native backward/communication overlap; the latency is measured
wall-clock from launch to future resolution (see ``comm.CommTimer``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def timed_allreduce_hook(timer):
    """DDP comm hook: divide by world size, time the async all-reduce, return a future."""

    def hook(state, bucket):
        process_group = state  # `state` IS the process group passed to register_comm_hook
        group = process_group if process_group is not None else dist.group.WORLD

        tensor = bucket.buffer()
        tensor.div_(group.size())
        return timer.timed_all_reduce_async(tensor, group=process_group, name="dp_grad_allreduce")

    return hook


def wrap_ddp(model, device_ids, measure_comm=False, timer=None):
    ddp = DDP(model, device_ids=device_ids, find_unused_parameters=False)
    if measure_comm and timer is not None:
        ddp.register_comm_hook(None, timed_allreduce_hook(timer))
    return ddp
