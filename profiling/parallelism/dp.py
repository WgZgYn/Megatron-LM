"""Data parallelism via DDP.

DDP's gradient all-reduce runs asynchronously on its own communication stream
and overlaps with the backward pass. It is *not* wrapped with a timing hook
here: the comm stream cannot be bracketed with CUDA events from Python, and
wall-clock timing resolves before the NCCL kernel finishes. DP communication
time is read from the torch.profiler NCCL kernel durations by ``analyze.py``.
"""

from __future__ import annotations

from torch.nn.parallel import DistributedDataParallel as DDP


def wrap_ddp(model, device_ids):
    return DDP(model, device_ids=device_ids, find_unused_parameters=False)
