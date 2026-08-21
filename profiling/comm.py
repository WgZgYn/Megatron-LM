"""Communication timing utilities.

``CommTimer`` wraps NCCL/GLOO collectives with CUDA-event (or wall-clock)
timing so every collective is logged as one JSONL record containing byte
count, latency and effective bandwidth. The per-rank logs are the primary
input to ``analyze.py`` for the DP / TP / PP communication comparison.

Bandwidth convention
--------------------
For ``all_reduce`` we report the *effective* bytes sent per rank of a ring
algorithm: ``2 * (n - 1) / n * numel * element_size``. This is the standard
approximation (no data actually leaves the node for n=1, ~2x the tensor for
large n). Point-to-point ``send``/``recv`` report ``numel * element_size``.
"""

from __future__ import annotations

import json
import time

import torch
import torch.distributed as dist


class CommTimer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        device_type: str = "cuda",
        enabled: bool = True,
        log_path: str | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device_type = device_type
        self.enabled = enabled
        self.log_path = log_path
        self.records = []
        self._step = 0
        self._seq = 0

    # -- lifecycle ----------------------------------------------------------

    def set_step(self, step: int) -> None:
        self._step = step

    def flush(self):
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")
        return self.records

    # -- internals ----------------------------------------------------------

    def _record(self, op, num_bytes, ms, src=None, dst=None, tag=None) -> None:
        gbps = (num_bytes / 1e9) / (ms / 1e3) if ms > 0 else 0.0
        self.records.append(
            {
                "rank": self.rank,
                "step": self._step,
                "seq": self._seq,
                "op": op,
                "bytes": int(num_bytes),
                "ms": round(ms, 6),
                "gbps": round(gbps, 3),
                "src": src,
                "dst": dst,
                "tag": tag,
            }
        )
        self._seq += 1

    # -- collective wrappers ------------------------------------------------

    def timed_all_reduce(self, tensor, group=None, name="all_reduce"):
        """Blocking all-reduce, timed. Returns the (in-place reduced) tensor."""
        world = dist.get_world_size(group) if group is not None else self.world_size
        factor = 2.0 * (world - 1) / world if world > 1 else 1.0
        num_bytes = int(tensor.numel() * tensor.element_size() * factor)

        if not self.enabled:
            dist.all_reduce(tensor, group=group)
            return tensor

        start, end, t0 = self._begin()
        dist.all_reduce(tensor, group=group)
        ms = self._finish(start, end, t0)
        self._record(name, num_bytes, ms)
        return tensor

    def timed_send(self, tensor, dst, tag=0, name="send") -> None:
        num_bytes = int(tensor.numel() * tensor.element_size())
        if not self.enabled:
            dist.send(tensor, dst, tag=tag)
            return
        start, end, t0 = self._begin()
        dist.send(tensor, dst, tag=tag)
        ms = self._finish(start, end, t0)
        self._record(name, num_bytes, ms, dst=dst, tag=tag)

    def timed_recv(self, tensor, src, tag=0, name="recv") -> None:
        num_bytes = int(tensor.numel() * tensor.element_size())
        if not self.enabled:
            dist.recv(tensor, src, tag=tag)
            return
        start, end, t0 = self._begin()
        dist.recv(tensor, src, tag=tag)
        ms = self._finish(start, end, t0)
        self._record(name, num_bytes, ms, src=src, tag=tag)

    # -- timing primitives --------------------------------------------------

    def _begin(self):
        if self.device_type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end, None
        return None, None, time.perf_counter()

    def _finish(self, start, end, t0) -> float:
        if self.device_type == "cuda":
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        return (time.perf_counter() - t0) * 1e3
