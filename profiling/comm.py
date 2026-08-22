"""Communication timing utilities.

``CommTimer`` logs every collective (op, byte count, latency, effective
bandwidth) to a per-rank JSONL consumed by ``analyze.py``.

Two timing paths, chosen by whether the collective is *blocking*:

* **Blocking collectives** (TP ``all_reduce``, PP ``send``/``recv``) — timed with
  CUDA events that bracket the kernel on the compute stream. Events are recorded
  per-op but resolved in one ``torch.cuda.synchronize()`` at flush time, so no
  extra sync barrier is inserted into the step (this used to distort the trace).
* **Async all-reduce** (DDP gradient buckets) — the all-reduce is non-blocking
  and runs on DDP's communication stream, so it cannot be bracketed with events
  on the compute stream. It is timed wall-clock from launch to future
  resolution instead, which preserves DDP's native backward/communication
  overlap.

Bandwidth convention: ``all_reduce`` reports effective bytes
``2 * (n - 1) / n * numel * element_size`` (ring approximation); point-to-point
reports ``numel * element_size``.
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
        self._pending = []  # (step, name, num_bytes, start_ev, end_ev, src, dst, tag)
        self._step = 0
        self._seq = 0

    # -- lifecycle ----------------------------------------------------------

    def set_step(self, step: int) -> None:
        self._step = step

    def flush(self):
        self.resolve_pending()
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

    def _add_pending(self, name, num_bytes, start, end, src=None, dst=None, tag=None):
        self._pending.append((self._step, name, num_bytes, start, end, src, dst, tag))

    def resolve_pending(self) -> None:
        """Resolve all deferred CUDA-event timings with a single device sync."""
        if self.device_type == "cuda" and self._pending:
            torch.cuda.synchronize()
            for step, name, num_bytes, start, end, src, dst, tag in self._pending:
                self._step = step
                self._record(name, num_bytes, start.elapsed_time(end), src, dst, tag)
            self._pending.clear()

    def _ring_bytes(self, tensor, group):
        world = dist.get_world_size(group) if group is not None else self.world_size
        factor = 2.0 * (world - 1) / world if world > 1 else 1.0
        return int(tensor.numel() * tensor.element_size() * factor)

    # -- blocking collectives (TP all-reduce, PP send/recv) -----------------

    def timed_all_reduce(self, tensor, group=None, name="all_reduce"):
        num_bytes = self._ring_bytes(tensor, group)
        if not self.enabled:
            dist.all_reduce(tensor, group=group)
            return tensor
        if self.device_type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor, group=group)
            end.record()
            self._add_pending(name, num_bytes, start, end)
        else:
            t0 = time.perf_counter()
            dist.all_reduce(tensor, group=group)
            self._record(name, num_bytes, (time.perf_counter() - t0) * 1e3)
        return tensor

    def timed_send(self, tensor, dst, tag=0, name="send"):
        num_bytes = int(tensor.numel() * tensor.element_size())
        if not self.enabled:
            dist.send(tensor, dst, tag=tag)
            return
        if self.device_type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.send(tensor, dst, tag=tag)
            end.record()
            self._add_pending(name, num_bytes, start, end, dst=dst, tag=tag)
        else:
            t0 = time.perf_counter()
            dist.send(tensor, dst, tag=tag)
            self._record(name, num_bytes, (time.perf_counter() - t0) * 1e3, dst=dst, tag=tag)

    def timed_recv(self, tensor, src, tag=0, name="recv"):
        num_bytes = int(tensor.numel() * tensor.element_size())
        if not self.enabled:
            dist.recv(tensor, src, tag=tag)
            return
        if self.device_type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.recv(tensor, src, tag=tag)
            end.record()
            self._add_pending(name, num_bytes, start, end, src=src, tag=tag)
        else:
            t0 = time.perf_counter()
            dist.recv(tensor, src, tag=tag)
            self._record(name, num_bytes, (time.perf_counter() - t0) * 1e3, src=src, tag=tag)

    # -- async all-reduce (DDP gradient buckets) ----------------------------

    def timed_all_reduce_async(self, tensor, group=None, name="all_reduce"):
        """Launch a non-blocking all-reduce and time it wall-clock until done.

        Returns a ``torch.futures.Future`` (for the DDP comm hook contract).
        Wall-clock is used because DDP runs the NCCL kernel on its own
        communication stream, which CUDA events on the compute stream cannot
        bracket.
        """
        num_bytes = self._ring_bytes(tensor, group)
        t0 = time.perf_counter()
        handle = dist.all_reduce(tensor, group=group, async_op=True)

        if not self.enabled:
            return handle.get_future().then(lambda f: f.value()[0])

        def _done(f):
            self._record(name, num_bytes, (time.perf_counter() - t0) * 1e3)
            return f.value()[0]

        return handle.get_future().then(_done)
