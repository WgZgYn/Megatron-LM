"""Parallelism implementations for the profiling harness.

- ``dp``: data parallelism via DDP (+ optional timed gradient all-reduce hook).
- ``tp``: Megatron-style column/row parallel linear layers.
- ``pp``: Gpipe-style pipeline parallelism with explicit blocking P2P.
"""

from . import dp, pp, tp  # noqa: F401
