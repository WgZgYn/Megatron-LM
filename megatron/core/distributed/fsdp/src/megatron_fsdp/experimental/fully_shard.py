# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Megatron-FSDP fully_shard entrypoint."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .module import FsdpContext, FsdpModule
from .placement import Placements


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    use_symm_mem: bool = False,
    grad_divisor: int = 1,
) -> None:
    """Apply FSDP to a module in place.

    This attaches the FSDP mixin to the original module instance, so parent
    modules do not need to replace existing child-module references.

    Args:
        module: Module whose currently unowned parameters are managed by FSDP.
        mesh: Device mesh used for sharding.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional precision policy. Defaults to FP32 main weights
            and parameter-dtype main gradients.
        use_symm_mem: Allocate all-gather and reduce-scatter staging buffers from
            PyTorch's NCCL symmetric-memory pool.
        grad_divisor: Additional divisor applied to the reduced gradient, on top of the
            averaging the mesh already performs. Defaults to 1, which is correct whenever
            each mesh rank contributes exactly one term to the gradient.

            Expert parallelism is the motivating case. A rank's experts process tokens
            routed to them from every rank in the expert-parallel group, and the backward
            pass routes those tokens' gradients back, so a rank's expert gradient already
            sums over ``ep_size`` ranks' data before any reduction happens. Averaging over
            the expert-data-parallel mesh alone therefore divides by too little, and
            ``grad_divisor=ep_size`` makes up the difference. Dense parameters see only
            their own rank's tokens and need no divisor.
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by FSDP.")

    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    original_cls = module.__class__
    _attach_mixin(module)
    try:
        assert isinstance(module, FsdpModule)
        FsdpModule.__init__(
            module,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=mixed_precision_policy,
            use_symm_mem=use_symm_mem,
            grad_divisor=grad_divisor,
        )
    except Exception:
        module.__class__ = original_cls
        raise


@contextmanager
def microbatch(module: nn.Module, is_last: bool) -> Iterator[None]:
    """Scope FSDP state to one microbatch.

    Args:
        module: Module tree whose FSDP roots should use this microbatch state.
        is_last: Whether forwards in this scope are for the last microbatch.
    """
    contexts: list[FsdpContext] = []
    _collect_fsdp_contexts(module, contexts)
    previous_states = [(context, context.is_last_microbatch) for context in contexts]
    for context in contexts:
        context.is_last_microbatch = is_last

    try:
        yield
    finally:
        for context, is_last_microbatch in previous_states:
            context.is_last_microbatch = is_last_microbatch


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls


def _collect_fsdp_contexts(module: nn.Module, contexts: list[FsdpContext]) -> None:
    if isinstance(module, FsdpModule):
        module._lazy_init_context()
        contexts.append(module.context)
        return

    for child in module.children():
        _collect_fsdp_contexts(child, contexts)
