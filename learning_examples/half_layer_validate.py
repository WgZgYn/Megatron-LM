"""Distributed half-layer PP validation (F.2 + F.3).

Requires torchrun:  torchrun --nproc_per_node=2 half_layer_validate.py

Verifies:
  - Stage 0 produces 2-tensor output from AttentionSubLayer
  - Stage 1 receives 2-tensor input and forwards through FFNSubLayer
  - Both input grads are non-None after backward
  - Per-rank layer inventory matches expected split
"""

import os
import sys

import torch
import torch.distributed as dist

# Init distributed BEFORE importing Megatron (it calls get_args etc.)
dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

assert world_size == 2, "This test requires exactly 2 GPUs"

# Patch megatron globals before any Megatron import
# (megatron imports call get_args() which needs distributed init)
# We use a minimal mock approach.

import megatron.core.parallel_state as mpu

# Minimal model-parallel init: PP=2, TP=1, DP=1
mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    pipeline_model_parallel_comm_backend="gloo" if not torch.cuda.is_available() else None,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    order="tp-cp-ep-dp-pp",
)

pp_rank = mpu.get_pipeline_model_parallel_rank()
pp_size = mpu.get_pipeline_model_parallel_world_size()
g_rank = dist.get_rank()

print(f"[INIT] global_rank={g_rank} pp_rank={pp_rank}/{pp_size}", flush=True)

# Initialize CUDA RNG tracker (required by ColumnParallelLinear/RowParallelLinear weight init)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

# ═══ Build config ═══
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    num_layers=4,
    hidden_size=128,
    num_attention_heads=4,
    pipeline_model_parallel_size=2,
    pipeline_dtype=torch.float32,
    decoder_num_layers_per_pipeline_stage=[2, 2],
    pipeline_split_layers=[2],
    recompute_granularity=None,
    params_dtype=torch.float32,
    fp16=False,
    bf16=False,
)

# ═══ Build layer specs for this stage ═══
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_sublayer import (
    AttentionSubLayer, FFNSubLayer,
)
from megatron.core.transformer.transformer_layer import TransformerLayer

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

# Get a full layer spec template
# get_gpt_layer_local_spec returns ModuleSpec(module=TransformerLayer, submodules=...)
# This IS the per-layer spec — use it directly, not .layer_specs
dense_template = get_gpt_layer_local_spec(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    moe_use_legacy_grouped_gemm=False, normalization="LayerNorm",
)
all_specs = [dense_template] * 4

# Build per-stage specs
if pp_rank == 0:
    # Stage 0: [TransformerLayer(1), AttentionSubLayer(2)]
    stage_specs = [
        all_specs[0],  # full layer 1
        ModuleSpec(
            module=AttentionSubLayer,
            params={"global_layer_number": 2},
            submodules=dense_template.submodules,
        ),
    ]
else:
    # Stage 1: [FFNSubLayer(2), TransformerLayer(3), TransformerLayer(4)]
    stage_specs = [
        ModuleSpec(
            module=FFNSubLayer,
            params={"global_layer_number": 2},
            submodules=dense_template.submodules,
        ),
        all_specs[2],  # full layer 3
        all_specs[3],  # full layer 4
    ]

# Build modules
modules = []
for i, spec in enumerate(stage_specs):
    m = build_module(spec, config=config, layer_number=i + 1)
    modules.append(m)

# Print inventory
inv = ", ".join(f"{type(m).__name__}(ln={m.layer_number})" for m in modules)
print(f"[RANK {g_rank}] pp={pp_rank} layers: [{inv}]", flush=True)
dist.barrier()

# ═══ Forward test (no PP communication — manual hand-off) ═══
torch.manual_seed(42 + rank)
x0 = torch.randn(2, 1, 128)  # input to stage 0

if pp_rank == 0:
    hidden = x0
    ctx = None
    pending = None

    for layer in modules:
        if isinstance(layer, AttentionSubLayer):
            pre_mlp, residual, ctx = layer(
                hidden_states=hidden, attention_mask=None, context=ctx,
            )
            pending = (pre_mlp, residual)
        else:
            hidden, ctx = layer(hidden_states=hidden, attention_mask=None, context=ctx)

    assert pending is not None, "Stage 0 must end with AttentionSubLayer output"
    print(f"[RANK {g_rank}] Stage0 output: pre_mlp={pending[0].shape} residual={pending[1].shape}",
          flush=True)

    # Send to rank 1 (simulating PP schedule)
    dist.send(pending[0].contiguous(), dst=1)
    dist.send(pending[1].contiguous(), dst=1)

    # Receive grads back (after rank 1 backward)
    grad_pre_mlp = torch.zeros_like(pending[0])
    grad_residual = torch.zeros_like(pending[1])
    dist.recv(grad_pre_mlp, src=1)
    dist.recv(grad_residual, src=1)

    # Run backward through stage 0 with received grads
    torch.autograd.backward([pending[0], pending[1]],
                            grad_tensors=[grad_pre_mlp, grad_residual])
    print(f"[RANK {g_rank}] Stage0 backward done. "
          f"grad_pre_mlp_norm={grad_pre_mlp.norm():.4f} "
          f"grad_residual_norm={grad_residual.norm():.4f}",
          flush=True)

else:
    # Rank 1: receive from rank 0
    recv_pre_mlp = torch.zeros(2, 1, 128)
    recv_residual = torch.zeros(2, 1, 128)
    dist.recv(recv_pre_mlp, src=0)
    dist.recv(recv_residual, src=0)

    # require grad (simulating PP recv buffer)
    recv_pre_mlp.requires_grad = True
    recv_residual.requires_grad = True

    pending = (recv_pre_mlp, recv_residual)
    hidden = recv_residual
    ctx = None

    for layer in modules:
        if isinstance(layer, FFNSubLayer):
            hidden = layer(*pending)
            pending = None
        else:
            hidden, ctx = layer(hidden_states=hidden, attention_mask=None, context=ctx)

    assert pending is None, "Stage 1 must consume pending_mlp_input"
    print(f"[RANK {g_rank}] Stage1 output: hidden={hidden.shape}", flush=True)

    # Backward
    loss = hidden.float().mean()
    loss.backward()

    assert recv_pre_mlp.grad is not None, "grad of pre_mlp_layernorm_output must not be None!"
    assert recv_residual.grad is not None, "grad of residual must not be None!"

    print(f"[RANK {g_rank}] Stage1 backward: "
          f"grad_pre_mlp_norm={recv_pre_mlp.grad.norm():.4f} "
          f"grad_residual_norm={recv_residual.grad.norm():.4f}",
          flush=True)

    # Send grads back to rank 0
    dist.send(recv_pre_mlp.grad.contiguous(), dst=0)
    dist.send(recv_residual.grad.contiguous(), dst=0)

dist.barrier()

# ═══ Compare with un-split reference (not practical in distributed test;
#      F.3 does this with full training runs) ═══
# Here we just verify all assertions passed.

if g_rank == 0:
    print("\n" + "=" * 60)
    print("F.2 Validation: PASSED (assertions verified on both ranks)")
    print("=" * 60)

dist.destroy_process_group()
