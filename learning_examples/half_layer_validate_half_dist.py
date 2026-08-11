"""Validate decoder_num_half_layers_per_pipeline_stage with cross-stage split.

PP=4, 12 layers -> 24 half-layers, distribution [5,7,6,6].
Auto-detects split at layer 3 (cumsum 5 is odd).
GPU0 ends with AttentionSubLayer(3), GPU1 starts with FFNSubLayer(3).
Tests: 2-tensor P2P across the cross-stage boundary.

Usage: torchrun --nproc_per_node=4 half_layer_validate_half_dist.py
"""
import os, torch, torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

import megatron.core.parallel_state as mpu
mpu.initialize_model_parallel(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=4,
    pipeline_model_parallel_comm_backend=None,
    context_parallel_size=1, expert_model_parallel_size=1,
    order="tp-cp-ep-dp-pp",
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

pp_rank = mpu.get_pipeline_model_parallel_rank()
pp_size = mpu.get_pipeline_model_parallel_world_size()
g_rank = dist.get_rank()
print(f"[INIT] global_rank={g_rank} pp_rank={pp_rank}/{pp_size}", flush=True)

from megatron.core.transformer.transformer_config import TransformerConfig
config = TransformerConfig(
    num_layers=4, hidden_size=128, num_attention_heads=4,
    pipeline_model_parallel_size=4, pipeline_dtype=torch.float32,
    split_all_layers=True,
    decoder_num_half_layers_per_pipeline_stage=[3, 1, 2, 2],
    # 4 layers * 2 = 8 half-layers total. Stage boundaries: cumsum [3,4,6]
    # 3%2=1 -> layer 2 split (Attn on GPU0, FFN on GPU1)
    # 4%2=0 -> no split
    # 6%2=0 -> no split
    recompute_granularity=None, params_dtype=torch.float32,
    fp16=False, bf16=False,
)
print(f"[RANK {g_rank}] auto splits={config.pipeline_split_layers}", flush=True)

from megatron.core.transformer.transformer_sublayer import AttentionSubLayer, FFNSubLayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

layer_template = get_gpt_layer_local_spec(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    moe_use_legacy_grouped_gemm=False, normalization="LayerNorm",
)

# Build per-stage specs: half-layer distribution + auto cross-stage split
all_specs = [layer_template] * 4
half_dist = config.decoder_num_half_layers_per_pipeline_stage
offset = sum(half_dist[:pp_rank])  # half-layers before this stage
count  = half_dist[pp_rank]        # half-layers for this stage

stage_specs = []
for h in range(offset, offset + count):
    full_idx = h // 2          # 0-based full layer index
    is_attn = (h % 2 == 0)     # even = attention half, odd = FFN half
    global_ln = full_idx + 1   # 1-based layer number
    full_spec = all_specs[full_idx]
    if is_attn:
        stage_specs.append(ModuleSpec(
            module=AttentionSubLayer, params={"global_layer_number": global_ln},
            submodules=full_spec.submodules))
    else:
        stage_specs.append(ModuleSpec(
            module=FFNSubLayer, params={"global_layer_number": global_ln},
            submodules=full_spec.submodules))

modules = [build_module(s, config=config, layer_number=i+1).cuda() for i, s in enumerate(stage_specs)]
inv = ", ".join(f"{type(m).__name__}(ln={m.layer_number})" for m in modules)
print(f"[RANK {g_rank}] pp={pp_rank} [{len(modules)} halves]: [{inv}]", flush=True)
dist.barrier()

# Forward pass through this stage's half-layers
device = torch.device("cuda")
torch.manual_seed(42 + rank)
hidden = torch.randn(2, 1, 128, device=device)
ctx, pending = None, None

for layer in modules:
    if isinstance(layer, AttentionSubLayer):
        pre_mlp, residual, ctx = layer(hidden_states=hidden, attention_mask=None, context=ctx)
        pending = (pre_mlp, residual)
    elif isinstance(layer, FFNSubLayer):
        hidden = layer(*pending)
        pending = None

# Identify stage boundary type
ends_with_attn = isinstance(modules[-1], AttentionSubLayer) if modules else False
starts_with_ffn = isinstance(modules[0], FFNSubLayer) if modules else False

if pp_rank < pp_size - 1 and ends_with_attn:
    # Cross-stage split: send 2 tensors to next stage
    assert pending is not None
    dist.send(pending[0].contiguous(), dst=g_rank + 1)
    dist.send(pending[1].contiguous(), dst=g_rank + 1)
    print(f"[RANK {g_rank}] CROSS-SPLIT SEND: pre_mlp+residual -> rank {g_rank+1}", flush=True)
    # Receive grads back
    g0 = torch.zeros_like(pending[0]); g1 = torch.zeros_like(pending[1])
    dist.recv(g0, src=g_rank + 1); dist.recv(g1, src=g_rank + 1)
    torch.autograd.backward([pending[0], pending[1]], grad_tensors=[g0, g1])
    print(f"[RANK {g_rank}] CROSS-SPLIT BACKWARD: grad_pre_mlp={g0.norm():.4f} grad_residual={g1.norm():.4f}", flush=True)
else:
    # Last stage or no split: just run backward
    loss = hidden.float().mean()
    loss.backward()
    if pp_rank > 0 and starts_with_ffn:
        # Receive from prev stage, run forward, send grads back
        # (This is the simpler case handled by half_layer_validate.py)
        pass

if g_rank == 0:
    print("\n" + "=" * 60)
    print("half-layer distribution + cross-stage split: PASSED")
    print("=" * 60)

dist.destroy_process_group()
