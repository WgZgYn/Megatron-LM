"""Distributed validation for split_all_layers mode.

torchrun --nproc_per_node=2 half_layer_validate_all.py
"""
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

import megatron.core.parallel_state as mpu
mpu.initialize_model_parallel(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=2,
    pipeline_model_parallel_comm_backend=None,
    context_parallel_size=1, expert_model_parallel_size=1,
    order="tp-cp-ep-dp-pp",
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

pp_rank = mpu.get_pipeline_model_parallel_rank()
g_rank = dist.get_rank()
print(f"[INIT] global_rank={g_rank} pp_rank={pp_rank}/2", flush=True)

from megatron.core.transformer.transformer_config import TransformerConfig
config = TransformerConfig(
    num_layers=4, hidden_size=128, num_attention_heads=4,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.float32,
    split_all_layers=True,
    recompute_granularity=None, params_dtype=torch.float32,
    fp16=False, bf16=False,
)

from megatron.core.transformer.transformer_sublayer import AttentionSubLayer, FFNSubLayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

layer_template = get_gpt_layer_local_spec(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    moe_use_legacy_grouped_gemm=False, normalization="LayerNorm",
)

# Build per-stage specs: each full layer → [Attn, FFN] pair
offset = pp_rank * 2  # 2 full layers per stage
all_specs = [layer_template] * 4
stage_specs = []
for g in range(offset + 1, offset + 3):  # 1-based layers this stage
    stage_specs.append(ModuleSpec(module=AttentionSubLayer, params={"global_layer_number": g},
                                  submodules=all_specs[g-1].submodules))
    stage_specs.append(ModuleSpec(module=FFNSubLayer, params={"global_layer_number": g},
                                  submodules=all_specs[g-1].submodules))

modules = [build_module(s, config=config, layer_number=i+1).cuda() for i, s in enumerate(stage_specs)]
inv = ", ".join(f"{type(m).__name__}(ln={m.layer_number})" for m in modules)
print(f"[RANK {g_rank}] pp={pp_rank} layers [{len(modules)}]: [{inv}]", flush=True)
dist.barrier()

# Forward + backward through all 4 half-layers in sequence
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

assert pending is None, f"Stage {pp_rank}: must end with FFNSubLayer"

loss = hidden.float().mean()
loss.backward()

# Check grads
grads_ok = all(p.grad is not None for p in modules[0].parameters() if p.requires_grad) and \
           all(p.grad is not None for p in modules[-1].parameters() if p.requires_grad)
print(f"[RANK {g_rank}] split_all_layers: forward OK, backward grads={'OK' if grads_ok else 'MISSING!'}", flush=True)

if g_rank == 0:
    print("\n" + "=" * 60)
    print("split_all_layers Validation: PASSED" if grads_ok else "FAILED")
    print("=" * 60)

dist.destroy_process_group()
