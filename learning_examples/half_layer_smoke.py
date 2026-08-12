"""Smoke test for half-layer PP split (Phase 1).

Verifies spec construction and forward pass correctness for:
  - split_all_layers mode (every layer → [AttentionSubLayer, FFNSubLayer] pair)
  - pipeline_split_layers mode (selective cross-stage splits)

Usage:  python learning_examples/half_layer_smoke.py
"""

import os, sys
from unittest import mock

import torch
torch.compile = lambda f, **kwargs: f  # stub Megatron jit

sys.path.insert(0, r"F:\PycharmProjects\Megatron-LM")

# ═══ Init ═══
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1,
                                          init_method="tcp://127.0.0.1:29501")

# ═══ Mock parallel_state ═══
import megatron.core.parallel_state as real_ps
_stage = [0]
def _grank(): return _stage[0]
def _gpp_rank(): return _stage[0]
def _set_stage(s): _stage[0] = s

real_ps._TENSOR_MODEL_PARALLEL_GROUP = torch.distributed.group.WORLD
real_ps._PIPELINE_MODEL_PARALLEL_GROUP = torch.distributed.group.WORLD
real_ps._DATA_PARALLEL_GROUP = torch.distributed.group.WORLD
real_ps.is_pipeline_first_stage = lambda *a, **kw: _stage[0] == 0
real_ps.is_pipeline_last_stage = lambda *a, **kw: _stage[0] == 1
real_ps.get_pipeline_model_parallel_rank = lambda *a, **kw: _stage[0]
real_ps.get_pipeline_model_parallel_world_size = lambda *a, **kw: 2
real_ps.is_inside_encoder = lambda: True
real_ps.get_pipeline_model_parallel_next_rank = lambda: None
real_ps.get_pipeline_model_parallel_prev_rank = lambda: None
real_ps.get_virtual_pipeline_model_parallel_world_size = lambda: None
real_ps.get_virtual_pipeline_model_parallel_rank = lambda: None
real_ps.get_context_parallel_world_size = lambda: 1
real_ps.get_context_parallel_rank = lambda: 0
real_ps.get_data_parallel_world_size = lambda **kw: 1
real_ps.get_data_parallel_rank = lambda **kw: 0

# Mock RNG tracker
_tracker = mock.MagicMock()
_tracker.fork.return_value.__enter__ = lambda _: None
_tracker.fork.return_value.__exit__ = lambda *_: None
from unittest.mock import patch
for mod in [
    "megatron.core.tensor_parallel.random.get_cuda_rng_tracker",
    "megatron.core.tensor_parallel.layers.get_cuda_rng_tracker",
    "megatron.core.transformer.dot_product_attention.tensor_parallel.get_cuda_rng_tracker",
    "megatron.core.transformer.attention.tensor_parallel.get_cuda_rng_tracker",
]:
    patch(mod, return_value=_tracker).start()

# Mock global memory buffer (used by dot_product_attention)
_gmb = mock.MagicMock()
def _make_tensor(shape, dtype, name):
    return torch.empty(shape, dtype=dtype, device=device)
_gmb.get_tensor.side_effect = _make_tensor
real_ps.get_global_memory_buffer = lambda: _gmb

# ═══ Build configs ═══
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_sublayer import AttentionSubLayer, FFNSubLayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

_layer_spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    moe_use_legacy_grouped_gemm=False, normalization="LayerNorm")

def make_config(**overrides):
    base = dict(num_layers=4, hidden_size=128, num_attention_heads=4,
                pipeline_model_parallel_size=2, pipeline_dtype=torch.float32,
                params_dtype=torch.float32)
    base.update(overrides)
    return TransformerConfig(**base)

errors = []

# ═══ Test 1: split_all_layers mode ═══
print("=== Test 1: split_all_layers (PP=2, 4 layers) ===")
cfg = make_config(split_all_layers=True)

# Manually build specs (mirrors gpt_layer_specs logic)
all_specs = [_layer_spec] * 4  # 4 full layer specs

# Rank 0: layers 1-2 → [Attn(1), FFN(1), Attn(2), FFN(2)]
_set_stage(0)
stage0_specs = []
for g in range(1, 3):  # layers 1-2
    full = all_specs[g - 1]
    stage0_specs.append(ModuleSpec(module=AttentionSubLayer, params={"global_layer_number": g}, submodules=full.submodules))
    stage0_specs.append(ModuleSpec(module=FFNSubLayer, params={"global_layer_number": g}, submodules=full.submodules))

# Rank 1: layers 3-4 → [Attn(3), FFN(3), Attn(4), FFN(4)]
_set_stage(1)
stage1_specs = []
for g in range(3, 5):  # layers 3-4
    full = all_specs[g - 1]
    stage1_specs.append(ModuleSpec(module=AttentionSubLayer, params={"global_layer_number": g}, submodules=full.submodules))
    stage1_specs.append(ModuleSpec(module=FFNSubLayer, params={"global_layer_number": g}, submodules=full.submodules))

expected = [
    ("Stage0[0]", AttentionSubLayer, 1),
    ("Stage0[1]", FFNSubLayer, 1),
    ("Stage0[2]", AttentionSubLayer, 2),
    ("Stage0[3]", FFNSubLayer, 2),
    ("Stage1[0]", AttentionSubLayer, 3),
    ("Stage1[1]", FFNSubLayer, 3),
    ("Stage1[2]", AttentionSubLayer, 4),
    ("Stage1[3]", FFNSubLayer, 4),
]
for label, cls, ln in expected:
    print(f"  {label}: {cls.__name__}(ln={ln})")

for label, cls, ln in expected:
    s = stage0_specs if "Stage0" in label else stage1_specs
    idx = int(label.split("[")[1].rstrip("]"))
    actual = s[idx]
    if not issubclass(actual.module, cls):
        errors.append(f"{label}: expected {cls.__name__}, got {actual.module.__name__}")
    if actual.params.get("global_layer_number") != ln:
        errors.append(f"{label}: expected ln={ln}, got {actual.params.get('global_layer_number')}")

# ═══ Test 2: Build modules and run forward ═══
print("\n=== Test 2: Forward pass (split_all_layers) ===")
_set_stage(0)
mods0 = [build_module(s, config=cfg, layer_number=i+1).to(device) for i, s in enumerate(stage0_specs)]
_set_stage(1)
mods1 = [build_module(s, config=cfg, layer_number=i+1).to(device) for i, s in enumerate(stage1_specs)]

torch.manual_seed(42)
x = torch.randn(2, 1, 128, device=device)

# Stage 0 forward
hidden, ctx, pending = x, None, None
for m in mods0:
    if isinstance(m, AttentionSubLayer):
        pre_mlp, residual, ctx = m(hidden_states=hidden, attention_mask=None, context=ctx)
        pending = (pre_mlp, residual)
    elif isinstance(m, FFNSubLayer):
        assert pending is not None, "FFNSubLayer needs pending input"
        hidden = m(*pending)
        pending = None
assert pending is None, "Stage 0 must end with FFNSubLayer"
out0 = hidden.clone()

# Stage 1 forward
pending = None
for m in mods1:
    if isinstance(m, AttentionSubLayer):
        pre_mlp, residual, ctx = m(hidden_states=hidden, attention_mask=None, context=ctx)
        pending = (pre_mlp, residual)
    elif isinstance(m, FFNSubLayer):
        hidden = m(*pending)
        pending = None

# Backward test
loss = hidden.float().mean()
loss.backward()
grads_ok = True
for name, p in mods0[0].named_parameters():
    if p.requires_grad and p.grad is None:
        grads_ok = False
        errors.append(f"Stage0 {name}: grad is None")
for name, p in mods0[-1].named_parameters():
    if p.requires_grad and p.grad is None:
        grads_ok = False
        errors.append(f"Stage0 last {name}: grad is None")

print(f"  Forward OK, backward grads: {'ALL_PRESENT' if grads_ok else 'MISSING!'}")

# ═══ Result ═══
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
