"""Minimal smoke check for per-stage layer distribution (mock parallel_state)."""
import sys
from unittest import mock

# torch 2.3.1 does not support Dynamo on Python 3.12; Megatron's jit_fuser calls
# torch.compile at import time. Stub it out for this logic-only smoke check.
import torch
torch.compile = lambda f, **kwargs: f

sys.path.insert(0, r"F:\PycharmProjects\Megatron-LM")

from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.transformer_config import TransformerConfig

cfg = TransformerConfig(
    num_layers=12,
    hidden_size=128,
    num_attention_heads=8,
    pipeline_model_parallel_size=4,
    pipeline_dtype=torch.float32,
    decoder_num_layers_per_pipeline_stage=[1, 5, 2, 4],
)

# expected: stage0 -> 1 layer (global 1), stage1 -> 5 (2-6), stage2 -> 2 (7-8), stage3 -> 4 (9-12)
exp_layers = [1, 5, 2, 4]
exp_offset = [0, 1, 6, 8]

ps = sys.modules["megatron.core.transformer.transformer_block"].parallel_state
ps2 = sys.modules["megatron.core.transformer.transformer_layer"].parallel_state

ok = True
for rank in range(4):
    with mock.patch.object(ps, "get_pipeline_model_parallel_rank", return_value=rank), \
         mock.patch.object(ps, "is_inside_encoder", return_value=True):
        n = get_num_layers_to_build(cfg)
    with mock.patch.object(ps2, "get_pipeline_model_parallel_rank", return_value=rank), \
         mock.patch.object(ps2, "is_inside_encoder", return_value=True):
        o = get_transformer_layer_offset(cfg)
    status = "OK" if (n, o) == (exp_layers[rank], exp_offset[rank]) else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"rank {rank}: layers={n} (exp {exp_layers[rank]}), offset={o} (exp {exp_offset[rank]}) -> {status}")

# regression: distribution=None -> uniform path unchanged (rank 1, 12 layers / 4 stages -> 3 layers, offset 3)
cfg2 = TransformerConfig(
    num_layers=12, hidden_size=128, num_attention_heads=8,
    pipeline_model_parallel_size=4, pipeline_dtype=torch.float32,
)
with mock.patch.object(ps, "get_pipeline_model_parallel_rank", return_value=1), \
     mock.patch.object(ps, "is_inside_encoder", return_value=True), \
     mock.patch.object(ps, "is_pipeline_first_stage", return_value=False), \
     mock.patch.object(ps, "is_pipeline_last_stage", return_value=False):
    n2 = get_num_layers_to_build(cfg2)
with mock.patch.object(ps2, "get_pipeline_model_parallel_rank", return_value=1), \
     mock.patch.object(ps2, "is_inside_encoder", return_value=True):
    o2 = get_transformer_layer_offset(cfg2)
print(f"uniform rank1: layers={n2} (exp 3), offset={o2} (exp 3) -> {'OK' if (n2, o2) == (3, 3) else 'FAIL'}")
ok = ok and (n2, o2) == (3, 3)

print("\nSMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
