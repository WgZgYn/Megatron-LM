"""Build-level validation of per-stage layer distribution (mock parallel_state, CPU only)."""
import sys
from unittest import mock

import torch
import torch.distributed as dist

# torch 2.3.1 + Py3.12: Megatron's jit_fuser calls torch.compile at import time; stub it.
torch.compile = lambda f, **kwargs: f

sys.path.insert(0, r"F:\PycharmProjects\Megatron-LM")

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

# CPU-only: no CUDA RNG state available; fake the rng tracker used at weight init.
from contextlib import nullcontext
import megatron.core.tensor_parallel.layers as _tp_layers

class _FakeRngTracker:
    def fork(self, *a, **k):
        return nullcontext()

_tp_layers.get_cuda_rng_tracker = lambda: _FakeRngTracker()

NUM_LAYERS, HIDDEN, HEADS, SEQ, VOCAB, PP = 12, 128, 8, 64, 256, 4
DIST = [1, 5, 2, 4]

# single-process gloo group so dist calls (get_world_size etc.) work on CPU
dist.init_process_group(
    backend="gloo", init_method="tcp://127.0.0.1:29501", rank=0, world_size=1
)
# real TP=1 / PP=1 groups (world_size=1), so tensor-parallel getters work untouched
ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

cfg = TransformerConfig(
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN,
    num_attention_heads=HEADS,
    pipeline_model_parallel_size=PP,
    pipeline_dtype=torch.float32,
    decoder_num_layers_per_pipeline_stage=DIST,
)
spec = get_gpt_layer_local_spec(
    num_experts=None, moe_grouped_gemm=False, qk_layernorm=False,
    multi_latent_attention=False, moe_use_legacy_grouped_gemm=False,
)

ok = True
for pp_rank in range(PP):
    with mock.patch.object(ps, "get_pipeline_model_parallel_rank", return_value=pp_rank), \
         mock.patch.object(ps, "get_pipeline_model_parallel_world_size", return_value=PP), \
         mock.patch.object(ps, "get_virtual_pipeline_model_parallel_world_size", return_value=None), \
         mock.patch.object(ps, "is_inside_encoder", return_value=True), \
         mock.patch.object(ps, "is_pipeline_first_stage", return_value=(pp_rank == 0)), \
         mock.patch.object(ps, "is_pipeline_last_stage", return_value=(pp_rank == PP - 1)), \
         mock.patch.object(ps, "get_pipeline_model_parallel_decoder_start", return_value=None):
        model = GPTModel(
            config=cfg, transformer_layer_spec=spec, vocab_size=VOCAB,
            max_sequence_length=SEQ,
            pre_process=(pp_rank == 0), post_process=(pp_rank == PP - 1),
        )

    local = [layer.layer_number for layer in model.decoder.layers]
    exp_start = sum(DIST[:pp_rank]) + 1
    expected = list(range(exp_start, exp_start + DIST[pp_rank]))
    good = local == expected
    ok = ok and good
    print(f"pp_rank={pp_rank}: n={len(local)} layer_numbers={local} expected={expected} "
          f"emb={'Y' if hasattr(model, 'embedding') else 'N'}(exp {'Y' if pp_rank==0 else 'N'}) "
          f"out={'Y' if hasattr(model, 'output_layer') else 'N'}(exp {'Y' if pp_rank==PP-1 else 'N'}) -> {'OK' if good else 'FAIL'}")

print("\nBUILD-LEVEL", "PASS" if ok else "FAIL")
ps.destroy_model_parallel()
dist.destroy_process_group()
sys.exit(0 if ok else 1)
