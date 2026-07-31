"""
Validate uneven per-pipeline-stage layer distribution.

Usage (on a GPU server, after pulling the feature branch):
    torchrun --nproc_per_node=4 --master_port=29500 pp_distribution_validate.py

Expected: each rank builds exactly the layers assigned by
--decoder-num-layers-per-pipeline-stage, with continuous global
layer numbers, and forward/backward runs on the last stage.

The script is self-contained: no dataset / tokenizer / checkpoint needed.
"""

import sys

import torch
import torch.distributed as dist

sys.path.insert(0, ".")

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

# ---- model / parallel config (edit freely) ----
NUM_LAYERS = 12
HIDDEN = 128
HEADS = 8
SEQ = 64
VOCAB = 256
TP = 1
PP = 4
DISTRIBUTION = [1, 5, 2, 4]  # must sum to NUM_LAYERS, length == PP
DTYPE = torch.bfloat16


def main():
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=TP, pipeline_model_parallel_size=PP)
    rank = dist.get_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        pipeline_model_parallel_size=PP,
        pipeline_dtype=DTYPE,
        decoder_num_layers_per_pipeline_stage=DISTRIBUTION,
    )
    spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    model.to(torch.cuda.current_device())

    # ---- check 1: per-stage layer count and global layer numbers ----
    local = [layer.layer_number for layer in model.decoder.layers]
    expected_n = DISTRIBUTION[pp_rank]
    expected_start = sum(DISTRIBUTION[:pp_rank]) + 1  # 1-based global numbering
    expected = list(range(expected_start, expected_start + expected_n))
    ok1 = local == expected
    print(
        f"[rank {rank}] pp_rank={pp_rank} n_layers={len(local)} "
        f"layer_numbers={local} expected={expected} -> {'OK' if ok1 else 'FAIL'}"
    )
    # first stage must hold embedding, last stage must hold output layer
    ok1 = ok1 and (mpu.is_pipeline_first_stage() == hasattr(model, "embedding"))
    ok1 = ok1 and (mpu.is_pipeline_last_stage() == hasattr(model, "output_layer"))

    # ---- check 2: forward (+ backward on last stage) ----
    torch.manual_seed(0)
    if mpu.is_pipeline_first_stage():
        tokens = torch.randint(0, VOCAB, (SEQ, 1), device="cuda")
        pos = torch.arange(SEQ, device="cuda").unsqueeze(1).expand(SEQ, 1)
        out = model(tokens, pos, None)
    else:
        hidden = torch.randn(SEQ, 1, HIDDEN, dtype=DTYPE, device="cuda")
        model.set_input_tensor(hidden)
        out = model(None, None, None)

    if mpu.is_pipeline_last_stage():
        assert out.shape == (SEQ, 1, VOCAB), f"unexpected logits shape {out.shape}"
        out.float().mean().backward()
        print(f"[rank {rank}] forward+backward OK, logits shape {tuple(out.shape)}")
    else:
        out.float().mean().backward()
        print(f"[rank {rank}] forward OK (hidden shape {tuple(out.shape)})")

    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()
    print(f"[rank {rank}] DONE ok1={ok1}")


if __name__ == "__main__":
    main()
