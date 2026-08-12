# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for half-layer PP split — config validation + layer construction.

No distributed required (config tests) or single-process gloo (build tests).
"""

import os
import pytest
import torch

# Set env before any megatron import
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_sublayer import (
    AttentionSubLayer,
    FFNSubLayer,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


# ═══ Helpers ═══

_backend = "gloo"  # Windows PyTorch may not have NCCL


def _init_dist():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=_backend, rank=0, world_size=1)


def _destroy_dist():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _make_config(**overrides):
    base = dict(
        num_layers=4, hidden_size=64, num_attention_heads=4,
        pipeline_model_parallel_size=2, pipeline_dtype=torch.float32,
        params_dtype=torch.float32, use_cpu_initialization=True,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _make_spec():
    return get_gpt_layer_local_spec(
        num_experts=None, moe_grouped_gemm=False,
        qk_layernorm=False, multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False, normalization="LayerNorm",
    )


class TestHalfLayerConfig:
    """Config validation — no distributed needed."""

    def test_split_all_layers_uniform_no_cross(self):
        cfg = _make_config(split_all_layers=True, decoder_num_layers_per_pipeline_stage=[4, 4])
        assert cfg.split_all_layers is True
        assert cfg.pipeline_split_layers is None

    def test_split_all_layers_auto_split_odd_prefix(self):
        cfg = _make_config(split_all_layers=True, decoder_num_layers_per_pipeline_stage=[3, 5])
        # cumsum 3 is odd → full layer (3+1)//2 = 2
        assert cfg.pipeline_split_layers == [2]

    def test_split_all_layers_multi_auto_split(self):
        cfg = _make_config(
            num_layers=4, split_all_layers=True, pipeline_model_parallel_size=4,
            decoder_num_layers_per_pipeline_stage=[3, 3, 1, 1],
        )  # sum=8=4*2. cumsum: 3(odd→2), 6(even), 7(odd→4)
        assert cfg.pipeline_split_layers == [2, 4]

    def test_split_all_layers_requires_vpp_raises(self):
        with pytest.raises(ValueError, match="VPP"):
            _make_config(split_all_layers=True, virtual_pipeline_model_parallel_size=2)

    def test_split_all_layers_bad_sum_raises(self):
        with pytest.raises(ValueError, match="must equal"):
            _make_config(split_all_layers=True, decoder_num_layers_per_pipeline_stage=[2, 4])

    def test_pipeline_split_layers_requires_distribution(self):
        with pytest.raises(ValueError, match="requires decoder_num_layers"):
            _make_config(pipeline_split_layers=[3])

    def test_pipeline_split_layers_non_boundary_raises(self):
        with pytest.raises(ValueError, match="not a stage boundary"):
            _make_config(
                num_layers=12, pipeline_model_parallel_size=4,
                decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
                pipeline_split_layers=[4],  # 4 not in {3, 6, 9}
            )

    def test_baseline_no_split(self):
        """Without any split flags, behavior unchanged."""
        cfg = _make_config()
        assert cfg.split_all_layers is False
        assert cfg.pipeline_split_layers is None


class TestHalfLayerBuild:
    """Layer construction — manual spec to avoid get_args() dependency."""

    @classmethod
    def setup_class(cls):
        _init_dist()
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        _destroy_dist()

    def _make_block(self, cfg):
        """Build a TransformerBlock with half-layer specs applied manually."""
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        raw_spec = _make_spec()  # ModuleSpec(module=TransformerLayer, submodules=...)
        sub = raw_spec.submodules

        if cfg.split_all_layers:
            # Manually build the half-layer spec list: each full layer → [Attn, FFN] pair
            layer_specs = []
            for g in range(1, cfg.num_layers + 1):
                layer_specs.append(ModuleSpec(
                    module=AttentionSubLayer, params={"global_layer_number": g},
                    submodules=sub))
                layer_specs.append(ModuleSpec(
                    module=FFNSubLayer, params={"global_layer_number": g},
                    submodules=sub))
        else:
            layer_specs = [raw_spec] * cfg.num_layers

        block_spec = TransformerBlockSubmodules(layer_specs=layer_specs)
        return TransformerBlock(cfg, block_spec)

    def test_split_all_layers_builds_attn_ffn_pairs(self):
        cfg = _make_config(num_layers=2, pipeline_model_parallel_size=1, split_all_layers=True)
        block = self._make_block(cfg)
        assert len(block.layers) == 4
        assert isinstance(block.layers[0], AttentionSubLayer)
        assert isinstance(block.layers[1], FFNSubLayer)
        assert isinstance(block.layers[2], AttentionSubLayer)
        assert isinstance(block.layers[3], FFNSubLayer)
        assert block.layers[0].layer_number == block.layers[1].layer_number == 1
        assert block.num_layers_per_pipeline_rank == 4

    def test_baseline_builds_full_layers(self):
        cfg = _make_config(num_layers=2, pipeline_model_parallel_size=1)
        block = self._make_block(cfg)
        assert len(block.layers) == 2
        assert isinstance(block.layers[0], TransformerLayer)


    # -- Forward pass smoke test (needs CUDA) --

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_split_forward_runs_without_error(self):
        """split_all_layers forward pass runs and produces correct output shape."""
        device = torch.device("cuda")
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        spec = _make_spec()
        sub = spec.submodules

        cfg_split = _make_config(num_layers=2, pipeline_model_parallel_size=1, split_all_layers=True)
        layer_specs = []
        for g in range(1, 3):
            layer_specs.append(ModuleSpec(module=AttentionSubLayer, params={"global_layer_number": g}, submodules=sub))
            layer_specs.append(ModuleSpec(module=FFNSubLayer, params={"global_layer_number": g}, submodules=sub))
        block = TransformerBlock(cfg_split, TransformerBlockSubmodules(layer_specs=layer_specs)).to(device)

        torch.manual_seed(42)
        x = torch.randn(2, 1, 64, device=device)

        hidden, ctx, pending = x, None, None
        for layer in block.layers:
            if isinstance(layer, AttentionSubLayer):
                pre_mlp, residual, ctx = layer(hidden_states=hidden, attention_mask=None, context=ctx)
                pending = (pre_mlp, residual)
            elif isinstance(layer, FFNSubLayer):
                hidden = layer(*pending)
                pending = None
        assert pending is None, "Must end with FFNSubLayer"
        assert hidden.shape == (2, 1, 64), f"Expected [2,1,64], got {hidden.shape}"

        # Backward must produce non-None grads
        loss = hidden.float().mean()
        loss.backward()
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Grad for {name} is None"
