# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for half-layer PP configuration, planning, and autograd."""

import pytest
import socket
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.pipeline_parallel.pipeline_partition import (
    build_pipeline_stage_partitions,
    get_cross_stage_split_layers,
)
from megatron.core.pipeline_parallel.schedules import (
    custom_backward,
    deallocate_output_tensor,
    get_tensor_shapes,
    send_forward_recv_backward,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_sublayer import AttentionSubLayer, FFNSubLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    values = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_split_all_full_layer_distribution_is_converted_to_halves():
    config = _make_config(
        split_all_layers=True,
        decoder_num_layers_per_pipeline_stage=[2, 2],
    )
    assert [part.num_half_layers for part in build_pipeline_stage_partitions(config)] == [4, 4]
    assert get_cross_stage_split_layers(config) == []


def test_half_layer_distribution_crosses_full_layer():
    config = _make_config(
        split_all_layers=True,
        decoder_num_half_layers_per_pipeline_stage=[3, 5],
    )
    partitions = build_pipeline_stage_partitions(config)
    assert (partitions[0].half_start, partitions[0].half_end) == (0, 3)
    assert partitions[0].ends_with_attention
    assert partitions[1].starts_with_ffn
    assert get_cross_stage_split_layers(config) == [2]
    assert config.pipeline_split_layers is None


def test_multiple_half_layer_boundaries():
    config = _make_config(
        pipeline_model_parallel_size=4,
        split_all_layers=True,
        decoder_num_half_layers_per_pipeline_stage=[3, 3, 1, 1],
    )
    assert get_cross_stage_split_layers(config) == [2, 4]


def test_real_gpt_spec_starts_second_stage_with_ffn(monkeypatch):
    config = _make_config(
        split_all_layers=True,
        decoder_num_half_layers_per_pipeline_stage=[3, 5],
    )
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 1)
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    assert [spec.module for spec in block_spec.layer_specs] == [
        FFNSubLayer,
        AttentionSubLayer,
        FFNSubLayer,
        AttentionSubLayer,
        FFNSubLayer,
    ]
    assert [spec.params["global_layer_number"] for spec in block_spec.layer_specs] == [
        2,
        3,
        3,
        4,
        4,
    ]


def test_transformer_block_builds_uniform_logical_layer_sequence(monkeypatch):
    config = _make_config(
        num_layers=2,
        pipeline_model_parallel_size=1,
        split_all_layers=True,
    )
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1', 0))
        port = port_socket.getsockname()[1]
    torch.distributed.init_process_group(
        'gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=0,
        world_size=1,
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    if torch.cuda.is_available():
        model_parallel_cuda_manual_seed(42)
    try:
        block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        block = TransformerBlock(config, block_spec, post_layer_norm=False)
        assert [type(layer) for layer in block.layers] == [
            AttentionSubLayer,
            FFNSubLayer,
            AttentionSubLayer,
            FFNSubLayer,
        ]
        if torch.cuda.is_available():
            block = block.cuda()
            hidden_states = torch.randn(
                3, 1, config.hidden_size, device='cuda', requires_grad=True
            )
            output = block(hidden_states=hidden_states, attention_mask=None)
            assert output.shape == hidden_states.shape
            output.sum().backward()
            assert hidden_states.grad is not None
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_p2p_shape_count_uses_same_partition_plan(monkeypatch):
    config = _make_config(
        split_all_layers=True,
        decoder_num_half_layers_per_pipeline_stage=[3, 5],
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    shape_args = dict(
        model_type=ModelType.encoder_or_decoder,
        seq_length=8,
        micro_batch_size=2,
        decoder_seq_length=None,
        config=config,
        encoder_decoder_xattn=False,
    )
    assert get_tensor_shapes(rank=0, **shape_args) == [(2, 8, 2, 64)]
    assert get_tensor_shapes(rank=1, **shape_args) == [(8, 2, 64)]


def test_pp2_11_13_contract(monkeypatch):
    config = _make_config(
        num_layers=12,
        split_all_layers=True,
        decoder_num_half_layers_per_pipeline_stage=[11, 13],
    )
    monkeypatch.setattr(parallel_state, 'get_context_parallel_world_size', lambda: 1)
    monkeypatch.setattr(parallel_state, 'get_pipeline_model_parallel_rank', lambda: 1)

    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    assert len(block_spec.layer_specs) == 13
    assert block_spec.layer_specs[0].module is FFNSubLayer
    assert block_spec.layer_specs[0].params['global_layer_number'] == 6
    assert block_spec.layer_specs[-1].module is FFNSubLayer
    assert block_spec.layer_specs[-1].params['global_layer_number'] == 12

    shape_args = dict(
        model_type=ModelType.encoder_or_decoder,
        seq_length=128,
        micro_batch_size=2,
        decoder_seq_length=None,
        config=config,
        encoder_decoder_xattn=False,
    )
    assert get_tensor_shapes(rank=0, **shape_args) == [(2, 128, 2, 64)]
    assert get_tensor_shapes(rank=1, **shape_args) == [(128, 2, 64)]


def test_logical_layers_pack_and_unpack_one_tensor(monkeypatch):
    import megatron.core.transformer.transformer_sublayer as logical_layers

    source = torch.randn(3, 2, 4, requires_grad=True)

    def fake_attention(module, hidden_states, **kwargs):
        return hidden_states * 2, hidden_states * 3, 'context'

    def fake_mlp(module, pre_mlp_layernorm_output, residual):
        return pre_mlp_layernorm_output + residual

    monkeypatch.setattr(logical_layers, '_run_attention', fake_attention)
    monkeypatch.setattr(logical_layers, '_run_mlp', fake_mlp)
    attention = AttentionSubLayer.__new__(AttentionSubLayer)
    ffn = FFNSubLayer.__new__(FFNSubLayer)
    packed, context = AttentionSubLayer.forward(attention, source)
    output, output_context = FFNSubLayer.forward(ffn, packed, context=context)

    assert packed.shape == (2, 3, 2, 4)
    assert context == 'context'
    assert output_context is None
    torch.testing.assert_close(output, source * 5)
    output.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 5))


def test_1f1b_boundary_uses_one_standard_p2p_call(monkeypatch):
    packed = torch.randn(2, 8, 2, 64)
    packed_grad = torch.randn_like(packed)
    calls = []

    def standard_p2p(output_tensor, tensor_shape, config):
        calls.append((output_tensor, tensor_shape))
        return packed_grad

    monkeypatch.setattr(
        'megatron.core.pipeline_parallel.schedules.p2p_communication.'
        'send_forward_recv_backward',
        standard_p2p,
    )
    result = send_forward_recv_backward(packed, [(2, 8, 2, 64)], object())
    assert result[0] is packed_grad
    assert len(calls) == 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            dict(split_all_layers=True, virtual_pipeline_model_parallel_size=2),
            "VPP",
        ),
        (
            dict(
                split_all_layers=True,
                decoder_num_half_layers_per_pipeline_stage=[2, 4],
            ),
            "must equal",
        ),
        (
            dict(
                split_all_layers=True,
                decoder_num_layers_per_pipeline_stage=[2, 2],
                decoder_num_half_layers_per_pipeline_stage=[4, 4],
            ),
            "mutually exclusive",
        ),
        (dict(pipeline_split_layers=[3]), "requires decoder_num_layers"),
    ],
)
def test_invalid_partition_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(**overrides)


def test_selective_split_must_be_full_layer_stage_boundary():
    with pytest.raises(ValueError, match="not a stage boundary"):
        _make_config(
            num_layers=12,
            pipeline_model_parallel_size=4,
            decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
            pipeline_split_layers=[4],
        )


def test_packed_boundary_custom_backward_after_deallocation():
    """The packed boundary follows the standard single-output schedule path."""
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    packed = torch.stack((source * 3, source * 5), dim=0)
    deallocate_output_tensor(packed, deallocate_pipeline_outputs=True)
    custom_backward(packed, torch.tensor([[7.0, 7.0], [11.0, 11.0]]))
    torch.testing.assert_close(source.grad, torch.full_like(source, 76.0))
