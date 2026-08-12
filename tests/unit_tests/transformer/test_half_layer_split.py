# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for half-layer PP configuration, planning, and autograd."""

import pytest
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
    send_backward_recv_forward,
    send_forward_recv_backward,
)
from megatron.core.transformer.transformer_sublayer import AttentionSubLayer, FFNSubLayer
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
    assert len(get_tensor_shapes(rank=0, **shape_args)) == 2
    assert len(get_tensor_shapes(rank=1, **shape_args)) == 1


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


def test_multi_output_custom_backward_after_deallocation():
    """Both attention outputs must remain roots of the backward graph."""
    source = torch.tensor(2.0, requires_grad=True)
    first = source * 3
    second = source * 5
    deallocate_output_tensor([first, second], deallocate_pipeline_outputs=True)
    custom_backward([first, second], [torch.tensor(7.0), torch.tensor(11.0)])
    assert source.grad.item() == 76.0


def test_multi_tensor_1f1b_uses_one_batched_p2p_transaction(monkeypatch):
    """Sequential combined calls deadlock across a two-tensor boundary."""
    shapes = [(8, 2, 64), (8, 2, 64)]
    outputs = [torch.tensor(1.0), torch.tensor(2.0)]
    grads = [torch.tensor(3.0), torch.tensor(4.0)]
    calls = []

    def fail_single(*args, **kwargs):
        raise AssertionError('single-tensor combined P2P must not be used')

    def forward_multi(tensors, tensor_shapes, config):
        calls.append(('forward', tensors, tensor_shapes))
        return grads

    def backward_multi(tensors, tensor_shapes, config):
        calls.append(('backward', tensors, tensor_shapes))
        return outputs

    monkeypatch.setattr(
        'megatron.core.pipeline_parallel.schedules.p2p_communication.'
        'send_forward_recv_backward',
        fail_single,
    )
    monkeypatch.setattr(
        'megatron.core.pipeline_parallel.schedules.p2p_communication.'
        'send_backward_recv_forward',
        fail_single,
    )
    monkeypatch.setattr(
        'megatron.core.pipeline_parallel.schedules.p2p_communication.'
        'send_forward_recv_backward_multi',
        forward_multi,
    )
    monkeypatch.setattr(
        'megatron.core.pipeline_parallel.schedules.p2p_communication.'
        'send_backward_recv_forward_multi',
        backward_multi,
    )

    config = object()
    assert send_forward_recv_backward(outputs, shapes, config) == grads
    assert send_backward_recv_forward(grads, shapes, config) == outputs
    assert [call[0] for call in calls] == ['forward', 'backward']
