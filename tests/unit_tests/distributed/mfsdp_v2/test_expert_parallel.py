# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore HybridModel.

Builds an EP ``HybridModel`` whose single ``"E"`` layer is a ``MoELayer`` (router +
all-to-all token dispatch + ``TEGroupedMLP`` experts) and shards the whole model with
mFSDP v2: the EP-partitioned experts over a 1-D expert-DP mesh (sliced from a ``(dp, ep)``
mesh), and the remaining EP-replicated ("dense") params over the full 1-D DP mesh. The
composition must be numerically transparent (matches an EP-only baseline that has no mFSDP
applied) and must place each expert weight on the expert-DP sub-mesh (ep excluded) while
dense params shard over all ranks.

Mesh topology and model shapes are test-local so different tests can pick different
``(ep, dp)`` splits.
"""


import dataclasses

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@dataclasses.dataclass(frozen=True)
class ModelParallelSizes:
    """Parallelism sizes for the ``model_parallel`` fixture (each defaults to 1)."""

    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


@pytest.fixture
def model_parallel(request, distributed_setup):
    """Set up and tear down model parallelism from a ``ModelParallelSizes`` request.param.

    The fixture skips when the world size is incompatible, yields the requested
    ``ModelParallelSizes`` and the resolved ``dp_size``, and tears down even if the test
    fails.
    """
    sizes: ModelParallelSizes = request.param
    non_dp = sizes.tp_size * sizes.pp_size * sizes.cp_size * sizes.ep_size
    if distributed_setup.world_size % non_dp != 0:
        pytest.skip(f"world_size {distributed_setup.world_size} is incompatible with {sizes}.")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=sizes.tp_size,
        pipeline_model_parallel_size=sizes.pp_size,
        context_parallel_size=sizes.cp_size,
        expert_model_parallel_size=sizes.ep_size,
    )
    dp_size = distributed_setup.world_size // non_dp
    yield sizes, dp_size
    Utils.destroy_model_parallel()


def _moe_config(
    num_routed_experts: int, expert_model_parallel_size: int, hidden_size: int, ffn_hidden_size: int
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_routed_experts,
        expert_model_parallel_size=expert_model_parallel_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=ffn_hidden_size,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def _build_hybrid_model(config: TransformerConfig) -> HybridModel:
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=8,
        hybrid_layer_pattern="E",
    )


@pytest.mark.parametrize("model_parallel", [ModelParallelSizes(ep_size=2)], indirect=True)
def test_hybrid_model_shards_experts_and_dense(distributed_setup, model_parallel):
    """mFSDP v2 shards an entire EP HybridModel: experts over expert-DP, dense over full DP.

    Composes two nested (bottom-up) fully_shard calls -- the experts over a 1-D expert-DP
    mesh (sliced from a (dp, ep) mesh), then the remaining EP-replicated ("dense") params
    over the full 1-D DP mesh -- and checks the composition is numerically transparent
    versus an EP-only baseline (same EP model, no mFSDP) across forward, backward, and
    optimizer step.
    """
    sizes, dp_size = model_parallel
    ep_size = sizes.ep_size
    num_routed_experts = 8
    hidden, ffn = 16, 64
    seq, batch, vocab = 8, 2, 128
    device = distributed_setup.device
    world_size = distributed_setup.world_size

    config = _moe_config(num_routed_experts, ep_size, hidden, ffn)
    # Disable dropout so the baseline and sharded forward passes are RNG-independent.
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    # data_parallel_random_init=False gives every DP rank the same seed, so weights
    # initialize identically across ranks. Coherent FSDP sharding requires this: each rank
    # shards its own copy, so the all-gather only reconstructs a consistent weight if all
    # ranks started equal. (It also makes the run deterministic.)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    baseline = _build_hybrid_model(config).cuda()
    model = _build_hybrid_model(config).cuda()
    # The model under test starts from the EP-only baseline's weights.
    model.load_state_dict(baseline.state_dict())

    # Bottom-up: shard each decoder layer's experts over this rank's expert-DP group. Build
    # the full 2-D (dp, ep) mesh (ep innermost, matching MCore's ep-fastest layout) only to
    # slice out the 1-D expert-DP sub-mesh -- a plain init_device_mesh((dp_size,)) would
    # group the wrong ranks ({0,1,2,3} instead of the expert-DP {0,2,4,6}).
    moe_mesh = init_device_mesh(device.type, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    for decoder_layer in model.decoder.layers:
        fully_shard(
            decoder_layer.mlp.experts,
            mesh=moe_mesh["dp"],
            placements=Placements(
                dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
            ),
        )
    # ...then the remaining EP-replicated params shard over the full DP mesh.
    dense_mesh = init_device_mesh(device.type, (world_size,))
    fully_shard(
        model,
        mesh=dense_mesh,
        placements=Placements(
            dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
        ),
    )

    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.02, foreach=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)

    # Re-anchor the RNG right before the draws so the inputs are identical across ranks --
    # the transparency invariant (identical data -> the DP gradient reduction is a no-op, so
    # the mFSDP-sharded model must reproduce the EP-only baseline exactly). Re-anchoring here
    # makes this independent of whatever RNG the two model builds above consumed (which could
    # differ per rank), and decouples the inputs from the weight-init seed. randint keeps the
    # inputs non-degenerate across batch/seq.
    torch.manual_seed(4321)
    input_ids = torch.randint(0, vocab, (batch, seq), dtype=torch.int64, device=device)
    position_ids = torch.arange(seq, dtype=torch.int64, device=device).repeat(batch, 1)
    attention_mask = torch.ones((batch, 1, seq, seq), dtype=torch.bool, device=device)
    target = torch.randn(batch, seq, vocab, device=device)

    def train(m, opt) -> list[torch.Tensor]:
        losses = []
        for _ in range(5):
            opt.zero_grad()
            logits = m(
                input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
            )
            loss = torch.nn.functional.mse_loss(logits, target)
            losses.append(loss.detach())
            loss.backward()
            opt.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer)
    sharded_losses = train(model, optimizer)

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        rtol=1e-4,
        atol=1e-5,
        msg="mFSDP sharding was not numerically transparent vs the EP-only baseline.",
    )
