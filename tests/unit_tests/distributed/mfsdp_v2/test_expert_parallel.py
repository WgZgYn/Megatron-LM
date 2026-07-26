# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore HybridModel.

Checks that an ``EP=2`` MoE ``HybridModel`` sharded with mFSDP v2 (experts over the
expert-DP sub-mesh, dense params over the full DP mesh), consuming its ``1/dp`` shard of a
global batch, reproduces a single **full-batch ``EP=1`` reference**.

The reference processes the whole global batch on every rank, so its gradients are
identical across ranks and need no reduction -- it has no distributed logic. The model's
gradients are reduced only by mFSDP. So this independently validates EP all-to-all
dispatch, FSDP sharding, and the gradient reduction/scaling: a broken reduction (or a
missing expert-grad scaling factor) would diverge from full-batch training.

Both models are built from explicit ``ProcessGroupCollection``s (no global
``parallel_state`` / ``initialize_model_parallel``): the reference with a size-1 ``ep``
group (all experts local), the model with the 2-way ``ep`` group. Model shapes and the
``(ep, dp)`` split are test-local so different tests can vary them.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

_FLAT_SHARD = Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _transformer_config(num_experts, ep_size, hidden, ffn_hidden):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        expert_model_parallel_size=ep_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=ffn_hidden,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.local,
    )


def _process_group_collection(one, world, ep, expert_dp):
    """A ProcessGroupCollection for a TP=PP=CP=1 MoE model with the given ep/expert-dp."""
    return ProcessGroupCollection(
        tp=one,
        expt_tp=one,
        cp=one,
        pp=one,
        tp_cp=one,
        tp_dp_cp=world,
        ep=ep,
        tp_ep=ep,
        expt_dp=expert_dp,
        dp=world,
        dp_cp=world,
        embd=None,
        pos_embd=None,
    )


def _build_hybrid_model(config, pg_collection, vocab, seq):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab,
        max_sequence_length=seq,
        hybrid_layer_pattern="E",
        pg_collection=pg_collection,
    ).cuda()


def _train(model, ids, pos, mask, target, loss_reduce_group=None):
    """Run 5 SGD steps; return the per-step losses (globally averaged if loss_reduce_group given)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(input_ids=ids, position_ids=pos, attention_mask=mask), target
        )
        loss.backward()
        optimizer.step()
        # The model sees a shard, so average the loss across ranks for the global loss.
        loss = loss.detach()
        if loss_reduce_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=loss_reduce_group)
        losses.append(loss)
    return losses


def test_ep_fsdp_matches_fullbatch_reference(distributed_setup):
    """EP=2 + mFSDP on 1/dp-sharded data reproduces single full-batch EP=1 training."""
    device = distributed_setup.device
    world_size, rank = distributed_setup.world_size, distributed_setup.rank

    num_experts, ep_size = 8, 2
    hidden, ffn_hidden, vocab, seq, b_local = 16, 64, 128, 8, 2
    if world_size % ep_size != 0 or num_experts % ep_size != 0:
        pytest.skip(f"world_size {world_size} is incompatible with EP={ep_size}.")
    dp_size = world_size // ep_size
    global_batch = world_size * b_local  # one shard per rank

    # Process groups (no global parallel_state). world_mesh: the full DP group; moe_mesh: the
    # ep (ep_size-way) and expert-DP (dp_size-way) groups for the EP=2 model. Meshes also
    # initialize the default process group, so build them before the size-1 group below.
    world_mesh = init_device_mesh(device.type, (world_size,))
    world = world_mesh.get_group()
    moe_mesh = init_device_mesh(device.type, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_group, expert_dp_group = moe_mesh.get_group("ep"), moe_mesh.get_group("dp")
    # This rank's size-1 group: the trivial TP=PP=CP axes and the EP=1 reference's ep group.
    one = dist.new_group([rank], use_local_synchronization=True)

    # Reference EP=1 (all experts local); model EP=2. Seed once so the reference is
    # deterministic and identical across ranks (CPU init); the model's own init is irrelevant
    # since its weights are copied from the reference below.
    torch.manual_seed(123)
    reference = _build_hybrid_model(
        _transformer_config(num_experts, 1, hidden, ffn_hidden),
        _process_group_collection(one, world, one, world),
        vocab,
        seq,
    )
    model = _build_hybrid_model(
        _transformer_config(num_experts, ep_size, hidden, ffn_hidden),
        _process_group_collection(one, world, ep_group, expert_dp_group),
        vocab,
        seq,
    )

    # Dense params line up by name (load_state_dict); the experts do not -- EP=1 stores all
    # experts as weight0.., EP=2 stores num_experts/EP as weight0.. per rank -- so patch them
    # by global index (model local weight i == reference global weight local_expert_indices[i]).
    model.load_state_dict(reference.state_dict(), strict=False)
    for model_layer, reference_layer in zip(model.decoder.layers, reference.decoder.layers):
        for fc in ("linear_fc1", "linear_fc2"):
            model_fc = getattr(model_layer.mlp.experts, fc)
            reference_fc = getattr(reference_layer.mlp.experts, fc)
            for local, global_ in enumerate(model_layer.mlp.local_expert_indices):
                getattr(model_fc, f"weight{local}").data.copy_(
                    getattr(reference_fc, f"weight{global_}").data
                )

    # Shard the model: experts over the expert-DP sub-mesh, dense params over the full DP mesh.
    for decoder_layer in model.decoder.layers:
        fully_shard(decoder_layer.mlp.experts, mesh=moe_mesh["dp"], placements=_FLAT_SHARD)
    fully_shard(model, mesh=world_mesh, placements=_FLAT_SHARD)

    # One global batch, identical on every rank; the reference sees all of it, the model its shard.
    torch.manual_seed(4321)
    ids = torch.randint(0, vocab, (global_batch, seq), dtype=torch.int64, device=device)
    pos = torch.arange(seq, dtype=torch.int64, device=device).repeat(global_batch, 1)
    mask = torch.ones((global_batch, 1, seq, seq), dtype=torch.bool, device=device)
    target = torch.randn(global_batch, seq, vocab, device=device)
    shard = slice(rank * b_local, (rank + 1) * b_local)

    reference_losses = _train(reference, ids, pos, mask, target)
    model_losses = _train(model, ids[shard], pos[shard], mask[shard], target[shard], loss_reduce_group=world)

    # rtol dominates; the residual ~2e-5 drift is benign EP-path numerics (alltoall token
    # reordering + grouped-GEMM over num_experts/EP vs all experts).
    torch.testing.assert_close(
        torch.stack(model_losses),
        torch.stack(reference_losses),
        rtol=1e-3,
        atol=0,
        msg="EP=2 mFSDP model did not reproduce full-batch EP=1 training.",
    )

    # Destroy the groups this test created; leave the default (world) group for later tests.
    for group in (one, ep_group, expert_dp_group):
        dist.destroy_process_group(group)
