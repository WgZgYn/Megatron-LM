# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore HybridModel.

Checks that an ``EP=2`` MoE ``HybridModel`` sharded with mFSDP v2 (experts over the
expert-DP sub-mesh, the remaining dense params over the full DP mesh), consuming its
``1/dp`` shard of a global batch, reproduces the same training as a single **full-batch
``EP=1`` reference**.

The reference processes the **whole** global batch on every rank, so its gradients are
identical across ranks and need no reduction -- it has no distributed logic. The model's
gradients are reduced only by mFSDP. So this is an independent check of EP all-to-all
dispatch, FSDP sharding, and the gradient reduction/scaling: a broken reduction (or a
missing expert-grad scaling factor) would diverge from full-batch training.

Both models are built from explicit ``ProcessGroupCollection``s (no global
``parallel_state`` / ``initialize_model_parallel``): the reference with a size-1 ``ep``
group (all experts local) and the model with the 2-way ``ep`` group.

Model shapes and the ``(ep, dp)`` split are test-local so different tests can vary them.
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


def _config(num_experts, ep_size, hidden, ffn):
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
        moe_ffn_hidden_size=ffn,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.local,
    )


def _pgc(one, world, ep, expt_dp):
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
        expt_dp=expt_dp,
        dp=world,
        dp_cp=world,
        embd=None,
        pos_embd=None,
    )


def _build(config, pgc, vocab, seq):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab,
        max_sequence_length=seq,
        hybrid_layer_pattern="E",
        pg_collection=pgc,
    ).cuda()


def test_ep_fsdp_matches_fullbatch_reference(distributed_setup):
    """EP=2 + mFSDP on 1/dp-sharded data reproduces single full-batch EP=1 training."""
    device = distributed_setup.device
    dev = device.type
    world_size = distributed_setup.world_size
    rank = distributed_setup.rank

    num_experts, ep_size = 8, 2
    hidden, ffn = 16, 64
    vocab, seq, b_local = 128, 8, 2
    if world_size % ep_size != 0 or num_experts % ep_size != 0:
        pytest.skip(f"world_size {world_size} is incompatible with EP={ep_size}.")
    dp_size = world_size // ep_size
    global_batch = world_size * b_local  # split one shard per rank

    # Process groups, all derived from device meshes (no global parallel_state):
    #   one     -> size-1 groups (TP=PP=CP=1, and the EP=1 reference's ep group)
    #   world   -> the full DP group (dense + EP=1 expert-DP)
    #   moe_mesh-> ep (ep_size-way) and expert-DP (dp_size-way) groups for the EP=2 model
    one = init_device_mesh(dev, (world_size, 1), mesh_dim_names=("w", "one"))["one"].get_group()
    world = init_device_mesh(dev, (world_size,)).get_group()
    moe_mesh = init_device_mesh(dev, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_grp, expt_dp_grp = moe_mesh["ep"].get_group(), moe_mesh["dp"].get_group()

    # Reference: EP=1 (all num_experts local). Model: EP=2 (num_experts/EP local per rank).
    # Same seed + use_cpu_initialization => weights are identical across ranks.
    torch.manual_seed(123)
    reference = _build(_config(num_experts, 1, hidden, ffn), _pgc(one, world, one, world), vocab, seq)
    torch.manual_seed(123)
    model = _build(
        _config(num_experts, ep_size, hidden, ffn),
        _pgc(one, world, ep_grp, expt_dp_grp),
        vocab,
        seq,
    )

    # Align model weights to the reference: dense params directly; expert params by their
    # GLOBAL index (the model's local weight i is global expert local_expert_indices[i]).
    reference_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        if ".experts." not in name:
            param.data.copy_(reference_params[name].data)
    moe = model.decoder.layers[0].mlp
    reference_experts = reference.decoder.layers[0].mlp.experts
    for fc in ("linear_fc1", "linear_fc2"):
        reference_fc, model_fc = getattr(reference_experts, fc), getattr(moe.experts, fc)
        for local, global_ in enumerate(moe.local_expert_indices):
            getattr(model_fc, f"weight{local}").data.copy_(
                getattr(reference_fc, f"weight{global_}").data
            )

    # Shard the model: experts over the expert-DP sub-mesh, dense params over the full DP mesh.
    for decoder_layer in model.decoder.layers:
        fully_shard(
            decoder_layer.mlp.experts,
            mesh=moe_mesh["dp"],
            placements=Placements(
                dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
            ),
        )
    dense_mesh = init_device_mesh(dev, (world_size,))
    fully_shard(
        model,
        mesh=dense_mesh,
        placements=Placements(
            dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
        ),
    )

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.02, foreach=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)

    # One global batch, identical on every rank (same seed). The reference consumes the whole
    # batch; the model consumes its 1/dp shard.
    torch.manual_seed(4321)
    input_ids = torch.randint(0, vocab, (global_batch, seq), dtype=torch.int64, device=device)
    position_ids = torch.arange(seq, dtype=torch.int64, device=device).repeat(global_batch, 1)
    attention_mask = torch.ones((global_batch, 1, seq, seq), dtype=torch.bool, device=device)
    target = torch.randn(global_batch, seq, vocab, device=device)
    shard = slice(rank * b_local, (rank + 1) * b_local)

    def train_reference():
        # Full batch on every rank -> identical grads across ranks, so no reduction is
        # needed. A plain single-model reference with no distributed logic.
        losses = []
        for _ in range(5):
            reference_optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(
                reference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask),
                target,
            )
            losses.append(loss.detach())
            loss.backward()
            reference_optimizer.step()
        return losses

    def train_model():
        # The model trains on its 1/dp shard; mFSDP reduces the gradients. All-reduce only
        # the loss (AVG) to recover the global full-batch loss for the comparison.
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(
                model(
                    input_ids=input_ids[shard],
                    position_ids=position_ids[shard],
                    attention_mask=attention_mask[shard],
                ),
                target[shard],
            )
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG, group=world)
            losses.append(global_loss)
            loss.backward()
            optimizer.step()
        return losses

    reference_losses = train_reference()
    sharded_losses = train_model()

    # rtol dominates the tolerance; the residual drift is benign EP-path numerics (alltoall
    # token reordering + grouped-GEMM over num_experts/EP vs all experts), ~2e-5 at 5 steps.
    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(reference_losses),
        rtol=1e-3,
        atol=1e-4,
        msg="EP=2 mFSDP model did not reproduce full-batch EP=1 training.",
    )
