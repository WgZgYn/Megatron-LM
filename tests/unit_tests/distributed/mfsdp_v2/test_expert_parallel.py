# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore HybridModel.

Checks that an ``EP=2`` MoE ``HybridModel`` sharded with mFSDP v2 (experts over the
expert-DP sub-mesh, the remaining dense params over the full DP mesh) reproduces a
**fully replicated ``EP=1`` baseline** trained with plain DDP-style gradient averaging.

The two are compared on **distinct data per rank**, so unlike an identical-data
transparency check this actually exercises the cross-rank gradient reduction (a broken
reduce-scatter would diverge) as well as EP alltoall dispatch and FSDP sharding.

Both models are built from explicit ``ProcessGroupCollection``s (no global
``parallel_state`` / ``initialize_model_parallel``): the baseline with a size-1 ``ep``
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


def test_ep_fsdp_matches_replicated_ddp_baseline(distributed_setup):
    """EP=2 + mFSDP reproduces a replicated EP=1 DDP baseline on distinct per-rank data."""
    device = distributed_setup.device
    dev = device.type
    world_size = distributed_setup.world_size
    rank = distributed_setup.rank

    num_experts, ep_size = 8, 2
    hidden, ffn = 16, 64
    vocab, seq, batch = 128, 8, 2
    if world_size % ep_size != 0 or num_experts % ep_size != 0:
        pytest.skip(f"world_size {world_size} is incompatible with EP={ep_size}.")
    dp_size = world_size // ep_size

    # Process groups, all derived from device meshes (no global parallel_state):
    #   one     -> size-1 groups (TP=PP=CP=1, and the EP=1 baseline's ep group)
    #   world   -> the full DP group (dense + EP=1 expert-DP)
    #   moe_mesh-> ep (ep_size-way) and expert-DP (dp_size-way) groups for the EP=2 model
    one = init_device_mesh(dev, (world_size, 1), mesh_dim_names=("w", "one"))["one"].get_group()
    world = init_device_mesh(dev, (world_size,)).get_group()
    moe_mesh = init_device_mesh(dev, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_grp, expt_dp_grp = moe_mesh["ep"].get_group(), moe_mesh["dp"].get_group()
    expt_dp_size = expt_dp_grp.size()

    # Baseline: EP=1 (all num_experts local, no dispatch). Model: EP=2 (num_experts/EP local).
    # Same seed + use_cpu_initialization => weights are identical across ranks.
    torch.manual_seed(123)
    baseline = _build(_config(num_experts, 1, hidden, ffn), _pgc(one, world, one, world), vocab, seq)
    torch.manual_seed(123)
    model = _build(
        _config(num_experts, ep_size, hidden, ffn),
        _pgc(one, world, ep_grp, expt_dp_grp),
        vocab,
        seq,
    )

    # Align model weights to the baseline: dense params directly; expert params by their
    # GLOBAL index (the model's local weight i is global expert local_expert_indices[i]).
    baseline_params = dict(baseline.named_parameters())
    for name, param in model.named_parameters():
        if ".experts." not in name:
            param.data.copy_(baseline_params[name].data)
    moe = model.decoder.layers[0].mlp
    baseline_experts = baseline.decoder.layers[0].mlp.experts
    for fc in ("linear_fc1", "linear_fc2"):
        baseline_fc, model_fc = getattr(baseline_experts, fc), getattr(moe.experts, fc)
        for local, global_ in enumerate(moe.local_expert_indices):
            getattr(model_fc, f"weight{local}").data.copy_(
                getattr(baseline_fc, f"weight{global_}").data
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

    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.02, foreach=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)

    # Distinct data per rank -- this is what makes the DP gradient reduction observable.
    torch.manual_seed(4321 + rank)
    input_ids = torch.randint(0, vocab, (batch, seq), dtype=torch.int64, device=device)
    position_ids = torch.arange(seq, dtype=torch.int64, device=device).repeat(batch, 1)
    attention_mask = torch.ones((batch, 1, seq, seq), dtype=torch.bool, device=device)
    target = torch.randn(batch, seq, vocab, device=device)

    def forward(m):
        return m(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    def train_baseline():
        losses = []
        for _ in range(5):
            baseline_optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(forward(baseline), target)
            losses.append(loss.detach())
            loss.backward()
            # Expert-aware DDP averaging that matches the sharded model's reductions: the
            # EP=2 forward already sums each ep group's tokens, so expert grads average over
            # expert-DP (size expt_dp_size) while dense grads average over the full DP world.
            for name, param in baseline.named_parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=world)
                param.grad /= expt_dp_size if ".experts." in name else world_size
            baseline_optimizer.step()
        return losses

    def train_model():
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(forward(model), target)
            losses.append(loss.detach())
            loss.backward()
            optimizer.step()
        return losses

    baseline_losses = train_baseline()
    sharded_losses = train_model()

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        rtol=1e-3,
        atol=1e-4,
        msg="EP=2 mFSDP model did not match the replicated EP=1 DDP baseline.",
    )
