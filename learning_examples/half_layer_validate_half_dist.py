"""Validate the PP=2, 12-layer, [11, 13] logical-layer partition.

The attention/FFN boundary is encoded as one [2, S, B, H] tensor. This script
builds the real per-rank GPT block spec, runs forward/backward, and performs one
forward send and one backward send at the split boundary.

Usage: torchrun --nproc_per_node=2 learning_examples/half_layer_validate_half_dist.py
"""

import os

import torch
import torch.distributed as dist


local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    order='tp-cp-ep-dp-pp',
)
model_parallel_cuda_manual_seed(42)

rank = dist.get_rank()
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
config = TransformerConfig(
    num_layers=12,
    hidden_size=128,
    num_attention_heads=4,
    pipeline_model_parallel_size=2,
    pipeline_dtype=torch.float32,
    params_dtype=torch.float32,
    split_all_layers=True,
    decoder_num_half_layers_per_pipeline_stage=[11, 13],
    recompute_granularity=None,
)

block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
block = TransformerBlock(
    config,
    block_spec,
    pre_process=(pp_rank == 0),
    post_process=(pp_rank == 1),
).cuda()

sequence_length, micro_batch_size = 2, 1
boundary_shape = (2, sequence_length, micro_batch_size, config.hidden_size)

if pp_rank == 0:
    hidden_states = torch.randn(
        sequence_length,
        micro_batch_size,
        config.hidden_size,
        device='cuda',
        requires_grad=True,
    )
    boundary = block(hidden_states=hidden_states, attention_mask=None)
    assert tuple(boundary.shape) == boundary_shape
    dist.send(boundary.contiguous(), dst=1)

    boundary_grad = torch.empty_like(boundary)
    dist.recv(boundary_grad, src=1)
    torch.autograd.backward(boundary, boundary_grad)
    assert hidden_states.grad is not None
    print('[rank0] packed forward send and backward receive passed', flush=True)
else:
    boundary = torch.empty(boundary_shape, device='cuda')
    dist.recv(boundary, src=0)
    boundary.requires_grad_()
    block.set_input_tensor(boundary)
    output = block(hidden_states=None, attention_mask=None)
    output.float().mean().backward()
    assert boundary.grad is not None
    dist.send(boundary.grad.contiguous(), dst=0)
    print('[rank1] packed forward receive and backward send passed', flush=True)

dist.barrier()
if rank == 0:
    print('PP=2 [11,13] packed-boundary validation: PASSED', flush=True)

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
