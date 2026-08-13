"""Exercise the PP=4 [5, 7, 6, 6] split with Megatron's real 1F1B schedule.

Usage:
  LOG_P2P_COMMS=1 torchrun --nproc_per_node=4 \
    learning_examples/half_layer_validate_schedule.py
"""

import os

import torch
import torch.distributed as dist


local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    order='tp-cp-ep-dp-pp',
)
model_parallel_cuda_manual_seed(42)

pp_rank = parallel_state.get_pipeline_model_parallel_rank()
config = TransformerConfig(
    num_layers=12,
    hidden_size=128,
    num_attention_heads=4,
    pipeline_model_parallel_size=4,
    pipeline_dtype=torch.float32,
    params_dtype=torch.float32,
    decoder_num_half_layers_per_pipeline_stage=[5, 7, 6, 6],
    deallocate_pipeline_outputs=True,
    recompute_granularity=None,
)

block = TransformerBlock(
    config,
    get_gpt_decoder_block_spec(config, use_transformer_engine=False),
    pre_process=(pp_rank == 0),
    post_process=(pp_rank == 3),
).cuda()


class ScheduleModel(torch.nn.Module):
    """Expose the same pipeline input contract as GPTModel."""

    def __init__(self, decoder, model_config):
        super().__init__()
        self.decoder = decoder
        self.config = model_config
        self.model_type = ModelType.encoder_or_decoder

    def set_input_tensor(self, input_tensor):
        assert len(input_tensor) == 1
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(self, hidden_states):
        return self.decoder(hidden_states=hidden_states, attention_mask=None)


model = ScheduleModel(block, config)

sequence_length = 2
micro_batch_size = 1
num_microbatches = 8
if pp_rank == 0:
    data_iterator = iter(
        [
            torch.randn(
                sequence_length,
                micro_batch_size,
                config.hidden_size,
                device='cuda',
            )
            for _ in range(num_microbatches)
        ]
    )
else:
    data_iterator = iter([None] * num_microbatches)


def forward_step(data, model):
    hidden_states = next(data) if pp_rank == 0 else None
    output = model(hidden_states)

    def loss_func(tensor):
        loss = tensor.float().square().mean()
        return loss, {'loss': loss.detach()}

    return output, loss_func


losses = forward_backward_pipelining_without_interleaving(
    forward_step_func=forward_step,
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=sequence_length,
    micro_batch_size=micro_batch_size,
    forward_only=False,
)

assert any(parameter.grad is not None for parameter in model.parameters())
dist.barrier()
if dist.get_rank() == 0:
    print('PP=4 [5,7,6,6] 8-microbatch 1F1B validation: PASSED', flush=True)

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
