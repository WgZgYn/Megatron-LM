# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Half-layer modules for fine-grained pipeline parallelism.

AttentionSubLayer and FFNSubLayer are logical layers with the same external
contract as TransformerLayer. Their intermediate state is one packed tensor,
so pipeline scheduling never handles multiple GPT boundary tensors.

When a layer is split across PP stages:
  Stage r:   ... TransformerLayer(n) -> AttentionSubLayer(n+1)
  Stage r+1: FFNSubLayer(n+1) -> TransformerLayer(n+2) ...
"""

from typing import Optional

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayerSubmodules,
    _run_attention,
    _run_mlp,
)


class AttentionSubLayer(MegatronModule, BaseTransformerLayer):
    """Attention half of a TransformerLayer (input_layernorm through pre_mlp_layernorm).

    Forward returns one packed boundary tensor and context.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        global_layer_number: Optional[int] = None,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules
        # Use the explicit global index if given, otherwise derive from layer_number+offset.
        if global_layer_number is not None:
            self.layer_number = global_layer_number
        else:
            from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            self.layer_number = layer_number + get_transformer_layer_offset(config)

        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.config = config
        self.bias_dropout_add_exec_handler = torch.enable_grad
        self.current_microbatch = -1

        # Build only attention-side submodules.
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config, layer_number=self.layer_number,
            **attention_optional_kwargs,
        )
        self.self_attn_bda = build_module(submodules.self_attn_bda)
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config, layer_number=self.layer_number,
        )
        self.cross_attn_bda = build_module(submodules.cross_attn_bda)
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Recompute flags: pre_mlp layernorm recompute is DISABLED because its
        # checkpoint hook fires inside _run_mlp (on the FFN side), which is on a
        # different PP stage when the layer is split.
        self.recompute_input_layernorm = getattr(
            config, 'recompute_input_layernorm', False
        ) and (config.recompute_granularity == 'selective' and 'input_layernorm' in config.recompute_modules)
        self.recompute_pre_mlp_layernorm = False  # disabled for split layers
        self.recompute_mlp = False  # not owned

    def forward(
        self, hidden_states, attention_mask=None, context=None, context_mask=None,
        rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
        attention_bias=None, inference_context=None, packed_seq_params=None,
        sequence_len_offset=None, *, inference_params=None,
    ):
        """Run attention and encode the FFN inputs as one boundary tensor."""
        from megatron.core.utils import deprecate_inference_params
        inference_context = deprecate_inference_params(inference_context, inference_params)
        pre_mlp_layernorm_output, residual, context = _run_attention(
            self, hidden_states,
            attention_mask=attention_mask, context=context, context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb, rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin, attention_bias=attention_bias,
            inference_context=inference_context, packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return torch.stack((pre_mlp_layernorm_output, residual), dim=0), context


class FFNSubLayer(MegatronModule, BaseTransformerLayer):
    """FFN half of a TransformerLayer (pre_mlp_layernorm -> mlp -> mlp_bda).

    Forward decodes the single packed tensor received from the attention half.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        global_layer_number: Optional[int] = None,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules
        if global_layer_number is not None:
            self.layer_number = global_layer_number
        else:
            from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            self.layer_number = layer_number + get_transformer_layer_offset(config)

        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.config = config
        self.bias_dropout_add_exec_handler = torch.enable_grad
        self.current_microbatch = -1

        # Build only MLP-side submodules.  MLP does not accept layer_number.
        self.mlp = build_module(
            submodules.mlp, config=self.config,
        )
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)
        self.mlp_bda = build_module(submodules.mlp_bda)

        # Recompute flags.
        self.recompute_mlp = getattr(
            config, 'recompute_mlp', False
        ) and (config.recompute_granularity == 'selective' and 'mlp' in config.recompute_modules)
        self.recompute_input_layernorm = False  # not owned
        self.recompute_pre_mlp_layernorm = False  # disabled for split layers

    def forward(self, hidden_states, context=None, **kwargs):
        """Decode one boundary tensor and run the MLP half."""
        if hidden_states.dim() != 4 or hidden_states.size(0) != 2:
            raise RuntimeError(
                'FFNSubLayer expects a packed [2, sequence, batch, hidden] tensor, '
                f'got shape {tuple(hidden_states.shape)}'
            )
        pre_mlp_layernorm_output, residual = hidden_states.unbind(dim=0)
        return _run_mlp(self, pre_mlp_layernorm_output, residual), None
