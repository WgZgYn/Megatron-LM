# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional, Union

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.pipeline_parallel.pipeline_partition import get_pipeline_stage_partition
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_sublayer import (
    AttentionSubLayer,
    FFNSubLayer,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TEColumnParallelLinear,
                        linear_q_up_proj=(
                            TELayerNormColumnParallelLinear
                            if qk_layernorm
                            else TEColumnParallelLinear
                        ),
                        linear_kv_down_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=(
                            TELayerNormColumnParallelLinear
                            if qk_layernorm
                            else TEColumnParallelLinear
                        ),
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        global LNImpl
        LNImpl = WrappedTorchNorm

    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=ColumnParallelLinear,
                        linear_q_down_proj=ColumnParallelLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ColumnParallelLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        kv_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (LNImpl if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (LNImpl if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                    'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                },
            ),
        )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    if use_transformer_engine:
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl

    # Layer specs.
    dense_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
        )
    )
    moe_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
        )
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    if config.split_all_layers:
        # ═══ Half-layer PP (all layers): produce [Attn, FFN] pair per layer ═══
        # Construction and P2P consume the same plan. A stage may start with
        # FFN or end with attention, so full-layer slicing is insufficient.
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        partition = get_pipeline_stage_partition(config, pipeline_rank)
        assert partition is not None
        stage_specs = []
        for half_layer_index in range(partition.half_start, partition.half_end):
            full_layer_index = half_layer_index // 2
            full_spec = layer_specs[full_layer_index]
            stage_specs.append(
                ModuleSpec(
                    module=(AttentionSubLayer if half_layer_index % 2 == 0 else FFNSubLayer),
                    params={"global_layer_number": full_layer_index + 1},
                    submodules=full_spec.submodules,
                )
            )
        layer_specs = stage_specs
        # ═══ END split-all-layers spec construction ═══
    elif config.pipeline_split_layers is not None:
        offset = get_transformer_layer_offset(config)
        num_layers_to_build = get_num_layers_to_build(config)
        # ═══ Half-layer PP (selective): per-stage specs with split layers ═══
        start_layer = offset + 1   # 1-based first layer index of this stage
        end_layer = offset + num_layers_to_build  # 1-based last layer index

        stage_specs = []
        # Left boundary: if THIS stage's FIRST layer is the FFN half of a split
        previous_layer = start_layer - 1
        if previous_layer in config.pipeline_split_layers:
            full_spec = layer_specs[previous_layer - 1]  # 0-indexed
            ffn_spec = ModuleSpec(
                module=FFNSubLayer,
                params={"global_layer_number": previous_layer},
                submodules=full_spec.submodules,
            )
            stage_specs.append(ffn_spec)

        # Full layers in the middle
        for g in range(start_layer, end_layer + 1):
            is_split_right = g in config.pipeline_split_layers
            if is_split_right:
                full_spec = layer_specs[g - 1]
                attn_spec = ModuleSpec(
                    module=AttentionSubLayer,
                    params={"global_layer_number": g},
                    submodules=full_spec.submodules,
                )
                stage_specs.append(attn_spec)
            else:
                stage_specs.append(layer_specs[g - 1])

        layer_specs = stage_specs
        # ═══ END selective-split spec construction ═══
    else:
        offset = get_transformer_layer_offset(config)
        num_layers_to_build = get_num_layers_to_build(config)
        layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm_impl)

    return block_spec


def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    use_transformer_engine: bool,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config)
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec(
        transformer_layer_spec=transformer_layer_spec, use_transformer_engine=use_transformer_engine
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert (
            len(mtp_layer_specs) == config.mtp_num_layers
        ), +f"currently all of the mtp layers must stage in the same pipeline stage."
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec
