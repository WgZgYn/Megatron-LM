# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline-stage partition planning for Transformer layers.

Megatron normally partitions a Transformer by full-layer indices. Half-layer
pipeline parallelism adds a second, finer coordinate system where each full
layer consists of two consecutive fragments: attention and FFN. Keeping the
conversion in this module prevents model construction and the pipeline
schedule from independently interpreting distribution arguments.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PipelineStagePartition:
    """Half-open half-layer interval assigned to one pipeline stage."""

    rank: int
    half_start: int
    half_end: int

    @property
    def num_half_layers(self) -> int:
        return self.half_end - self.half_start

    @property
    def starts_with_ffn(self) -> bool:
        return self.half_start % 2 == 1

    @property
    def ends_with_attention(self) -> bool:
        return self.half_end % 2 == 1

    @property
    def first_full_layer_offset(self) -> int:
        """Zero-based full-layer index touched by this stage."""
        return self.half_start // 2


def get_half_layer_distribution(config) -> Optional[List[int]]:
    """Return per-stage half-layer counts, or ``None`` for full-layer mode.

    ``decoder_num_layers_per_pipeline_stage`` always remains in full-layer
    units. ``decoder_num_half_layers_per_pipeline_stage`` is the only argument
    expressed in half-layer units.
    """

    if not config.split_all_layers:
        return None
    if config.decoder_num_half_layers_per_pipeline_stage is not None:
        return list(config.decoder_num_half_layers_per_pipeline_stage)
    if config.decoder_num_layers_per_pipeline_stage is not None:
        return [2 * count for count in config.decoder_num_layers_per_pipeline_stage]

    if config.num_layers % config.pipeline_model_parallel_size != 0:
        raise ValueError(
            'split_all_layers without an explicit distribution requires num_layers '
            'to be divisible by pipeline_model_parallel_size'
        )
    full_layers_per_stage = config.num_layers // config.pipeline_model_parallel_size
    return [2 * full_layers_per_stage] * config.pipeline_model_parallel_size


def build_pipeline_stage_partitions(config) -> Optional[List[PipelineStagePartition]]:
    """Build the canonical half-layer interval for every pipeline rank."""

    distribution = get_half_layer_distribution(config)
    if distribution is None:
        return None

    partitions = []
    half_start = 0
    for rank, count in enumerate(distribution):
        partitions.append(PipelineStagePartition(rank, half_start, half_start + count))
        half_start += count
    return partitions


def get_pipeline_stage_partition(config, rank: int) -> Optional[PipelineStagePartition]:
    partitions = build_pipeline_stage_partitions(config)
    return None if partitions is None else partitions[rank]


def get_cross_stage_split_layers(config) -> List[int]:
    """Return 1-based full-layer indices cut across adjacent stages."""

    partitions = build_pipeline_stage_partitions(config)
    if partitions is None:
        return list(config.pipeline_split_layers or [])
    return [part.half_end // 2 + 1 for part in partitions[:-1] if part.ends_with_attention]
