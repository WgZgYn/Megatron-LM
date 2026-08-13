# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical pipeline partition plans for Transformer layers.

The planner uses half-layer coordinates internally: attention is the even
fragment and FFN is the odd fragment of a full Transformer layer. Consumers
must use the resulting plan instead of interpreting distribution arguments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class FragmentKind(str, Enum):
    FULL = "full"
    ATTENTION = "attention"
    FFN = "ffn"


class BoundaryKind(str, Enum):
    HIDDEN = "hidden"
    PACKED_ATTENTION = "packed_attention"


@dataclass(frozen=True)
class LayerFragment:
    layer_number: int
    kind: FragmentKind


@dataclass(frozen=True)
class PipelineStagePlan:
    """Resolved module sequence and communication contract for one PP rank."""

    rank: int
    half_start: int
    half_end: int
    fragments: Tuple[LayerFragment, ...]
    input_boundary: BoundaryKind
    output_boundary: BoundaryKind

    @property
    def num_half_layers(self) -> int:
        return self.half_end - self.half_start

    @property
    def starts_with_ffn(self) -> bool:
        return self.input_boundary is BoundaryKind.PACKED_ATTENTION

    @property
    def ends_with_attention(self) -> bool:
        return self.output_boundary is BoundaryKind.PACKED_ATTENTION

    @property
    def first_full_layer_offset(self) -> int:
        return self.half_start // 2


# Compatibility for existing external validation scripts.
PipelineStagePartition = PipelineStagePlan


@dataclass(frozen=True)
class PipelinePlan:
    """Complete partition plan shared by construction and schedules."""

    num_layers: int
    stages: Tuple[PipelineStagePlan, ...]

    def stage(self, rank: int) -> PipelineStagePlan:
        return self.stages[rank]


def _fragments_for_interval(half_start: int, half_end: int) -> Tuple[LayerFragment, ...]:
    """Collapse co-located attention/FFN pairs back into full layers."""

    fragments = []
    cursor = half_start
    while cursor < half_end:
        layer_number = cursor // 2 + 1
        if cursor % 2 == 0 and cursor + 1 < half_end:
            fragments.append(LayerFragment(layer_number, FragmentKind.FULL))
            cursor += 2
        else:
            kind = FragmentKind.ATTENTION if cursor % 2 == 0 else FragmentKind.FFN
            fragments.append(LayerFragment(layer_number, kind))
            cursor += 1
    return tuple(fragments)


def _half_layer_distribution(config) -> Optional[List[int]]:
    """Normalize supported user configurations to half-layer counts."""

    half_distribution = config.decoder_num_half_layers_per_pipeline_stage
    full_distribution = config.decoder_num_layers_per_pipeline_stage

    if half_distribution is not None:
        return list(half_distribution)

    if full_distribution is not None:
        boundaries = []
        cumulative_layers = 0
        split_layers = set(config.pipeline_split_layers or [])
        for count in full_distribution[:-1]:
            cumulative_layers += count
            boundary = 2 * cumulative_layers
            if cumulative_layers in split_layers:
                boundary -= 1
            boundaries.append(boundary)
        points = [0, *boundaries, 2 * config.num_layers]
        return [end - start for start, end in zip(points, points[1:])]

    return None


def build_pipeline_plan(config) -> Optional[PipelinePlan]:
    """Build the sole partition representation used after configuration."""

    distribution = _half_layer_distribution(config)
    if distribution is None:
        return None

    stages = []
    half_start = 0
    for rank, count in enumerate(distribution):
        half_end = half_start + count
        stages.append(
            PipelineStagePlan(
                rank=rank,
                half_start=half_start,
                half_end=half_end,
                fragments=_fragments_for_interval(half_start, half_end),
                input_boundary=(
                    BoundaryKind.PACKED_ATTENTION if half_start % 2 else BoundaryKind.HIDDEN
                ),
                output_boundary=(
                    BoundaryKind.PACKED_ATTENTION if half_end % 2 else BoundaryKind.HIDDEN
                ),
            )
        )
        half_start = half_end

    if half_start != 2 * config.num_layers:
        raise ValueError(
            f'pipeline plan covers {half_start} half-layers, expected {2 * config.num_layers}'
        )
    return PipelinePlan(config.num_layers, tuple(stages))


def build_pipeline_stage_partitions(config) -> Optional[List[PipelineStagePlan]]:
    plan = build_pipeline_plan(config)
    return None if plan is None else list(plan.stages)


def get_pipeline_stage_partition(config, rank: int) -> Optional[PipelineStagePlan]:
    plan = build_pipeline_plan(config)
    return None if plan is None else plan.stage(rank)


def get_cross_stage_split_layers(config) -> List[int]:
    plan = build_pipeline_plan(config)
    if plan is None:
        return []
    return [
        stage.half_end // 2 + 1
        for stage in plan.stages[:-1]
        if stage.ends_with_attention
    ]
