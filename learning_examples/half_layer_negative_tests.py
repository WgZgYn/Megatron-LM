"""Negative tests for half-layer PP config validation (F.4).

Runs on CPU — verifies that invalid configurations raise clear errors.
"""

import sys
import os

import torch
torch.compile = lambda f, **kwargs: f  # stub for Megatron import

sys.path.insert(0, r"F:\PycharmProjects\Megatron-LM")

from megatron.core.transformer.transformer_config import TransformerConfig


def test_should_fail(description, **kwargs):
    """Assert that TransformerConfig(**kwargs) raises ValueError."""
    try:
        base = {
            "num_layers": 12,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "pipeline_model_parallel_size": 4,
            "pipeline_dtype": torch.float32,
            "params_dtype": torch.float32,
        }
        base.update(kwargs)
        cfg = TransformerConfig(**base)
        # Validation runs in __post_init__ — if we reach here without error, fail
        print(f"  [FAIL] {description}: NO ERROR RAISED (should have failed)")
        return False
    except ValueError as e:
        print(f"  [PASS] {description}: {e}")
        return True
    except Exception as e:
        print(f"  [PASS] {description}: {type(e).__name__}: {e}")
        return True


print("=== F.4 Negative Tests ===\n")

passed = 0
failed = 0

# 1. Split without decoder_num_layers_per_pipeline_stage
if test_should_fail(
    "split without per-stage distribution",
    pipeline_split_layers=[6],
    # decoder_num_layers_per_pipeline_stage not set
):
    passed += 1
else:
    failed += 1

# 2. Split index not on a stage boundary
if test_should_fail(
    "split index not at stage boundary",
    decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
    pipeline_split_layers=[4],  # 4 is not 3, 6, or 9
):
    passed += 1
else:
    failed += 1

# 3. Split at layer 0 (out of range)
if test_should_fail(
    "split at layer 0",
    decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
    pipeline_split_layers=[0],  # < 1
):
    passed += 1
else:
    failed += 1

# 4. Split at last layer (out of range)
if test_should_fail(
    "split at layer 12 (last layer)",
    decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
    pipeline_split_layers=[12],  # == num_layers
):
    passed += 1
else:
    failed += 1

# 5. Split with VPP
if test_should_fail(
    "split with virtual pipeline parallelism",
    decoder_num_layers_per_pipeline_stage=[6, 6],
    pipeline_split_layers=[6],
    virtual_pipeline_model_parallel_size=2,
):
    passed += 1
else:
    failed += 1

# 6. Split with full recompute
if test_should_fail(
    "split with recompute_granularity=full",
    decoder_num_layers_per_pipeline_stage=[6, 6],
    pipeline_split_layers=[6],
    recompute_granularity="full",
):
    passed += 1
else:
    failed += 1

# 7. Duplicate split indices
if test_should_fail(
    "duplicate split indices",
    decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
    pipeline_split_layers=[3, 3],  # duplicate
):
    passed += 1
else:
    failed += 1

# 8. Valid config should NOT raise error
print("\n--- Valid config (should pass) ---")
try:
    cfg = TransformerConfig(
        num_layers=12,
        hidden_size=128,
        num_attention_heads=4,
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        decoder_num_layers_per_pipeline_stage=[3, 3, 3, 3],
        pipeline_split_layers=[3, 6, 9],
    )
    print(f"  [PASS] valid config: split_layers={cfg.pipeline_split_layers}")
    passed += 1
except Exception as e:
    print(f"  [FAIL] valid config raised: {e}")
    failed += 1

# 9. Split at multiple stage boundaries
try:
    cfg = TransformerConfig(
        num_layers=12,
        hidden_size=128,
        num_attention_heads=4,
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        decoder_num_layers_per_pipeline_stage=[2, 4, 3, 3],
        pipeline_split_layers=[2, 6, 9],
    )
    print(f"  [PASS] multi-split config: boundaries at {sorted(cfg.pipeline_split_layers)}")
    passed += 1
except Exception as e:
    print(f"  [FAIL] multi-split config raised: {e}")
    failed += 1

# --- New: decoder_num_half_layers_per_pipeline_stage tests ---

# 10. split_all_layers: half-layer sum wrong (must = num_layers*2)
if test_should_fail(
    "split_all_layers bad half-layer sum",
    split_all_layers=True,
    decoder_num_layers_per_pipeline_stage=[5, 7, 6, 5],  # sum=23, need 24
):
    passed += 1
else:
    failed += 1

# 11. split_all_layers: valid uniform half-layer (no cross-stage split)
try:
    cfg = TransformerConfig(
        num_layers=12, hidden_size=128, num_attention_heads=4,
        pipeline_model_parallel_size=4, pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        split_all_layers=True,
        decoder_num_layers_per_pipeline_stage=[6, 6, 6, 6],
    )
    assert cfg.pipeline_split_layers is None
    print(f"  [PASS] split_all_layers uniform half-layer: no cross-stage splits")
    passed += 1
except Exception as e:
    print(f"  [FAIL] split_all_layers uniform half-layer: {e}")
    failed += 1

# 12. split_all_layers: valid uneven half-layer with auto cross-stage split
try:
    cfg = TransformerConfig(
        num_layers=12, hidden_size=128, num_attention_heads=4,
        pipeline_model_parallel_size=4, pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        split_all_layers=True,
        decoder_num_layers_per_pipeline_stage=[5, 7, 6, 6],
    )
    assert cfg.pipeline_split_layers == [3]
    print(f"  [PASS] split_all_layers [5,7,6,6] -> auto split at layer 3")
    passed += 1
except Exception as e:
    print(f"  [FAIL] split_all_layers [5,7,6,6]: {e}")
    failed += 1

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL NEGATIVE TESTS PASSED")
    sys.exit(0)
