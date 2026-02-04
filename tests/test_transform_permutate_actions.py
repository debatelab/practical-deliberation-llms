import types

import pytest

from experiments.re_sampling_stability.transform import permutate_actions
from practical_deliberation_llms.model import PracticalProblem


class SimpleConfig(types.SimpleNamespace):
    """Lightweight stand-in for ExperimentConfig with only needed field."""

    max_transformations_per_problem: int


@pytest.mark.asyncio
async def test_permutate_actions_respects_max_and_includes_original():
    # Problem with three actions so that multiple distinct permutations exist.
    problem = PracticalProblem(
        decision_situation="test situation",
        actions=["A", "B", "C"],
    )

    config = SimpleConfig(max_transformations_per_problem=2)

    variants = await permutate_actions(config, problem)

    # Original problem is always first and returned unchanged.
    assert variants[0] is problem
    assert variants[0].actions == ["A", "B", "C"]

    # We requested at most 2 transformed variants.
    assert 1 <= len(variants) <= 3
    assert len(variants) - 1 <= 2

    # All transformed variants must have permuted (not identical) action order.
    original_actions = variants[0].actions
    transformed_orders = [v.actions for v in variants[1:]]
    for acts in transformed_orders:
        assert acts != original_actions

    # Within a single call, all transformed action orders should be distinct.
    assert len({tuple(acts) for acts in transformed_orders}) == len(transformed_orders)
