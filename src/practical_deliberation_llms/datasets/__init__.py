"""Dataset adapters and sampling utilities package.

This package exposes the same public API that previously lived in the
monolithic ``practical_deliberation_llms.datasets`` module, but with a
more modular internal structure.
"""

from .base import DatasetAdapter, make_problem_from_row
from .registry import ADAPTER_REGISTRY, load_normalized_dataset
from .sampling import sample_problems
from .daily_dilemmas import DailyDilemmasAdapter
from .aita import AITAAdapter
from .role_conflict_bench import RoleConflictBenchAdapter
from .airisk_dilemmas import AIRiskDilemmasAdapter
from .multi_step_moral_dilemmas import MultiStepMoralDilemmasAdapter

__all__ = [
    "DatasetAdapter",
    "make_problem_from_row",
    "ADAPTER_REGISTRY",
    "load_normalized_dataset",
    "sample_problems",
    "DailyDilemmasAdapter",
    "AITAAdapter",
    "RoleConflictBenchAdapter",
    "AIRiskDilemmasAdapter",
    "MultiStepMoralDilemmasAdapter",
]
