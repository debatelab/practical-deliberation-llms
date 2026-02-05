from __future__ import annotations

from typing import Any, Dict, Type

import pandas as pd

from .base import DatasetAdapter
from .daily_dilemmas import DailyDilemmasAdapter
from .aita import AITAAdapter
from .role_conflict_bench import RoleConflictBenchAdapter
from .airisk_dilemmas import AIRiskDilemmasAdapter
from .multi_step_moral_dilemmas import MultiStepMoralDilemmasAdapter


ADAPTER_REGISTRY: Dict[str, Type[DatasetAdapter]] = {
    "daily_dilemmas": DailyDilemmasAdapter,
    "AITA": AITAAdapter,
    "RoleConflictBench": RoleConflictBenchAdapter,
    "AIRiskDilemmas": AIRiskDilemmasAdapter,
    "MSMDilemmas": MultiStepMoralDilemmasAdapter,
}


def load_normalized_dataset(
    dataset_name: str,
    adapter_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a dataset by name using the adapter registry.

    Returns a normalized DataFrame as defined by the corresponding adapter.
    """

    if dataset_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    adapter_cls = ADAPTER_REGISTRY[dataset_name]
    adapter = adapter_cls(dataset_name=dataset_name, **(adapter_kwargs or {}))
    df = adapter.load()
    return df
