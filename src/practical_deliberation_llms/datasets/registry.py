from __future__ import annotations

from typing import Any, Dict, Type

import pandas as pd

from .airisk_dilemmas import AIRiskDilemmasAdapter
from .aita import AITAAdapter
from .base import DatasetAdapter
from .daily_dilemmas import DailyDilemmasAdapter
from .legalbench_corporate_lobbying import LegalBenchCorporateLobbyingAdapter
from .legalbench_insurance_policy_interpretation import (
    LegalBenchInsurancePolicyInterpretationAdapter,
)
from .multi_step_moral_dilemmas import MultiStepMoralDilemmasAdapter
from .role_conflict_bench import RoleConflictBenchAdapter

ADAPTER_REGISTRY: Dict[str, Type[DatasetAdapter]] = {
    "daily_dilemmas": DailyDilemmasAdapter,
    "AITA": AITAAdapter,
    "RoleConflictBench": RoleConflictBenchAdapter,
    "AIRiskDilemmas": AIRiskDilemmasAdapter,
    "MSMDilemmas": MultiStepMoralDilemmasAdapter,
    "legalbench_corporate_lobbying": LegalBenchCorporateLobbyingAdapter,
    "legalbench_insurance_policy_interpretation": LegalBenchInsurancePolicyInterpretationAdapter,
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
