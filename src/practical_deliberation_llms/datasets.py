"""Dataset adapters and sampling utilities for practical deliberation experiments.

This module collects the code that was previously embedded in the
`notebooks/draft_notebook.md` under section 4 (Dataset Adapters).

It provides:

- A base `DatasetAdapter` class that normalizes datasets to a common schema.
- A concrete `DailyDilemmasAdapter` for the `kellycyy/DailyDilemmas` dataset.
- Stubs for additional datasets (AITA, RoleConflictBench, AIRiskDilemmas).
- Helper functions to load normalized datasets and sample `PracticalProblem` instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset

from .model import PracticalProblem


class DatasetAdapter(ABC):
    """Base class for dataset adapters returning a normalized DataFrame.

    Adapters should implement `load` and return a `pandas.DataFrame` with
    at least the following columns:

    - `decision_situation`: str
    - `actions`: list[str]
    - `metadata`: dict
    - `problem_uid`: str (optional but recommended)
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the normalized dataset as a DataFrame.

        Expected columns:

        - `decision_situation`: str
        - `actions`: list[str]
        - `metadata`: dict
        - `problem_uid`: str (optional, will be set if missing)
        """


def make_problem_from_row(row: pd.Series) -> PracticalProblem:
    """Convert a normalized DataFrame row into a `PracticalProblem`.

    Expected row keys:

    - `decision_situation`: str
    - `actions`: list[str]
    - `metadata`: dict
    - `problem_uid`: str (optional, will be set deterministically if missing)

    The function attaches metadata fields to the returned `PracticalProblem`
    instance. These fields line up with the extended dataclass fields defined
    in `model.PracticalProblem`, but we use `setattr` to remain robust while
    the dataclass is being evolved.
    """

    decision_situation = row["decision_situation"]
    actions = row["actions"]
    metadata = row.get("metadata", {}) or {}

    # Create or reuse stable problem UID
    problem_uid = row.get("problem_uid")
    if problem_uid is None:
        problem_uid = f"{metadata.get('source_dataset', 'unknown')}::{metadata.get('source_id', row.name)}"

    problem = PracticalProblem(
        decision_situation=decision_situation,
        actions=list(actions),
    )

    # Attach metadata dynamically, so older code that does not yet declare
    # these fields continues to work.
    setattr(problem, "problem_uid", problem_uid)
    setattr(problem, "source_dataset", metadata.get("source_dataset"))
    setattr(problem, "source_id", metadata.get("source_id"))
    setattr(problem, "metadata", metadata)
    setattr(problem, "base_problem_uid", metadata.get("base_problem_uid"))
    setattr(problem, "transformation_type", metadata.get("transformation_type"))
    setattr(
        problem,
        "transformation_params",
        metadata.get("transformation_params", {}),
    )

    return problem


class DailyDilemmasAdapter(DatasetAdapter):
    """Adapter for the `kellycyy/DailyDilemmas` dataset.

    Uses the `Dilemmas_with_values_aggregated` configuration and collapses
    each dilemma into a single record with two actions.
    """

    HF_DATASET_ID = "kellycyy/DailyDilemmas"
    HF_CONFIG = "Dilemmas_with_values_aggregated"

    def load(self) -> pd.DataFrame:
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split="test")
        df = ds.to_pandas()

        # Assumptions about column names; adjust if needed after inspecting
        # the dataset in a notebook:
        # - "dilemma" is the situation text
        # - "action_1" / "action_2" are the two options
        # - "dilemma_id" or "id" is the unique identifier
        decision_col_candidates = ["dilemma", "dilemma_text", "scenario"]
        action1_candidates = ["action_1", "action1", "option_1"]
        action2_candidates = ["action_2", "action2", "option_2"]
        id_candidates = ["dilemma_id", "id", "index"]

        def pick_first(cols, candidates):
            for c in candidates:
                if c in cols:
                    return c
            raise KeyError(f"None of {candidates} found in dataset columns {cols}")

        cols = list(df.columns)
        decision_col = pick_first(cols, decision_col_candidates)
        action1_col = pick_first(cols, action1_candidates)
        action2_col = pick_first(cols, action2_candidates)
        id_col = pick_first(cols, id_candidates)

        records = []
        for _, row in df.iterrows():
            decision_situation = str(row[decision_col])
            actions = [str(row[action1_col]), str(row[action2_col])]
            source_id = row[id_col]

            metadata = {
                "source_dataset": "daily_dilemmas",
                "source_id": source_id,
                "hf_dataset_id": self.HF_DATASET_ID,
                "hf_config": self.HF_CONFIG,
            }

            records.append(
                {
                    "decision_situation": decision_situation,
                    "actions": actions,
                    "metadata": metadata,
                    "problem_uid": f"daily_dilemmas::{source_id}",
                }
            )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df


class AITAAdapter(DatasetAdapter):
    """Stub adapter for AITA-style datasets.

    To be implemented using, e.g., `derek-thomas/dataset-creator-reddit-amitheasshole`.
    """

    def load(self) -> pd.DataFrame:  # pragma: no cover - placeholder
        raise NotImplementedError("AITAAdapter is not implemented yet.")


class RoleConflictBenchAdapter(DatasetAdapter):
    """Stub adapter for RoleConflictBench.

    To be implemented once the dataset format is finalized.
    """

    def load(self) -> pd.DataFrame:  # pragma: no cover - placeholder
        raise NotImplementedError("RoleConflictBenchAdapter is not implemented yet.")


class AIRiskDilemmasAdapter(DatasetAdapter):
    """Stub adapter for `kellycyy/AIRiskDilemmas`."""

    def load(self) -> pd.DataFrame:  # pragma: no cover - placeholder
        raise NotImplementedError("AIRiskDilemmasAdapter is not implemented yet.")


ADAPTER_REGISTRY = {
    "daily_dilemmas": DailyDilemmasAdapter,
    "AITA": AITAAdapter,
    "RoleConflictBench": RoleConflictBenchAdapter,
    "AIRiskDilemmas": AIRiskDilemmasAdapter,
    # "MSMDilemmas": MSMDilemmasAdapter,  # TODO if needed
}


def load_normalized_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a dataset by name using the adapter registry.

    Returns a normalized DataFrame as defined by the corresponding adapter.
    """

    if dataset_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    adapter_cls = ADAPTER_REGISTRY[dataset_name]
    adapter = adapter_cls(dataset_name=dataset_name)
    df = adapter.load()
    return df


def sample_problems(
    dataset_name: str, n_problems: int, seed: int
) -> List[PracticalProblem]:
    """Sample a fixed-size subset of problems from the given dataset.

    Uses a reproducible random seed and converts rows into `PracticalProblem`
    instances via `make_problem_from_row`.
    """

    df = load_normalized_dataset(dataset_name)
    n_total = len(df)
    n = min(n_problems, n_total)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(n_total, size=n, replace=False)

    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

    problems: List[PracticalProblem] = []
    for _, row in sampled_df.iterrows():
        problem = make_problem_from_row(row)
        problems.append(problem)

    return problems
