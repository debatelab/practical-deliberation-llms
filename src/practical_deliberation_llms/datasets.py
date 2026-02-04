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
from typing import Any, Dict, List

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

    # Default configuration for this adapter instance. Subclasses should
    # override this with adapter-specific defaults (e.g. HF config, split),
    # and interpret the merged ``adapter_kwargs`` in ``load``.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {}

    def __init__(self, dataset_name: str, **adapter_kwargs: Any):
        self.dataset_name = dataset_name

        # Merge class-level defaults with user-provided overrides from YAML
        # or other configuration sources.
        merged: Dict[str, Any] = dict(self.DEFAULT_ADAPTER_KWARGS)
        merged.update(adapter_kwargs or {})
        self.adapter_kwargs = merged

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

    Semantics of identifiers used in experiments
    -------------------------------------------

    - `problem_uid` identifies this *specific* problem instance as it appears
      in an experiment run. For base problems this typically encodes the
      dataset and source row (e.g. ``"daily_dilemmas::42"``).
    - `base_problem_uid` (if present in `metadata`) points to the
      *underlying original* dilemma from which a problem instance was
      derived. For base problems this is often absent/``None``; for
      transformed variants it is expected to equal the base problem's
      `problem_uid` so that metrics can group variants with their baseline.
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

    # Default kwargs passed to ``datasets.load_dataset`` for this adapter.
    # Callers may override these via ``adapter_kwargs`` in the experiment
    # configuration.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "name": "Dilemmas_with_values_aggregated",
        "split": "test",
    }

    def load(self) -> pd.DataFrame:
        """Load `DailyDilemmas` in its aggregated (long) layout.

        The `Dilemmas_with_values_aggregated` configuration provides multiple
        rows per dilemma index (`dilemma_idx`). This adapter groups rows by
        `dilemma_idx`, selects a representative dilemma description, and
        collapses the available actions into exactly two options per dilemma.
        """

        ds = load_dataset(self.HF_DATASET_ID, **self.adapter_kwargs)
        df = ds.to_pandas()

        cols = list(df.columns)

        # Expected columns (as of writing):
        #   ['idx', 'dilemma_idx', 'basic_situation', 'dilemma_situation',
        #    'action_type', 'action', 'negative_consequence',
        #    'values_aggregated', 'topic', 'topic_group']

        if "dilemma_idx" not in cols:
            raise KeyError(
                f"Expected 'dilemma_idx' in DailyDilemmas columns, got {cols}"
            )

        # Prefer the more detailed dilemma_situation text if present.
        if "dilemma_situation" in cols:
            situation_col = "dilemma_situation"
        elif "basic_situation" in cols:
            situation_col = "basic_situation"
        else:
            raise KeyError(
                "None of ['dilemma_situation', 'basic_situation'] found in "
                f"dataset columns {cols}"
            )

        if "action" not in cols:
            raise KeyError(f"Expected 'action' column in DailyDilemmas, got {cols}")

        records = []
        # Group rows by dilemma index; each group corresponds to a single
        # decision situation with multiple candidate actions.
        grouped = df.groupby("dilemma_idx", dropna=False)

        for dilemma_idx, group in grouped:
            # Use the first non-null situation text in the group.
            situation_series = group[situation_col].dropna()
            if situation_series.empty:
                # Skip groups without any usable text.
                continue
            decision_situation = str(situation_series.iloc[0])

            # Collect unique action texts in a stable order.
            actions_raw = group["action"].astype(str).tolist()
            seen = set()
            actions: list[str] = []
            for a in actions_raw:
                if a not in seen:
                    seen.add(a)
                    actions.append(a)

            # Require at least two actions to form a dilemma.
            if len(actions) < 2:
                continue

            # For now, take the first two distinct actions. If the
            # dataset evolves to include more than two per dilemma, we
            # can revisit this and sample / rank them.
            actions = actions[:2]

            first = group.iloc[0]
            source_id = first.get("dilemma_idx", dilemma_idx)

            metadata = {
                "source_dataset": self.dataset_name,
                "source_id": source_id,
                "hf_dataset_id": self.HF_DATASET_ID,
                "hf_config": self.adapter_kwargs.get("name"),
                "hf_split": self.adapter_kwargs.get("split"),
                "topic": first.get("topic"),
                "topic_group": first.get("topic_group"),
                "values_aggregated": first.get("values_aggregated"),
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


ADAPTER_REGISTRY: Dict[str, type[DatasetAdapter]] = {
    "daily_dilemmas": DailyDilemmasAdapter,
    "AITA": AITAAdapter,
    "RoleConflictBench": RoleConflictBenchAdapter,
    "AIRiskDilemmas": AIRiskDilemmasAdapter,
    # "MSMDilemmas": MSMDilemmasAdapter,  # TODO if needed
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


def sample_problems(
    dataset_name: str,
    n_problems: int,
    seed: int,
    adapter_kwargs: Dict[str, Any] | None = None,
) -> List[PracticalProblem]:
    """Sample a fixed-size subset of problems from the given dataset.

    Uses a reproducible random seed and converts rows into `PracticalProblem`
    instances via `make_problem_from_row`.
    """

    df = load_normalized_dataset(dataset_name, adapter_kwargs=adapter_kwargs or {})
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
