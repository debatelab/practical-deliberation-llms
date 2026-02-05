from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

from .base import DatasetAdapter


class RoleConflictBenchAdapter(DatasetAdapter):
    """Adapter for the `DebateLabKIT/role-conflict-bench` dataset.

    This adapter loads the benchmark directly from Hugging Face Datasets
    and normalizes each row into a single practical decision problem with
    two actions corresponding to the competing role expectations.
    """

    HF_DATASET_ID = "DebateLabKIT/role-conflict-bench"

    # Default kwargs passed through to ``datasets.load_dataset``. Callers
    # may override these via ``adapter_kwargs`` (e.g. a different split),
    # but for now the dataset exposes a single ``default`` config with a
    # ``train`` split.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "name": "default",
        "split": "train",
    }

    def load(self) -> pd.DataFrame:
        ds = load_dataset(self.HF_DATASET_ID, **self.adapter_kwargs)
        df = ds.to_pandas()

        # Expected columns from the dataset card / viewer.
        required_cols = {
            "Story",
            "Role1",
            "Expectation1",
            "Situation1",
            "Role2",
            "Expectation2",
            "Situation2",
            "key",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(
                "RoleConflictBenchAdapter expected columns %s, missing: %s; "
                "columns present: %s"
                % (
                    sorted(required_cols),
                    sorted(missing),
                    list(df.columns),
                )
            )

        records: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            story = str(row["Story"]).strip()
            if not story:
                continue

            # Minimal mapping as per PLAN_DATASETS_INTEGRATION.md: use the
            # narrative story as the decision situation, and the two role
            # expectations as candidate actions.
            decision_situation = story

            exp1 = str(row["Expectation1"]).strip()
            exp2 = str(row["Expectation2"]).strip()
            if not exp1 or not exp2:
                # Require two non-empty expectations to form a dilemma.
                continue

            actions = [exp1, exp2]

            key = str(row["key"]).strip()

            metadata: Dict[str, Any] = {
                "source_dataset": self.dataset_name,
                "source_id": key,
                "hf_dataset_id": self.HF_DATASET_ID,
                "hf_config": self.adapter_kwargs.get("name"),
                "hf_split": self.adapter_kwargs.get("split"),
                # Role / situation context preserved for downstream analysis.
                "Role1": row.get("Role1"),
                "Role2": row.get("Role2"),
                "Expectation_No1": row.get("Expectation_No1"),
                "Expectation_No2": row.get("Expectation_No2"),
                "Obligation1": row.get("Obligation1"),
                "Obligation2": row.get("Obligation2"),
                "Situation1": row.get("Situation1"),
                "Situation2": row.get("Situation2"),
                "key": key,
            }

            records.append(
                {
                    "decision_situation": decision_situation,
                    "actions": actions,
                    "metadata": metadata,
                    "problem_uid": f"role_conflict::{key}",
                }
            )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df
