from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

from .base import DatasetAdapter


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
