from __future__ import annotations

from typing import Any, Dict

import hashlib

import pandas as pd
from datasets import load_dataset

from .base import DatasetAdapter


class AIRiskDilemmasAdapter(DatasetAdapter):
    """Adapter for the `kellycyy/AIRiskDilemmas` dataset.

    This adapter groups rows by ``dilemma`` so that each resulting record
    corresponds to a single multi-action decision problem. For each dilemma we
    construct:

    - ``decision_situation``: the shared dilemma text.
    - ``actions``: the list of action texts for that dilemma.
    - ``metadata``: source/HF bookkeeping plus per-action values/targets.
    - ``problem_uid``: a stable identifier derived from the subset name and
      a deterministic source id.
    """

    HF_DATASET_ID = "kellycyy/AIRiskDilemmas"

    # Default adapter kwargs. ``subset`` is mapped to the HF configuration
    # name, and ``split`` is passed through directly.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "subset": "model_eval",  # HF config name
        "split": "test",  # split for the model_eval subset
    }

    def load(self) -> pd.DataFrame:
        subset = str(self.adapter_kwargs.get("subset", "model_eval"))
        split = str(self.adapter_kwargs.get("split", "test"))

        # Map our adapter-level ``subset`` concept to the HF ``name``
        # argument. This keeps YAML configs stable even if we later add
        # additional adapter-level options.
        ds = load_dataset(self.HF_DATASET_ID, name=subset, split=split)
        df = ds.to_pandas()

        # Basic schema sanity check. We expect the model_eval/full subsets to
        # expose at least these columns.
        required_cols = {"dilemma", "action", "values", "targets"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(
                "AIRiskDilemmasAdapter expected columns %s, missing: %s; "
                "columns present: %s"
                % (
                    sorted(required_cols),
                    sorted(missing),
                    list(df.columns),
                )
            )

        records: list[dict[str, Any]] = []

        # Group rows by dilemma text; each group corresponds to one decision
        # situation with multiple candidate actions.
        grouped = df.groupby("dilemma", dropna=False)

        for dilemma_text, group in grouped:
            dilemma_str = str(dilemma_text).strip()
            if not dilemma_str:
                # Skip groups without a usable dilemma description.
                continue

            # Collect action texts in a stable order, de-duplicated.
            actions_raw = group["action"].astype(str).tolist()
            seen_actions: set[str] = set()
            actions: list[str] = []
            for a in actions_raw:
                a_stripped = a.strip()
                if not a_stripped:
                    continue
                if a_stripped not in seen_actions:
                    seen_actions.add(a_stripped)
                    actions.append(a_stripped)

            # Require at least two actions to form a dilemma.
            if len(actions) < 2:
                continue

            # Per-action value and target annotations; we preserve the order so
            # that downstream analysis can align them with ``actions``.
            values_per_action = group["values"].tolist()
            targets_per_action = group["targets"].tolist()

            # Derive a deterministic source_id. Prefer any explicit id-like
            # column if present; otherwise fall back to a stable hash of the
            # dilemma text.
            first_row = group.iloc[0]
            source_id = (
                first_row.get("id")
                or first_row.get("idx")
                or hashlib.sha1(dilemma_str.encode("utf-8")).hexdigest()[:12]
            )

            metadata: Dict[str, Any] = {
                "source_dataset": self.dataset_name,
                "source_id": source_id,
                "hf_dataset_id": self.HF_DATASET_ID,
                "hf_subset": subset,
                "hf_split": split,
                "num_actions": len(actions),
                "values_per_action": values_per_action,
                "targets_per_action": targets_per_action,
            }

            records.append(
                {
                    "decision_situation": dilemma_str,
                    "actions": actions,
                    "metadata": metadata,
                    "problem_uid": f"airisk::{subset}::{source_id}",
                }
            )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df
