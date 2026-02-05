from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

from .base import DatasetAdapter


class MultiStepMoralDilemmasAdapter(DatasetAdapter):
    """Adapter for `DebateLabKIT/multi-step-moral-dilemmas` (MMDs).

    Each HF row corresponds to a single multi-step scenario with five
    sequential decision steps. This adapter flattens each scenario into five
    normalized records, one per step, while accumulating prior step
    descriptions into the `decision_situation` text.

    The HF dataset exposes three configs:

    - ``default``: combined view of both value frameworks.
    - ``MFT``: scenarios annotated with Moral Foundations Theory values.
    - ``Schwartz``: scenarios annotated with Schwartz values.

    Callers can select the config via ``adapter_kwargs['framework']`` or
    ``adapter_kwargs['config_name']``; if neither is provided, ``"default"``
    is used.
    """

    HF_DATASET_ID = "DebateLabKIT/multi-step-moral-dilemmas"

    # Default kwargs passed to ``datasets.load_dataset``. The "framework"
    # key is an adapter-level concept that we map to the HF config name.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "framework": "default",  # one of {"default", "MFT", "Schwartz"}
        "split": "train",
    }

    def _resolve_hf_kwargs(self) -> tuple[Dict[str, Any], str]:
        """Return (kwargs, framework) for ``load_dataset`` based on settings."""

        framework = str(self.adapter_kwargs.get("framework", "default"))
        config_name = str(self.adapter_kwargs.get("config_name", framework))

        # Allow callers to override split if needed, but default to "train",
        # which is the only split currently exposed by the dataset card.
        split = str(self.adapter_kwargs.get("split", "train"))

        hf_kwargs: Dict[str, Any] = {"name": config_name, "split": split}
        return hf_kwargs, framework

    def load(self) -> pd.DataFrame:
        hf_kwargs, framework = self._resolve_hf_kwargs()

        ds = load_dataset(self.HF_DATASET_ID, **hf_kwargs)
        df = ds.to_pandas()

        # Basic schema sanity check for key columns. We tolerate additional
        # columns or future extensions.
        expected_base_cols = {"norm", "idx"}
        missing_base = expected_base_cols - set(df.columns)
        if missing_base:
            raise KeyError(
                "MultiStepMoralDilemmasAdapter expected columns 'norm' and 'idx', "
                f"missing: {sorted(missing_base)}; columns present: {list(df.columns)}"
            )

        records: list[dict[str, Any]] = []

        # Each row is a single scenario; we turn it into five problems, one
        # per step, with accumulated history in the situation text.
        for _, row in df.iterrows():
            norm_text = str(row["norm"]).strip()
            scenario_id = row["idx"]

            # Pre-extract step-wise situation texts to avoid repeated lookups.
            step_situations: list[str] = []
            for i in range(1, 6):
                col = f"step {i}_situation"
                step_situations.append(str(row.get(col, "")).strip())

            for step_idx in range(1, 6):
                # Build accumulated situation text up to this step.
                situations_so_far = [s for s in step_situations[:step_idx] if s]

                dilemma_col = f"step {step_idx}_dilemma"
                choiceA_col = f"step {step_idx}_choiceA"
                choiceB_col = f"step {step_idx}_choiceB"
                choiceA_val_col = f"step {step_idx}_choiceA_value"
                choiceB_val_col = f"step {step_idx}_choiceB_value"
                choiceA_val_j_col = f"step {step_idx}_choiceA_value_judgement"
                choiceB_val_j_col = f"step {step_idx}_choiceB_value_judgement"
                conflict_col = f"step {step_idx}_conflict"

                dilemma_text = str(row.get(dilemma_col, "")).strip()
                choiceA_text = str(row.get(choiceA_col, "")).strip()
                choiceB_text = str(row.get(choiceB_col, "")).strip()

                # Skip steps that do not contain a well-formed dilemma.
                if not dilemma_text or not choiceA_text or not choiceB_text:
                    continue

                # Construct the accumulated decision situation as discussed
                # in PLAN_DATASETS_INTEGRATION.md.
                lines: list[str] = []
                lines.append(f'Norm: "{norm_text}"')
                lines.append("")
                lines.append("Situation so far:")
                for idx, sit in enumerate(situations_so_far, start=1):
                    lines.append(f"- Step {idx}: {sit}")
                lines.append("")
                lines.append(f"Current dilemma (step {step_idx}):")
                lines.append(dilemma_text)

                decision_situation = "\n".join(lines)

                actions = [choiceA_text, choiceB_text]

                metadata: Dict[str, Any] = {
                    "source_dataset": self.dataset_name,
                    "source_id": scenario_id,
                    "hf_dataset_id": self.HF_DATASET_ID,
                    "hf_config": hf_kwargs.get("name"),
                    "hf_split": hf_kwargs.get("split"),
                    "norm": norm_text,
                    "framework": framework,
                    "scenario_id": scenario_id,
                    "stage_index": step_idx,
                    "step_conflict": row.get(conflict_col),
                    "values_per_action": [
                        row.get(choiceA_val_col),
                        row.get(choiceB_val_col),
                    ],
                    "value_judgement_per_action": [
                        row.get(choiceA_val_j_col),
                        row.get(choiceB_val_j_col),
                    ],
                    "situations_so_far": situations_so_far,
                }

                records.append(
                    {
                        "decision_situation": decision_situation,
                        "actions": actions,
                        "metadata": metadata,
                        "problem_uid": f"mmd::{framework}::{scenario_id}::step{step_idx}",
                    }
                )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df
