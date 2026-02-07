from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset, load_dataset

from .base import DatasetAdapter

logger = logging.getLogger(__name__)


class LegalBenchInsurancePolicyInterpretationAdapter(DatasetAdapter):
    """Adapter for the `arcee-ai/legalbench_tasks` dataset (config
    `insurance_policy_interpretation`).

    This simplified adapter only accepts inputs that match a strict
    instruction template. If a row does not match the expected template
    it is skipped.

    For accepted rows the adapter strips the leading and trailing
    instructions, prepends a short generic header, and treats the
    remaining text as the decision situation. Actions are never parsed
    from the inputs; the adapter always uses the canonical ambiguity
    scale (default) or a binary YES/NO set when ``use_ambiguous=False``.

    The HF ``answer`` field is not mapped to action indices; it is
    preserved verbatim in ``metadata["gt_raw"]``.
    """

    HF_DATASET_ID = "arcee-ai/legalbench_tasks"

    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "name": "insurance_policy_interpretation",
        "split": "test",
        # Whether to use ambiguity scale or binary choice.
        "use_ambiguous": True,
    }

    CANONICAL_AMBIGUOUS = [
        "YES",
        "NO",
        "It's ambiguous",
    ]

    GENERIC_HEADER = (
        "Read the insurance policy and the insurance claim. "
        "Decide whether the claim is covered under the policy."
    )

    def load(self) -> pd.DataFrame:
        hf_kwargs = {}
        for key in ("name", "split"):
            if key in self.adapter_kwargs:
                hf_kwargs[key] = self.adapter_kwargs[key]

        ds = load_dataset(self.HF_DATASET_ID, **hf_kwargs)
        assert isinstance(ds, Dataset)
        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame)
        return self.normalize_dataframe(df)

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []

        if "inputs" not in df.columns:
            raise KeyError(
                "LegalBenchInsurancePolicyInterpretationAdapter expected column 'inputs' in dataframe"
            )

        for idx, row in df.iterrows():
            try:
                source_id = row.get("index") or row.get("id") or str(idx)
                inputs = row.get("inputs")
                if not inputs:
                    inputs = ""

                orig_fields = {
                    "answer": row.get("answer"),
                    "index": row.get("index"),
                    "task_type": row.get("task_type"),
                    "task_name": row.get("task_name"),
                    "multiple_choice_targets": row.get("multiple_choice_targets"),
                }

                decision_situation = self.parse_inputs(inputs)

                if decision_situation is None:
                    logger.warning(
                        "Skipping row with idx %s: inputs did not match strict insurance template",
                        row.get("index"),
                    )
                    continue

                use_ambiguous = self.adapter_kwargs.get("use_ambiguous", True)

                if use_ambiguous:
                    actions = list(self.CANONICAL_AMBIGUOUS)
                else:
                    actions = ["YES", "NO"]

                metadata: Dict[str, Any] = {
                    "source_dataset": self.dataset_name,
                    "source_id": source_id,
                    "hf_dataset_id": self.HF_DATASET_ID,
                    "hf_config": self.adapter_kwargs.get("name"),
                    "hf_split": self.adapter_kwargs.get("split"),
                    "task_type": row.get("task_type"),
                    "task_name": row.get("task_name"),
                    "original_fields": orig_fields,
                    "gt_raw": row.get("answer"),
                }

                record: Dict[str, Any] = {
                    "decision_situation": decision_situation,
                    "actions": actions,
                    "metadata": metadata,
                    "problem_uid": f"legalbench_insurance_policy_interpretation::{source_id}",
                }

                records.append(record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error normalizing row %s: %s", idx, exc)

        norm_df = pd.DataFrame.from_records(records)
        return norm_df

    def parse_inputs(self, inputs: str) -> str | None:
        """Parse only rows that match the strict instruction template.

        We expect the text to start with a fixed instruction prefix and end
        with a fixed instruction suffix. If either is missing, or if there
        is no content between them, return ``None`` so the row will be
        skipped.

        For matching rows, return the cleaned decision_situation (generic
        header + middle content).
        """

        text = inputs.strip("\n ")
        if not text:
            return None

        # NOTE: The apostrophe in "Its ambiguous" matches the HF dataset text
        # and the existing tests; we intentionally keep this non-ASCII form
        # here so that the strict template check passes.
        expected_prefix = (
            "Read the insurance policy and the insurance claim. "
            "Answer whether the claim is covered under the policy with "
            "[A: Yes; B: No; C: It’s ambiguous]"
        )
        expected_suffix = "Answer by only outputting the letter A, B or C."

        if not text.startswith(expected_prefix):
            logger.warning(
                "Inputs did not start with expected insurance prefix; skipping row."
            )
            return None
        if not text.endswith(expected_suffix):
            logger.warning(
                "Inputs did not end with expected insurance suffix; skipping row."
            )
            return None

        text = text[len(expected_prefix) : -len(expected_suffix)].strip("\n ")

        if not text:
            return None

        # Prepend generic header
        decision = f"{self.GENERIC_HEADER}\n\n{text}"

        return decision

    # removed helpers and mapping function: this adapter does not parse actions nor map answers
