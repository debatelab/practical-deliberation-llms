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

    This simplified adapter only accepts inputs that match the strict
    three-or-more-paragraph template described in the project notes.
    If a row does not match the expected template it is skipped.

    For accepted rows the adapter drops the first and last paragraph and
    prepends a short generic instruction. Actions are never parsed from
    the inputs; the adapter always uses the canonical Likert (default)
    or a binary YES/NO set when `use_likert=False`.
    """

    HF_DATASET_ID = "arcee-ai/legalbench_tasks"

    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "name": "insurance_policy_interpretation",
        "split": "test",
        # Interpret numeric answers as 0-based indices by default.
        "answer_numeric_base": 0,
        # Whether to use Likert scale or binary choice.
        "use_likert": True,
    }

    CANONICAL_LIKERT = [
        "Surely YES (claim covered)",
        "Rather YES (claim rather covered)",
        "Unclear (it's ambiguous)",
        "Rather NO (claim rather not covered)",
        "Surely NO (claim not covered)",
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

                use_likert = self.adapter_kwargs.get("use_likert", True)

                if use_likert:
                    actions = list(self.CANONICAL_LIKERT)
                else:
                    actions = ["YES", "NO"]

                metadata: Dict[str, Any] = {
                    "source_dataset": "legalbench_insurance_policy_interpretation",
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
                metadata = {
                    "source_dataset": "legalbench_insurance_policy_interpretation",
                    "source_id": row.get("index") or row.get("id") or str(idx),
                    "hf_dataset_id": self.HF_DATASET_ID,
                    "parsing_error": str(exc),
                }
                records.append(
                    {
                        "decision_situation": str(row.get("inputs") or ""),
                        "actions": list(self.CANONICAL_LIKERT),
                        "metadata": metadata,
                        "problem_uid": f"legalbench_insurance_policy_interpretation::{metadata['source_id']}",
                    }
                )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df

    def parse_inputs(self, inputs: str) -> str | None:
        """Parse only rows that match the strict template.

        Expected form (paragraphs separated by blank lines):
        - first paragraph starts with 'Read the insurance policy and the insurance claim.'
        - middle paragraphs describe policy/claim (one or more paragraphs)
        - last paragraph is an instruction like 'Answer by only outputting the letter A, B or C.'

        If the input does not match this structure, return (None)
        so the row will be skipped.

        For matching rows return the cleaned decision_situation (generic header + middle
        content), no instruction, and an empty parsed_actions list.
        """

        text = inputs.strip("\n ")
        if not text:
            return None

        expected_prefix = "Read the insurance policy and the insurance claim. Answer whether the claim is covered under the policy with [A: Yes; B: No; C: It’s ambiguous]"
        expected_suffix = "Answer by only outputting the letter A, B or C."

        if not text.startswith(expected_prefix):
            logger.warning(
                f"Inputs '{text[:50]}...' does not start with {expected_prefix}"
            )
            return None
        if not text.endswith(expected_suffix):
            logger.warning(
                f"Inputs '{text[:50]}...' does not end with {expected_suffix}"
            )
            return None

        text = text[len(expected_prefix) : -len(expected_suffix)].strip("\n ")

        if not text:
            return None

        # Prepend generic header
        decision = f"{self.GENERIC_HEADER}\n\n{text}"

        return decision

    # removed helpers and mapping function: this adapter does not parse actions nor map answers
