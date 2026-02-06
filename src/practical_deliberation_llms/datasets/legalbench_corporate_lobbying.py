from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset, load_dataset

from .base import DatasetAdapter

logger = logging.getLogger(__name__)


class LegalBenchCorporateLobbyingAdapter(DatasetAdapter):
    """Adapter for the `arcee-ai/legalbench_tasks` dataset (config `corporate_lobbying`).

    Mapping rules (summary):
    - `decision_situation` is derived from the `inputs` field by splitting off
      any question/prompt or options block.
    - `actions` come from HF `multiple_choice_targets` when available and
      non-empty, otherwise from choices parsed from `inputs`, otherwise a
      canonical 5-point Likert: ["Strongly oppose", "Oppose", "Neutral", "Support", "Strongly support"].
    - Conservative GT handling: attempt to map `answer` to an index only when
      unambiguous. Otherwise store raw GT in metadata and set parsing flags.
    """

    HF_DATASET_ID = "arcee-ai/legalbench_tasks"

    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "name": "corporate_lobbying",
        "split": "test",
        # Interpret numeric answers as 0-based indices by default. Can be
        # overridden via adapter_kwargs.
        "answer_numeric_base": 0,
        # Whether to use Likert scale or binary choice.
        "use_likert": True,
    }

    CANONICAL_LIKERT = [
        "Surely YES",
        "Rather YES",
        "Unclear",
        "Rather NO",
        "Surely NO",
    ]

    def load(self) -> pd.DataFrame:
        # Only pass HF-recognized arguments to `load_dataset` (e.g., name/config, split).
        # Adapter-specific options (like answer_numeric_base, use_likert) should not be
        # forwarded to the datasets builder as they can cause builder config errors.
        hf_kwargs = {}
        for key in ("name", "split"):  # supported config args for this dataset
            if key in self.adapter_kwargs:
                hf_kwargs[key] = self.adapter_kwargs[key]

        ds = load_dataset(self.HF_DATASET_ID, **hf_kwargs)
        assert isinstance(ds, Dataset)
        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame)
        return self.normalize_dataframe(df)

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a pandas DataFrame with HF-like columns into the
        canonical DataFrame expected by the system.

        This helper exists so tests can exercise normalization without
        contacting the Hugging Face hub.
        """
        records: List[Dict[str, Any]] = []

        # Minimal column presence check
        if "inputs" not in df.columns:
            raise KeyError(
                "LegalBenchCorporateLobbyingAdapter expected column 'inputs' in dataframe"
            )

        for idx, row in df.iterrows():
            try:
                source_id = row.get("index") or row.get("id") or str(idx)
                inputs = row.get("inputs")
                if not inputs:
                    inputs = ""

                # Preserve original HF fields in metadata
                orig_fields = {
                    "answer": row.get("answer"),
                    "index": row.get("index"),
                    "task_type": row.get("task_type"),
                    "task_name": row.get("task_name"),
                    "multiple_choice_targets": row.get("multiple_choice_targets"),
                }

                # Parse inputs
                decision_situation = self.parse_inputs(inputs)

                if decision_situation is None:
                    logger.warning(f"Skipping row with idx {row.get('index')}")
                    continue

                use_likert = self.adapter_kwargs.get("use_likert", True)

                if use_likert:
                    actions = list(self.CANONICAL_LIKERT)
                else:
                    actions = [
                        "YES",
                        "NO",
                    ]

                metadata: Dict[str, Any] = {
                    "source_dataset": "legalbench_corporate_lobbying",
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
                    "problem_uid": f"legalbench_corporate_lobbying::{source_id}",
                }

                records.append(record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error normalizing row %s: %s", idx, exc)
                metadata = {
                    "source_dataset": "legalbench_corporate_lobbying",
                    "source_id": row.get("index") or row.get("id") or str(idx),
                    "hf_dataset_id": self.HF_DATASET_ID,
                    "parsing_error": str(exc),
                }
                records.append(
                    {
                        "decision_situation": str(row.get("inputs") or ""),
                        "actions": list(self.CANONICAL_LIKERT),
                        "metadata": metadata,
                        "problem_uid": f"legalbench_corporate_lobbying::{metadata['source_id']}",
                    }
                )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df

    def parse_inputs(self, inputs: str) -> str | None:
        """Split `inputs` into (decision_situation, instruction, parsed_actions, parsing_flags).

        Simplified parser per request:
        1. Normalize newlines.
        2. If the raw text contains the substring "\n " (newline + space), split at the first occurrence and keep the left part.
        3. If the resulting text ends with the exact sentence
           "Answer by only replying to Yes or No.", remove that sentence and trim again.
        4. Return the remaining text as the decision_situation; no parsed actions or instruction.
        """

        text = inputs or ""
        if not text.strip():
            return None

        # Normalize newlines
        text = re.sub(r"\r\n?", "\n", text)

        # Step 1: strip
        text = text.strip("\n ")

        # Step 2: detect and remove the trailing yes/no instruction line
        yes_no_sentence = "Answer by only replying to Yes or No."
        if text.endswith(yes_no_sentence):
            text = text[: -len(yes_no_sentence)].strip()
        else:
            logger.warning(
                f"Inputs '{inputs[:50]}...{inputs[-50:]}' does not end with '{yes_no_sentence}'."
            )
            return None

        # Step 3: remove embedded instruction
        yes_no_instruction = " (by saying YES or NO; note the all-caps)"
        if yes_no_instruction in text:
            text = text.replace(yes_no_instruction, "")
        else:
            logger.warning(
                f"Inputs '{inputs[:50]}...' does not contain '{yes_no_instruction}'."
            )
            text = None

        # Use the remaining text as the decision situation
        return text
