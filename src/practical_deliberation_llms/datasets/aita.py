from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

from .base import DatasetAdapter


class AITAAdapter(DatasetAdapter):
    """Adapter for Reddit AITA-style datasets.

    Uses the `derek-thomas/dataset-creator-reddit-amitheasshole` dataset
    from Hugging Face Datasets as the canonical source.

    Each row is normalized into a single `decision_situation` plus a fixed
    set of four judgment options as `actions`:

    - OP is in the wrong (YTA)
    - OP is not in the wrong (NTA)
    - Everyone behaved badly (ESH)
    - No one behaved badly (NAH)
    """

    HF_DATASET_ID = "derek-thomas/dataset-creator-reddit-amitheasshole"

    # Default kwargs passed through to ``datasets.load_dataset``. Callers
    # can override these via ``adapter_kwargs`` (e.g. split, filters).
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {
        "split": "train",
    }

    def load(self) -> pd.DataFrame:
        ds = load_dataset(self.HF_DATASET_ID, **self.adapter_kwargs)
        df = ds.to_pandas()

        # Basic schema sanity check; the exact set of additional columns may
        # evolve, but we rely on at least ``title`` and ``content``.
        missing = [col for col in ("title", "content") if col not in df.columns]
        if missing:
            raise KeyError(
                "AITAAdapter expected columns 'title' and 'content', missing: "
                f"{missing}; columns present: {list(df.columns)}"
            )

        records = []
        for _, row in df.iterrows():
            # Construct decision situation as requested.
            title = str(row["title"]).strip()
            content = str(row["content"]).strip()
            decision_situation = (
                "You are asked to respond to the following message posted "
                f"under the title '{title}': '''{content}'''"
            )

            actions = [
                "You judge that the original poster is the one in the wrong (YTA).",
                "You judge that the original poster is not the one in the wrong (NTA).",
                "You judge that everyone involved behaved badly (ESH).",
                "You judge that no one behaved badly (NAH).",
            ]

            source_id = row.get("id") or row.get("_id") or row.name

            metadata = {
                "source_dataset": self.dataset_name,
                "source_id": source_id,
                "hf_dataset_id": self.HF_DATASET_ID,
                "hf_split": self.adapter_kwargs.get("split"),
                "permalink": row.get("permalink"),
                "score": row.get("score"),
                "date_utc": row.get("date_utc"),
                # No ground-truth "true_verdict" available in this dataset.
            }

            records.append(
                {
                    "decision_situation": decision_situation,
                    "actions": actions,
                    "metadata": metadata,
                    "problem_uid": f"aita::{source_id}",
                }
            )

        norm_df = pd.DataFrame.from_records(records)
        return norm_df
