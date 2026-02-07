from __future__ import annotations

import pandas as pd

from practical_deliberation_llms.datasets.legalbench_corporate_lobbying import (
    LegalBenchCorporateLobbyingAdapter,
)


def test_use_ambiguous_default_trims_instruction_and_uses_ambiguous():
    # Inputs end with the exact Yes/No instruction sentence -> should be removed
    df = pd.DataFrame(
        [
            {
                "index": "row-001",
                "inputs": "Answer (by saying YES or NO; note the all-caps). A short scenario about lobbying. Answer by only replying to Yes or No.",
                "answer": "Yes",
            }
        ]
    )

    adapter = LegalBenchCorporateLobbyingAdapter(
        dataset_name="legalbench_corporate_lobbying"
    )
    out = adapter.normalize_dataframe(df)

    assert len(out) == 1
    row = out.iloc[0]
    # The trailing instruction is removed
    assert row["decision_situation"] == "Answer. A short scenario about lobbying."
    # Default behavior (current code) uses the canonical Likert scale
    assert row["actions"] == LegalBenchCorporateLobbyingAdapter.CANONICAL_AMBIGUOUS
    # Raw GT preserved in metadata
    assert row["metadata"]["gt_raw"] == "Yes"


def test_use_ambiguous_false_produces_binary_choice():
    # Note: the adapter currently reads the adapter keyword 'use_ambiguous' (typo)
    # to determine binary vs likert behavior. Tests reflect the current logic.
    df = pd.DataFrame(
        [
            {
                "index": "row-002",
                "inputs": "Answer (by saying YES or NO; note the all-caps). Policy scenario. Answer by only replying to Yes or No.",
                "answer": "No",
            }
        ]
    )

    # Pass the actual key the adapter reads ('use_ambiguous') to disable Ambiguity option
    adapter = LegalBenchCorporateLobbyingAdapter(
        dataset_name="legalbench_corporate_lobbying",
        use_ambiguous=False,
    )
    out = adapter.normalize_dataframe(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["decision_situation"] == "Answer. Policy scenario."
    assert row["actions"] == ["YES", "NO"]
    assert row["metadata"]["gt_raw"] == "No"


def test_missing_yes_no_instruction_skips_row():
    # If the inputs do NOT end with the exact instruction sentence, the parser
    # marks the decision as None and the row is skipped by normalize_dataframe.
    df = pd.DataFrame(
        [
            {
                "index": "row-003",
                "inputs": "Answer (by saying YES or NO; note the all-caps). A scenario that does not include the required trailing sentence.",
                "answer": "Yes",
            }
        ]
    )

    adapter = LegalBenchCorporateLobbyingAdapter(
        dataset_name="legalbench_corporate_lobbying"
    )
    out = adapter.normalize_dataframe(df)

    # Row should be skipped because the parser expects the exact trailing
    # sentence 'Answer by only replying to Yes or No.'
    assert len(out) == 0
