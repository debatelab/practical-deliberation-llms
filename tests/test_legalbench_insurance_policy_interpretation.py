from __future__ import annotations

import pandas as pd

from practical_deliberation_llms.datasets.legalbench_insurance_policy_interpretation import (
    LegalBenchInsurancePolicyInterpretationAdapter,
)


def test_use_likert_default_trims_instruction_and_uses_likert():
    df = pd.DataFrame(
        [
            {
                "index": "row-001",
                "inputs": "Read the insurance policy and the insurance claim. Answer whether the claim is covered under the policy with [A: Yes; B: No; C: It’s ambiguous] SOME CASE Answer by only outputting the letter A, B or C.",
                "answer": "A",
            }
        ]
    )

    adapter = LegalBenchInsurancePolicyInterpretationAdapter(
        dataset_name="legalbench_insurance_policy_interpretation"
    )
    out = adapter.normalize_dataframe(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert (
        row["decision_situation"]
        == LegalBenchInsurancePolicyInterpretationAdapter.GENERIC_HEADER
        + "\n\nSOME CASE"
    )
    assert (
        row["actions"]
        == LegalBenchInsurancePolicyInterpretationAdapter.CANONICAL_LIKERT
    )
    assert row["metadata"]["gt_raw"] == "A"


def test_use_likert_false_produces_binary_choice():
    df = pd.DataFrame(
        [
            {
                "index": "row-002",
                "inputs": "Read the insurance policy and the insurance claim. Answer whether the claim is covered under the policy with [A: Yes; B: No; C: It’s ambiguous] SOME CASE Answer by only outputting the letter A, B or C.",
                "answer": "B",
            }
        ]
    )

    adapter = LegalBenchInsurancePolicyInterpretationAdapter(
        dataset_name="legalbench_insurance_policy_interpretation",
        use_likert=False,
    )
    out = adapter.normalize_dataframe(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert (
        row["decision_situation"]
        == LegalBenchInsurancePolicyInterpretationAdapter.GENERIC_HEADER
        + "\n\nSOME CASE"
    )
    assert row["actions"] == ["YES", "NO"]
    assert row["metadata"]["gt_raw"] == "B"


def test_strict_template_skips_nonmatching_rows():
    df = pd.DataFrame(
        [
            {
                "index": "row-003",
                "inputs": "This input does not follow the expected template.",
                "answer": "B",
            }
        ]
    )

    adapter = LegalBenchInsurancePolicyInterpretationAdapter(
        dataset_name="legalbench_insurance_policy_interpretation"
    )
    out = adapter.normalize_dataframe(df)

    assert len(out) == 0
