from practical_deliberation_llms.datasets import (
    ADAPTER_REGISTRY,
    load_normalized_dataset,
    sample_problems,
)
from practical_deliberation_llms.model import PracticalProblem


def _assert_problem_basic_shape(problem: PracticalProblem) -> None:
    """Common sanity checks for PracticalProblem instances."""

    assert isinstance(problem, PracticalProblem)
    assert isinstance(problem.decision_situation, str)
    assert problem.decision_situation.strip()

    assert isinstance(problem.actions, list)
    assert problem.actions, "actions list must not be empty"
    assert all(isinstance(a, str) and a.strip() for a in problem.actions)

    # Attributes set by make_problem_from_row
    assert getattr(problem, "problem_uid", None)
    assert getattr(problem, "source_dataset", None)
    assert isinstance(getattr(problem, "metadata", {}), dict)


def test_daily_dilemmas_adapter_full_integration() -> None:
    """Download DailyDilemmas from HF, normalize, and cast to PracticalProblems."""

    assert "daily_dilemmas" in ADAPTER_REGISTRY

    # Load normalized DataFrame
    df = load_normalized_dataset(
        "daily_dilemmas",
        adapter_kwargs={"split": "test"},
    )
    assert not df.empty
    for col in ("decision_situation", "actions", "metadata"):
        assert col in df.columns

    # Sample a few problems and ensure they look correct
    n = 3
    problems = sample_problems(
        dataset_name="daily_dilemmas",
        n_problems=n,
        seed=0,
        adapter_kwargs={"split": "test"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        # DailyDilemmas adapter always truncates to exactly two actions
        assert len(p.actions) == 2
        assert p.source_dataset == "daily_dilemmas"
        assert isinstance(p.problem_uid, str)
        assert p.problem_uid.startswith("daily_dilemmas::")


def test_aita_adapter_full_integration() -> None:
    """Download AITA HF dataset, normalize, and cast to PracticalProblems."""

    assert "AITA" in ADAPTER_REGISTRY

    # Load normalized DataFrame
    df = load_normalized_dataset(
        "AITA",
        adapter_kwargs={"split": "train"},
    )
    assert not df.empty
    for col in ("decision_situation", "actions", "metadata", "problem_uid"):
        assert col in df.columns

    # Basic checks on the normalized row schema
    first = df.iloc[0]
    assert isinstance(first["decision_situation"], str)
    assert isinstance(first["actions"], list)
    assert len(first["actions"]) == 4

    # Decision situation should follow the prompt-like template
    ds = first["decision_situation"]
    prefix = "You are asked to respond to the following message posted under the title "
    assert ds.startswith(prefix)
    # Rough sanity check for the embedded triple quotes
    assert "'''" in ds

    # Actions should be the fixed four verdict options
    expected_actions = [
        "You judge that the original poster is the one in the wrong (YTA).",
        "You judge that the original poster is not the one in the wrong (NTA).",
        "You judge that everyone involved behaved badly (ESH).",
        "You judge that no one behaved badly (NAH).",
    ]
    assert first["actions"] == expected_actions

    meta = first["metadata"]
    assert isinstance(meta, dict)
    assert meta.get("source_dataset") == "AITA"
    assert (
        meta.get("hf_dataset_id") == "derek-thomas/dataset-creator-reddit-amitheasshole"
    )
    assert meta.get("hf_split") == "train"
    assert meta.get("source_id") is not None

    problem_uid = first["problem_uid"]
    assert isinstance(problem_uid, str)
    assert problem_uid.startswith("aita::")

    # Now go end-to-end through sample_problems -> PracticalProblem
    n = 3
    problems = sample_problems(
        dataset_name="AITA",
        n_problems=n,
        seed=0,
        adapter_kwargs={"split": "train"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        assert len(p.actions) == 4
        assert p.actions == expected_actions
        assert p.source_dataset == "AITA"
        assert isinstance(p.problem_uid, str)
        assert p.problem_uid.startswith("aita::")

        # Ensure the template survived into the PracticalProblem instance
        assert p.decision_situation.startswith(prefix)
        assert "'''" in p.decision_situation
