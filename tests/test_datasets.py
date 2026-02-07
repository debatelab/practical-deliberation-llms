from practical_deliberation_llms.datasets import (
    ADAPTER_REGISTRY,
    load_normalized_dataset,
    sample_problems,
)
from practical_deliberation_llms.datasets.legalbench_corporate_lobbying import (
    LegalBenchCorporateLobbyingAdapter,
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


def test_msmdilemmas_adapter_accumulates_history() -> None:
    """Multi-step Moral Dilemmas adapter should accumulate prior situations."""

    assert "MSMDilemmas" in ADAPTER_REGISTRY

    df = load_normalized_dataset(
        "MSMDilemmas",
        adapter_kwargs={"framework": "MFT"},
    )
    assert not df.empty

    for col in ("decision_situation", "actions", "metadata", "problem_uid"):
        assert col in df.columns

    # Find a later step (>= 3) so that there is meaningful history to check.
    later_row = None
    for _, row in df.iterrows():
        meta_obj = row["metadata"]
        assert isinstance(meta_obj, dict)
        stage_index = meta_obj.get("stage_index")
        if isinstance(stage_index, int) and stage_index >= 3:
            later_row = row
            break

    assert later_row is not None, "Expected at least one scenario with >= 3 steps"

    meta_obj = later_row["metadata"]
    assert isinstance(meta_obj, dict)
    stage_index = meta_obj["stage_index"]
    situations_so_far = meta_obj.get("situations_so_far")

    assert isinstance(situations_so_far, list)
    assert len(situations_so_far) == stage_index

    ds_text = later_row["decision_situation"]

    # The accumulated situation text should include numbered steps for all
    # situations up to the current stage.
    for idx, sit in enumerate(situations_so_far, start=1):
        if sit:
            snippet = f"- Step {idx}: {sit}"
            assert snippet in ds_text

    # The dilemma header should reference the current step index.
    assert f"Current dilemma (step {stage_index}):" in ds_text

    # Basic adapter metadata and action sanity checks.
    assert len(later_row["actions"]) == 2
    assert meta_obj.get("source_dataset") == "MSMDilemmas"
    assert meta_obj.get("framework") == "MFT"
    assert meta_obj.get("scenario_id") == meta_obj.get("source_id")

    problem_uid = later_row["problem_uid"]
    assert isinstance(problem_uid, str)
    assert problem_uid.startswith("mmd::MFT::")
    assert f"::step{stage_index}" in problem_uid

    # End-to-end through sample_problems -> PracticalProblem
    n = 3
    problems = sample_problems(
        dataset_name="MSMDilemmas",
        n_problems=n,
        seed=0,
        adapter_kwargs={"framework": "MFT"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        assert len(p.actions) == 2
        assert p.source_dataset == "MSMDilemmas"
        assert isinstance(p.problem_uid, str)
        assert p.problem_uid.startswith("mmd::MFT::")


def test_role_conflict_bench_adapter_basic_integration() -> None:
    """Load RoleConflictBench from HF and normalize to PracticalProblems."""

    assert "RoleConflictBench" in ADAPTER_REGISTRY

    df = load_normalized_dataset(
        "RoleConflictBench",
        adapter_kwargs={"name": "default", "split": "train"},
    )
    assert not df.empty

    for col in ("decision_situation", "actions", "metadata", "problem_uid"):
        assert col in df.columns

    first = df.iloc[0]
    assert isinstance(first["decision_situation"], str)
    assert isinstance(first["actions"], list)
    assert len(first["actions"]) == 2

    meta = first["metadata"]
    assert isinstance(meta, dict)
    assert meta.get("source_dataset") == "RoleConflictBench"
    assert meta.get("hf_dataset_id") == "DebateLabKIT/role-conflict-bench"
    assert meta.get("hf_config") == "default"
    assert meta.get("hf_split") == "train"
    assert meta.get("source_id") is not None

    problem_uid = first["problem_uid"]
    assert isinstance(problem_uid, str)
    assert problem_uid.startswith("role_conflict::")

    # End-to-end through sample_problems -> PracticalProblem
    n = 3
    problems = sample_problems(
        dataset_name="RoleConflictBench",
        n_problems=n,
        seed=0,
        adapter_kwargs={"name": "default", "split": "train"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        assert len(p.actions) == 2
        assert p.source_dataset == "RoleConflictBench"
        assert isinstance(p.problem_uid, str)
        assert p.problem_uid.startswith("role_conflict::")


def test_airisk_dilemmas_adapter_basic_integration() -> None:
    """Load AIRiskDilemmas from HF and normalize to PracticalProblems."""

    assert "AIRiskDilemmas" in ADAPTER_REGISTRY

    df = load_normalized_dataset(
        "AIRiskDilemmas",
        adapter_kwargs={"subset": "model_eval", "split": "test"},
    )
    assert not df.empty

    for col in ("decision_situation", "actions", "metadata", "problem_uid"):
        assert col in df.columns

    first = df.iloc[0]
    assert isinstance(first["decision_situation"], str)
    assert isinstance(first["actions"], list)
    assert len(first["actions"]) >= 2
    assert all(isinstance(a, str) and a.strip() for a in first["actions"])

    meta = first["metadata"]
    assert isinstance(meta, dict)
    assert meta.get("source_dataset") == "AIRiskDilemmas"
    assert meta.get("hf_dataset_id") == "kellycyy/AIRiskDilemmas"
    assert meta.get("hf_subset") == "model_eval"
    assert meta.get("hf_split") == "test"
    assert meta.get("source_id") is not None

    # Per-action annotations should line up with the number of actions.
    values_per_action = meta.get("values_per_action")
    targets_per_action = meta.get("targets_per_action")
    assert isinstance(values_per_action, list)
    assert isinstance(targets_per_action, list)
    assert len(values_per_action) == meta.get("num_actions")
    assert len(targets_per_action) == meta.get("num_actions")

    problem_uid = first["problem_uid"]
    assert isinstance(problem_uid, str)
    assert problem_uid.startswith("airisk::model_eval::")

    # End-to-end through sample_problems -> PracticalProblem
    n = 3
    problems = sample_problems(
        dataset_name="AIRiskDilemmas",
        n_problems=n,
        seed=0,
        adapter_kwargs={"subset": "model_eval", "split": "test"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        assert len(p.actions) >= 2
        assert p.source_dataset == "AIRiskDilemmas"
        assert isinstance(p.problem_uid, str)
        assert p.problem_uid.startswith("airisk::model_eval::")


def test_legalbench_corporate_lobbying_integration() -> None:
    """Integration checks for the legalbench corporate lobbying adapter.

    This test is stronger and more deterministic now that the adapter
    consistently uses the `use_ambiguous` kwarg.

    - Verifies adapter registration and normalized DataFrame shape.
    - Asserts canonical ambiguity scale in default mode.
    - Asserts binary actions when `use_ambiguous=False` is passed.
    """

    assert "legalbench_corporate_lobbying" in ADAPTER_REGISTRY

    # Load normalized DataFrame (may download from HF); skip if empty to avoid
    # flaky failures in offline CI environments or when the adapter filters all rows.
    df = load_normalized_dataset(
        "legalbench_corporate_lobbying",
        adapter_kwargs={"split": "test"},
    )

    assert not df.empty, "legalbench normalized dataset must not be empty"

    # Basic normalized columns
    for col in ("decision_situation", "actions", "metadata", "problem_uid"):
        assert col in df.columns

    first = df.iloc[0]
    assert isinstance(first["decision_situation"], str)
    assert "Answer by only replying to Yes or No." not in first["decision_situation"]
    assert isinstance(first["actions"], list) and first["actions"]
    assert isinstance(first["metadata"], dict)
    assert first["metadata"].get("source_dataset") == "legalbench_corporate_lobbying"
    assert isinstance(first["problem_uid"], str)
    assert first["problem_uid"].startswith("legalbench_corporate_lobbying::")

    # End-to-end via sample_problems: default should produce the canonical ambiguity scale
    n = 3
    problems = sample_problems(
        dataset_name="legalbench_corporate_lobbying",
        n_problems=n,
        seed=0,
        adapter_kwargs={"split": "test"},
    )
    assert len(problems) == n

    for p in problems:
        _assert_problem_basic_shape(p)
        # With the kwarg naming fixed, default mode must equal the canonical AMBIGUITY scale
        assert p.actions == LegalBenchCorporateLobbyingAdapter.CANONICAL_AMBIGUOUS
        assert p.source_dataset == "legalbench_corporate_lobbying"

    # Binary mode: pass use_ambiguous=False to obtain YES/NO actions
    problems_bin = sample_problems(
        dataset_name="legalbench_corporate_lobbying",
        n_problems=3,
        seed=1,
        adapter_kwargs={"split": "test", "use_ambiguous": False},
    )
    assert len(problems_bin) == 3
    for p in problems_bin:
        _assert_problem_basic_shape(p)
        assert "Answer by only replying to Yes or No." not in p.decision_situation
        assert p.actions == ["YES", "NO"]
        assert p.source_dataset == "legalbench_corporate_lobbying"
