import asyncio
import json
from typing import Any, Dict, List

import pytest

from experiments.re_sampling_stability.config import DatasetSpec, ExperimentConfig
from experiments.re_sampling_stability.run_experiment import run_experiment_async
from practical_deliberation_llms.inference import InferenceError


class DummyInferenceClient:
    def __init__(self) -> None:
        self.calls_generate: List[Dict[str, Any]] = []
        self.calls_score: List[Dict[str, Any]] = []

    # We do not use this in the test because we inject dummy pluggables
    # that bypass the real InferenceClient methods, but the attributes
    # must exist on the object created in run_experiment_async.
    def generate_trace(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - not used
        self.calls_generate.append({"args": args, "kwargs": kwargs})
        return {"think": "", "label_json": '{"label": "a"}', "label": "a"}

    def score_label_given_trace(
        self, context_messages: List[Dict[str, Any]], reasoning: str, labels: List[str]
    ) -> Dict[str, float]:  # pragma: no cover - not used
        self.calls_score.append(
            {
                "context_messages": context_messages,
                "reasoning": reasoning,
                "labels": labels,
            }
        )
        return {label: 1.0 / len(labels) for label in labels}


@pytest.mark.asyncio
async def test_run_experiment_async_with_dummy_pluggables(monkeypatch, tmp_path):
    # Construct a tiny ExperimentConfig
    out_dir = tmp_path / "results"

    cfg = ExperimentConfig(
        candidate_model="cand",
        assistant_model="assist",
        openai_base_url="http://localhost:8000/v1",
        api_token="key",
        seed=0,
        datasets=[
            DatasetSpec(name="daily_dilemmas", n_problems=1, adapter_kwargs={}),
        ],
        n_traces_per_problem=1,
        temperature=0.7,
        top_p=0.95,
        max_transformations_per_problem=0,
        output_dir=str(out_dir),
        make_plots=False,
        transform_problems_fn=None,
        generate_reasoning_trace_fn="dummy.reasoning",
        score_choice_labels_fn="dummy.scoring",
    )

    # Monkeypatch OpenAI and InferenceClient construction inside the module
    import experiments.re_sampling_stability.run_experiment as mod

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(mod, "AsyncOpenAI", DummyClient)

    dummy_inference = DummyInferenceClient()

    def dummy_inference_client_factory(client: Any, model: str) -> DummyInferenceClient:
        return dummy_inference

    monkeypatch.setattr(mod, "InferenceClient", dummy_inference_client_factory)

    # Dummy dataset sampler: one trivial problem
    class DummyProblem:
        def __init__(self) -> None:
            self.decision_situation = "Test situation"
            self.actions = ["do A", "do B"]
            self.problem_uid = "p0"

    def sample_problems_stub(
        dataset_name: str, n_problems: int, seed: int, adapter_kwargs=None
    ):
        return [DummyProblem()]

    monkeypatch.setattr(mod, "sample_problems", sample_problems_stub)

    # Dummy pluggables: no transform, fixed trace/score
    async def transform_stub(config: Any, problem: Any):
        return [problem]

    async def reasoning_stub(config: Any, inference_client: Any, problem: Any):
        return [
            {
                "problem_uid": problem.problem_uid,
                "trace_id": "p0::trace_0",
                "decision_situation": problem.decision_situation,
                "actions": list(problem.actions),
            }
        ]

    async def scoring_stub(config: Any, inference_client: Any, trace: Dict[str, Any]):
        return [
            {
                "problem_uid": trace["problem_uid"],
                "trace_id": trace["trace_id"],
                "label": "a",
                "prob": 1.0,
            }
        ]

    await run_experiment_async(
        config=cfg,
        transform_problems_fn=transform_stub,
        generate_reasoning_trace_fn=reasoning_stub,
        score_choice_labels_fn=scoring_stub,
    )

    # Basic sanity: result files should exist
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "scores.jsonl").exists()


@pytest.mark.asyncio
async def test_run_experiment_async_writes_jsonl_when_configured(monkeypatch, tmp_path):
    # Same setup as the parquet test, but with results_file_format=jsonl
    out_dir = tmp_path / "results"

    cfg = ExperimentConfig(
        candidate_model="cand",
        assistant_model="assist",
        openai_base_url="http://localhost:8000/v1",
        api_token="key",
        seed=0,
        datasets=[
            DatasetSpec(name="daily_dilemmas", n_problems=1, adapter_kwargs={}),
        ],
        n_traces_per_problem=1,
        temperature=0.7,
        top_p=0.95,
        max_transformations_per_problem=0,
        output_dir=str(out_dir),
        make_plots=False,
        transform_problems_fn=None,
        generate_reasoning_trace_fn="dummy.reasoning",
        score_choice_labels_fn="dummy.scoring",
        results_file_format="jsonl",
    )

    import experiments.re_sampling_stability.run_experiment as mod

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(mod, "AsyncOpenAI", DummyClient)

    dummy_inference = DummyInferenceClient()

    def dummy_inference_client_factory(client: Any, model: str) -> DummyInferenceClient:
        return dummy_inference

    monkeypatch.setattr(mod, "InferenceClient", dummy_inference_client_factory)

    class DummyProblem:
        def __init__(self) -> None:
            self.decision_situation = "Test situation"
            self.actions = ["do A", "do B"]
            self.problem_uid = "p0"

    def sample_problems_stub(
        dataset_name: str, n_problems: int, seed: int, adapter_kwargs=None
    ):
        return [DummyProblem()]

    monkeypatch.setattr(mod, "sample_problems", sample_problems_stub)

    async def transform_stub(config: Any, problem: Any):
        return [problem]

    async def reasoning_stub(config: Any, inference_client: Any, problem: Any):
        return [
            {
                "problem_uid": problem.problem_uid,
                "trace_id": "p0::trace_0",
                "decision_situation": problem.decision_situation,
                "actions": list(problem.actions),
            }
        ]

    async def scoring_stub(config: Any, inference_client: Any, trace: Dict[str, Any]):
        return [
            {
                "problem_uid": trace["problem_uid"],
                "trace_id": trace["trace_id"],
                "label": "a",
                "prob": 1.0,
            }
        ]

    await run_experiment_async(
        config=cfg,
        transform_problems_fn=transform_stub,
        generate_reasoning_trace_fn=reasoning_stub,
        score_choice_labels_fn=scoring_stub,
    )

    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "scores.jsonl").exists()


@pytest.mark.asyncio
async def test_run_experiment_async_fails_if_output_dir_exists(monkeypatch, tmp_path):
    # Create an output directory ahead of time to trigger the guard.
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    cfg = ExperimentConfig(
        candidate_model="cand",
        assistant_model="assist",
        openai_base_url="http://localhost:8000/v1",
        api_token="key",
        seed=0,
        datasets=[
            DatasetSpec(name="daily_dilemmas", n_problems=1, adapter_kwargs={}),
        ],
        n_traces_per_problem=1,
        temperature=0.7,
        top_p=0.95,
        max_transformations_per_problem=0,
        output_dir=str(out_dir),
        make_plots=False,
        transform_problems_fn=None,
        generate_reasoning_trace_fn="dummy.reasoning",
        score_choice_labels_fn="dummy.scoring",
    )

    import experiments.re_sampling_stability.run_experiment as mod

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(mod, "AsyncOpenAI", DummyClient)

    dummy_inference = DummyInferenceClient()

    def dummy_inference_client_factory(client: Any, model: str) -> DummyInferenceClient:
        return dummy_inference

    monkeypatch.setattr(mod, "InferenceClient", dummy_inference_client_factory)

    class DummyProblem:
        def __init__(self) -> None:
            self.decision_situation = "Test situation"
            self.actions = ["do A", "do B"]
            self.problem_uid = "p0"

    def sample_problems_stub(
        dataset_name: str, n_problems: int, seed: int, adapter_kwargs=None
    ):
        return [DummyProblem()]

    monkeypatch.setattr(mod, "sample_problems", sample_problems_stub)

    async def transform_stub(config: Any, problem: Any):
        return [problem]

    async def reasoning_stub(config: Any, inference_client: Any, problem: Any):
        return [
            {
                "problem_uid": problem.problem_uid,
                "trace_id": "p0::trace_0",
                "decision_situation": problem.decision_situation,
                "actions": list(problem.actions),
            }
        ]

    async def scoring_stub(config: Any, inference_client: Any, trace: Dict[str, Any]):
        return [
            {
                "problem_uid": trace["problem_uid"],
                "trace_id": trace["trace_id"],
                "label": "a",
                "prob": 1.0,
            }
        ]

    with pytest.raises(FileExistsError):
        await run_experiment_async(
            config=cfg,
            transform_problems_fn=transform_stub,
            generate_reasoning_trace_fn=reasoning_stub,
            score_choice_labels_fn=scoring_stub,
        )


@pytest.mark.asyncio
async def test_run_experiment_async_skips_on_inference_error(monkeypatch, tmp_path):
    """Problems / traces raising InferenceError are skipped, others succeed."""

    out_dir = tmp_path / "results"

    cfg = ExperimentConfig(
        candidate_model="cand",
        assistant_model="assist",
        openai_base_url="http://localhost:8000/v1",
        api_token="key",
        seed=0,
        datasets=[
            DatasetSpec(name="daily_dilemmas", n_problems=3, adapter_kwargs={}),
        ],
        n_traces_per_problem=1,
        temperature=0.7,
        top_p=0.95,
        max_transformations_per_problem=0,
        output_dir=str(out_dir),
        make_plots=False,
        transform_problems_fn=None,
        generate_reasoning_trace_fn="dummy.reasoning",
        score_choice_labels_fn="dummy.scoring",
        results_file_format="jsonl",
    )

    import experiments.re_sampling_stability.run_experiment as mod

    class DummyClient:
        def __init__(
            self, *args: Any, **kwargs: Any
        ) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr(mod, "AsyncOpenAI", DummyClient)

    dummy_inference = DummyInferenceClient()

    def dummy_inference_client_factory(client: Any, model: str) -> DummyInferenceClient:
        return dummy_inference

    monkeypatch.setattr(mod, "InferenceClient", dummy_inference_client_factory)

    class DummyProblem:
        def __init__(self, uid: str) -> None:
            self.decision_situation = f"Situation {uid}"
            self.actions = ["do A", "do B"]
            self.problem_uid = uid

    def sample_problems_stub(
        dataset_name: str, n_problems: int, seed: int, adapter_kwargs=None
    ):
        # Return three problems with distinct UIDs. We intentionally ignore
        # n_problems here to keep the stub simple and deterministic.
        return [
            DummyProblem("p_ok"),
            DummyProblem("p_reason_error"),
            DummyProblem("p_score_error"),
        ]

    monkeypatch.setattr(mod, "sample_problems", sample_problems_stub)

    async def transform_stub(config: Any, problem: Any):
        return [problem]

    async def reasoning_stub(config: Any, inference_client: Any, problem: Any):
        if problem.problem_uid == "p_reason_error":
            # Simulate an inference failure during reasoning.
            raise InferenceError("synthetic reasoning failure")
        return [
            {
                "problem_uid": problem.problem_uid,
                "trace_id": f"{problem.problem_uid}::trace_0",
                "decision_situation": problem.decision_situation,
                "actions": list(problem.actions),
            }
        ]

    async def scoring_stub(config: Any, inference_client: Any, trace: Dict[str, Any]):
        if trace["problem_uid"] == "p_score_error":
            # Simulate an inference failure during scoring for a single trace.
            raise InferenceError("synthetic scoring failure")
        return [
            {
                "problem_uid": trace["problem_uid"],
                "trace_id": trace["trace_id"],
                "label": "a",
                "prob": 1.0,
            }
        ]

    await run_experiment_async(
        config=cfg,
        transform_problems_fn=transform_stub,
        generate_reasoning_trace_fn=reasoning_stub,
        score_choice_labels_fn=scoring_stub,
    )

    # Read back traces and scores written by save_results.
    traces_path = out_dir / "traces.jsonl"
    scores_path = out_dir / "scores.jsonl"

    assert traces_path.exists()
    assert scores_path.exists()

    with open(traces_path, "r", encoding="utf-8") as f:
        trace_records = [json.loads(line) for line in f if line.strip()]

    with open(scores_path, "r", encoding="utf-8") as f:
        score_records = [json.loads(line) for line in f if line.strip()]

    # Reasoning failure: problem "p_reason_error" should be completely absent
    # from traces and scores.
    trace_problem_uids = {rec["problem_uid"] for rec in trace_records}
    score_problem_uids = {rec["problem_uid"] for rec in score_records}

    assert "p_reason_error" not in trace_problem_uids
    assert "p_reason_error" not in score_problem_uids

    # Scoring failure: problem "p_score_error" should appear in traces but
    # not in scores.
    assert "p_score_error" in trace_problem_uids
    assert "p_score_error" not in score_problem_uids

    # The healthy problem should appear in both.
    assert "p_ok" in trace_problem_uids
    assert "p_ok" in score_problem_uids
