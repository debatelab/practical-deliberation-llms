import asyncio
from typing import Any, Dict, List

import pytest

from experiments.re_sampling_stability.config import ExperimentConfig, DatasetSpec
from experiments.re_sampling_stability.run_experiment import run_experiment_async


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
        self, prompt: str, labels: List[str]
    ) -> Dict[str, float]:  # pragma: no cover - not used
        self.calls_score.append({"prompt": prompt, "labels": labels})
        return {label: 1.0 / len(labels) for label in labels}


@pytest.mark.asyncio
async def test_run_experiment_async_with_dummy_pluggables(monkeypatch, tmp_path):
    # Construct a tiny ExperimentConfig
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
        output_dir=str(tmp_path),
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

    monkeypatch.setattr(mod, "OpenAI", DummyClient)

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
    assert (tmp_path / "traces.parquet").exists()
    assert (tmp_path / "scores.parquet").exists()
