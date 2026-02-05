"""CLI entry point for judgment stability / re-sampling experiments.

This script wires together the high-level workflow:

    1. Parse an :class:`ExperimentConfig` from the command line via
       :func:`chz.nested_entrypoint`.
    2. If ``config_yaml`` is provided, load a YAML configuration and merge
       it with CLI values (CLI taking precedence) to obtain the final
       :class:`ExperimentConfig`.
    3. Initialize an OpenAI-compatible client and the InferenceClient wrapper.
    4. Load decision problems from a configured dataset.
    5. Apply a *pluggable* async transformation function to each problem.
    6. Apply a *pluggable* async reasoning generation function to each
       (possibly transformed) problem to obtain reasoning traces.
    7. Apply a *pluggable* async scoring function to each reasoning trace to
       obtain label probabilities.
    8. Convert the accumulated Python data structures into DataFrames and
       hand them off to fixed analysis / plotting / persistence helpers.

Usage examples (chz CLI)
------------------------

Run with a YAML configuration file::

    python -m experiments.re_sampling_stability.run_experiment \
        config_yaml=experiments/re_sampling_stability/configs/minimal.yaml

Override a single field from the YAML via CLI (CLI wins for that field)::

    python -m experiments.re_sampling_stability.run_experiment \
        config_yaml=experiments/re_sampling_stability/configs/minimal.yaml \
        candidate_model=Qwen/Qwen2.5-7B-Instruct

Specify custom pluggable functions for transform / reasoning / scoring::

    python -m experiments.re_sampling_stability.run_experiment \
        config_yaml=experiments/re_sampling_stability/configs/with_transform.yaml \
        generate_reasoning_trace_fn=\
          experiments.re_sampling_stability.reasoning.generate_reasoning_traces_for_problem \
        score_choice_labels_fn=\
          experiments.re_sampling_stability.judgment.score_choice_labels_for_trace
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from typing import Any, Awaitable, Callable, List, Optional, Sequence

import chz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from practical_deliberation_llms.datasets import sample_problems
from practical_deliberation_llms.inference import InferenceClient
from practical_deliberation_llms.io import ensure_fresh_output_dir, save_results
from practical_deliberation_llms.plotting import plot_results

from .config import CLIConfig, ExperimentConfig, build_config_from_cli

logger = logging.getLogger(__name__)


# Type aliases for pluggable async function signatures.

TransformProblemsFn = Callable[[ExperimentConfig, Any], Awaitable[Sequence[Any]]]
"""Async function that transforms a single problem.

The function receives an `ExperimentConfig` and a single "problem" object,
typically a `PracticalProblem` instance produced by `sample_problems`.
It must return a sequence (possibly empty, typically non-empty) of
problems. The default implementation will at least include the input
problem and may append transformed variants.

NOTE: We intentionally keep the type as `Any` here to avoid over-constraining
the interface while the rest of the experiment code is in flux.
"""


GenerateReasoningTraceFn = Callable[
    [ExperimentConfig, InferenceClient, Any], Awaitable[Sequence[dict]]
]
"""Async function that generates reasoning traces for a single problem.

The function receives the experiment configuration, an `InferenceClient`
instance, and a single problem object. It must return a sequence of
trace records represented as plain dicts. Each dict should at least
contain keys needed by the scoring function (e.g. problem identifiers,
options, and the generated reasoning text).

TODO: Once the schema stabilizes, introduce a lightweight TraceRecord
dataclass and replace the `dict` here.
"""


ScoreChoiceLabelsFn = Callable[
    [ExperimentConfig, InferenceClient, dict], Awaitable[Sequence[dict]]
]
"""Async function that scores labels for a single reasoning trace.

The function receives the experiment configuration, an `InferenceClient`,
and a single trace record (as produced by the reasoning function). It
must return a sequence of score records, typically one per label. Each
record should at least contain a `problem_uid`, `trace_id`, `label`, and
`prob` field.

TODO: As with traces, once the schema is stable, we can introduce a
ScoreRecord dataclass.
"""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def resolve_callable(path: str) -> Callable[..., Any]:
    """Resolve a dotted-path string to a Python callable.

    Parameters
    ----------
    path:
        Dotted import path, e.g.
        ``"experiments.re_sampling_stability.transform.transform_problems"``.

    Returns
    -------
    callable
        The resolved Python object. A `TypeError` is raised if the
        resolved attribute is not callable.
    """

    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):  # pragma: no cover - defensive
        raise TypeError(f"Object at {path!r} is not callable")
    return fn


def set_global_seeds(seed: int) -> None:
    """Set global random seeds for Python and NumPy.

    Note that this does *not* control determinism of the underlying
    language model; it only helps make sampling and problem selection
    deterministic where we rely on Python/NumPy RNGs.
    """

    import random

    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: str | int | None = None) -> None:
    """Configure a basic logging setup for the CLI.

    Parameters
    ----------
    level:
        Logging level to use. May be a standard level name (e.g. "DEBUG",
        "info", "WARNING") or an integer (e.g. 10 for DEBUG). If None,
        defaults to INFO.
    """

    if isinstance(level, str):
        level_name = level.upper()
        numeric_level = getattr(logging, level_name, None)
        if not isinstance(numeric_level, int):
            raise ValueError(
                "Invalid log level name {!r}; expected one of DEBUG, INFO, "
                "WARNING, ERROR, CRITICAL or an integer value".format(level)
            )
    elif isinstance(level, int):
        numeric_level = level
    else:
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Core async orchestration
# ---------------------------------------------------------------------------


async def run_experiment_async(
    config: ExperimentConfig,
    transform_problems_fn: Optional[TransformProblemsFn],
    generate_reasoning_trace_fn: GenerateReasoningTraceFn,
    score_choice_labels_fn: ScoreChoiceLabelsFn,
) -> None:
    """Run a single experiment end-to-end.

    This function wires together dataset loading, problem transformation,
    reasoning trace generation, scoring, and downstream analysis.

    All pluggable functions are expected to be async. Transformations are
    currently applied sequentially per problem. Reasoning and scoring are
    executed with bounded concurrency controlled by `config.max_concurrency`
    via an `asyncio.Semaphore` and `asyncio.gather`.
    """

    logger.info("Experiment configuration: %s", chz.asdict(config))

    # Ensure output directory is fresh so we do not overwrite existing results.
    ensure_fresh_output_dir(config.output_dir)

    # Concurrency control for reasoning and scoring stages.
    max_concurrency = getattr(config, "max_concurrency", 64)
    if max_concurrency <= 0:  # pragma: no cover - defensive; config validates
        max_concurrency = 64
    semaphore = asyncio.Semaphore(max_concurrency)

    # Initialize client and wrapper. We load a .env file (if present) so
    # that OPENAI_BASE_URL and OPENAI_API_KEY can be configured without
    # touching git-tracked files. CLI/config values act as optional
    # overrides for these environment variables.
    load_dotenv()

    client_kwargs: dict[str, Any] = {}
    if config.effective_openai_base_url:
        client_kwargs["base_url"] = config.effective_openai_base_url
    if config.effective_api_token:
        client_kwargs["api_key"] = config.effective_api_token

    client = AsyncOpenAI(**client_kwargs)
    inference_client = InferenceClient(client=client, model=config.candidate_model)

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base problems (potentially from multiple datasets)
    # ------------------------------------------------------------------
    base_problems: List[Any] = []

    for ds_idx, ds_spec in enumerate(config.datasets):
        effective_seed = int(config.seed) + ds_idx

        logger.info(
            "Loading %d problems from dataset '%s' with adapter_kwargs=%s",
            ds_spec.n_problems,
            ds_spec.name,
            ds_spec.adapter_kwargs,
        )

        problems_for_ds = sample_problems(
            dataset_name=ds_spec.name,
            n_problems=ds_spec.n_problems,
            seed=effective_seed,
            adapter_kwargs=ds_spec.adapter_kwargs,
        )

        logger.info(
            "Loaded %d base problems from dataset '%s'",
            len(problems_for_ds),
            ds_spec.name,
        )

        base_problems.extend(problems_for_ds)

    logger.info(
        "Loaded %d base problems across %d datasets",
        len(base_problems),
        len(config.datasets),
    )

    # ------------------------------------------------------------------
    # 2. Apply transformations (optional, pluggable, async, per-problem)
    # ------------------------------------------------------------------
    all_problems: List[Any] = []

    # If `config.transform_problems_fn` is None, we skip the transformation
    # step entirely and only use the base problems. This allows callers to
    # disable the transform hook even if a transform function was configured
    # in an older YAML file.
    if config.transform_problems_fn is None or transform_problems_fn is None:
        logger.info("No transform_problems_fn configured; skipping transformations.")
        all_problems.extend(base_problems)
    else:
        # NOTE: For now we apply transformations sequentially per problem to
        # keep the control flow simple and avoid concurrency issues while we
        # iterate on the design. Once the default implementations are stable,
        # we can consider grouping problems into batches and using
        # `asyncio.gather`.
        for idx, problem in enumerate(base_problems):
            if idx % 10 == 0:
                logger.debug("Transforming problem %d / %d", idx, len(base_problems))

            try:
                transformed_seq = await transform_problems_fn(config, problem)
            except Exception:  # pragma: no cover - defensive logging, re-raise
                logger.exception(
                    "Transform failed for problem_idx=%d problem_uid=%s",
                    idx,
                    getattr(problem, "problem_uid", None),
                )
                raise

            logger.debug(
                "Transform result problem_idx=%d n_variants=%d",
                idx,
                len(transformed_seq),
            )

            # TODO: Validate that the original problem is included in
            # `transformed_seq` (or decide that this is the
            # transformer's responsibility and document the contract).
            all_problems.extend(transformed_seq)

    if not all_problems:
        logger.warning("No problems after transformations; nothing to run.")

    logger.info("Total problems after transformations: %d", len(all_problems))

    # ------------------------------------------------------------------
    # 3. Generate reasoning traces (pluggable, async, per-problem)
    # ------------------------------------------------------------------
    trace_records: List[dict] = []

    async def _reason_for_problem(idx: int, problem: Any) -> List[dict]:
        if idx % 10 == 0:
            logger.debug(
                "Generating traces for problem %d / %d", idx, len(all_problems)
            )

        async with semaphore:
            try:
                traces_for_problem = await generate_reasoning_trace_fn(
                    config,
                    inference_client,
                    problem,
                )
            except Exception:  # pragma: no cover - defensive logging, re-raise
                logger.exception(
                    "Reasoning failed for problem_idx=%d problem_uid=%s",
                    idx,
                    getattr(problem, "problem_uid", None),
                )
                raise

        if traces_for_problem is None:
            raise TypeError(
                "generate_reasoning_trace_fn must return a non-None sequence; "
                "got None instead"
            )

        # NOTE: We intentionally coerce to list here to avoid surprises if the
        # pluggable returns a non-list sequence.
        return list(traces_for_problem)

    if all_problems:
        reasoning_results = await asyncio.gather(
            *[
                _reason_for_problem(idx, problem)
                for idx, problem in enumerate(all_problems)
            ]
        )
        for traces_for_problem in reasoning_results:
            # TODO: The default implementation is expected to generate
            # `config.n_traces_per_problem` traces, but we do not enforce that
            # here. It is responsibility of the pluggable function to respect
            # this hyperparameter where appropriate.
            trace_records.extend(traces_for_problem)

    if not trace_records:
        logger.warning(
            "No reasoning traces generated; downstream scoring and metrics will be empty."
        )

    logger.info("Generated %d reasoning traces", len(trace_records))

    # ------------------------------------------------------------------
    # 4. Score choice labels (pluggable, async, per-trace)
    # ------------------------------------------------------------------
    score_records: List[dict] = []

    async def _score_single_trace(idx: int, trace: dict) -> List[dict]:
        if idx % 50 == 0:
            logger.debug("Scoring labels for trace %d / %d", idx, len(trace_records))

        async with semaphore:
            try:
                scores_for_trace = await score_choice_labels_fn(
                    config,
                    inference_client,
                    trace,
                )
            except Exception:  # pragma: no cover - defensive logging, re-raise
                logger.exception(
                    "Scoring failed for trace_idx=%d trace_id=%s problem_uid=%s",
                    idx,
                    trace.get("trace_id"),
                    trace.get("problem_uid"),
                )
                raise

        if scores_for_trace is None:
            raise TypeError(
                "score_choice_labels_fn must return a non-None sequence; "
                "got None instead"
            )

        return list(scores_for_trace)

    if trace_records:
        scoring_results = await asyncio.gather(
            *[
                _score_single_trace(idx, trace)
                for idx, trace in enumerate(trace_records)
            ]
        )
        for scores_for_trace in scoring_results:
            score_records.extend(scores_for_trace)

    if not score_records:
        logger.warning(
            "No label scores computed; metrics and plots will be empty or missing."
        )

    logger.info("Computed %d label scores", len(score_records))

    # ------------------------------------------------------------------
    # 5. Convert to DataFrames and run fixed analysis / plotting / saving
    # ------------------------------------------------------------------

    from .metrics import compute_metrics

    trace_df = pd.DataFrame(trace_records)
    scores_df = pd.DataFrame(score_records)

    d_within_df, baseline_vs_trans_df = compute_metrics(trace_df, scores_df)

    save_results(
        config=config,
        output_dir=config.output_dir,
        trace_df=trace_df,
        scores_df=scores_df,
        d_within_df=d_within_df,
        baseline_vs_trans_df=baseline_vs_trans_df,
        results_file_format=config.results_file_format,
    )

    if config.make_plots:
        plot_results(
            output_dir=config.output_dir,
            d_within_df=d_within_df,
            baseline_vs_trans_df=baseline_vs_trans_df,
        )

    logger.info(
        "Experiment run complete (traces=%d, scores=%d, metrics_within=%d, kl_pairs=%d)",
        len(trace_records),
        len(score_records),
        len(d_within_df) if d_within_df is not None else 0,
        len(baseline_vs_trans_df) if baseline_vs_trans_df is not None else 0,
    )


def main_chz(cli_cfg: CLIConfig) -> None:
    """CLI entry point powered by :mod:`chz`.

    ``cli_cfg`` is parsed from the command line via
    :func:`chz.nested_entrypoint`. The referenced YAML configuration is
    loaded and combined with any non-``None`` CLI overrides to obtain the
    final :class:`ExperimentConfig` passed to
    :func:`run_experiment_async`.
    """

    config = build_config_from_cli(cli_cfg)

    # Configure logging based on the experiment configuration.
    setup_logging(config.log_level)

    # Seed RNGs based on the final configuration so that YAML / override
    # changes always affect the experiment deterministically.
    set_global_seeds(config.seed)

    transform_problems_fn: Optional[TransformProblemsFn] = None
    if config.transform_problems_fn:
        transform_problems_fn = resolve_callable(config.transform_problems_fn)

    generate_reasoning_trace_fn = resolve_callable(config.generate_reasoning_trace_fn)
    score_choice_labels_fn = resolve_callable(config.score_choice_labels_fn)

    try:
        asyncio.run(
            run_experiment_async(
                config=config,
                transform_problems_fn=transform_problems_fn,
                generate_reasoning_trace_fn=generate_reasoning_trace_fn,
                score_choice_labels_fn=score_choice_labels_fn,
            )
        )
    except FileExistsError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # NOTE: This module is intended to be invoked via
    #   `python -m experiments.re_sampling_stability.run_experiment`
    # so that :mod:`chz` can parse a :class:`CLIConfig` from the command
    # line and run the experiment.
    chz.nested_entrypoint(main_chz)
