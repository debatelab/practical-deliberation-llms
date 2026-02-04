"""CLI entry point for judgment stability / re-sampling experiments.

This script wires together the high-level workflow:

    1. Parse CLI arguments into an experiment configuration.
    2. Initialize an OpenAI-compatible client and the InferenceClient wrapper.
    3. Load decision problems from a configured dataset.
    4. Apply a *pluggable* async transformation function to each problem.
    5. Apply a *pluggable* async reasoning generation function to each
       (possibly transformed) problem to obtain reasoning traces.
    6. Apply a *pluggable* async scoring function to each reasoning trace to
       obtain label probabilities.
    7. Convert the accumulated Python data structures into DataFrames and
       hand them off to fixed analysis / plotting / persistence helpers.

The goal is to keep this file responsible for overall orchestration and
dataset plumbing (mapping async callables over collections), while keeping
the "scientific" logic in separate, testable modules that can be swapped via
CLI arguments.

All *pluggable* functions are expected to be async callables. The main
entrypoint is therefore also async and is wrapped via asyncio.run(...).

NOTE: The first draft deliberately avoids over-engineering around custom
row/data structures. We work with plain Python dicts and lists here and
can introduce lightweight dataclasses later once the interfaces stabilize.

Usage examples
--------------

Run with default settings (uses CLI defaults only)::

    uv run python -m experiments.re_sampling_stability.run_experiment

Run with a YAML configuration file (YAML overrides CLI defaults)::

    uv run python -m experiments.re_sampling_stability.run_experiment \
        --config-yaml experiments/re_sampling_stability/configs/minimal.yaml

Override a single field from the YAML via CLI (YAML still wins for
other fields)::

    uv run python -m experiments.re_sampling_stability.run_experiment \
        --config-yaml experiments/re_sampling_stability/configs/minimal.yaml \
        --candidate-model Qwen/Qwen2.5-7B-Instruct

Specify custom pluggable functions for transform / reasoning / scoring::

    uv run python -m experiments.re_sampling_stability.run_experiment \
        --transform-problems-fn \
          experiments.re_sampling_stability.transform.transform_problems \
        --generate-reasoning-trace-fn \
          experiments.re_sampling_stability.reasoning.generate_reasoning_traces_for_problem \
        --score-choice-labels-fn \
          experiments.re_sampling_stability.judgment.score_choice_labels_for_trace
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
from dataclasses import asdict
from typing import Any, Awaitable, Callable, List, Optional, Sequence

import chz
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from practical_deliberation_llms.datasets import sample_problems
from practical_deliberation_llms.inference import InferenceClient
from practical_deliberation_llms.io import save_results
from practical_deliberation_llms.plotting import plot_results

from .chz_config import LauncherConfig
from .config import ExperimentConfig, build_config, build_config_from_launcher


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


def setup_logging(verbose: bool = False) -> None:
    """Configure a basic logging setup for the CLI.

    Parameters
    ----------
    verbose:
        If True, use DEBUG level; otherwise INFO.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


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

    All pluggable functions are expected to be async and are awaited
    sequentially for now. If/when we introduce parallelism (e.g. via
    `asyncio.gather` on batches), this is the place to do it.
    """

    logger.info("Experiment configuration: %s", asdict(config))

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

    client = OpenAI(**client_kwargs)
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

    for idx, problem in enumerate(all_problems):
        if idx % 10 == 0:
            logger.debug(
                "Generating traces for problem %d / %d", idx, len(all_problems)
            )

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

    for idx, trace in enumerate(trace_records):
        if idx % 50 == 0:
            logger.debug("Scoring labels for trace %d / %d", idx, len(trace_records))

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


# ---------------------------------------------------------------------------
# CLI parsing and entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the experiment script.

    Parameters
    ----------
    argv:
        Optional sequence of argument strings. If omitted, `sys.argv[1:]`
        is used implicitly by `argparse`.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run judgment stability / re-sampling experiment with pluggable "
            "transform, reasoning, and scoring functions."
        )
    )

    # Optional YAML configuration
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help=(
            "Path to a YAML file with experiment configuration. "
            "Values from the YAML file override the corresponding CLI "
            "options for ExperimentConfig fields."
        ),
    )

    # Model and API settings
    parser.add_argument(
        "--candidate-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Name of the model under test (served via OpenAI-compatible API)",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Name of the assistant / transformation model (currently unused in this draft)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help=(
            "Optional override for OPENAI_BASE_URL. If omitted, the "
            "OpenAI client will use environment variables (which can be "
            "populated from a .env file)."
        ),
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help=(
            "Optional override for OPENAI_API_KEY. If omitted, the "
            "OpenAI client will use environment variables (which can be "
            "populated from a .env file)."
        ),
    )

    # Dataset and sampling. Dataset selection and adapter configuration are
    # provided via the YAML config file; the CLI only exposes the global
    # random seed here.
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling and other RNG-based steps",
    )

    # Generation parameters
    parser.add_argument(
        "--n-traces-per-problem",
        type=int,
        default=4,
        help="Number of reasoning traces to generate per problem",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for trace generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter for trace generation",
    )

    # Transformations
    parser.add_argument(
        "--max-transformations-per-problem",
        type=int,
        default=2,
        help=(
            "Upper bound on the number of transformed variants per problem. "
            "Exact semantics are up to the transform function."
        ),
    )

    # Analysis / IO
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/re_sampling_stability/results",
        help="Directory where results (and later plots) will be stored",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of plots (once implemented)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging output",
    )

    # Pluggable async functions (dotted paths)
    parser.add_argument(
        "--transform-problems-fn",
        type=str,
        default=None,
        help=(
            "Optional dotted path to async function transforming a single "
            "problem. If omitted, no transformations are applied. "
            "Signature: (config, problem) -> awaitable[sequence[problem]]."
        ),
    )
    parser.add_argument(
        "--generate-reasoning-trace-fn",
        type=str,
        default="experiments.re_sampling_stability.reasoning.generate_reasoning_traces_for_problem",
        help=(
            "Dotted path to async function generating traces for a problem. "
            "Signature: (config, inference_client, problem) -> awaitable[sequence[dict]]."
        ),
    )
    parser.add_argument(
        "--score-choice-labels-fn",
        type=str,
        default="experiments.re_sampling_stability.judgment.score_choice_labels_for_trace",
        help=(
            "Dotted path to async function scoring labels for a trace. "
            "Signature: (config, inference_client, trace_dict) -> awaitable[sequence[dict]]."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    This function parses arguments, sets up logging and seeds, resolves
    the pluggable async functions, builds the experiment configuration,
    and finally runs the async experiment via `asyncio.run`.
    """

    args = parse_args(argv)
    setup_logging(verbose=args.verbose)
    set_global_seeds(args.seed)

    # Resolve pluggable async functions from dotted paths.
    transform_problems_fn = None
    if args.transform_problems_fn:
        transform_problems_fn = resolve_callable(args.transform_problems_fn)
    generate_reasoning_trace_fn = resolve_callable(args.generate_reasoning_trace_fn)
    score_choice_labels_fn = resolve_callable(args.score_choice_labels_fn)

    config = build_config(args)

    asyncio.run(
        run_experiment_async(
            config=config,
            transform_problems_fn=transform_problems_fn,
            generate_reasoning_trace_fn=generate_reasoning_trace_fn,
            score_choice_labels_fn=score_choice_labels_fn,
        )
    )


def main_chz(launcher: LauncherConfig) -> None:
    """CLI entry point powered by :mod:`chz`.

    This variant parses a :class:`LauncherConfig` from the command line
    using :func:`chz.nested_entrypoint`, combines YAML configuration with
    CLI overrides via :mod:`chz`'s :class:`~chz.Blueprint`, and then runs
    :func:`run_experiment_async`.
    """

    setup_logging(verbose=False)

    config = build_config_from_launcher(launcher)

    # Seed RNGs based on the final configuration so that YAML / override
    # changes always affect the experiment deterministically.
    set_global_seeds(config.seed)

    transform_problems_fn: Optional[TransformProblemsFn] = None
    if config.transform_problems_fn:
        transform_problems_fn = resolve_callable(config.transform_problems_fn)

    generate_reasoning_trace_fn = resolve_callable(config.generate_reasoning_trace_fn)
    score_choice_labels_fn = resolve_callable(config.score_choice_labels_fn)

    asyncio.run(
        run_experiment_async(
            config=config,
            transform_problems_fn=transform_problems_fn,
            generate_reasoning_trace_fn=generate_reasoning_trace_fn,
            score_choice_labels_fn=score_choice_labels_fn,
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # NOTE: The :mod:`chz`-powered entrypoint is the default when running
    # this module as a script. Legacy consumers that rely on the
    # argparse-based CLI should continue to call :func:`main` directly.
    chz.nested_entrypoint(main_chz)
