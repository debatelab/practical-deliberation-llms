"""chz-based configuration objects for re-sampling stability experiments.

This module provides the canonical configuration datamodel for
``experiments.re_sampling_stability`` based on :mod:`chz`.

The legacy :mod:`config` module re-exports the public types from here and
offers a small compatibility shim for callers that still construct
``ExperimentConfig`` via ``build_config(argparse.Namespace)``.
"""

from __future__ import annotations

from typing import Any

import chz
from chz import field
from chz import validate


@chz.chz
class DatasetSpec:
    """Configuration for a single dataset used in an experiment.

    Each dataset is identified by a logical name (key in
    ``ADAPTER_REGISTRY``) and may carry adapter-specific kwargs that are
    forwarded to the corresponding :class:`DatasetAdapter`.
    """

    name: str
    n_problems: int
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    @validate
    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Each dataset entry must define a non-empty 'name'")
        if not isinstance(self.n_problems, int) or self.n_problems <= 0:
            raise ValueError("Each dataset entry must define positive 'n_problems'")
        if not isinstance(self.adapter_kwargs, dict):
            raise ValueError("adapter_kwargs must be a mapping if provided")


@chz.chz
class ExperimentConfig:
    """Configuration for a single experiment run.

    This mirrors the high-level parameters described in
    ``experiments/re_sampling_stability/PLAN_COLAB_NOTEBOOK.md`` but is kept
    minimal on purpose. If you extend the CLI, add corresponding fields
    here so that downstream code can access them without additional
    parameters.
    """

    # Model and API settings
    candidate_model: str
    assistant_model: str
    # These are optional overrides. If left as None, the OpenAI client
    # (or callers using :attr:`effective_openai_base_url` /
    # :attr:`effective_api_token`) will read configuration from
    # environment variables (possibly populated via a .env file):
    # OPENAI_BASE_URL and OPENAI_API_KEY.
    openai_base_url: str | None = None
    api_token: str | None = None

    # Dataset and sampling
    seed: int
    datasets: list[DatasetSpec]

    # Generation parameters
    n_traces_per_problem: int
    temperature: float
    top_p: float

    # Transformations
    max_transformations_per_problem: int

    # Analysis / IO
    output_dir: str
    make_plots: bool

    # For reproducibility / logging it is useful to also store the string
    # representation of the pluggable functions that were used. The
    # transform function may be None to indicate that no transformation
    # step should be applied (i.e. we only use the base problems).
    transform_problems_fn: str | None
    generate_reasoning_trace_fn: str
    score_choice_labels_fn: str

    @chz.init_property
    def effective_openai_base_url(self) -> str | None:
        """Return the base URL, falling back to ``OPENAI_BASE_URL``.

        Callers that need the actual value to use for API calls should
        prefer this property over :attr:`openai_base_url` so that
        environment variables are honoured when no explicit override is
        provided.
        """

        import os

        return self.openai_base_url or os.getenv("OPENAI_BASE_URL")

    @chz.init_property
    def effective_api_token(self) -> str | None:
        """Return the API token, falling back to ``OPENAI_API_KEY``."""

        import os

        return self.api_token or os.getenv("OPENAI_API_KEY")

    @validate
    def _validate(self) -> None:
        """Basic sanity checks for experiment configuration.

        This mirrors the invariants previously enforced in the
        dataclasses-based configuration helpers.
        """

        if not isinstance(self.candidate_model, str) or not self.candidate_model:
            raise ValueError("candidate_model must be a non-empty string")
        if not isinstance(self.assistant_model, str) or not self.assistant_model:
            raise ValueError("assistant_model must be a non-empty string")

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        if not isinstance(self.datasets, list) or not self.datasets:
            raise ValueError("datasets must be a non-empty list")

        if (
            not isinstance(self.n_traces_per_problem, int)
            or self.n_traces_per_problem <= 0
        ):
            raise ValueError("n_traces_per_problem must be a positive integer")

        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise ValueError("temperature must be a positive number")

        if not isinstance(self.top_p, (int, float)) or not (0 < float(self.top_p) <= 1):
            raise ValueError("top_p must be in the interval (0, 1]")

        if (
            not isinstance(self.max_transformations_per_problem, int)
            or self.max_transformations_per_problem < 0
        ):
            raise ValueError(
                "max_transformations_per_problem must be a non-negative integer"
            )

        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")


# The chz CLI entrypoint uses these two helper classes. They are kept in
# this module so that all configuration-related types live in one place.


@chz.chz
class ExperimentOverrides:
    """Optional CLI overrides for :class:`ExperimentConfig`.

    All fields are optional. When provided, they override the corresponding
    values coming from a YAML configuration file.
    """

    candidate_model: str | None = None
    assistant_model: str | None = None
    openai_base_url: str | None = None
    api_token: str | None = None
    seed: int | None = None
    n_traces_per_problem: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_transformations_per_problem: int | None = None
    output_dir: str | None = None
    make_plots: bool | None = None
    transform_problems_fn: str | None = None
    generate_reasoning_trace_fn: str | None = None
    score_choice_labels_fn: str | None = None


@chz.chz
class LauncherConfig:
    """Top-level configuration used by the chz-powered CLI.

    ``config_yaml`` points to the YAML file from which we derive the base
    :class:`ExperimentConfig`. The nested ``overrides`` object can be used
    to override individual fields via the CLI while keeping most settings
    in YAML.
    """

    config_yaml: str
    overrides: ExperimentOverrides = field(default_factory=ExperimentOverrides)
