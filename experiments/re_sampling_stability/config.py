"""chz-based configuration and helpers for re-sampling stability experiments.

This module provides the canonical configuration datamodel for
``experiments.re_sampling_stability`` based on :mod:`chz`, as well as
helpers for constructing :class:`ExperimentConfig` instances from YAML
configuration files.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import chz
from chz import field, validate


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
    seed: int = 1234
    datasets: list[DatasetSpec]

    # Generation parameters
    n_traces_per_problem: int
    temperature: float
    top_p: float

    # Concurrency
    max_concurrency: int = 64

    # Transformations
    max_transformations_per_problem: int

    # Analysis / IO
    output_dir: str
    make_plots: bool = False
    # Output format for tabular results written by the experiment pipeline.
    # Supported values:
    #   - "jsonl": write newline-delimited JSON files with *.jsonl suffix (default).
    #   - "parquet": write *.parquet files.
    results_file_format: str = "jsonl"

    # Logging
    # Default to INFO to avoid overly verbose logs for typical runs.
    log_level: str = "INFO"

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

        if not isinstance(self.max_concurrency, int) or self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")

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

        allowed_formats = {"parquet", "jsonl"}
        if self.results_file_format not in allowed_formats:
            raise ValueError(
                "results_file_format must be one of "
                f"{sorted(allowed_formats)}, got {self.results_file_format!r}"
            )

        # Validate log_level: must be a standard logging level name.
        if not isinstance(self.log_level, str):
            raise ValueError("log_level must be a string")

        level_name = self.log_level.upper()
        level_value = getattr(logging, level_name, None)
        if not isinstance(level_value, int):
            raise ValueError(
                "log_level must be a standard logging level name like "
                "'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'; "
                f"got {self.log_level!r}"
            )


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load a YAML mapping from ``path`` or return an empty mapping.

    The legacy experiment setup assumes that a YAML file is always provided
    and that it defines a non-empty ``datasets`` list. We keep that
    assumption here to avoid surprising changes in behavior.
    """

    cfg_from_yaml: dict[str, Any] = {}

    if not path:
        return cfg_from_yaml

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load experiment configuration from YAML, "
            "but it is not installed. Install it with `pip install pyyaml`."
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        cfg_from_yaml = {}
    elif isinstance(loaded, dict):
        cfg_from_yaml = loaded
    else:
        raise ValueError(
            "Config YAML must contain a mapping at the top level; "
            f"got {type(loaded)!r} instead."
        )

    return cfg_from_yaml


def _normalize_and_validate_datasets(cfg_from_yaml: dict[str, Any]) -> None:
    """Ensure that ``datasets`` is present and non-empty.

    We rely on :class:`DatasetSpec` validation for the per-entry checks.
    """

    raw_datasets = cfg_from_yaml.get("datasets")
    if not raw_datasets:
        raise ValueError("Config YAML must define a non-empty 'datasets' list")

    # Convert YAML dataset entries (list of mappings) into DatasetSpec
    # instances so that the Blueprint sees values with the correct type.
    cfg_from_yaml["datasets"] = [DatasetSpec(**ds) for ds in raw_datasets]


@chz.chz
class CLIConfig:
    """CLI-facing configuration used by the chz-powered entrypoint.

    ``config_yaml`` points to the YAML file from which we derive the base
    :class:`ExperimentConfig`. The remaining fields are optional CLI
    overrides. When provided, they override the corresponding values
    coming from YAML.
    """

    config_yaml: str

    candidate_model: str | None = None
    assistant_model: str | None = None
    openai_base_url: str | None = None
    api_token: str | None = None
    seed: int | None = None
    n_traces_per_problem: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_concurrency: int | None = None
    max_transformations_per_problem: int | None = None
    output_dir: str | None = None
    make_plots: bool | None = None
    results_file_format: str | None = None
    transform_problems_fn: str | None = None
    generate_reasoning_trace_fn: str | None = None
    score_choice_labels_fn: str | None = None

    # Logging override (e.g. "DEBUG", "INFO", ...). When provided, this
    # overrides the value from the YAML configuration.
    log_level: str | None = None


def build_config_from_cli(cli_cfg: CLIConfig) -> ExperimentConfig:
    """Resolve final :class:`ExperimentConfig` from a CLI-provided :class:`CLIConfig`.

    YAML (``config_yaml``) is used as the base configuration; any non-``None``
    fields on ``cli_cfg`` are applied on top via :class:`~chz.Blueprint`.
    """

    cfg_from_yaml = _load_yaml_config(cli_cfg.config_yaml)
    _normalize_and_validate_datasets(cfg_from_yaml)

    blueprint = chz.Blueprint(ExperimentConfig)
    blueprint.apply(cfg_from_yaml)

    override_payload: dict[str, Any] = {}
    cli_values = chz.asdict(cli_cfg)
    # Remove config_yaml, which is only meaningful for loading YAML.
    cli_values.pop("config_yaml", None)
    for key, value in cli_values.items():
        if value is None:
            continue
        override_payload[key] = value

    if override_payload:
        blueprint.apply(override_payload)

    config = blueprint.make()

    if (
        isinstance(config.transform_problems_fn, str)
        and not config.transform_problems_fn.strip()
    ):
        config = chz.replace(config, transform_problems_fn=None)

    return config
