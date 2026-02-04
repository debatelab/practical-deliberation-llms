"""Experiment configuration utilities (legacy shim around :mod:`chz_config`).

Historically this module defined ``dataclasses``-based configuration objects and
performed manual merging of CLI arguments with a YAML configuration file via
``build_config``.

Configuration is now modelled using :mod:`chz` in :mod:`chz_config`. This module
re-exports the public types and keeps ``build_config`` as a thin compatibility
helper used in a few tests and scripts.
"""

from __future__ import annotations

from typing import Any
import argparse
import os
import sys

import chz

from .chz_config import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentOverrides,
    LauncherConfig,
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
            "PyYAML is required to use --config-yaml, but it is not "
            "installed. Install it with `pip install pyyaml` or remove "
            "the --config-yaml option."
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


def _build_experiment_config_from_yaml_and_overrides(
    cfg_from_yaml: dict[str, Any], overrides: ExperimentOverrides
) -> ExperimentConfig:
    """Combine YAML mapping and overrides into an :class:`ExperimentConfig`.

    YAML provides the base configuration; non-``None`` fields on
    ``overrides`` take precedence. The final configuration object is
    normalised such that an empty string for ``transform_problems_fn`` is
    treated as ``None``.
    """

    _normalize_and_validate_datasets(cfg_from_yaml)

    blueprint = chz.Blueprint(ExperimentConfig)
    blueprint.apply(cfg_from_yaml)

    override_values = chz.asdict(overrides)
    # Apply non-None overrides on top of the YAML configuration.
    override_payload: dict[str, Any] = {}
    for key, value in override_values.items():
        if value is None:
            continue
        override_payload[key] = value

    if override_payload:
        blueprint.apply(override_payload)

    config = blueprint.make()

    # Normalise empty strings for transform_problems_fn to None. This
    # preserves the legacy behaviour where either YAML or CLI can disable
    # the transform hook via an empty string.
    if (
        isinstance(config.transform_problems_fn, str)
        and not config.transform_problems_fn.strip()
    ):
        config = chz.replace(config, transform_problems_fn=None)

    return config


def build_config_from_launcher(launcher: LauncherConfig) -> ExperimentConfig:
    """Construct an :class:`ExperimentConfig` from a :class:`LauncherConfig`.

    The launcher specifies the YAML configuration file and an
    :class:`ExperimentOverrides` object. YAML provides the base
    configuration; non-``None`` overrides take precedence. The resulting
    configuration mirrors the behaviour of :func:`build_config` but
    without relying on ``argparse``.
    """

    cfg_from_yaml = _load_yaml_config(launcher.config_yaml)
    return _build_experiment_config_from_yaml_and_overrides(
        cfg_from_yaml, launcher.overrides
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Construct an :class:`ExperimentConfig` from CLI args and optional YAML.

    This is a legacy helper used by tests and some scripts. New code should
    prefer :mod:`chz`-based configuration wiring in ``run_experiment``.
    """

    cfg_from_yaml = _load_yaml_config(getattr(args, "config_yaml", None))

    # Helper to preserve the original precedence rules used in the argparse
    # based CLI. For fields that are exposed as CLI flags, we only treat the
    # CLI value as an override if the corresponding flag is present on
    # ``sys.argv``; otherwise the YAML value (if any) wins.

    argv = sys.argv[1:]

    def was_flag_provided(cli_flag: str) -> bool:
        return any((arg == cli_flag) or arg.startswith(cli_flag + "=") for arg in argv)

    def resolve_field(field_name: str, cli_value: Any, cli_flag: str) -> Any:
        if was_flag_provided(cli_flag):
            return cli_value
        if field_name in cfg_from_yaml:
            return cfg_from_yaml[field_name]
        return cli_value

    overrides = ExperimentOverrides(
        candidate_model=resolve_field(
            "candidate_model", args.candidate_model, "--candidate-model"
        ),
        assistant_model=resolve_field(
            "assistant_model", args.assistant_model, "--assistant-model"
        ),
        openai_base_url=resolve_field(
            "openai_base_url", args.openai_base_url, "--openai-base-url"
        ),
        api_token=resolve_field("api_token", args.api_token, "--api-token"),
        seed=int(resolve_field("seed", args.seed, "--seed")),
        n_traces_per_problem=int(
            resolve_field(
                "n_traces_per_problem",
                args.n_traces_per_problem,
                "--n-traces-per-problem",
            )
        ),
        temperature=float(
            resolve_field("temperature", args.temperature, "--temperature")
        ),
        top_p=float(resolve_field("top_p", args.top_p, "--top-p")),
        max_transformations_per_problem=int(
            resolve_field(
                "max_transformations_per_problem",
                args.max_transformations_per_problem,
                "--max-transformations-per-problem",
            )
        ),
        output_dir=resolve_field("output_dir", args.output_dir, "--output-dir"),
    )

    # Fields that are not part of ExperimentOverrides are handled directly on
    # the resulting ExperimentConfig: ``make_plots`` is derived from
    # ``--no-plots`` and the pluggable function paths follow the original
    # precedence rules (YAML fallback to CLI default).

    config = _build_experiment_config_from_yaml_and_overrides(cfg_from_yaml, overrides)

    make_plots = not getattr(args, "no_plots", False)
    generate_reasoning_trace_fn = cfg_from_yaml.get(
        "generate_reasoning_trace_fn", args.generate_reasoning_trace_fn
    )
    score_choice_labels_fn = cfg_from_yaml.get(
        "score_choice_labels_fn", args.score_choice_labels_fn
    )

    config = chz.replace(
        config,
        make_plots=make_plots,
        generate_reasoning_trace_fn=generate_reasoning_trace_fn,
        score_choice_labels_fn=score_choice_labels_fn,
    )

    return config
