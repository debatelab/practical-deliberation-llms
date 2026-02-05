"""IO helpers for saving experiment results to disk.

The main entry point is :func:`save_results`, which writes the
configuration and tabular result DataFrames to the output directory.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

import pandas as pd

try:  # optional dependency; used when available
    import chz  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional
    chz = None


logger = logging.getLogger(__name__)


def ensure_fresh_output_dir(output_dir: str) -> None:
    """Create ``output_dir``, failing if it already exists.

    This is used by experiment entry points to avoid accidentally
    overwriting existing results in a previously used directory.
    """

    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Output directory already exists: {output_dir!r}. "
            "Refuse to overwrite existing results."
        )

    os.makedirs(output_dir, exist_ok=False)


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` that is safe to write with pyarrow.

    In particular, pyarrow cannot handle struct-typed columns with no
    child fields (e.g. ``struct<transformation_params: struct<>``). This
    can happen when all values in a dict-like column are empty mappings.

    For now we special-case ``transformation_params`` and either drop it
    (if it is completely empty) or convert entries to JSON strings.
    """

    if "transformation_params" not in df.columns:
        return df

    series = df["transformation_params"]

    # If the column is entirely missing / null / empty dicts, drop it.
    is_all_na = bool(series.isna().all())
    if is_all_na or all((isinstance(v, dict) and not v) or v is None for v in series):
        return df.drop(columns=["transformation_params"])

    # Otherwise, convert to JSON strings for robust parquet serialization.
    new_df = df.copy()
    new_df["transformation_params"] = series.apply(
        lambda v: (
            json.dumps(v) if isinstance(v, dict) else ("null" if v is None else str(v))
        )
    )
    return new_df


def _write_table_single_format(
    df: pd.DataFrame,
    base_path: str,
    file_format: str,
) -> None:
    """Write a single non-empty DataFrame to disk in the requested format.

    ``base_path`` is the path without file extension; ``file_format`` is
    either ``"parquet"`` or ``"jsonl"``.
    """

    if df.empty:
        return

    if file_format == "parquet":
        df = _sanitize_for_parquet(df)
        path = base_path + ".parquet"
        df.to_parquet(path)
        logger.info("save_results parquet path=%s rows=%d", path, len(df))
    elif file_format == "jsonl":
        path = base_path + ".jsonl"
        df.to_json(path, orient="records", lines=True)
        logger.info("save_results jsonl path=%s rows=%d", path, len(df))
    else:
        raise ValueError(
            f"Unsupported results file format: {file_format!r} (expected 'parquet' or 'jsonl')"
        )


def save_results(
    config: Any,
    output_dir: str,
    trace_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    d_within_df: pd.DataFrame,
    baseline_vs_trans_df: pd.DataFrame,
    results_file_format: str = "jsonl",
) -> None:
    """Save configuration and result tables to ``output_dir``.

    Files written (all optional based on non-emptiness of the inputs):

    - ``config.json``: JSON-serialized ``asdict(config)`` if possible.
    - ``traces.parquet`` / ``scores.parquet`` or ``traces.jsonl`` /
      ``scores.jsonl``: raw trace/score tables.
    - ``metrics_within.parquet`` / ``metrics_within.jsonl``: within-context
      disagreement metrics.
    - ``metrics_baseline_vs_trans.parquet`` /
      ``metrics_baseline_vs_trans.jsonl``: baseline vs transformed KL.
    """

    logger.info("save_results output_dir=%s", output_dir)

    # Best-effort config serialization. We first try chz.asdict for
    # chz-based configuration objects, then fall back to
    # dataclasses.asdict, and finally to the object's __dict__.
    cfg_path = os.path.join(output_dir, "config.json")
    cfg_dict: dict[str, Any]
    if chz is not None:
        try:
            cfg_dict = chz.asdict(config)  # type: ignore[assignment]
        except TypeError:
            # Not a chz config; fall back to dataclasses / __dict__.
            try:
                cfg_dict = asdict(config)
            except TypeError:
                cfg_dict = getattr(config, "__dict__", {})
    else:
        try:
            cfg_dict = asdict(config)
        except TypeError:
            cfg_dict = getattr(config, "__dict__", {})

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)

    logger.debug(
        "save_results config_written path=%s trace_rows=%d score_rows=%d within_rows=%d baseline_vs_trans_rows=%d",
        cfg_path,
        len(trace_df),
        len(scores_df),
        len(d_within_df),
        len(baseline_vs_trans_df),
    )

    if not trace_df.empty:
        _write_table_single_format(
            trace_df,
            os.path.join(output_dir, "traces"),
            results_file_format,
        )

    if not scores_df.empty:
        _write_table_single_format(
            scores_df,
            os.path.join(output_dir, "scores"),
            results_file_format,
        )

    if not d_within_df.empty:
        _write_table_single_format(
            d_within_df,
            os.path.join(output_dir, "metrics_within"),
            results_file_format,
        )

    if not baseline_vs_trans_df.empty:
        _write_table_single_format(
            baseline_vs_trans_df,
            os.path.join(output_dir, "metrics_baseline_vs_trans"),
            results_file_format,
        )
