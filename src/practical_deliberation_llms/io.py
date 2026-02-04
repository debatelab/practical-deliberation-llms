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
    if series.isna().all() or all(
        (isinstance(v, dict) and not v) or v is None for v in series
    ):
        return df.drop(columns=["transformation_params"])

    # Otherwise, convert to JSON strings for robust parquet serialization.
    new_df = df.copy()
    new_df["transformation_params"] = series.apply(
        lambda v: json.dumps(v)
        if isinstance(v, dict)
        else ("null" if v is None else str(v))
    )
    return new_df


def save_results(
    config: Any,
    output_dir: str,
    trace_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    d_within_df: pd.DataFrame,
    baseline_vs_trans_df: pd.DataFrame,
) -> None:
    """Save configuration and result tables to ``output_dir``.

    Files written (all optional based on non-emptiness of the inputs):

    - ``config.json``: JSON-serialized ``asdict(config)`` if possible.
    - ``traces.parquet`` / ``scores.parquet``: raw trace/score tables.
    - ``metrics_within.parquet``: within-context disagreement metrics.
    - ``metrics_baseline_vs_trans.parquet``: baseline vs transformed KL.
    """

    os.makedirs(output_dir, exist_ok=True)

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
        trace_df = _sanitize_for_parquet(trace_df)
        trace_path = os.path.join(output_dir, "traces.parquet")
        trace_df.to_parquet(trace_path)
        logger.info("save_results traces path=%s rows=%d", trace_path, len(trace_df))
    if not scores_df.empty:
        scores_path = os.path.join(output_dir, "scores.parquet")
        scores_df.to_parquet(scores_path)
        logger.info("save_results scores path=%s rows=%d", scores_path, len(scores_df))
    if not d_within_df.empty:
        within_path = os.path.join(output_dir, "metrics_within.parquet")
        d_within_df.to_parquet(within_path)
        logger.info(
            "save_results metrics_within path=%s rows=%d", within_path, len(d_within_df)
        )
    if not baseline_vs_trans_df.empty:
        baseline_path = os.path.join(output_dir, "metrics_baseline_vs_trans.parquet")
        baseline_vs_trans_df.to_parquet(baseline_path)
        logger.info(
            "save_results metrics_baseline_vs_trans path=%s rows=%d",
            baseline_path,
            len(baseline_vs_trans_df),
        )
