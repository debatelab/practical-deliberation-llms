"""Metric computation helpers for re-sampling stability experiments.

This module mirrors the analysis steps from the draft notebook:

- Construct per-trace label distributions ``Q_i`` from score records.
- Compute within-context disagreement ``D_within`` per problem using
  ``util.within_context_disagreement``.
- Compute KL divergence between baseline and transformed variants using
  ``util.kl_divergence``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import logging
import numpy as np
import pandas as pd

from practical_deliberation_llms.util import kl_divergence, within_context_disagreement


logger = logging.getLogger(__name__)


def _scores_to_distributions(scores_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Pivot score records into per-trace distributions.

    Returns a tuple ``(dist_df, label_cols)`` where ``dist_df`` has one
    row per ``(problem_uid, trace_id)`` and columns for each label
    containing probabilities, and ``label_cols`` is the list of label
    column names in sorted order.
    """

    dist_df = scores_df.pivot_table(
        index=["problem_uid", "trace_id"],
        columns="label",
        values="prob",
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()

    label_cols = sorted(
        [c for c in dist_df.columns if c not in ("problem_uid", "trace_id")]
    )

    logger.debug(
        "_scores_to_distributions dist_shape=%s label_cols=%s",
        dist_df.shape,
        label_cols,
    )
    return dist_df, label_cols


def compute_metrics(
    trace_records: Iterable[Dict[str, Any]] | pd.DataFrame,
    score_records: Iterable[Dict[str, Any]] | pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-context disagreement and baseline-vs-transform metrics.

    Parameters
    ----------
    trace_records:
        Iterable of trace dicts or a DataFrame. Traces are expected to
        include at least ``problem_uid`` and the metadata fields attached
        to :class:`PracticalProblem` instances by the dataset adapter and
        transformation functions (e.g. ``source_dataset``, ``source_id``,
        ``base_problem_uid``, ``transformation_type``). For minimal or
        synthetic setups (e.g. certain unit tests), ``base_problem_uid``
        may be absent, in which case baseline-vs-transform metrics are
        skipped and an empty DataFrame is returned for that part.
    score_records:
        Iterable of score dicts or a DataFrame with at least
        ``problem_uid``, ``trace_id``, ``label``, and ``prob`` columns.
    """

    trace_df = (
        trace_records
        if isinstance(trace_records, pd.DataFrame)
        else pd.DataFrame(list(trace_records))
    )
    scores_df = (
        score_records
        if isinstance(score_records, pd.DataFrame)
        else pd.DataFrame(list(score_records))
    )

    if trace_df.empty or scores_df.empty:
        logger.warning(
            "compute_metrics empty_input trace_rows=%d score_rows=%d",
            len(trace_df),
            len(scores_df),
        )
        return pd.DataFrame(), pd.DataFrame()

    dist_df, label_cols = _scores_to_distributions(scores_df)

    # ------------------------------------------------------------------
    # Within-context disagreement per problem_uid
    # ------------------------------------------------------------------
    d_within_records: List[Dict[str, Any]] = []

    for problem_uid, group in dist_df.groupby("problem_uid"):
        dists: List[np.ndarray] = []
        for _, row in group[label_cols].iterrows():
            p_vec = np.asarray(row, dtype=float)
            if p_vec.sum() > 0:
                p_vec = p_vec / p_vec.sum()
            dists.append(p_vec)

        d_within = within_context_disagreement(dists)
        n_traces = len(dists)

        # Recover metadata from the first matching trace row if available.
        trace_rows = trace_df[trace_df["problem_uid"] == problem_uid]
        meta_row = trace_rows.iloc[0] if not trace_rows.empty else None

        record: Dict[str, Any] = {
            "problem_uid": problem_uid,
            "n_traces": n_traces,
            "D_within": float(d_within),
        }

        if meta_row is not None:
            for col in [
                "base_problem_uid",
                "transformation_type",
                "source_dataset",
                "source_id",
            ]:
                if col in trace_df.columns:
                    record[col] = meta_row.get(col)

        d_within_records.append(record)

    d_within_df = pd.DataFrame(d_within_records)

    logger.debug("compute_metrics d_within_rows=%d", len(d_within_df))

    # ------------------------------------------------------------------
    # Baseline vs transformed KL divergence per base_problem_uid
    # ------------------------------------------------------------------
    qbar_records: List[Dict[str, Any]] = []

    for problem_uid, group in dist_df.groupby("problem_uid"):
        trace_rows = trace_df[trace_df["problem_uid"] == problem_uid]
        meta_row = trace_rows.iloc[0] if not trace_rows.empty else None

        mean_vec = np.asarray(group[label_cols].mean(axis=0), dtype=float)
        if mean_vec.sum() > 0:
            mean_vec = mean_vec / mean_vec.sum()

        record: Dict[str, Any] = {
            "problem_uid": problem_uid,
            "Q_bar": mean_vec,
        }

        if meta_row is not None:
            for col in [
                "base_problem_uid",
                "transformation_type",
                "source_dataset",
                "source_id",
            ]:
                if col in trace_df.columns:
                    record[col] = meta_row.get(col)

        qbar_records.append(record)

    qbar_df = pd.DataFrame(qbar_records)

    transform_pairs: List[Dict[str, Any]] = []

    # Group by base_problem_uid to find baseline and transformed variants.
    #
    # In some minimal or synthetic setups (e.g. unit tests) the trace
    # records may not carry a ``base_problem_uid`` column at all. In that
    # case we skip baseline-vs-transform metrics and return an empty
    # DataFrame instead of raising a KeyError.
    if "base_problem_uid" not in qbar_df.columns:
        logger.warning(
            "compute_metrics no_base_problem_uid_column; "
            "baseline_vs_trans_df will be empty"
        )
    else:
        for base_uid, group in qbar_df.groupby("base_problem_uid"):
            if base_uid is None or (isinstance(base_uid, float) and np.isnan(base_uid)):
                continue

            baseline_rows = qbar_df[qbar_df["problem_uid"] == base_uid]
            if baseline_rows.empty:
                logger.warning(
                    "compute_metrics missing_baseline_for_base_uid base_uid=%s",
                    base_uid,
                )
                continue

            qbar_base = baseline_rows.iloc[0]["Q_bar"]

            for _, row in group.iterrows():
                # Skip the baseline itself if it appears in the group.
                if row["problem_uid"] == base_uid:
                    continue

                qbar_base_arr = np.asarray(qbar_base, dtype=float)
                qbar_trans_arr = np.asarray(row["Q_bar"], dtype=float)

                d_kl = kl_divergence(qbar_base_arr, qbar_trans_arr)

                pair_record: Dict[str, Any] = {
                    "base_problem_uid": base_uid,
                    "transformed_problem_uid": row["problem_uid"],
                    "D_KL_base_to_trans": float(d_kl),
                }

                for col in ["source_dataset", "source_id", "transformation_type"]:
                    if col in row:
                        pair_record[col] = row[col]

                transform_pairs.append(pair_record)

    baseline_vs_trans_df = pd.DataFrame(transform_pairs)

    logger.info(
        "compute_metrics done d_within_rows=%d baseline_vs_trans_rows=%d",
        len(d_within_df),
        len(baseline_vs_trans_df),
    )

    return d_within_df, baseline_vs_trans_df
