"""Plotting helpers for experiment metrics.

These functions produce simple diagnostic plots from the metrics DataFrame
produced by :mod:`experiments.re_sampling_stability.metrics`.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_results(
    output_dir: str,
    d_within_df: pd.DataFrame,
    baseline_vs_trans_df: pd.DataFrame,
) -> None:
    """Generate basic plots and save them as PNGs under ``output_dir``."""

    os.makedirs(output_dir, exist_ok=True)

    logger.info("plot_results output_dir=%s", output_dir)

    if not d_within_df.empty and "D_within" in d_within_df.columns:
        plt.figure(figsize=(10, 6))
        histplot_kwargs = {
            "data": d_within_df,
            "x": "D_within",
            "bins": 30,
            "kde": True,
        }
        if "source_dataset" in d_within_df.columns:
            histplot_kwargs["hue"] = "source_dataset"
        sns.histplot(**histplot_kwargs)
        plt.title("Within-context disagreement (D_within)")
        plt.xlabel("D_within")
        plt.tight_layout()
        within_path = os.path.join(output_dir, "within_disagreement_hist.png")
        plt.savefig(within_path)
        plt.close()
        logger.info("plot_results within_disagreement_hist path=%s", within_path)
    else:
        logger.warning("plot_results skipping_within_plot empty_or_missing_column")

    if (
        not baseline_vs_trans_df.empty
        and "D_KL_base_to_trans" in baseline_vs_trans_df.columns
    ):
        plt.figure(figsize=(6, 4))
        sns.histplot(baseline_vs_trans_df["D_KL_base_to_trans"], bins=30, kde=True)
        plt.title("KL divergence between baseline and transformed variants")
        plt.xlabel("D_KL_base_to_trans")
        plt.tight_layout()
        kl_path = os.path.join(output_dir, "baseline_vs_trans_kl_hist.png")
        plt.savefig(kl_path)
        plt.close()
        logger.info("plot_results baseline_vs_trans_kl_hist path=%s", kl_path)
    else:
        logger.warning("plot_results skipping_kl_plot empty_or_missing_column")
