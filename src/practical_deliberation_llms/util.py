"""Utility functions for logprob processing and disagreement metrics.

This module collects small, model-agnostic helpers used across the
project and in the Colab notebook, including:

- Converting token-level logprobs into label-level probabilities.
- Parsing `<think>...</think>` fences and JSON label answers.
- Computing KL divergence and within-context disagreement metrics.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from openai.types.chat.chat_completion_token_logprob import TopLogprob

logger = logging.getLogger(__name__)


def logprobs_to_label_probs(
    logprob_records: Iterable[TopLogprob], labels: List[str]
) -> Dict[str, float]:
    """Convert token-level logprobs to a label->probability mapping.

    Parameters
    ----------
    logprob_records:
        Iterable of OpenAI ``TopLogprob`` objects for a single token
        position, as returned by ``chat.completions`` with
        ``logprobs=True``.
    labels:
        Candidate label strings (e.g. ``["a", "b"]``) that we want to
        assign probabilities to.

    Returns
    -------
    dict[str, float]
        Mapping from label to (unnormalized) probability mass, where
        probabilities are derived via a softmax over the provided
        logprobs and then summed for tokens matching each label.
    """

    records = list(logprob_records)
    if not records or not labels:
        return {label: 0.0 for label in labels}

    lower_labels = [l.lower() for l in labels]

    # We keep an intermediate mutable representation annotated with
    # inferred label and probability for clarity when aggregating.
    enriched: List[Dict[str, Any]] = []
    for rec in records:
        raw_token = rec.token
        token_stripped = raw_token.strip().strip('"').lower()

        # First pass: exact match against labels.
        exact_matches = [label for label in lower_labels if token_stripped == l]
        if len(exact_matches) == 1:
            label = labels[lower_labels.index(exact_matches[0])]
        else:
            # Second pass: substring match, mirroring original logic.
            token_lower = raw_token.lower()
            substring_matches = [label for label in lower_labels if l in token_lower]
            if len(substring_matches) == 1:
                label = labels[lower_labels.index(substring_matches[0])]
            else:
                label = None

        enriched.append(
            {"token": raw_token, "logprob": float(rec.logprob), "label": label}
        )

    logprobs = np.array([float(rec["logprob"]) for rec in enriched], dtype=float)
    # Numerically stable softmax.
    max_lp = np.max(logprobs)
    probs = np.exp(logprobs - max_lp)
    probs = probs / probs.sum()

    for rec, prob in zip(enriched, probs):
        rec["prob"] = float(prob)

    label_probs: Dict[str, float] = {label: 0.0 for label in labels}
    for rec in enriched:
        label = rec["label"]
        if label in label_probs:
            label_probs[label] += float(rec["prob"])

    return label_probs


THINK_REGEX = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
LABEL_JSON_REGEX = re.compile(r"\{[\s\S]*?\"label\"[\s\S]*?\}")


def parse_think_and_label(text: str | None) -> Tuple[str | None, str | None]:
    """Extract `<think>...</think>` content and a JSON-like label snippet.

    Returns a pair ``(think_text, label_json_str)``, where either element
    may be ``None`` if parsing fails. When multiple matches are present,
    the *last* occurrence is returned, which matches the notebook
    behaviour of using the most recent answer.
    """

    if text is None:
        return None, None

    think_matches = list(THINK_REGEX.finditer(text))
    think_text: str | None
    if think_matches:
        think_text = think_matches[-1].group(1).strip()
    else:
        think_text = None

    label_match = None
    for match in LABEL_JSON_REGEX.finditer(text):
        label_match = match

    label_json: str | None
    if label_match is not None:
        label_json = label_match.group(0).strip()
    else:
        label_json = None

    if think_text is None or label_json is None:
        logger.debug(
            "parse_think_and_label missing_parts think_missing=%s label_json_missing=%s",
            think_text is None,
            label_json is None,
        )

    return think_text, label_json


def extract_label_from_json(json_str: str) -> str | None:
    """Parse a label value from a JSON-like string.

    The function tries ``json.loads`` first and falls back to a simple
    regex that searches for a ``"label"`` field. On any failure, it
    returns ``None`` instead of raising.
    """

    if not json_str:
        logger.debug("extract_label_from_json empty_input")
        return None

    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict) and "label" in obj:
            val = obj["label"]
            return str(val) if val is not None else None
    except json.JSONDecodeError:
        logger.debug("extract_label_from_json json_decode_error")

    # Fallback: regex the label out of the string.
    match = re.search(r"\"label\"\s*:\s*\"(.*?)\"", json_str)
    if match:
        return match.group(1)

    logger.debug("extract_label_from_json regex_failed")
    return None


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute a numerically safe KL divergence ``KL(p || q)``.

    Both ``p`` and ``q`` are interpreted as probability vectors; they are
    normalized internally and clipped to ``[eps, 1]`` to avoid
    ``log(0)``. The result is returned as a Python ``float``.
    """

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("p and q must be 1D arrays of the same shape")

    # Normalize in case inputs are not perfectly normalized.
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))


def within_context_disagreement(dists: List[np.ndarray]) -> float:
    """Compute within-context disagreement ``D_within`` over distributions.

    The metric is defined as::

        D_within = (1/n) * sum_i KL(Q_i || Q_bar)

    where ``Q_i`` are the input distributions and ``Q_bar`` is their
    mean distribution. This mirrors the definition used in the Colab
    notebook and `PLAN_COLAB_NOTEBOOK.md`.
    """

    if not dists:
        return 0.0

    arr = np.stack([np.asarray(d, dtype=float) for d in dists], axis=0)
    # Compute mean distribution and normalize for safety.
    q_bar = arr.mean(axis=0)
    if q_bar.sum() > 0:
        q_bar = q_bar / q_bar.sum()

    n = arr.shape[0]
    total = 0.0
    for i in range(n):
        total += kl_divergence(arr[i], q_bar)

    return float(total / n)
