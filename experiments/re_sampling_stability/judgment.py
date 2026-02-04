"""Judgment / scoring functions for re-sampling stability experiments.

This module defines the default async pluggable function
``score_choice_labels_for_trace`` used by ``run_experiment_async`` to
turn a single reasoning trace into label probability records.

Given a trace dict (as produced by
``generate_reasoning_traces_for_problem``), the scoring function
reconstructs a prompt that conditions on both the original dilemma and
the model's reasoning, then calls
``InferenceClient.score_label_given_trace`` to obtain a probability
distribution over labels.

The function returns a list of plain dicts, one per label, with at least
the following keys:

- ``problem_uid``: str
- ``trace_id``: str
- ``label``: str (e.g. "a")
- ``prob``: float, normalized so that the probabilities across labels
  sum to 1.0. Degenerate all-zero cases fall back to a uniform
  distribution.

Additionally, where present in the input trace, we propagate
``base_problem_uid``, ``transformation_type``,
``transformation_params``, ``source_dataset``, and ``source_id``. We
also attach ``candidate_model`` and ``assistant_model`` from the
experiment config for easier downstream analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

import logging
from jinja2 import Template

from practical_deliberation_llms.inference import InferenceClient


logger = logging.getLogger(__name__)


SCORE_PROMPT_TEMPLATE = Template(
    """You will evaluate how well each labeled option fits the reasoning
below about a practical dilemma. Your job is *not* to generate new
reasoning but to interpret the existing reasoning and decide which
option it most strongly supports.

Decision situation:
{{ decision_situation }}

Available actions:
{% for label, action in labeled_actions %}
- {{ label }}) {{ action }}
{% endfor %}

Reasoning (provided):
{{ think or "<no_think_block>" }}

Based only on this reasoning, output a single token corresponding to the
best-supported option label (one of: {{ labels|join(", ") }}).
"""
)


async def score_choice_labels_for_trace(
    config: Any, inference_client: InferenceClient, trace: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Score labels for a single reasoning trace.

    Parameters
    ----------
    config:
        Experiment configuration object. Used here for metadata (e.g.
        ``candidate_model``, ``assistant_model``) only.
    inference_client:
        ``InferenceClient`` wrapper used to obtain label probabilities
        given a fixed prompt.
    trace:
        Trace record as produced by
        ``generate_reasoning_traces_for_problem``.
    """

    actions = list(trace.get("actions", []))
    label_letters = [chr(ord("a") + i) for i in range(len(actions))]
    labeled_actions = list(zip(label_letters, actions))

    problem_uid = trace.get("problem_uid")
    trace_id = trace.get("trace_id")

    if not actions:
        logger.warning(
            "score_trace empty_actions problem_uid=%s trace_id=%s",
            problem_uid,
            trace_id,
        )

    decision_situation = trace.get("decision_situation", "")
    think = trace.get("think")

    logger.debug(
        "score_trace start problem_uid=%s trace_id=%s n_actions=%d",
        problem_uid,
        trace_id,
        len(actions),
    )

    prompt = SCORE_PROMPT_TEMPLATE.render(
        decision_situation=decision_situation,
        labeled_actions=labeled_actions,
        think=think,
        labels=label_letters,
    )

    label_probs = inference_client.score_label_given_trace(
        prompt=prompt, labels=label_letters
    )

    # Ensure we have a proper probability distribution; fall back to
    # uniform if the model failed to assign any mass to the candidate
    # labels.
    probs = [float(label_probs.get(label, 0.0)) for label in label_letters]
    total = sum(probs)
    if total <= 0.0:
        if label_letters:
            logger.warning(
                "score_trace uniform_fallback_zero_mass problem_uid=%s trace_id=%s labels=%s",
                problem_uid,
                trace_id,
                label_letters,
            )
        probs = [1.0 / len(label_letters)] * len(label_letters) if label_letters else []
    else:
        probs = [p / total for p in probs]

    records: List[Dict[str, Any]] = []
    for label, prob in zip(label_letters, probs):
        record: Dict[str, Any] = {
            "problem_uid": problem_uid,
            "trace_id": trace_id,
            "label": label,
            "prob": prob,
            "candidate_model": getattr(config, "candidate_model", None),
            "assistant_model": getattr(config, "assistant_model", None),
        }

        for attr in [
            "base_problem_uid",
            "transformation_type",
            "transformation_params",
            "source_dataset",
            "source_id",
        ]:
            if attr in trace:
                record[attr] = trace[attr]

        records.append(record)

    if probs:
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_label = label_letters[max_idx]
    else:
        best_label = None

    logger.info(
        "score_trace done problem_uid=%s trace_id=%s best_label=%s",
        problem_uid,
        trace_id,
        best_label,
    )

    logger.debug(
        "score_trace probs problem_uid=%s trace_id=%s probs=%s",
        problem_uid,
        trace_id,
        {label: prob for label, prob in zip(label_letters, probs)},
    )

    return records
