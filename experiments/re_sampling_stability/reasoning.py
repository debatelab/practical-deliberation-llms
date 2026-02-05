"""Reasoning trace generation for re-sampling stability experiments.

This module defines the default async pluggable function
``generate_reasoning_traces_for_problem`` used by
``run_experiment_async``. It is responsible for turning a single
``PracticalProblem`` into a list of trace records by calling
``InferenceClient.generate_trace`` multiple times with prompts derived
from the problem.

Each returned trace record is a plain ``dict`` with (at least) the
following keys:

- ``problem_uid``: str, UID of the (possibly transformed) problem
- ``trace_id``: str, stable identifier for this trace
- ``decision_situation``: str
- ``actions``: list[str]
- ``think``: str | None, extracted <think>...</think> content
- ``label_json``: str | None, raw JSON-like answer substring
- ``label``: str | None, parsed label chosen by the model

Where available on the input problem, we also propagate:

- ``base_problem_uid``
- ``transformation_type``
- ``transformation_params``
- ``source_dataset``
- ``source_id``

This lightweight schema mirrors the informal contract in the notebook
while keeping the door open to introduce small dataclasses later.
"""

from __future__ import annotations

from typing import Any, Dict, List

import logging
from jinja2 import Template

from practical_deliberation_llms.inference import InferenceClient
from practical_deliberation_llms.model import PracticalProblem


logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = Template(
    """You are a careful deliberative assistant.

You will reason step by step inside <think></think> tags before
committing to a final choice among the available options. After
thinking, you will output a short JSON object of the form:

{"label": "a"}

where the label is one of: {{ labels|join(", ") }}.
"""
)


USER_PROMPT_TEMPLATE = Template(
    """You are considering the following situation.

Decision situation:
{{ decision_situation }}

Available actions:
{% for label, action in labeled_actions %}
- {{ label }}) {{ action }}
{% endfor %}

Think carefully about the pros and cons of each action inside
<think>...</think> tags. After you finish thinking, output a JSON object
of the form {"label": "a"} indicating which labeled option you judge
to be best.
"""
)


async def generate_reasoning_traces_for_problem(
    config: Any, inference_client: InferenceClient, problem: PracticalProblem
) -> List[Dict[str, Any]]:
    """Generate reasoning traces for a single problem.

    Parameters
    ----------
    config:
        Experiment configuration object. Must expose at least
        ``n_traces_per_problem``, ``temperature``, ``top_p``, and a
        numeric ``seed``.
    inference_client:
        ``InferenceClient`` wrapper around an OpenAI-compatible API.
    problem:
        ``PracticalProblem`` instance describing the decision situation
        and available actions.
    """

    # Map actions to simple letter labels ("a", "b", ...).
    actions = list(problem.actions)
    label_letters = [chr(ord("a") + i) for i in range(len(actions))]
    labeled_actions = list(zip(label_letters, actions))

    system_prompt = SYSTEM_PROMPT_TEMPLATE.render(labels=label_letters)
    user_prompt = USER_PROMPT_TEMPLATE.render(
        decision_situation=problem.decision_situation,
        labeled_actions=labeled_actions,
    )

    traces: List[Dict[str, Any]] = []

    base_uid = getattr(problem, "problem_uid", None)
    seed_base = int(getattr(config, "seed", 0))
    n_traces = int(getattr(config, "n_traces_per_problem", 1))

    logger.debug(
        "generate_reasoning_traces problem_uid=%s n_traces=%d", base_uid, n_traces
    )

    for idx in range(n_traces):
        # Derive a per-trace seed in a reproducible way, without relying
        # on loop ordering across potential future concurrency models.
        per_trace_seed = (
            seed_base + (hash(base_uid) % 1_000_000 if base_uid else 0) + idx
        )

        logger.debug(
            "generate_reasoning_traces call idx=%d seed=%d problem_uid=%s",
            idx,
            per_trace_seed,
            base_uid,
        )

        trace_info = await inference_client.generate_trace(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(config, "temperature", 0.7)),
            top_p=float(getattr(config, "top_p", 0.95)),
            seed=per_trace_seed,
        )

        trace_id = f"{base_uid or 'unknown_problem'}::trace_{idx}"

        record: Dict[str, Any] = {
            "problem_uid": base_uid,
            "trace_id": trace_id,
            "decision_situation": problem.decision_situation,
            "actions": actions,
            "think": trace_info.get("think"),
            "label_json": trace_info.get("label_json"),
            "label": trace_info.get("label"),
        }

        think_text = record.get("think")
        label_json = record.get("label_json")
        label = record.get("label")

        if think_text is None:
            logger.warning(
                "generate_reasoning_traces missing_think problem_uid=%s trace_id=%s",
                base_uid,
                trace_id,
            )
        if label_json is None:
            logger.warning(
                "generate_reasoning_traces missing_label_json problem_uid=%s trace_id=%s",
                base_uid,
                trace_id,
            )
        if label is None:
            logger.warning(
                "generate_reasoning_traces missing_label problem_uid=%s trace_id=%s",
                base_uid,
                trace_id,
            )

        logger.debug(
            "generate_reasoning_traces lengths problem_uid=%s trace_id=%s think_len=%d",
            base_uid,
            trace_id,
            len(think_text or ""),
        )

        # Propagate optional metadata fields if present on the problem.
        for attr in [
            "base_problem_uid",
            "transformation_type",
            "transformation_params",
            "source_dataset",
            "source_id",
        ]:
            if hasattr(problem, attr):
                record[attr] = getattr(problem, attr)

        traces.append(record)

    logger.info("generated_traces problem_uid=%s count=%d", base_uid, len(traces))

    return traces
