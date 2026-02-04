"""Inference utilities wrapping the OpenAI-compatible client.

This module provides the `InferenceClient` used in the Colab notebook to:

- Generate reasoning traces and JSON label answers via `chat.completions`.
- Score labels given a fixed reasoning trace via `completions` with logprobs.

NOTE/TODO: The project has standardized on the chat API (`/v1/chat/completions`).
`score_label_given_trace` still uses the legacy `/v1/completions` endpoint and
should be adapted to use `chat.completions` (with logprobs) once the scoring
pipeline is updated accordingly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import logging
from openai import OpenAI

from .grammar import make_structured_label_grammar
from .util import (
    logprobs_to_label_probs,
    parse_think_and_label,
    extract_label_from_json,
)


logger = logging.getLogger(__name__)


class InferenceClient:
    """Thin wrapper around an OpenAI-compatible client.

    - `generate_trace` uses `chat.completions.create` to sample `<think>`
      reasoning plus a JSON label answer.
    - `score_label_given_trace` currently uses `completions.create` with
      `logprobs=True` to obtain probabilities for candidate labels given a
      fixed prompt.

    TODO: Migrate `score_label_given_trace` to use `chat.completions.create`
    in line with the project-wide decision to avoid `/v1/completions`.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_trace(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """Generate a single reasoning trace and label choice.

        Returns a dict with keys:

        - `raw_response`: the full client response object
        - `content`: raw text content from the assistant
        - `think`: extracted `<think>...</think>` block (if any)
        - `label_json`: extracted JSON-like substring for the label answer
        - `label`: parsed `label` value (or `None` on failure)
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "generate_trace call model=%s temperature=%.3f top_p=%.3f seed=%s",
            self.model,
            temperature,
            top_p,
            seed,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=1,
        )

        content = response.choices[0].message.content
        think_text, label_json_str = parse_think_and_label(content)
        label = extract_label_from_json(label_json_str) if label_json_str else None

        if think_text is None or label_json_str is None or label is None:
            logger.debug(
                "generate_trace parse_issues think_missing=%s label_json_missing=%s label_missing=%s",
                think_text is None,
                label_json_str is None,
                label is None,
            )

        logger.debug(
            "generate_trace result think_len=%d content_len=%d",
            len(think_text or ""),
            len(content or ""),
        )

        return {
            "raw_response": response,
            "content": content,
            "think": think_text,
            "label_json": label_json_str,
            "label": label,
        }

    def score_label_given_trace(
        self,
        context_messages: List[dict],
        reasoning: str,
        labels: List[str],
    ) -> Dict[str, float]:
        """Score ``P(label | context, reasoning)`` using structured_outputs.

        This method calls the OpenAI chat completions endpoint with
        ``logprobs=True`` and a structured_outputs grammar that forces the
        assistant to emit ``<think>{reasoning}</think>{"label": "<LABEL>"}``,
        where ``<LABEL>`` must be one of the provided ``labels``. It then
        extracts the token-level logprobs at the *first* label token and
        converts them into label probabilities via
        :func:`logprobs_to_label_probs`.
        """

        logger.debug(
            "score_label_given_trace call model=%s n_labels=%d reasoning_len=%d",
            self.model,
            len(labels),
            len(reasoning or ""),
        )

        logger.debug("score_label_given_trace context_messages=%s", context_messages)

        extra_body = make_structured_label_grammar(reasoning, labels)
        logger.debug("score_label_given_trace extra_body=%s", extra_body)

        # Conservative upper bound for generated tokens: reasoning plus
        # tags and a small JSON object. This intentionally over-allocates
        # a bit to avoid truncation.
        max_tokens = max(len(reasoning) // 2 + 32, 64)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=context_messages,
            temperature=0.0,
            top_p=1.0,
            logprobs=True,
            top_logprobs=max(5, len(labels)),
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        logger.debug("score_label_given_trace raw_response=%s", response)

        choice = response.choices[0]
        message_logprobs = getattr(choice, "logprobs", None)

        if not message_logprobs or not getattr(message_logprobs, "content", None):
            logger.warning("score_label_given_trace missing_logprobs choice=%s", choice)
            return {label: 0.0 for label in labels}

        # vLLM chat logprobs format: logprobs.content is a list of
        # per-token entries, each with .top_logprobs containing token
        # logprob records. We reconstruct the generated text incrementally
        # and locate the step where the label value begins.
        content_entries = message_logprobs.content
        logger.debug("score_label_given_trace logprobs_content=%s", content_entries)
        accum = ""
        label_prefix = '{"label": "'
        step_label_start = None

        for idx, entry in enumerate(content_entries):
            token = entry.token  # type: ignore[attr-defined]
            accum += token
            if accum.endswith(label_prefix):
                step_label_start = idx + 1
                break

        if step_label_start is None or step_label_start >= len(content_entries):
            logger.warning(
                "score_label_given_trace could_not_locate_label_token prefix_seen=%s accum_tail=%r",
                label_prefix in accum,
                accum[-80:],
            )
            return {label: 0.0 for label in labels}

        top_logprobs_step = content_entries[step_label_start].top_logprobs  # type: ignore[attr-defined]
        label_probs = logprobs_to_label_probs(top_logprobs_step, labels)

        logger.debug("score_label_given_trace label_probs=%s", label_probs)

        return label_probs
