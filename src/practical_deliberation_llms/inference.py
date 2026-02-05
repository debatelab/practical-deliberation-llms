"""Inference utilities wrapping the OpenAI-compatible client.

This module provides the `InferenceClient` used in the Colab notebook to:

- Generate reasoning traces and JSON label answers via `chat.completions`.
- Score labels given a fixed reasoning trace via `chat.completions` with logprobs.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)

from .formats import LABEL_FIELD_NAME
from .grammar import make_structured_label_grammar
from .util import (
    LABEL_PREFIX_REGEX,
    extract_label_from_json,
    logprobs_to_label_probs,
    parse_think_and_label,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Thin wrapper around an OpenAI-compatible client.

    - `generate_trace` uses `chat.completions.create` to sample `<think>`
      reasoning plus a JSON label answer.
    - `score_label_given_trace` uses `chat.completions.create` with
      ``logprobs=True`` and a structured_outputs grammar to estimate
      ``P(label | context, reasoning)`` over a fixed set of candidate labels.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def generate_trace(
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

        messages: list[ChatCompletionMessageParam] = [
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

        response: ChatCompletion = await self.client.chat.completions.create(
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

    async def score_label_given_trace(
        self,
        context_messages: list[ChatCompletionMessageParam],
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

        response: ChatCompletion = await self.client.chat.completions.create(
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

        # Chat logprobs format: logprobs.content is a list of
        # per-token entries, each with ``.top_logprobs`` containing
        # `TopLogprob` records. We reconstruct the generated text
        # incrementally and locate the step where the label value begins.
        # The prefix we look for must stay in sync with the JSON schema
        # defined in ``formats.make_label_json_schema``.
        content_entries: List[ChatCompletionTokenLogprob] = message_logprobs.content
        logger.debug("score_label_given_trace logprobs_content=%s", content_entries)
        accum = ""
        # Allow for arbitrary whitespace between the opening brace,
        # the label field name (see ``LABEL_FIELD_NAME``), the colon,
        # and the opening quote of the value. This is robust to
        # formats like:
        #   {"label":"a"}
        #   { "label" : "a" }
        #   {\n  "label": "a"\n}
        step_label_start: int | None = None

        for idx, entry in enumerate(content_entries):
            token: str = entry.token
            accum += token
            tail = accum[-128:]
            # We only care whether the tail ends with a label prefix.
            # Using a regex here makes us insensitive to whitespace
            # choices in the JSON object produced by the model.
            if LABEL_PREFIX_REGEX.search(tail) and tail.rstrip().endswith('"'):
                step_label_start = idx + 1
                break

        if step_label_start is None or step_label_start >= len(content_entries):
            tail = accum[-80:]
            logger.warning(
                "score_label_given_trace could_not_locate_label_token accum_tail=%r",
                tail,
            )
            return {label: 0.0 for label in labels}

        token_step: ChatCompletionTokenLogprob = content_entries[step_label_start]
        top_logprobs_step: List[TopLogprob] = token_step.top_logprobs
        label_probs = logprobs_to_label_probs(top_logprobs_step, labels)

        logger.debug("score_label_given_trace label_probs=%s", label_probs)

        return label_probs
