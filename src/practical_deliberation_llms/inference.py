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

from openai import OpenAI

from .util import (
    logprobs_to_label_probs,
    parse_think_and_label,
    extract_label_from_json,
)


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

        return {
            "raw_response": response,
            "content": content,
            "think": think_text,
            "label_json": label_json_str,
            "label": label,
        }

    def score_label_given_trace(
        self, prompt: str, labels: List[str]
    ) -> Dict[str, float]:
        """Score `P(label | prompt_with_reasoning)` using logprobs.

        Calls the OpenAI completions endpoint with `max_tokens=1` and
        `logprobs=True`, then converts the first-step `top_logprobs` into
        label probabilities using `logprobs_to_label_probs`.
        """

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=max(5, len(labels)),
        )

        choice = response.choices[0]

        # vLLM OpenAI-completions format: logprobs.top_logprobs is
        # list[list[dict]], where each dict has at least "token" and
        # "logprob" keys. We normalize to a flat list of records.
        logprobs_obj = choice.logprobs
        try:
            logprob_records = logprobs_obj.top_logprobs[0]
        except AttributeError:  # pragma: no cover - defensive fallback
            logprob_records = logprobs_obj[0] if isinstance(logprobs_obj, list) else []

        label_probs = logprobs_to_label_probs(logprob_records, labels)
        return label_probs
