from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from practical_deliberation_llms.inference import InferenceClient


class DummyChoiceLogprobs:
    def __init__(self, content: List[Any]) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, logprobs: DummyChoiceLogprobs) -> None:
        self.logprobs = logprobs
        # ``score_label_given_trace`` only accesses ``choice.logprobs``,
        # so we do not need to populate ``message``.


class DummyChatCompletion:
    def __init__(self, choices: List[DummyChoice]) -> None:
        self.choices = choices


class DummyChatCompletions:
    def __init__(self, response: DummyChatCompletion) -> None:
        self._response = response

    async def create(self, *args: Any, **kwargs: Any) -> DummyChatCompletion:  # type: ignore[override]
        return self._response


class DummyClient:
    def __init__(self, response: DummyChatCompletion) -> None:
        # Mimic ``client.chat.completions.create`` used in
        # ``score_label_given_trace``.
        self.chat = SimpleNamespace(completions=DummyChatCompletions(response))


@pytest.mark.asyncio
async def test_score_label_given_trace_handles_newline_label_json() -> None:
    """Ensure label-prefix detection works with pretty-printed JSON.

    This mirrors cases where the model emits::

        </think>{\n  "label": "a"\n}

    and we still want ``score_label_given_trace`` to locate the first
    label token and assign non-zero probability mass to it.
    """

    labels = ["a", "b"]
    reasoning = "Because reasons"

    # Tokens that reconstruct:
    #   <think>Because reasons</think>{\n  "label": "a"\n}
    tokens = [
        "<think>",
        reasoning,
        "</think>",
        "{\n  ",
        '"label"',
        ": ",
        '"',
        "a",  # first label token where we will attach top_logprobs
        '"',
        "\n}",
    ]

    content_entries: List[Any] = []
    for tok in tokens:
        if tok == "a":
            # Provide logprobs only at the first label token.
            top_logprobs = [
                TopLogprob(token="a", bytes=None, logprob=0.0),
                TopLogprob(token="b", bytes=None, logprob=-5.0),
            ]
        else:
            top_logprobs = []
        content_entries.append(SimpleNamespace(token=tok, top_logprobs=top_logprobs))

    response = DummyChatCompletion([DummyChoice(DummyChoiceLogprobs(content_entries))])
    dummy_client = DummyClient(response)  # type: ignore[arg-type]

    inference = InferenceClient(client=dummy_client, model="test-model")

    context_messages: List[Dict[str, str]] = [
        {"role": "user", "content": "prompt"},
    ]

    label_probs = await inference.score_label_given_trace(  # type: ignore[arg-type]
        context_messages=context_messages,
        reasoning=reasoning,
        labels=labels,
    )

    # We should assign essentially all mass to "a" and none to "b".
    assert label_probs["a"] > 0.0
    assert label_probs["b"] >= 0.0
    total = label_probs["a"] + label_probs["b"]
    assert total > 0.0
