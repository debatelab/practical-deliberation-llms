import math
from typing import List

import numpy as np
import pytest
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from practical_deliberation_llms.util import (
    extract_label_from_json,
    logprobs_to_label_probs,
    parse_think_and_label,
)


def make_toplogprob(token: str, logprob: float) -> TopLogprob:
    """Helper to construct a minimal TopLogprob instance for tests."""

    return TopLogprob(token=token, bytes=None, logprob=logprob)


def test_logprobs_to_label_probs_exact_match_and_softmax():
    # Two labels, exact token matches. We expect a proper softmax over
    # the provided logprobs and aggregation per label.
    labels = ["a", "b"]
    records: List[TopLogprob] = [
        make_toplogprob("a", math.log(0.8)),
        make_toplogprob("b", math.log(0.2)),
    ]

    probs = logprobs_to_label_probs(records, labels)

    # Probabilities should be close to the original 0.8 / 0.2 split.
    assert set(probs.keys()) == {"a", "b"}
    assert probs["a"] == pytest.approx(0.8, rel=1e-6)
    assert probs["b"] == pytest.approx(0.2, rel=1e-6)
    assert probs["a"] + probs["b"] == pytest.approx(1.0, rel=1e-6)


def test_logprobs_to_label_probs_exact_vs_substring_matching():
    # Mixture of tokens; we ensure that exact matches on stripped tokens
    # take precedence over substring matches and that unmatched tokens
    # do not contribute probability mass.
    labels = ["a", "b"]
    records: List[TopLogprob] = [
        make_toplogprob(' "a" ', 0.0),  # exact match (after stripping)
        make_toplogprob(
            "something about B", -2.0
        ),  # substring_matches` is `["a", "b"]` → length 2 → ambiguous → `label = None`
        make_toplogprob("irrelevant", -5.0),  # no label match
    ]

    probs = logprobs_to_label_probs(records, labels)

    # All mass should be assigned to either "a" or "b"; "irrelevant"
    # contributes nothing. We only assert qualitative ordering here,
    # but also ensure probabilities sum to 1.
    print(probs)
    assert probs["b"] == pytest.approx(0.0, rel=1e-6)
    assert probs["a"] > 0.8

    records = [
        make_toplogprob(' "a" ', 0.0),  # exact match (after stripping)
        make_toplogprob("B?!", -2.0),  # substring match for "b"
        make_toplogprob("irrelevant", -5.0),  # no label match
    ]

    probs = logprobs_to_label_probs(records, labels)

    # All mass should be assigned to either "a" or "b"; "irrelevant"
    # contributes nothing. We only assert qualitative ordering here,
    # but also ensure probabilities sum to 1.
    print(probs)
    total = probs["a"] + probs["b"]
    assert 0.0 < total <= 1.0
    assert total == pytest.approx(1.0, rel=1e-6)
    assert probs["a"] > 0.8
    assert probs["b"] > 0.1


def test_parse_think_and_label_and_extract_label_from_json():
    text = (
        "Intro text. <think> reason 1 </think> middle "
        '<think> final reason </think> trailing {"label": "b"} noise'
    )

    think, label_json = parse_think_and_label(text)
    assert think == "final reason"
    assert label_json == '{"label": "b"}'

    label = extract_label_from_json(label_json)
    assert label == "b"
