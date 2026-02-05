import math
from typing import List

import numpy as np
import pytest
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from practical_deliberation_llms.util import (
    extract_label_from_json,
    kl_divergence,
    logprobs_to_label_probs,
    parse_think_and_label,
    within_context_disagreement,
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


def test_kl_divergence_identical_non_normalized_is_zero():
    # Non-normalized but identical distributions
    p = np.array([2.0, 3.0, 5.0])
    q = np.array([4.0, 6.0, 10.0])  # same proportions as p

    value = kl_divergence(p, q)

    assert isinstance(value, float)
    assert np.isclose(value, 0.0, atol=1e-12)


def test_kl_divergence_handles_zeros_and_clipping():
    # Extreme case with zeros that exercises normalization + clipping
    eps = 1e-12
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])

    value = kl_divergence(p, q, eps=eps)

    # Manually replicate the normalization + clipping logic for a strong check
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    p_safe = np.clip(p_norm, eps, 1.0)
    q_safe = np.clip(q_norm, eps, 1.0)
    expected = float(np.sum(p_safe * np.log(p_safe / q_safe)))

    assert np.isclose(value, expected, rtol=1e-9, atol=1e-12)
    # Also check this is large and positive as intuition suggests
    assert value > 20.0


def test_within_context_disagreement_two_opposite_dists_equals_log2():
    # Two one-hot distributions on opposite categories
    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])

    value = within_context_disagreement([d1, d2])

    # Mean distribution is [0.5, 0.5], so each KL = log(2); average is log(2)
    expected = float(np.log(2.0))

    assert isinstance(value, float)
    assert np.isclose(value, expected, rtol=1e-9, atol=1e-12)


def test_within_context_disagreement_three_asymmetric_dists():
    d1 = np.array([1.0, 0.0, 0.0])
    d2 = np.array([0.2, 0.4, 0.4])
    d3 = np.array([0.0, 0.5, 0.5])

    wcdis12 = within_context_disagreement([d1, d2])
    wcdis23 = within_context_disagreement([d2, d3])
    wcdis13 = within_context_disagreement([d1, d3])
    wcdis123 = within_context_disagreement([d1, d2, d3])

    assert wcdis12 < wcdis13, (
        "d2 is closer to d1 than d3 is, so disagreement for (d1, d2) "
        "should be smaller than for the more separated pair (d1, d3)"
    )
    assert wcdis23 < wcdis13, (
        "Both d2 and d3 put mass on the same two coordinates, so they "
        "should disagree less than the extreme pair (d1, d3)"
    )
    assert wcdis12 < wcdis123, (
        "Adding d3, which is farther from d1 than d2 is, should increase "
        "the average disagreement compared to just (d1, d2)"
    )
    assert wcdis23 < wcdis123, (
        "Adding the outlier d1 to the closer pair (d2, d3) should increase "
        "the average disagreement"
    )
    assert wcdis123 < wcdis13, (
        "Including the intermediate distribution d2 between d1 and d3 "
        "should reduce average disagreement relative to the extreme pair (d1, d3)"
    )
