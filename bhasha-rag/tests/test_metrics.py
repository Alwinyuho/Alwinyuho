import math

import pytest

from bhasharag.metrics import (
    choose_threshold,
    paired_bootstrap_ci,
    refusal_metrics,
    retrieval_metrics,
    token_f1,
)


def test_retrieval_metrics_by_hand():
    rankings = [["a", "b"], ["x", "c"], ["y", "z"]]
    result = retrieval_metrics(rankings, ["a", "c", "missing"])
    assert result["recall@1"] == pytest.approx(1 / 3)
    assert result["recall@5"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert result["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 3)


def test_empty_rankings_are_rejected():
    with pytest.raises(ValueError):
        retrieval_metrics([], [])


def test_choose_threshold_respects_false_answer_limit():
    scores = [0.1 * i for i in range(1, 11)]
    threshold = choose_threshold(scores, 0.2)
    assert sum(s >= threshold for s in scores) == 2


def test_choose_threshold_with_zero_tolerance_rejects_everything_seen():
    scores = [0.2, 0.9]
    assert all(s < choose_threshold(scores, 0.0) for s in scores)


def test_refusal_metrics():
    result = refusal_metrics([0.9, 0.8, 0.1], [True, False, True], [0.7, 0.2], 0.5)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["false_answer_rate"] == pytest.approx(0.5)
    assert result["precision_when_answering"] == pytest.approx(0.5)


def test_bootstrap_interval_contains_the_mean():
    mean, low, high = paired_bootstrap_ci([0.0, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0])
    assert low <= mean <= high
    assert mean == pytest.approx(0.25)


def test_token_f1():
    assert token_f1("भारत की राजधानी", "भारत की राजधानी") == pytest.approx(1.0)
    assert token_f1("दिल्ली", "मुंबई") == 0.0
    assert token_f1("", "मुंबई") == 0.0
