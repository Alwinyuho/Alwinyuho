import numpy as np
import pytest

from shelfrank.metrics import ndcg_at, recall_at, summarize, target_ranks


def test_target_rank_counts_better_items():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    assert target_ranks(scores, np.array([2]), [[]]).tolist() == [2]


def test_history_items_are_excluded_but_not_the_target():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    # item 1 was already reviewed, so the target (item 2) moves to first place
    assert target_ranks(scores, np.array([2]), [[1]]).tolist() == [1]
    # a re-purchase of a history item is still scored, not hidden
    assert target_ranks(scores, np.array([1]), [[1]]).tolist() == [1]


def test_ties_do_not_give_the_best_rank():
    scores = np.zeros((1, 5))  # a model that cannot tell items apart
    assert target_ranks(scores, np.array([0]), [[]]).tolist() == [3.0]


def test_input_scores_are_not_modified():
    scores = np.array([[0.1, 0.9]])
    target_ranks(scores, np.array([0]), [[1]])
    assert scores[0, 1] == pytest.approx(0.9)


def test_recall_and_ndcg_by_hand():
    ranks = np.array([1, 3, 20])
    assert recall_at(ranks, 10) == pytest.approx(2 / 3)
    assert ndcg_at(ranks, 10) == pytest.approx((1 + 0.5) / 3)
    assert summarize(ranks, (10,))["n"] == 3
