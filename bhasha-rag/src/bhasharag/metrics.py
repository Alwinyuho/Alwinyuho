"""Retrieval, refusal and answer metrics. Pure functions, covered by unit tests."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .text import tokenize


def gold_rank(ranked: Sequence[str], gold: str) -> int | None:
    """Return the 1-based position of the gold document, or None if absent."""
    for position, doc in enumerate(ranked, start=1):
        if doc == gold:
            return position
    return None


def retrieval_metrics(rankings: list[Sequence[str]], golds: Sequence[str]) -> dict[str, float]:
    """Mean Recall@k, MRR and nDCG@10 with one relevant document per query."""
    if not rankings:
        raise ValueError("no rankings to score")
    ranks = [gold_rank(ranked, gold) for ranked, gold in zip(rankings, golds, strict=True)]
    n = len(ranks)
    return {
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
        "ndcg@10": sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= 10) / n,
    }


def paired_bootstrap_ci(
    baseline: Sequence[float], improved: Sequence[float], n_boot: int = 2000, seed: int = 13
) -> tuple[float, float, float]:
    """Mean per-query improvement with a 95% paired bootstrap confidence interval."""
    diff = np.asarray(improved, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[samples].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def choose_threshold(unanswerable_scores: Sequence[float], max_false_answer: float) -> float:
    """Pick the lowest threshold whose false-answer rate stays within the limit.

    A query is answered when ``score >= threshold``. Call this with validation
    scores only, and then report performance on the test split.
    """
    scores = np.sort(np.asarray(unanswerable_scores, dtype=float))[::-1]
    allowed = int(math.floor(max_false_answer * len(scores)))
    if allowed >= len(scores):
        return float(scores[-1])
    return float(np.nextafter(scores[allowed], np.inf))


def refusal_metrics(
    answerable_scores: Sequence[float],
    answerable_correct: Sequence[bool],
    unanswerable_scores: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    """Behaviour at one threshold.

    coverage: share of answerable queries the system answers.
    false_answer_rate: share of unanswerable queries it answers anyway.
    precision_when_answering: answered queries whose top passage was the gold one.
    """
    answered = np.asarray(answerable_scores, dtype=float) >= threshold
    falsely_answered = np.asarray(unanswerable_scores, dtype=float) >= threshold
    correct = np.asarray(answerable_correct, dtype=bool)
    return {
        "threshold": float(threshold),
        "coverage": float(answered.mean()),
        "false_answer_rate": float(falsely_answered.mean()),
        "precision_when_answering": float(correct[answered].mean()) if answered.any() else 0.0,
    }


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between a generated and a reference answer."""
    pred, ref = Counter(tokenize(prediction)), Counter(tokenize(reference))
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)
