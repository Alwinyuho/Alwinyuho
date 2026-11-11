"""Ranking metrics with one relevant item per query, plus bootstrap intervals."""

from __future__ import annotations

import numpy as np


def target_ranks(scores: np.ndarray, targets: np.ndarray, exclude: list[list[int]]) -> np.ndarray:
    """1-based rank of each target among all items, with already-seen items removed.

    ``scores`` is (queries, items). Excluding the user's history is standard for
    this task: recommending something the user already reviewed is not useful.
    """
    scores = scores.astype(np.float32, copy=True)
    for row, seen in enumerate(exclude):
        seen = [s for s in seen if s != targets[row]]
        if seen:
            scores[row, seen] = -np.inf
    target_scores = scores[np.arange(len(targets)), targets]
    better = (scores > target_scores[:, None]).sum(axis=1)
    # Ties are broken at random in expectation: the target sits in the middle of its tie
    # group. Ranking it first among ties would reward models that score many items equally.
    tied = (scores == target_scores[:, None]).sum(axis=1) - 1
    return better + tied / 2.0 + 1


def recall_at(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def ndcg_at(ranks: np.ndarray, k: int) -> float:
    gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return float(gains.mean())


def summarize(ranks: np.ndarray, ks=(10, 50)) -> dict[str, float]:
    out = {"n": int(len(ranks))}
    for k in ks:
        out[f"recall@{k}"] = recall_at(ranks, k)
        out[f"ndcg@{k}"] = ndcg_at(ranks, k)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 13) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
