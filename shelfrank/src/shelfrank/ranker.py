"""Second stage: re-rank about 150 candidates per user with LightGBM LambdaRank.

Candidates come from three sources (two-tower, item-KNN, popularity), so the
ranker can combine signals that no single retriever has.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

FEATURES = [
    "tt_score",
    "tt_rank",
    "knn_score",
    "knn_rank",
    "log_popularity",
    "seen_in_train",
    "average_rating",
    "log_rating_count",
    "price",
    "text_sim_last",
    "same_store_last",
    "history_length",
]


@dataclass
class Signals:
    """Precomputed per-batch scores and per-item attributes the ranker reads."""

    tt: np.ndarray  # (queries, items)
    knn: np.ndarray  # (queries, items)
    popularity: np.ndarray  # (items,)
    seen: np.ndarray  # (items,) bool
    average_rating: np.ndarray
    rating_count: np.ndarray
    price: np.ndarray
    text: np.ndarray  # (items, dim) normalised text embeddings
    store: np.ndarray  # (items,) integer store codes, -1 if unknown


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """Position (1 = best) of every item within each row."""
    order = np.argsort(-scores, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    positions[rows, order] = np.arange(1, scores.shape[1] + 1)
    return positions


def candidates(signals: Signals, histories: list[list[int]], n_tt=100, n_knn=50, n_pop=20):
    tt = signals.tt.copy()
    knn = signals.knn.copy()
    for row, history in enumerate(histories):
        tt[row, history] = -np.inf
        knn[row, history] = -np.inf
    popular = np.argsort(-signals.popularity)[: n_pop + 50]
    lists = []
    tt_top, knn_top = top_k(tt, n_tt), top_k(knn, n_knn)
    for row, history in enumerate(histories):
        seen = set(history)
        pool = list(dict.fromkeys([*tt_top[row], *knn_top[row]]))
        pool += [p for p in popular if p not in seen and p not in pool][:n_pop]
        lists.append(np.array(pool, dtype=np.int64))
    return lists


def build_features(signals: Signals, histories, candidate_lists) -> np.ndarray:
    tt_pos = rank_positions(signals.tt)
    knn_pos = rank_positions(signals.knn)
    rows = []
    for q, (history, cands) in enumerate(zip(histories, candidate_lists, strict=True)):
        last = history[-1] if history else -1
        text_sim = signals.text[cands] @ signals.text[last] if last >= 0 else np.zeros(len(cands))
        same_store = (
            (signals.store[cands] == signals.store[last]) & (signals.store[cands] >= 0)
            if last >= 0
            else np.zeros(len(cands), dtype=bool)
        )
        rows.append(
            np.column_stack(
                [
                    signals.tt[q, cands],
                    tt_pos[q, cands],
                    signals.knn[q, cands],
                    knn_pos[q, cands],
                    np.log1p(signals.popularity[cands]),
                    signals.seen[cands],
                    signals.average_rating[cands],
                    np.log1p(signals.rating_count[cands]),
                    signals.price[cands],
                    text_sim,
                    same_store,
                    np.full(len(cands), len(history)),
                ]
            )
        )
    return np.vstack(rows).astype(np.float32)


def train_ranker(features: np.ndarray, labels: np.ndarray, groups: list[int], seed: int = 13):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(features, labels, group=groups, feature_name=FEATURES)
    return model
