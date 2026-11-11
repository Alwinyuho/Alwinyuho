"""Baselines every recommender must beat: popularity and item-to-item co-occurrence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def popularity_scores(train: pd.DataFrame, n_items: int, half_life_days: float = 60.0):
    """Interaction counts, weighted so that recent interactions count more."""
    newest = train["timestamp"].max()
    age_days = (newest - train["timestamp"].to_numpy()) / 86_400_000
    weights = 0.5 ** (age_days / half_life_days)
    return np.bincount(train["item"].to_numpy(), weights=weights, minlength=n_items)


def user_item_matrix(train: pd.DataFrame, n_items: int) -> sparse.csr_matrix:
    users = train["user_id"].astype("category").cat.codes.to_numpy()
    matrix = sparse.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (users, train["item"].to_numpy())),
        shape=(users.max() + 1, n_items),
    )
    matrix.data[:] = 1.0  # count each user-item pair once
    return matrix


class ItemKNN:
    """Cosine similarity between items, from which users reviewed both."""

    def __init__(self, train: pd.DataFrame, n_items: int, neighbours: int = 200):
        matrix = user_item_matrix(train, n_items)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()) + 1e-9
        normalized = matrix.multiply(1.0 / norms).tocsc()
        similarity = (normalized.T @ normalized).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        self.similarity = _keep_top(similarity, neighbours)

    def score(self, histories: list[list[int]], recent: int = 20) -> np.ndarray:
        rows, cols = [], []
        for row, history in enumerate(histories):
            for item in history[-recent:]:
                rows.append(row)
                cols.append(item)
        selector = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(histories), self.similarity.shape[0]),
        )
        return np.asarray((selector @ self.similarity).todense(), dtype=np.float32)


def _keep_top(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        data, idx = matrix.data[start:end], matrix.indices[start:end]
        if len(data) > k:
            top = np.argpartition(-data, k)[:k]
            data, idx = data[top], idx[top]
        rows.extend([row] * len(data))
        cols.extend(idx)
        vals.extend(data)
    return sparse.csr_matrix((vals, (rows, cols)), shape=matrix.shape, dtype=np.float32)
