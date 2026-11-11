"""Off-policy evaluation, validated against a real online result.

The Open Bandit Dataset (ZOZOTOWN) logged the same fashion e-commerce traffic
under two policies: uniform random, and Bernoulli Thompson Sampling (BTS).
So we can estimate BTS's click rate *offline* from the random logs, and then
check the estimate against the click rate BTS actually achieved online.

BTS here does not use user context, so its action probabilities per slot
position equal how often it showed each item in that slot in its own logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_ITEMS = 80
N_POSITIONS = 3


@dataclass
class LoggedData:
    item: np.ndarray  # chosen item, 0..79
    position: np.ndarray  # slot, 0..2
    click: np.ndarray  # 0/1 reward
    propensity: np.ndarray  # logging policy's probability of the chosen item
    affinity: np.ndarray  # (rounds, items) user-item affinity scores
    users: np.ndarray  # (rounds, 4) integer codes of the hashed user features

    def reward_features(self, items: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Features of showing ``items[i]`` in round ``rows[i]``: one compact row each."""
        return np.column_stack(
            [
                items,
                self.position[rows],
                self.affinity[rows, items],
                self.users[rows],
            ]
        ).astype(np.float32)

    def predict_all_items(self, model, batch: int = 20_000) -> np.ndarray:
        """Modelled click probability of every item in every round, shape (rounds, items)."""
        out = np.empty((len(self.click), N_ITEMS), dtype=np.float32)
        for start in range(0, len(self.click), batch):
            rows = np.arange(start, min(start + batch, len(self.click)))
            grid_rows = np.repeat(rows, N_ITEMS)
            grid_items = np.tile(np.arange(N_ITEMS), len(rows))
            out[rows] = model.predict(self.reward_features(grid_items, grid_rows)).reshape(
                len(rows), N_ITEMS
            )
        return out


def action_distribution(items: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Empirical P(item | position), shape (positions, items)."""
    counts = np.zeros((N_POSITIONS, N_ITEMS))
    np.add.at(counts, (positions, items), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def ips(data: LoggedData, target: np.ndarray) -> np.ndarray:
    """Per-round inverse propensity scores. Unbiased, but can have high variance."""
    return data.click * target[data.position, data.item] / data.propensity


def snips(data: LoggedData, target: np.ndarray) -> float:
    weights = target[data.position, data.item] / data.propensity
    return float((weights * data.click).sum() / weights.sum())


def direct_method(predicted: np.ndarray, data: LoggedData, target: np.ndarray) -> np.ndarray:
    """``predicted`` is (rounds, items): modelled click probability of every item."""
    return (predicted * target[data.position]).sum(axis=1)


def doubly_robust(predicted: np.ndarray, data: LoggedData, target: np.ndarray) -> np.ndarray:
    """Model estimate, corrected by importance-weighted model error on the logged item."""
    rows = np.arange(len(data.click))
    weights = target[data.position, data.item] / data.propensity
    correction = weights * (data.click - predicted[rows, data.item])
    return direct_method(predicted, data, target) + correction


def bootstrap(values: np.ndarray, n_boot: int = 500, seed: int = 13) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def read_logs(path, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, nrows=max_rows)


def to_logged(frame: pd.DataFrame) -> LoggedData:
    users = frame.filter(like="user_feature")
    codes = np.column_stack([users[c].astype("category").cat.codes for c in users.columns])
    return LoggedData(
        item=frame["item_id"].to_numpy(),
        position=frame["position"].to_numpy() - 1,
        click=frame["click"].to_numpy(dtype=float),
        propensity=frame["propensity_score"].to_numpy(dtype=float),
        affinity=frame.filter(like="user-item_affinity").to_numpy(dtype=np.float32),
        users=codes,
    )
