"""Online recommendations from the trained two-tower model.

The demo serves the retrieval stage only (two-tower over every item). The
LightGBM ranker is evaluated offline in scripts/run_recsys.py.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config
from .data import load_dataset
from .two_tower import TwoTower, score_all


class Recommender:
    def __init__(self, device: str = "cpu"):
        ds = load_dataset()
        self.items = ds.items
        self.index = {asin: i for i, asin in enumerate(ds.items)}
        self.titles = [t.split(" | ")[0] for t in ds.meta["text"].tolist()]
        self.seen = ds.train_items
        text = np.load(config.ARTIFACT_DIR / "text_embeddings.npy")
        self.model = TwoTower(text, ds.train_items)
        state = torch.load(config.ARTIFACT_DIR / "two_tower.pt", map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = device

    def search(self, query: str, limit: int = 20) -> list[dict]:
        query = query.lower().strip()
        if not query:
            return []
        hits = [i for i, title in enumerate(self.titles) if query in title.lower()]
        return [self._item(i) for i in hits[:limit]]

    def recommend(self, history: list[str], k: int = 10) -> list[dict]:
        known = [self.index[a] for a in history if a in self.index]
        if not known:
            return []
        scores = score_all(self.model, [known], device=self.device)[0]
        scores[known] = -np.inf
        top = np.argsort(-scores)[:k]
        return [{**self._item(int(i)), "score": float(scores[i])} for i in top]

    def _item(self, i: int) -> dict:
        return {
            "asin": self.items[i],
            "title": self.titles[i] or self.items[i],
            "new_item": not bool(self.seen[i]),
        }
