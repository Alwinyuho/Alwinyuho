"""Load the official chronological split of Amazon Reviews 2023 and item metadata.

The benchmark files are already split by global timestamp: everything in
"valid" happens after everything in "train", and "test" comes last. Each row
says which item a user reviewed next and lists the items they reviewed before
(``history``), which is exactly the next-item recommendation task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Dataset:
    items: list[str]  # index -> parent_asin
    train: pd.DataFrame  # user_id, item, timestamp, history (list[int])
    valid: pd.DataFrame
    test: pd.DataFrame
    train_items: np.ndarray  # bool mask: item appears as a target in train
    meta: pd.DataFrame  # one row per item index

    @property
    def n_items(self) -> int:
        return len(self.items)


def _read_split(name: str) -> pd.DataFrame:
    path = config.AMAZON_DIR / f"{config.CATEGORY}.{name}.csv"
    frame = pd.read_csv(path, dtype={"history": str})
    frame["history"] = frame["history"].fillna("").str.split()
    return frame


def item_text(row: dict) -> str:
    """Short description used to embed an item, so new items can be recommended."""
    parts = [row.get("title") or "", row.get("store") or ""]
    parts += (row.get("categories") or [])[-2:]
    parts += (row.get("features") or [])[:2]
    return " | ".join(str(p)[:200] for p in parts if p)


def load_meta(asins: set[str]) -> pd.DataFrame:
    rows = []
    with (config.AMAZON_DIR / f"meta_{config.CATEGORY}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            asin = record.get("parent_asin")
            if asin not in asins:
                continue
            price = record.get("price")
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = np.nan
            rows.append(
                {
                    "asin": asin,
                    "text": item_text(record),
                    "store": record.get("store") or "",
                    "price": price,
                    "average_rating": record.get("average_rating") or np.nan,
                    "rating_number": record.get("rating_number") or 0,
                }
            )
    return pd.DataFrame(rows)


def load_dataset() -> Dataset:
    splits = {name: _read_split(name) for name in ("train", "valid", "test")}
    asins = set()
    for frame in splits.values():
        asins.update(frame["parent_asin"])
        for history in frame["history"]:
            asins.update(history)
    items = sorted(asins)
    index = {asin: i for i, asin in enumerate(items)}

    for frame in splits.values():
        frame["item"] = frame["parent_asin"].map(index).astype(np.int64)
        frame["history"] = frame["history"].map(lambda h: [index[a] for a in h])

    train_items = np.zeros(len(items), dtype=bool)
    train_items[splits["train"]["item"].to_numpy()] = True
    for history in splits["train"]["history"]:
        train_items[history] = True

    meta = load_meta(asins).set_index("asin").reindex(items)
    meta["text"] = meta["text"].fillna("")
    meta["store"] = meta["store"].fillna("")
    meta["rating_number"] = meta["rating_number"].fillna(0)
    meta = meta.reset_index(drop=True)

    columns = ["user_id", "item", "timestamp", "history"]
    return Dataset(
        items,
        *(splits[name][columns].reset_index(drop=True) for name in ("train", "valid", "test")),
        train_items=train_items,
        meta=meta,
    )
