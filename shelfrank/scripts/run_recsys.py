"""Train and evaluate every retrieval and ranking method on Amazon Video Games.

Protocol: all models train on the "train" split. The ranker trains on the
"valid" split, using retrieval models that never saw it. Every number reported
comes from the later "test" split, with results split by warm items (seen in
training) and cold items (never seen in training).

Usage: python scripts/run_recsys.py [--device cpu|cuda]
Output: reports/recsys.json
"""

import argparse
import json
import time

import numpy as np
import torch

from shelfrank import config
from shelfrank.baselines import ItemKNN, popularity_scores
from shelfrank.data import load_dataset
from shelfrank.metrics import bootstrap_ci, ndcg_at, summarize, target_ranks
from shelfrank.ranker import FEATURES, Signals, build_features, candidates, train_ranker
from shelfrank.two_tower import TwoTower, score_all, train_two_tower

BATCH = 2000
MISS_RANK = 10**6


def text_embeddings(texts: list[str], device: str) -> np.ndarray:
    path = config.ARTIFACT_DIR / "text_embeddings.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.TEXT_MODEL, device=device)
    vectors = model.encode(
        texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    )
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors.astype(np.float32))
    return vectors.astype(np.float32)


def ranks_for(score_fn, frame) -> np.ndarray:
    histories, targets = frame["history"].tolist(), frame["item"].to_numpy()
    out = []
    for start in range(0, len(frame), BATCH):
        h = histories[start : start + BATCH]
        out.append(target_ranks(score_fn(h), targets[start : start + BATCH], h))
    return np.concatenate(out)


def report_block(ranks: np.ndarray, cold: np.ndarray) -> dict:
    return {
        "all": summarize(ranks, config.EVAL_KS),
        "warm_items": summarize(ranks[~cold], config.EVAL_KS),
        "cold_items": summarize(ranks[cold], config.EVAL_KS) if cold.any() else None,
        "ndcg@10_ci95": bootstrap_ci(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = time.perf_counter()

    ds = load_dataset()
    cold_test = ~ds.train_items[ds.test["item"].to_numpy()]
    print(
        f"{ds.n_items:,} items | train {len(ds.train):,} valid {len(ds.valid):,} "
        f"test {len(ds.test):,} | cold test targets {cold_test.mean():.1%}",
        flush=True,
    )

    text = text_embeddings(ds.meta["text"].tolist(), args.device)
    pop = popularity_scores(ds.train, ds.n_items)
    knn = ItemKNN(ds.train, ds.n_items)
    counts = np.bincount(ds.train["item"], minlength=ds.n_items).astype(float)

    with_history = ds.train[ds.train["history"].map(len) > 0]
    models = {}
    for name, text_input in (("two_tower", text), ("two_tower_id_only", np.zeros_like(text))):
        model = TwoTower(text_input, ds.train_items)
        weights = config.ARTIFACT_DIR / f"{name}.pt"
        if weights.exists():
            print(f"loading {name}", flush=True)
            model.load_state_dict(torch.load(weights, map_location=args.device))
            models[name] = model.to(args.device).eval()
            continue
        print(f"training {name}", flush=True)
        models[name] = train_two_tower(
            model,
            with_history["history"].tolist(),
            with_history["item"].to_numpy(),
            counts,
            device=args.device,
        )
        config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(models[name].state_dict(), weights)

    def tt_scores(name):
        return lambda h: score_all(models[name], h, device=args.device)

    retrievers = {
        "popularity": lambda h: np.tile(pop, (len(h), 1)),
        "item_knn": lambda h: knn.score(h) + 1e-6 * pop,  # popularity breaks ties
        "two_tower_id_only": tt_scores("two_tower_id_only"),
        "two_tower": tt_scores("two_tower"),
    }
    results = {
        "dataset": {
            "category": config.CATEGORY,
            "items": ds.n_items,
            "train": len(ds.train),
            "valid": len(ds.valid),
            "test": len(ds.test),
            "cold_test_share": float(cold_test.mean()),
        },
        "methods": {},
    }
    for name, fn in retrievers.items():
        ranks = ranks_for(fn, ds.test)
        results["methods"][name] = report_block(ranks, cold_test)
        print(name, json.dumps(results["methods"][name]["all"]), flush=True)

    store_codes = ds.meta["store"].astype("category").cat.codes.to_numpy().copy()
    store_codes[ds.meta["store"].to_numpy() == ""] = -1

    def signals_for(histories):
        return Signals(
            tt=retrievers["two_tower"](histories),
            knn=retrievers["item_knn"](histories),
            popularity=pop,
            seen=ds.train_items,
            average_rating=ds.meta["average_rating"].to_numpy(dtype=float),
            rating_count=ds.meta["rating_number"].to_numpy(dtype=float),
            price=ds.meta["price"].to_numpy(dtype=float),
            text=text,
            store=store_codes,
        )

    def candidate_features(frame):
        feats, labels, groups, cand_lists = [], [], [], []
        histories, targets = frame["history"].tolist(), frame["item"].to_numpy()
        for start in range(0, len(frame), BATCH):
            h = histories[start : start + BATCH]
            sig = signals_for(h)
            cands = candidates(sig, h)
            feats.append(build_features(sig, h, cands))
            for c, t in zip(cands, targets[start : start + BATCH], strict=True):
                labels.append((c == t).astype(int))
                groups.append(len(c))
                cand_lists.append(c)
        return np.vstack(feats), labels, groups, cand_lists

    print("building ranker training data from the valid split", flush=True)
    feats, labels, groups, _ = candidate_features(ds.valid)
    has_positive = [lab.any() for lab in labels]
    keep_rows = np.concatenate([np.full(g, k) for g, k in zip(groups, has_positive, strict=True)])
    ranker = train_ranker(
        feats[keep_rows],
        np.concatenate([lab for lab, k in zip(labels, has_positive, strict=True) if k]),
        [g for g, k in zip(groups, has_positive, strict=True) if k],
    )

    print("scoring the test split with the ranker", flush=True)
    feats, labels, groups, cand_lists = candidate_features(ds.test)
    predictions = ranker.predict(feats)
    ranks, offset = [], 0
    for lab, g in zip(labels, groups, strict=True):
        scores = predictions[offset : offset + g]
        offset += g
        if lab.any():
            target_score = scores[lab.argmax()]
            tied = int((scores == target_score).sum()) - 1
            ranks.append(int((scores > target_score).sum()) + tied / 2 + 1)
        else:
            ranks.append(MISS_RANK)
    ranks = np.array(ranks)
    results["methods"]["two_stage_ranker"] = report_block(ranks, cold_test)
    results["candidate_recall"] = float(np.mean([lab.any() for lab in labels]))
    results["mean_candidates"] = float(np.mean(groups))
    importance = ranker.booster_.feature_importance(importance_type="gain")
    results["ranker_feature_gain"] = {
        f: float(v) for f, v in sorted(zip(FEATURES, importance, strict=True), key=lambda x: -x[1])
    }
    print("two_stage_ranker", json.dumps(results["methods"]["two_stage_ranker"]["all"]))

    baseline = ranks_for(retrievers["popularity"], ds.test)
    results["ranker_vs_popularity_ndcg@10_lift"] = (
        ndcg_at(ranks, 10) / max(ndcg_at(baseline, 10), 1e-9) - 1
    )
    results["seconds"] = time.perf_counter() - started
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "recsys.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"done in {results['seconds']:.0f}s")


if __name__ == "__main__":
    main()
