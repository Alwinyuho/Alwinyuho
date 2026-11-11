"""Estimate BTS's click rate from random-policy logs, and compare with BTS's real result.

Usage: python scripts/run_ope.py
Output: reports/ope.json
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfrank import config
from shelfrank.ope import (
    bootstrap,
    direct_method,
    doubly_robust,
    ips,
    read_logs,
    snips,
    to_logged,
)

CHUNK = 500_000


def bts_ground_truth():
    """Online CTR of BTS and its per-slot action distribution, read in chunks (6 GB file)."""
    path = config.OBD_DIR / "extracted" / "bts" / "all" / "all.csv"
    clicks, rounds = 0.0, 0
    counts = np.zeros((3, 80))
    for chunk in pd.read_csv(path, usecols=["item_id", "position", "click"], chunksize=CHUNK):
        clicks += chunk["click"].sum()
        rounds += len(chunk)
        np.add.at(counts, (chunk["position"].to_numpy() - 1, chunk["item_id"].to_numpy()), 1)
    target = counts / counts.sum(axis=1, keepdims=True)
    return clicks / rounds, rounds, target


def main() -> None:
    truth, bts_rounds, target = bts_ground_truth()
    print(f"BTS online CTR {truth:.5f} over {bts_rounds:,} impressions", flush=True)

    random_logs = read_logs(config.OBD_DIR / "extracted" / "random" / "all" / "all.csv")
    data = to_logged(random_logs)
    n = len(data.click)
    print(f"random-policy log: {n:,} impressions, CTR {data.click.mean():.5f}", flush=True)

    # Reward model for DM and DR, trained on one half and applied to the other half.
    rng = np.random.default_rng(config.SEED)
    fold = rng.integers(0, 2, n)
    rows = np.arange(n)
    predicted = np.empty((n, 80), dtype=np.float32)
    for k in (0, 1):
        train_rows = rows[fold != k]
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=200,
            verbose=-1,
            random_state=config.SEED,
        )
        model.fit(
            data.reward_features(data.item[train_rows], train_rows),
            data.click[train_rows],
            categorical_feature=[0, 1, 3, 4, 5, 6],
        )
        held_out = rows[fold == k]

        class Prob:
            def predict(self, x, _m=model):
                return _m.predict_proba(x)[:, 1]

        subset = to_subset(data, held_out)
        predicted[held_out] = subset.predict_all_items(Prob())
        print(f"reward model fold {k} done", flush=True)

    per_round = {
        "IPS": ips(data, target),
        "DM": direct_method(predicted, data, target),
        "DR": doubly_robust(predicted, data, target),
    }
    results = {}
    for name, values in per_round.items():
        estimate = float(values.mean())
        results[name] = {
            "estimate": estimate,
            "ci95": bootstrap(values),
            "relative_error": abs(estimate - truth) / truth,
        }
    snips_value = snips(data, target)
    results["SNIPS"] = {"estimate": snips_value, "relative_error": abs(snips_value - truth) / truth}

    report = {
        "ground_truth_bts_ctr": truth,
        "bts_impressions": int(bts_rounds),
        "random_impressions": int(n),
        "random_policy_ctr": float(data.click.mean()),
        "true_relative_lift_of_bts": truth / float(data.click.mean()) - 1,
        "estimators": results,
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "ope.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def to_subset(data, rows):
    from shelfrank.ope import LoggedData

    return LoggedData(
        item=data.item[rows],
        position=data.position[rows],
        click=data.click[rows],
        propensity=data.propensity[rows],
        affinity=data.affinity[rows],
        users=data.users[rows],
    )


if __name__ == "__main__":
    main()
