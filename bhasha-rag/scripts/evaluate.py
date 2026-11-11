"""Score stored runs. Tunes refusal thresholds on validation and reports on test only.

Usage: python scripts/evaluate.py
Output: reports/retrieval.json, reports/refusal.json, reports/thresholds.json
"""

import json

import numpy as np

from bhasharag import config
from bhasharag.metrics import (
    choose_threshold,
    gold_rank,
    paired_bootstrap_ci,
    refusal_metrics,
    retrieval_metrics,
)

METHODS = ["bm25", "dense_e5small", "hybrid_e5small", "dense_bgem3", "hybrid_bgem3", "rerank"]
# Each refusal signal: name, ranked list field, score field, profile that deploys it.
SIGNALS = {
    "cos_e5small": ("dense_e5small", "dense_e5small", "cos_e5small", "cpu"),
    "cos_bgem3": ("hybrid_bgem3", "dense_bgem3", "cos_bgem3", None),
    "reranker": ("rerank", "rerank", "rerank_scores", "gpu"),
}


def load_runs(lang):
    return json.loads((config.ARTIFACT_DIR / lang / "runs.json").read_text(encoding="utf-8"))


def scores_for(rows, ranking_field, score_field):
    """Top score with the gold passage present, and with it removed from the results."""
    answerable, unanswerable = [], []
    for row in rows:
        ranked, scores = row[ranking_field], row[score_field]
        answerable.append(scores[0])
        pairs = zip(ranked, scores, strict=True)
        unanswerable.append(next(s for d, s in pairs if d != row["gold"]))
    return answerable, unanswerable


def reciprocal_ranks(rows, field):
    ranks = [gold_rank(row[field], row["gold"]) for row in rows]
    return [1 / r if r else 0.0 for r in ranks]


def main() -> None:
    all_runs = {lang: load_runs(lang) for lang in config.LANGUAGES}
    test = {lang: [r for r in rows if r["split"] == "test"] for lang, rows in all_runs.items()}
    val = {lang: [r for r in rows if r["split"] == "val"] for lang, rows in all_runs.items()}

    retrieval = {"per_language": {}, "overall": {}}
    for lang, rows in test.items():
        golds = [r["gold"] for r in rows]
        retrieval["per_language"][lang] = {
            m: retrieval_metrics([r[m] for r in rows], golds) for m in METHODS
        }
    pooled = [r for rows in test.values() for r in rows]
    golds = [r["gold"] for r in pooled]
    retrieval["overall"] = {m: retrieval_metrics([r[m] for r in pooled], golds) for m in METHODS}
    retrieval["n_test_queries"] = len(pooled)
    # How often the gold passage is anywhere in the 30 candidates handed to the reranker.
    retrieval["candidate_recall@30"] = {
        m: sum(r["gold"] in r[m][: config.RERANK_DEPTH] for r in pooled) / len(pooled)
        for m in METHODS
    }

    mean, low, high = paired_bootstrap_ci(
        reciprocal_ranks(pooled, "bm25"), reciprocal_ranks(pooled, "rerank")
    )
    retrieval["rerank_vs_bm25_mrr_gain"] = {"mean": mean, "ci95": [low, high]}
    mean, low, high = paired_bootstrap_ci(
        reciprocal_ranks(pooled, "hybrid_bgem3"), reciprocal_ranks(pooled, "rerank")
    )
    retrieval["rerank_vs_hybrid_mrr_gain"] = {"mean": mean, "ci95": [low, high]}

    # Rerank fewer candidates: exact, because scores for all 30 candidates are stored.
    depth_tradeoff = {}
    for depth in (5, 10, 20, config.RERANK_DEPTH):
        rankings = []
        for r in pooled:
            allowed = set(r["hybrid_bgem3"][:depth])
            reranked = [d for d in r["rerank"] if d in allowed]
            rankings.append(reranked + [d for d in r["hybrid_bgem3"][depth:]])
        depth_tradeoff[depth] = retrieval_metrics(rankings, golds)
    retrieval["rerank_depth_tradeoff"] = depth_tradeoff

    refusal = {"target_false_answer_rate": config.TARGET_FALSE_ANSWER_RATE, "signals": {}}
    thresholds = {}
    for name, (order_field, score_order, score_field, profile) in SIGNALS.items():
        per_lang, lang_thresholds, val_u_pool = {}, {}, []
        pooled_a, pooled_c, pooled_u, answered_flags, false_flags = [], [], [], [], []
        for lang in config.LANGUAGES:
            _, val_u = scores_for(val[lang], score_order, score_field)
            val_u_pool += val_u
            threshold = choose_threshold(val_u, config.TARGET_FALSE_ANSWER_RATE)
            lang_thresholds[lang] = threshold

            test_a, test_u = scores_for(test[lang], score_order, score_field)
            correct = [r[order_field][0] == r["gold"] for r in test[lang]]
            per_lang[lang] = refusal_metrics(test_a, correct, test_u, threshold)
            pooled_a += test_a
            pooled_c += correct
            pooled_u += test_u
            answered_flags += [s >= threshold for s in test_a]
            false_flags += [s >= threshold for s in test_u]

        global_threshold = choose_threshold(val_u_pool, config.TARGET_FALSE_ANSWER_RATE)
        answered = np.array(answered_flags)
        per_language_summary = {
            "coverage": float(answered.mean()),
            "false_answer_rate": float(np.mean(false_flags)),
            "precision_when_answering": float(np.array(pooled_c)[answered].mean()),
        }
        refusal["signals"][name] = {
            "per_language_thresholds_summary": per_language_summary,
            "global_threshold": refusal_metrics(pooled_a, pooled_c, pooled_u, global_threshold),
            "per_language": per_lang,
            "no_refusal_precision": float(np.mean(pooled_c)),
        }
        if profile:
            thresholds[profile] = lang_thresholds

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("retrieval", retrieval),
        ("refusal", refusal),
        ("thresholds", thresholds),
    ):
        (config.REPORT_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    print(f"Test queries: {retrieval['n_test_queries']}")
    print(f"{'method':<16}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG10':>8}")
    for method in METHODS:
        m = retrieval["overall"][method]
        print(
            f"{method:<16}{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}"
            f"{m['recall@10']:>8.3f}{m['mrr']:>8.3f}{m['ndcg@10']:>8.3f}"
        )
    for depth, m in depth_tradeoff.items():
        print(f"rerank depth {depth:>2}: R@5 {m['recall@5']:.3f}  MRR {m['mrr']:.3f}")
    print("rerank vs bm25 MRR:", retrieval["rerank_vs_bm25_mrr_gain"])
    print("rerank vs hybrid MRR:", retrieval["rerank_vs_hybrid_mrr_gain"])
    print("\nRecall@5 per language")
    print(f"{'lang':<6}" + "".join(f"{m[:13]:>15}" for m in METHODS))
    for lang in config.LANGUAGES:
        values = "".join(
            f"{retrieval['per_language'][lang][m]['recall@5']:>15.3f}" for m in METHODS
        )
        print(f"{lang:<6}{values}")
    print("\nRefusal, tuned on validation, measured on test")
    for name, payload in refusal["signals"].items():
        print(name, json.dumps(payload["per_language_thresholds_summary"]))
        print("   global threshold:", json.dumps(payload["global_threshold"]))
        print("   no refusal precision:", round(payload["no_refusal_precision"], 3))


if __name__ == "__main__":
    main()
