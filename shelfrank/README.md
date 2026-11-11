# ShelfRank

**A two-stage recommender that can suggest brand-new products, with an off-policy evaluation checked against a real online A/B result.**

There are two parts:

1. **Recommendations** on Amazon Reviews 2023 (Video Games). A two-tower retrieval model with a text path for new items is followed by a LightGBM LambdaRank ranker. Everything is evaluated on a strictly chronological split.
2. **Off-policy evaluation (OPE)** on the Open Bandit Dataset. The click rate of a new policy is estimated from logs collected by a different policy, and the estimate is compared with the click rate that policy actually achieved online.

> **Live demo:** not deployed yet. Run it locally with `python -m shelfrank.ui`.

## Results

### Recommendations (test split: 43,249 next-purchase events)

The split is chronological: train comes first, then validation, then test. **60.5% of test purchases are items that never appear in training.** In real catalogues new products arrive constantly, so cold start is most of the problem, not a side case.

| Method | NDCG@10 | Recall@10 | Recall@50 | Recall@10, warm items | Recall@10, cold items |
|---|---|---|---|---|---|
| Popularity, recency weighted (baseline) | 0.0076 | 0.0136 | 0.0359 | 0.0345 | 0 |
| Item-KNN, co-review cosine | 0.0061 | 0.0110 | 0.0263 | 0.0279 | 0 |
| Two-tower, ID only | 0.0096 | 0.0175 | 0.0401 | 0.0442 | 0 |
| Two-tower, text + ID | 0.0119 | 0.0226 | 0.0626 | 0.0530 | 0.0027 |
| **Two-stage: candidates + LightGBM ranker** | **0.0195** | **0.0354** | **0.0793** | **0.0742** | **0.0101** |

- **NDCG@10 is 156% higher than popularity.** The 95% bootstrap intervals are 0.0184 to 0.0206 for the ranker and 0.0070 to 0.0083 for popularity, so they don't overlap.
- **Only the text path can recommend new items.** Every ID-based method scores exactly 0 on cold items. The text tower reaches 2.4% Recall@50 on them, and the ranker raises cold Recall@10 from 0.27% to 1.0%.
- **The bottleneck is retrieval, not ranking.** The correct item is among the roughly 163 candidates only 11.7% of the time. Any gain in candidate recall carries straight through to the final ranking.
- The ranker's most useful features (by total gain) were text similarity to the last item, the two-tower rank, and the KNN rank. Combining signals is what lifts it above each retriever on its own.

Absolute numbers look small because the task is hard: the model must pick 1 correct item out of 25,612 with no filtering by candidate. Compare methods with each other, not with numbers from easier protocols such as random splits or 100 sampled negatives.

### Off-policy evaluation (Open Bandit Dataset, "all" campaign)

ZOZOTOWN ran two policies on the same traffic: uniform random, and Bernoulli Thompson Sampling (BTS). Using only the 1.37M random-policy impressions, the estimators predict BTS's click rate. The truth is BTS's real click rate over its own 12.36M impressions.

| Estimator | Estimated BTS click rate | 95% CI | Error vs truth |
|---|---|---|---|
| IPS | 0.00504 | 0.00479 to 0.00529 | 1.7% |
| SNIPS | 0.00503 | not computed | 1.6% |
| DM (LightGBM reward model) | 0.00498 | 0.004980 to 0.004986 \* | 0.6% |
| **DR** | **0.00498** | **0.00472 to 0.00522** | **0.5%** |
| *Online truth* | *0.00495* | | |

The random policy's own click rate was 0.00347, so BTS's real lift was +42.8%. Offline estimation would have predicted that lift before any live test.

\* DM's interval is misleadingly narrow, because the bootstrap resamples rounds but not the reward model's own uncertainty. DR's interval is the honest one.

## How it works

```
user history ─┬─> two-tower (text + ID item tower) ── top 100 ─┐
              ├─> item-KNN (co-review cosine) ─────── top 50 ──┼─> ~163 candidates ─> LightGBM LambdaRank ─> top 10
              └─> recency-weighted popularity ─────── top 20 ──┘   (12 features)
```

- **Item tower** = MLP(text embedding of title, store, categories and features) + an ID embedding. The ID part is used only for items seen in training, and it is **randomly dropped 30% of the time while training**. That forces the text path to work on its own, which is what makes new items retrievable.
- **User tower** = a recency-weighted mean of the last 20 items, joined with the last item, then projected.
- **Loss:** in-batch sampled softmax with **logQ correction**. In-batch negatives over-sample popular items, and subtracting log item frequency corrects for that.
- **Ranker:** trained on the validation split, with retrieval models that never saw it, so its training data looks like test-time data.
- **Fair metrics:** items the user already reviewed are excluded, and **tied scores count as random tie-breaks**. An early version ranked the target first among ties, which inflated the ID-only model's Recall@10 from 0.017 to 0.044, because all cold items scored identically. A test now guards against that.
- **OPE:** BTS doesn't use user context, so its action probabilities per slot equal how often it showed each item in that slot. DM and DR use a LightGBM click model, cross-fitted on two halves so no round is scored by a model that saw it.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or /whl/cpu
pip install -e ".[ml,ui,dev]"
```

Download into `../_downloads/` (or set `SHELFRANK_DATA`):
- From [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023): `benchmark/5core/timestamp_w_his/Video_Games.{train,valid,test}.csv` and `raw/meta_categories/meta_Video_Games.jsonl`, into `amazon/`.
- The [Open Bandit Dataset](https://research.zozo.com/data.html) zip, with `random/all` and `bts/all` extracted into `obd/extracted/`.

Then run:

```bash
python scripts/run_recsys.py   # about 25 minutes on CPU; writes reports/recsys.json
python scripts/run_ope.py      # about 15 minutes; writes reports/ope.json
python -m shelfrank.ui         # demo at http://127.0.0.1:7862
uvicorn shelfrank.api:app      # POST /recommend {"history": ["B003XNU1LE"], "k": 10}, GET /search?q=zelda
```

The demo serves the two-tower stage only, and responds in about 90 ms on CPU for 25,612 items. The ranker is evaluated offline.

## Limitations

- **One category, one split.** The results are for Video Games only, with no hyperparameter search. Two-tower settings were chosen once, not tuned.
- **Reviews stand in for purchases.** A review is a delayed, biased signal of a purchase.
- **Context-free policy only.** The OPE check uses one context-free policy. Contextual policies would need the policy's own action probabilities per round.
- **No deployment yet.** The demo loads the full dataset at startup. A deployed version should ship a compact artifact (item ids, titles, embeddings, weights) instead. A [Dockerfile](Dockerfile) for the API is included, but it has not been built or tested, because Docker was not available on the development machine.

## Data credits

- Amazon Reviews 2023: Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (2024).
- Open Bandit Dataset: Saito et al., *Open Bandit Dataset and Pipeline* (NeurIPS Datasets and Benchmarks, 2021), released by ZOZO Research.
- Check each dataset's licence and terms before any commercial use.
