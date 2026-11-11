"""Run every retrieval method and store rankings, so evaluation is instant and repeatable.

Usage: python scripts/run_retrieval.py
Output: artifacts/<lang>/runs.json
"""

import json
import time
from functools import partial

import numpy as np

from bhasharag import config
from bhasharag.data import load_language
from bhasharag.pipeline import load_or_build_embeddings
from bhasharag.retrieval import BM25Index, DenseIndex, Reranker, TextEncoder, rrf_fuse

STORED_DEPTH = config.RERANK_DEPTH


def top_ids(doc_ids, ranking):
    return [doc_ids[d] for d in ranking[:STORED_DEPTH]]


def main() -> None:
    encoders = {e.key: TextEncoder(e) for e in (config.E5_SMALL, config.BGE_M3)}
    reranker = Reranker()

    for lang in config.LANGUAGES:
        began = time.perf_counter()
        corpus, queries = load_language(lang)
        doc_ids, texts = corpus["doc_id"].tolist(), corpus["text"].tolist()
        questions = queries["query"].tolist()

        bm25 = BM25Index(texts)
        bm25_rankings = [bm25.search(q) for q in questions]

        dense_rankings, dense_scores = {}, {}
        for key, encoder in encoders.items():
            index = DenseIndex(load_or_build_embeddings(lang, texts, encoder))
            scores, ids = index.search(encoder.encode_queries(questions))
            dense_rankings[key], dense_scores[key] = ids.tolist(), scores

        hybrid = {
            key: [rrf_fuse(pair) for pair in zip(bm25_rankings, dense_rankings[key], strict=True)]
            for key in encoders
        }

        candidates = [ranking[: config.RERANK_DEPTH] for ranking in hybrid["bgem3"]]
        pairs = [(q, texts[d]) for q, cand in zip(questions, candidates, strict=True) for d in cand]
        flat_scores = reranker.score(pairs)

        rows, offset = [], 0
        for i, query in enumerate(queries.itertuples(index=False)):
            cand = candidates[i]
            scores = flat_scores[offset : offset + len(cand)]
            offset += len(cand)
            order = np.argsort(-scores, kind="stable")
            ids_of = partial(top_ids, doc_ids)
            rows.append(
                {
                    "query_id": query.query_id,
                    "split": query.split,
                    "gold": query.gold_doc_id,
                    "query_type": query.query_type,
                    "bm25": ids_of(bm25_rankings[i]),
                    "dense_e5small": ids_of(dense_rankings["e5small"][i]),
                    "dense_bgem3": ids_of(dense_rankings["bgem3"][i]),
                    "hybrid_e5small": ids_of(hybrid["e5small"][i]),
                    "hybrid_bgem3": ids_of(hybrid["bgem3"][i]),
                    "rerank": [doc_ids[cand[j]] for j in order],
                    "rerank_scores": [round(float(scores[j]), 6) for j in order],
                    "cos_e5small": [
                        round(float(x), 6) for x in dense_scores["e5small"][i][:STORED_DEPTH]
                    ],
                    "cos_bgem3": [
                        round(float(x), 6) for x in dense_scores["bgem3"][i][:STORED_DEPTH]
                    ],
                }
            )

        out = config.ARTIFACT_DIR / lang
        out.mkdir(parents=True, exist_ok=True)
        (out / "runs.json").write_text(json.dumps(rows), encoding="utf-8")
        print(f"{lang}: {len(rows)} queries, {time.perf_counter() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
