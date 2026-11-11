"""Download the benchmark and build a corpus and query table for each language.

Corpus = labelled passages from IndicMSMARCO + unrelated Wikipedia-derived
paragraphs from Indic-Rag-Suite. The distractors make retrieval harder, since
1,000 passages alone would be an unrealistically easy search problem.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import config
from .text import doc_id, tokenize

MIN_DISTRACTOR_TOKENS = 20


def assign_split(query_id: str) -> str:
    """Deterministic validation/test split, stable across machines and reruns."""
    bucket = int(hashlib.sha1(str(query_id).encode()).hexdigest(), 16) % 10
    return "val" if bucket < config.VAL_FRACTION_IN_TENTHS else "test"


def prepare_benchmark(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn raw rows into a deduplicated gold-passage table and a query table."""
    docs: dict[str, str] = {}
    queries: dict[str, dict] = {}
    for row in rows:
        query, passage = (row.get("query") or "").strip(), (row.get("passage") or "").strip()
        qid = str(row.get("query_id") or "")
        if not (query and passage and qid) or qid in queries:
            continue
        did = doc_id(passage)
        docs.setdefault(did, passage)
        queries[qid] = {
            "query_id": qid,
            "query": query,
            "gold_doc_id": did,
            "answer": (row.get("answer") or "").strip(),
            "query_type": row.get("query_type") or "",
            "split": assign_split(qid),
        }
    gold = pd.DataFrame({"doc_id": list(docs), "text": list(docs.values()), "source": "msmarco"})
    return gold, pd.DataFrame(list(queries.values()))


def collect_distractors(paragraphs, n: int, exclude: set[str]) -> pd.DataFrame:
    """Keep the first ``n`` unique, reasonably long paragraphs that are not gold passages."""
    docs: dict[str, str] = {}
    for paragraph in paragraphs:
        if len(docs) >= n:
            break
        text = (paragraph or "").strip()
        if len(tokenize(text)) < MIN_DISTRACTOR_TOKENS:
            continue
        did = doc_id(text)
        if did not in exclude:
            docs.setdefault(did, text)
    return pd.DataFrame({"doc_id": list(docs), "text": list(docs.values()), "source": "wiki"})


def sample_paragraphs(lang: str):
    """Yield paragraphs from randomly chosen row groups of the remote Parquet file.

    Reading whole row groups over HTTP is fast, and random groups spread the
    sample across the file instead of taking only the first articles.
    """
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    path = f"datasets/ai4bharat/Indic-Rag-Suite/data/{lang}_data.parquet"
    parquet = pq.ParquetFile(HfFileSystem().open(path, block_size=8 * 2**20))
    groups = list(range(parquet.metadata.num_row_groups))
    random.Random(config.SEED).shuffle(groups)
    for group in groups:
        yield from parquet.read_row_group(group, columns=["paragraph"]).column(0).to_pylist()


def language_dir(lang: str) -> Path:
    return config.DATA_DIR / lang


def build_language(lang: str, n_distractors: int = config.N_DISTRACTORS) -> dict:
    """Download one language and write corpus.parquet and queries.parquet."""
    from datasets import load_dataset

    rows = list(load_dataset("ai4bharat/IndicMSMARCO", lang, split="train"))
    gold, queries = prepare_benchmark(rows)

    distractors = collect_distractors(sample_paragraphs(lang), n_distractors, set(gold["doc_id"]))

    corpus = pd.concat([gold, distractors], ignore_index=True)
    out = language_dir(lang)
    out.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out / "corpus.parquet", index=False)
    queries.to_parquet(out / "queries.parquet", index=False)
    return {
        "lang": lang,
        "gold_docs": len(gold),
        "distractors": len(distractors),
        "queries": len(queries),
        "val": int((queries["split"] == "val").sum()),
        "test": int((queries["split"] == "test").sum()),
    }


def load_language(lang: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = language_dir(lang)
    return (
        pd.read_parquet(directory / "corpus.parquet"),
        pd.read_parquet(directory / "queries.parquet"),
    )
