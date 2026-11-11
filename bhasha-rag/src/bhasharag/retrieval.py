"""BM25, dense search, reciprocal rank fusion and cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .text import tokenize


def rrf_fuse(rankings: Iterable[Sequence[int]], k: int = config.RRF_K) -> list[int]:
    """Merge rankings using reciprocal rank fusion.

    RRF uses positions only, so BM25 scores and cosine similarities, which live
    on different scales, can be combined without any normalisation.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        self._bm25 = BM25Okapi([tokenize(text) or ["<empty>"] for text in texts])

    def search(self, query: str, top_k: int = config.CANDIDATES) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_k = min(top_k, len(scores))
        top = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return [int(i) for i in ordered if scores[i] > 0]


class DenseIndex:
    """Exact inner-product search. At a few thousand documents, no ANN is needed."""

    def __init__(self, embeddings: np.ndarray):
        import faiss

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(np.asarray(embeddings, dtype=np.float32))

    def search(self, query_embeddings: np.ndarray, top_k: int = config.CANDIDATES):
        scores, ids = self._index.search(np.asarray(query_embeddings, dtype=np.float32), top_k)
        return scores, ids


class TextEncoder:
    def __init__(self, encoder: config.Encoder, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.encoder = encoder
        self.model = SentenceTransformer(encoder.model, device=device, model_kwargs=kwargs)

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> np.ndarray:
        return self.model.encode(
            [prefix + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._encode(texts, self.encoder.query_prefix, batch_size)

    def encode_passages(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, self.encoder.passage_prefix, batch_size)


class Reranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL, device: str | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(model_name, device=device, max_length=512, model_kwargs=kwargs)

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """Return one relevance score in the range 0 to 1 for each (query, passage) pair."""
        if not pairs:
            return np.empty(0, dtype=np.float32)
        return np.asarray(
            self.model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        )
