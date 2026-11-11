"""The online path: retrieve, decide whether to refuse, then answer with citations."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config
from .data import build_language, language_dir, load_language
from .generation import Generator, ParsedAnswer, build_messages, extractive_answer, parse_answer
from .retrieval import BM25Index, DenseIndex, Reranker, TextEncoder, rrf_fuse

CONTEXT_PASSAGES = 3


@dataclass
class Source:
    number: int
    doc_id: str
    text: str


@dataclass
class RagResponse:
    answer: str
    refused: bool
    refusal_reason: str | None
    confidence: float
    threshold: float
    mode: str
    sources: list[Source] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def embedding_path(lang: str, encoder_key: str):
    return config.ARTIFACT_DIR / lang / f"{encoder_key}.npy"


def load_or_build_embeddings(lang: str, texts: list[str], encoder: TextEncoder) -> np.ndarray:
    path = embedding_path(lang, encoder.encoder.key)
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    embeddings = encoder.encode_passages(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def load_thresholds() -> dict:
    path = config.REPORT_DIR / "thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class LanguageIndex:
    def __init__(self, lang: str, encoder: TextEncoder):
        if not (language_dir(lang) / "corpus.parquet").exists():
            build_language(lang)
        corpus, _ = load_language(lang)
        self.doc_ids = corpus["doc_id"].tolist()
        self.texts = corpus["text"].tolist()
        self.bm25 = BM25Index(self.texts)
        self.dense = DenseIndex(load_or_build_embeddings(lang, self.texts, encoder))


class BhashaRAG:
    def __init__(
        self,
        languages: list[str],
        profile: str = "cpu",
        generator: Generator | None = None,
        device: str | None = None,
    ):
        self.profile = config.PROFILES[profile]
        self.generator = generator
        self.encoder = TextEncoder(self.profile.encoder, device=device)
        self.reranker = Reranker(device=device) if self.profile.use_reranker else None
        self.indexes = {lang: LanguageIndex(lang, self.encoder) for lang in languages}
        self.thresholds = load_thresholds().get(profile, {})

    def retrieve(self, question: str, lang: str):
        """Return ranked document positions, the confidence score and stage timings."""
        index = self.indexes[lang]
        timings: dict[str, float] = {}

        start = time.perf_counter()
        scores, ids = index.dense.search(self.encoder.encode_queries([question]))
        fused = [int(i) for i in ids[0] if i >= 0]
        confidence = float(scores[0][0])
        timings["dense"] = (time.perf_counter() - start) * 1000

        if self.profile.use_bm25:
            start = time.perf_counter()
            fused = rrf_fuse([index.bm25.search(question), fused])
            timings["bm25"] = (time.perf_counter() - start) * 1000

        if self.reranker is not None:
            start = time.perf_counter()
            candidates = fused[: config.RERANK_DEPTH]
            rerank_scores = self.reranker.score([(question, index.texts[i]) for i in candidates])
            order = np.argsort(-rerank_scores, kind="stable")
            fused = [candidates[i] for i in order]
            confidence = float(rerank_scores[order[0]])
            timings["rerank"] = (time.perf_counter() - start) * 1000
        return fused, confidence, timings

    def answer(self, question: str, lang: str) -> RagResponse:
        if lang not in self.indexes:
            raise KeyError(f"language not loaded: {lang}")
        started = time.perf_counter()
        ranking, confidence, timings = self.retrieve(question, lang)
        index = self.indexes[lang]
        threshold = float(self.thresholds.get(lang, 0.0))

        def finish(response: RagResponse) -> RagResponse:
            response.timings_ms = {
                **{k: round(v, 1) for k, v in timings.items()},
                "total": round((time.perf_counter() - started) * 1000, 1),
            }
            return response

        def refuse(reason: str, mode: str, sources: list[Source]) -> RagResponse:
            return finish(
                RagResponse(
                    answer=config.REFUSAL_TEXT.get(lang, "No answer found in the sources."),
                    refused=True,
                    refusal_reason=reason,
                    confidence=confidence,
                    threshold=threshold,
                    mode=mode,
                    sources=sources,
                )
            )

        mode = self.generator.name if self.generator else "extractive"
        if not ranking or confidence < threshold:
            return refuse("low_retrieval_confidence", mode, [])

        chosen = ranking[:CONTEXT_PASSAGES]
        passages = [index.texts[i] for i in chosen]
        sources = [
            Source(number, index.doc_ids[i], index.texts[i])
            for number, i in enumerate(chosen, start=1)
        ]

        if self.generator is None:
            parsed: ParsedAnswer = extractive_answer(passages)
        else:
            start = time.perf_counter()
            raw = self.generator.generate(build_messages(question, passages, lang))
            timings["generate"] = (time.perf_counter() - start) * 1000
            parsed = parse_answer(raw, len(passages))
            if parsed.refused:
                return refuse("model_found_no_answer", mode, sources)
            if not parsed.grounded:
                return refuse("answer_had_no_valid_citation", mode, sources)

        return finish(
            RagResponse(
                answer=parsed.text,
                refused=False,
                refusal_reason=None,
                confidence=confidence,
                threshold=threshold,
                mode=mode,
                sources=[s for s in sources if s.number in parsed.citations],
            )
        )
