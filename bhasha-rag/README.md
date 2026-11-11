# BhashaRAG

**Retrieval-augmented question answering for 8 Indian languages that cites its sources and refuses when the evidence is weak.**

Most RAG demos are tested in English and answer every question with confidence. BhashaRAG is built around measurement instead. It reports retrieval quality per language, tests refusal in a controlled way, and states plainly where it falls short.

> **Live demo:** not deployed yet. See [Deploy](#deploy) for the free Hugging Face Spaces setup.

## Results

The test set has 5,647 questions across Hindi, Bengali, Tamil, Telugu, Marathi, Malayalam, Kannada and Gujarati. Each language has its own index of 6,000 passages: about 1,000 labelled passages plus 5,000 Wikipedia distractors. All thresholds were chosen on a separate validation split of 2,352 questions.

### Retrieval

| Method | Recall@1 | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 (baseline) | 0.449 | 0.611 | 0.523 | 0.555 |
| Dense, multilingual-e5-small | 0.663 | 0.816 | 0.732 | 0.760 |
| Dense, BGE-M3 | 0.755 | 0.885 | 0.813 | 0.836 |
| Hybrid (BM25 + BGE-M3, RRF) | 0.617 | 0.785 | 0.695 | 0.729 |
| **Hybrid + bge-reranker-v2-m3** | **0.789** | **0.903** | **0.840** | **0.859** |

- Reranking raises MRR by 0.316 over BM25 (95% paired bootstrap CI 0.305 to 0.328) and by 0.144 over the hybrid it reranks (CI 0.135 to 0.153).
- **Equal-weight hybrid search hurt.** Adding BM25 through RRF lowered MRR for both embedding models. A weight sweep on validation confirmed this: BM25 never helped BGE-M3, and helped e5-small by less than 0.01 MRR. The reason is that BM25 is far weaker here (Recall@30 of 0.73, against 0.94 for BGE-M3), so equal weight lets its mistakes push good passages down. The CPU demo therefore uses dense search only.

Recall@5 per language:

| | hi | bn | ta | te | mr | ml | kn | gu |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.748 | 0.670 | 0.562 | 0.637 | 0.557 | 0.555 | 0.574 | 0.582 |
| Reranked | 0.931 | 0.905 | 0.888 | 0.919 | 0.905 | 0.892 | 0.885 | 0.895 |

BM25 does worst on Dravidian languages and Marathi, where words take many suffixes and exact word matches are rarer. Reranking closes most of that gap.

### Refusal

**How the test works.** Each test question is asked twice. The first time, its labelled passage is in the index, so the question is answerable. The second time, that passage is removed, so the question should be refused. A threshold on the top retrieval score decides whether to answer. It is tuned on validation to allow at most 10% wrong answers on unanswerable questions, then measured on test.

| Confidence signal | Answers given (answerable) | Wrong answers (unanswerable) | Top passage correct when answering | Same, with no refusal |
|---|---|---|---|---|
| e5-small cosine (CPU profile) | 38.2% | 8.8% | 88.0% | 66.3% |
| BGE-M3 cosine | 47.8% | 10.0% | 74.9% | 61.7% |
| **Reranker score (GPU profile)** | **59.8%** | **8.7%** | **93.3%** | 78.9% |

The reranker score is the best signal for deciding when to refuse. At the same wrong-answer rate, it answers far more questions and is right more often.

### Answer generation

The model was Qwen3-4B, run locally in 4-bit. The sample was 200 test questions (25 per language), each given the top 3 reranked passages.

| | Grounded answer given | Cites the correct passage | Token F1 vs reference | Answers when the evidence was removed |
|---|---|---|---|---|
| Qwen3-4B | 88.5% | 70.5% | 0.122 | **68.0%** |
| Extractive baseline (first sentence of top passage) | 100% | 88.5% | 0.086 | 100% |

**Lesson:** the small LLM rarely says "I don't know" on its own. It answered 68% of the questions whose supporting passage had been removed. This is why the system decides on refusal *before* generation, using the retrieval score, instead of trusting the model to abstain. Token F1 is low for both methods, because the reference answers are machine-translated and the model paraphrases, so read it as a comparison between the two, not as an absolute quality score.

### Latency

Measured over 100 Hindi queries, one at a time, on a laptop with an RTX 5050 (8 GB). These are retrieval timings only, before any generation.

| Profile | Stages | p50 | p95 |
|---|---|---|---|
| CPU: e5-small dense | dense search | 27 ms | 31 ms |
| GPU: BGE-M3 hybrid + rerank 30 | dense 40 + BM25 21 + rerank 433 | 494 ms | 727 ms |

The reranker accounts for nearly all of the GPU profile's time. Reranking fewer candidates is the main lever for speed (exact, because the scores for all 30 candidates are stored):

| Rerank depth | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| Recall@5 | 0.785 | 0.845 | 0.894 | 0.903 |

## How it works

```
question ─┬─> dense search (FAISS, BGE-M3 or e5-small) ─┐
          └─> BM25 (Indic-aware tokenizer)  [GPU only] ─┴─> RRF fusion
                                                              │
                                     cross-encoder rerank (top 30) [GPU only]
                                                              │
                         top score < per-language threshold? ─┴─> refuse, in the user's language
                                                              │
                     LLM answer with [n] citations, or extractive answer if no LLM
                                                              │
                   citation check: no valid citation, or the model says NO_ANSWER ─> refuse
```

Design choices:
- **Indic-aware tokenization.** Python's `\w` does not match combining vowel signs, so `re.findall(r"\w+", "की")` splits the word apart. The tokenizer uses Unicode letter, mark and number classes, applies NFC normalisation, and strips zero-width joiners. A test documents this ([tests/test_text.py](tests/test_text.py)).
- **Rank fusion instead of score addition.** BM25 scores and cosine similarities are on different scales, and RRF uses positions only.
- **Threshold per language.** Score distributions differ by language. Global and per-language thresholds are both reported in [reports/refusal.json](reports/refusal.json).
- **No data leakage.** Thresholds and design decisions use the validation split only, and the split is a fixed hash of the query id.
- **Pluggable generation.** Answers can come from a local model, any OpenAI-compatible API (Groq, Gemini, Ollama), or no LLM at all.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or /whl/cpu
pip install -e ".[ml,ui,dev,local-llm]"

python scripts/build_data.py        # download data for 8 languages (about 40 minutes)
python scripts/run_retrieval.py     # all retrieval methods, about 35 minutes on an 8 GB GPU
python scripts/evaluate.py          # metrics and thresholds, written to reports/
python -m bhasharag.ui              # demo at http://127.0.0.1:7860
```

Optional benchmarks: `scripts/run_generation.py` (needs a GPU) and `scripts/benchmark_latency.py`.

API:

```bash
uvicorn bhasharag.api:app --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "भारत की राजधानी क्या है?", "lang": "hi"}'
```

The response includes the answer, whether it refused and why, the confidence score and threshold, the cited sources, and timing for each stage.

## Deploy

The [Dockerfile](Dockerfile) builds the CPU profile on port 7860, which suits a free Hugging Face Docker Space. On first start it downloads data for the languages in `BHASHARAG_LANGS` and builds the embeddings. To get generated answers instead of extractive ones, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` as Space secrets.

Not yet verified: the Docker image has not been built or tested, because Docker was not available on the development machine.

## Limitations

- **Simulated unanswerable questions.** Removing the labelled passage does not guarantee that no other passage answers the question. Some answers counted as wrong may actually be correct, so the wrong-answer rates are pessimistic.
- **Easy distractors.** They come from Wikipedia, while the questions come from translated MS MARCO, so many distractors are off-topic. A same-domain corpus would be harder.
- **One relevant passage per question.** Other passages that also answer count as misses, which understates Recall.
- **Small samples.** Thresholds come from about 294 validation questions per language, and generation was scored on 200 questions.
- **Same-language retrieval only.** Cross-lingual questions, and Romanized input such as Hinglish, are not handled yet.

## Repository layout

```
src/bhasharag/   text.py (tokenizer) · data.py · retrieval.py · metrics.py · generation.py · pipeline.py · api.py · ui.py
scripts/         build_data · run_retrieval · evaluate · run_generation · benchmark_latency
tests/           tokenizer, metrics, data, retrieval, generation and API tests (run in CI)
reports/         JSON results behind every number in this README
```

## Data and model credits

- [ai4bharat/IndicMSMARCO](https://huggingface.co/datasets/ai4bharat/IndicMSMARCO) and [ai4bharat/Indic-Rag-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite) (MIT), from AI4Bharat.
- Models: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
- Check each dataset and model licence before any commercial use.
