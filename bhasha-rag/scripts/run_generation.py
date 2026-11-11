"""Evaluate answer generation on a sample of test queries, with and without the gold passage.

Uses the reranked passages stored by run_retrieval.py, so results reflect the GPU profile.
Usage: python scripts/run_generation.py [--per-language 25]
Output: reports/generation.json
"""

import argparse
import json
import random

import numpy as np

from bhasharag import config
from bhasharag.data import load_language
from bhasharag.generation import (
    LocalGenerator,
    build_messages,
    extractive_answer,
    parse_answer,
)
from bhasharag.metrics import token_f1
from bhasharag.pipeline import CONTEXT_PASSAGES


def score(parsed, gold_position, reference):
    answered = not parsed.refused and parsed.grounded
    return {
        "answered": answered,
        "f1": token_f1(parsed.text, reference) if answered else 0.0,
        "cites_gold": answered and gold_position in parsed.citations,
    }


def summarise(answerable, unanswerable):
    return {
        "n": len(answerable),
        "answer_rate": float(np.mean([r["answered"] for r in answerable])),
        "token_f1": float(np.mean([r["f1"] for r in answerable])),
        "cites_gold_rate": float(np.mean([r["cites_gold"] for r in answerable])),
        "false_answer_rate": float(np.mean([r["answered"] for r in unanswerable])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", type=int, default=25)
    args = parser.parse_args()

    generator = LocalGenerator()
    rng = random.Random(config.SEED)
    results = {"model": generator.name, "per_language": {}}
    pooled = {"llm": ([], []), "extractive": ([], [])}

    for lang in config.LANGUAGES:
        corpus, queries = load_language(lang)
        text_of = dict(zip(corpus["doc_id"], corpus["text"], strict=True))
        reference_of = dict(zip(queries["query_id"], queries["answer"], strict=True))
        question_of = dict(zip(queries["query_id"], queries["query"], strict=True))
        runs = json.loads((config.ARTIFACT_DIR / lang / "runs.json").read_text(encoding="utf-8"))
        # Only queries where retrieval put the gold passage in context, to isolate generation.
        eligible = [
            r for r in runs if r["split"] == "test" and r["gold"] in r["rerank"][:CONTEXT_PASSAGES]
        ]
        sample = rng.sample(eligible, min(args.per_language, len(eligible)))

        per_lang = {"llm": ([], []), "extractive": ([], [])}
        for row in sample:
            question, reference = question_of[row["query_id"]], reference_of[row["query_id"]]
            with_gold = row["rerank"][:CONTEXT_PASSAGES]
            without_gold = [d for d in row["rerank"] if d != row["gold"]][:CONTEXT_PASSAGES]
            gold_position = with_gold.index(row["gold"]) + 1

            for condition, doc_list in ((0, with_gold), (1, without_gold)):
                passages = [text_of[d] for d in doc_list]
                raw = generator.generate(build_messages(question, passages, lang))
                position = gold_position if condition == 0 else -1
                per_lang["llm"][condition].append(
                    score(parse_answer(raw, len(passages)), position, reference)
                )
                per_lang["extractive"][condition].append(
                    score(extractive_answer(passages), position, reference)
                )

        results["per_language"][lang] = {k: summarise(*v) for k, v in per_lang.items()}
        for key in pooled:
            pooled[key][0].extend(per_lang[key][0])
            pooled[key][1].extend(per_lang[key][1])
        print(lang, json.dumps(results["per_language"][lang]["llm"]), flush=True)

    results["overall"] = {k: summarise(*v) for k, v in pooled.items()}
    (config.REPORT_DIR / "generation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results["overall"], indent=2))


if __name__ == "__main__":
    main()
