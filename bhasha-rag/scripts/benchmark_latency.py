"""Measure per-stage latency of the real online pipeline, one query at a time.

The CPU profile is measured on the CPU, the way a free hosting tier would run it.
Usage: python scripts/benchmark_latency.py [--lang hi] [--n 100]
Output: reports/latency.json
"""

import argparse
import json
import platform

import numpy as np

from bhasharag import config
from bhasharag.data import load_language
from bhasharag.pipeline import BhashaRAG

WARMUP = 5


def percentiles(values) -> dict[str, float]:
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    _, queries = load_language(args.lang)
    questions = queries[queries["split"] == "test"]["query"].tolist()[: args.n]
    report = {"lang": args.lang, "n": len(questions), "cpu": platform.processor(), "profiles": {}}

    for profile, device in (("cpu", "cpu"), ("gpu", "cuda")):
        rag = BhashaRAG([args.lang], profile=profile, device=device)
        for question in questions[:WARMUP]:
            rag.retrieve(question, args.lang)
        timings = [rag.retrieve(q, args.lang)[2] for q in questions]
        stages = {stage: percentiles([t[stage] for t in timings]) for stage in timings[0]}
        stages["total"] = percentiles([sum(t.values()) for t in timings])
        report["profiles"][profile] = {"device": device, "stages_ms": stages}
        print(profile, json.dumps(stages), flush=True)
        del rag

    (config.REPORT_DIR / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
