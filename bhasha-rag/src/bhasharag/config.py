"""Central settings. Every experiment reads from here so runs stay comparable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("BHASHARAG_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

LANGUAGES: dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
}

SEED = 13
N_DISTRACTORS = 5000
VAL_FRACTION_IN_TENTHS = 3

CANDIDATES = 100
RERANK_DEPTH = 30
RRF_K = 60
TARGET_FALSE_ANSWER_RATE = 0.10


@dataclass(frozen=True)
class Encoder:
    key: str
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""


E5_SMALL = Encoder(
    "e5small", "intfloat/multilingual-e5-small", query_prefix="query: ", passage_prefix="passage: "
)
BGE_M3 = Encoder("bgem3", "BAAI/bge-m3")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LOCAL_LLM_MODEL = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class Profile:
    """A deployable combination of models and the score used for the refusal decision."""

    name: str
    encoder: Encoder
    use_bm25: bool
    use_reranker: bool


# Chosen from the validation results in reports/retrieval.json: fusing BM25 with
# equal weight lowered MRR for both encoders, so the CPU profile uses dense search only.
PROFILES = {
    "cpu": Profile("cpu", E5_SMALL, use_bm25=False, use_reranker=False),
    "gpu": Profile("gpu", BGE_M3, use_bm25=True, use_reranker=True),
}

REFUSAL_TEXT = {
    "hi": "दिए गए स्रोतों में इस प्रश्न का उत्तर नहीं मिला।",
    "bn": "প্রদত্ত উৎসগুলিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।",
    "ta": "கொடுக்கப்பட்ட ஆதாரங்களில் இந்தக் கேள்விக்கான பதில் கிடைக்கவில்லை.",
    "te": "ఇచ్చిన మూలాల్లో ఈ ప్రశ్నకు సమాధానం కనుగొనబడలేదు.",
    "mr": "दिलेल्या स्रोतांमध्ये या प्रश्नाचे उत्तर सापडले नाही.",
    "ml": "നൽകിയ ഉറവിടങ്ങളിൽ ഈ ചോദ്യത്തിനുള്ള ഉത്തരം കണ്ടെത്തിയില്ല.",
    "kn": "ನೀಡಿದ ಮೂಲಗಳಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.",
    "gu": "આપેલા સ્ત્રોતોમાં આ પ્રશ્નનો જવાબ મળ્યો નથી.",
}
