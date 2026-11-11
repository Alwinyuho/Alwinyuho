"""Paths and settings shared by every experiment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SHELFRANK_HOME", Path(__file__).resolve().parents[2]))
DOWNLOADS = Path(os.environ.get("SHELFRANK_DATA", ROOT.parent / "_downloads"))
AMAZON_DIR = DOWNLOADS / "amazon"
OBD_DIR = DOWNLOADS / "obd"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

CATEGORY = "Video_Games"
SEED = 13
TEXT_MODEL = "BAAI/bge-small-en-v1.5"

HISTORY_LEN = 20
RETRIEVE_K = 100
EVAL_KS = (10, 50)
