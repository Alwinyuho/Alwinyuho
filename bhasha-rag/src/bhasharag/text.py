"""Text handling for Indian scripts.

Python's built-in ``\\w`` does not match combining vowel signs (for example the
matra in Hindi "की"), so ``re.findall(r"\\w+", text)`` silently splits Indic words
into fragments. Using Unicode categories with the ``regex`` package avoids that.
"""

from __future__ import annotations

import hashlib
import unicodedata

import regex

_TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))


def normalize(text: str) -> str:
    """Return NFC text without zero-width joiners, lower-cased and trimmed."""
    return unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).lower().strip()


def tokenize(text: str) -> list[str]:
    """Split text into word tokens, keeping combining marks attached."""
    return _TOKEN.findall(normalize(text))


def doc_id(text: str) -> str:
    """Stable id based on content, so duplicate passages collapse to one document."""
    collapsed = " ".join(tokenize(text))
    return hashlib.sha1(collapsed.encode("utf-8")).hexdigest()[:16]
