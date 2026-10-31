"""Text cleaning and chunking shared by training and inference."""
from __future__ import annotations

import re
import unicodedata

from .config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS

_ESCAPES = re.compile(r"\\[nrt]|\\x[0-9a-fA-F]{2}")      # literal "\n", "\xa0" found in the raw CSV
_URL = re.compile(r"https?://\S+|www\.\S+")
_BRACKETS = re.compile(r"\[[^\]]{0,60}\]")                 # wiki citations such as [12] or [citation needed]
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?\u0964\u0965\u061F\u2026])\s+")  # . ! ? | || Arabic ? ...


def clean_text(text: object) -> str:
    """Normalise raw text. Keeps letters and punctuation, drops noise (URLs, citations, digits)."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _ESCAPES.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _BRACKETS.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def vectorizer_preprocessor(text: object) -> str:
    """Used inside the sklearn pipeline, so the saved model accepts raw strings."""
    return clean_text(text).lower()


def chunk_text(text: object, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-aligned chunks of at most ``max_chars`` characters."""
    text = clean_text(text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:                   # very long sentence: cut at a space
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            flush()
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            flush()
        current = f"{current} {sentence}".strip()
    flush()
    return [c for c in chunks if len(c) >= min_chars]
