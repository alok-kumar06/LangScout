"""Unicode script detection.

Language detection is much easier once the writing system is known: Tamil text can only
be Tamil (among our 17 languages), while Latin text needs the statistical model to choose
between ten candidates.  This module maps characters to scripts and scripts to the
languages supported by the project.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

# (start, end, script) - inclusive code-point ranges.
_RANGES = [
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"), (0x00C0, 0x024F, "Latin"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"), (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"), (0xAC00, 0xD7AF, "Hangul"),
    (0x3040, 0x309F, "Hiragana"), (0x30A0, 0x30FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"),
]

# Scripts of the 17 supported languages -> candidate languages.
SCRIPT_TO_LANGUAGES = {
    "Latin": ["Danish", "Dutch", "English", "French", "German", "Italian",
              "Portuguese", "Spanish", "Swedish", "Turkish", "Hinglish"],
    "Cyrillic": ["Russian"],
    "Greek": ["Greek"],
    "Arabic": ["Arabic"],
    "Devanagari": ["Hindi"],
    "Tamil": ["Tamil"],
    "Kannada": ["Kannada"],
    "Malayalam": ["Malayalam"],
}

# The script each language is normally written in (used for data-quality checks).
LANGUAGE_SCRIPT = {lang: script for script, langs in SCRIPT_TO_LANGUAGES.items() for lang in langs}

# Scripts shared with languages we do NOT support - the app warns the user about these.
SHARED_SCRIPT_NOTES = {
    "Arabic": "Arabic script is also used by Persian, Urdu and others, which this model does not know.",
    "Cyrillic": "Cyrillic is also used by Ukrainian, Bulgarian, Serbian and others, which this model does not know.",
    "Devanagari": "Devanagari is also used by Marathi, Nepali and Sanskrit, which this model does not know.",
}

_SORTED = sorted(_RANGES)


def char_script(ch: str) -> str | None:
    """Script of a single character, or None for digits / punctuation / spaces / symbols."""
    category = unicodedata.category(ch)
    if category[0] not in ("L", "M"):  # letters and combining marks (Indic vowel signs are marks)
        return None
    cp = ord(ch)
    for start, end, script in _SORTED:
        if start <= cp <= end:
            return script
    return "Other"


def script_profile(text: str) -> Counter:
    """Count letters per script in ``text``."""
    return Counter(s for s in map(char_script, text) if s is not None)


def dominant_script(text: str) -> tuple[str | None, float]:
    """Return (script, share_of_letters). ``(None, 0.0)`` when the text has no letters."""
    counts = script_profile(text)
    total = sum(counts.values())
    if total == 0:
        return None, 0.0
    script, n = counts.most_common(1)[0]
    return script, n / total


def letter_count(text: str) -> int:
    return sum(script_profile(text).values())
