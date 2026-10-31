"""Turn Devanagari Hindi into 'Hinglish' (Hindi typed with English letters).

People rarely follow a fixed romanisation scheme, so every text gets its own random "spelling style"
(aa/a, ee/i, oo/u, v/w, ph/f ...) to expose the classifier to the variety it will meet in real chats.
"""
from __future__ import annotations

import random
import re

from indic_transliteration import sanscript

_CONS = "bcdfghjklmnpqrstvwxyz"
_MEDIAL_SCHWA = re.compile(rf"(?<=[aeiouAIU][{_CONS}])a(?=[{_CONS}]+[aeiouAIU])")
_WORD_VARIANTS = {
    "yah": ["yeh", "ye", "yah"], "vah": ["woh", "wo", "vo"], "nahIM": ["nahi", "nahin", "nahii"],
    "hai": ["hai", "hai", "hai", "he"], "haiM": ["hain", "hai", "hein"], "kyA": ["kya", "kyaa", "kya"],
    "aura": ["aur", "or", "aur"], "ko": ["ko", "ko", "ku"],
}


def _fix_word(word: str, rng: random.Random, style: dict) -> str:
    core, tail = re.match(r"^(.*?)([?!.,]*)$", word).groups()
    return _fix_core(core.replace(".", ""), rng) + tail


def _fix_core(word: str, rng: random.Random) -> str:
    if word in _WORD_VARIANTS:
        word = rng.choice(_WORD_VARIANTS[word])
    elif word.endswith("a") and len(word) > 2 and word[-2] != "a":      # final inherent 'a' is silent
        word = word[:-1]
    if rng.random() < 0.8:
        word = _MEDIAL_SCHWA.sub("", word)
    word = re.sub(r"ie$", "iye", word)
    return word


def hindi_to_hinglish(text: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    style = {
        "aa": rng.random() < 0.35, "ee": rng.random() < 0.3, "oo": rng.random() < 0.3,
        "w": rng.random() < 0.25, "f": rng.random() < 0.5, "s": rng.random() < 0.1,
    }
    s = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    s = s.replace("||", ".").replace("|", ".").replace("RRi", "ri").replace(".N", "n").replace("~N", "n").replace("~n", "n")
    s = s.replace("M", "n").replace("H", "h")
    s = re.sub(r"[^\w\s?!.,]", "", s)
    words = [_fix_word(w, rng, style) for w in s.split()]
    s = " ".join(words)
    s = s.replace("Ch", "chh").replace("ksh", "ksh").replace("x", "ksh").replace("GY", "gy")
    s = s.replace("A", "aa" if style["aa"] else "a").replace("I", "ee" if style["ee"] else "i")
    s = s.replace("U", "oo" if style["oo"] else "u").replace("E", "e").replace("O", "o")
    s = s.lower()
    if style["w"]:
        s = re.sub(r"v(?=[aeiou])", "w", s)
    if style["f"]:
        s = s.replace("ph", "f")
    if style["s"]:
        s = s.replace("sh", "s")
    return re.sub(r"\s+", " ", s).strip()
