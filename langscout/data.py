"""Dataset loading, quality audit, leakage-free splitting and chunking."""
from __future__ import annotations

import random

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (CONVERSATIONAL_HINDI, HINGLISH, LABEL_FIXES, SEED_VARIANTS, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, RAW_CSV, SEED,
                     TEST_SIZE, TRAIN_SIZE, VAL_SIZE)
from .preprocess import chunk_text, clean_text
from .scripts import LANGUAGE_SCRIPT, dominant_script, letter_count


def load_clean_dataset(path=RAW_CSV) -> tuple[pd.DataFrame, dict]:
    """Load the Kaggle CSV and apply every data-quality fix. Returns (dataframe, audit_log)."""
    raw = pd.read_csv(path)
    df = raw.rename(columns={"Text": "text", "Language": "language"})[["text", "language"]].copy()
    audit: dict = {"raw_rows": len(df)}

    fixed = df["language"].isin(LABEL_FIXES).sum()
    df["language"] = df["language"].str.strip().replace(LABEL_FIXES)
    audit["label_typos_fixed"] = int(fixed)

    df["text"] = df["text"].map(clean_text)

    few_letters = df["text"].map(letter_count) < 3
    audit["dropped_no_linguistic_content"] = int(few_letters.sum())
    df = df[~few_letters]

    key = df["text"].str.lower()
    conflict = df.groupby(key)["language"].transform("nunique") > 1
    audit["dropped_conflicting_labels"] = int(conflict.sum())
    df = df[~conflict]

    dup = df.assign(_k=df["text"].str.lower()).duplicated(subset=["_k", "language"])
    audit["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    script_ok = df.apply(lambda r: dominant_script(r["text"])[0] == LANGUAGE_SCRIPT[r["language"]], axis=1)
    audit["dropped_script_label_mismatch"] = int((~script_ok).sum())
    df = df[script_ok]

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index)
    audit["clean_rows"] = len(df)
    audit["languages"] = int(df["language"].nunique())
    audit["rows_per_language"] = df["language"].value_counts().to_dict()
    audit["median_chars_per_language"] = (
        df.assign(n=df["text"].str.len()).groupby("language")["n"].median().astype(int).to_dict()
    )
    return df, audit


def split_rows(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split at ROW level (before chunking, so no article leaks across splits)."""
    train, rest = train_test_split(df, test_size=VAL_SIZE + TEST_SIZE, stratify=df["language"], random_state=seed)
    val, test = train_test_split(rest, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
                                 stratify=rest["language"], random_state=seed)
    assert abs(len(train) / len(df) - TRAIN_SIZE) < 0.01
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def chunk_frame(df: pd.DataFrame, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS) -> pd.DataFrame:
    """Cut each row into sentence-aligned chunks (keeps ``row_id`` so grouping stays traceable)."""
    out = df.assign(text=df["text"].map(lambda t: chunk_text(t, max_chars, min_chars)))
    out = out.explode("text").dropna(subset=["text"])
    return out.reset_index(drop=True)[["row_id", "text", "language"]]


def add_hinglish(frame: pd.DataFrame, with_seed: bool = False, seed: int = SEED) -> pd.DataFrame:
    """Add the Hinglish class by transliterating the Hindi chunks of ``frame``.

    ``with_seed=True`` (training only) also adds a few dozen conversational Hindi sentences, in both
    Devanagari (Hindi) and several romanised spellings (Hinglish), because Wikipedia Hindi is very formal.
    """
    from .translit import hindi_to_hinglish

    rng = random.Random(seed)
    hindi = frame[frame["language"] == "Hindi"]
    extra = pd.DataFrame({"row_id": hindi["row_id"].values,
                          "text": [hindi_to_hinglish(t, rng) for t in hindi["text"]],
                          "language": HINGLISH})
    parts = [frame, extra]
    if with_seed:
        sentences = [l.strip() for l in CONVERSATIONAL_HINDI.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = -1 - pd.Series(range(len(sentences)))
        parts.append(pd.DataFrame({"row_id": ids.values, "text": sentences, "language": "Hindi"}))
        romanised = [(i, hindi_to_hinglish(s, rng)) for i, s in zip(ids, sentences) for _ in range(SEED_VARIANTS)]
        parts.append(pd.DataFrame({"row_id": [i for i, _ in romanised], "text": [t for _, t in romanised],
                                   "language": HINGLISH}))
    return pd.concat(parts, ignore_index=True)
