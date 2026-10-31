"""Compare LangScout with two popular open-source detectors on identical data.

    pip install langdetect langid
    python -m langscout.benchmark

Fairness notes
* Only the 17 languages LangScout knows are scored (the Hinglish class is left out of the test data).
* ``langid`` is restricted to those 17 languages.  ``langdetect`` cannot be restricted, so a prediction outside the
  17 counts as wrong.
* All tools see exactly the same texts: held-out test chunks, short truncations of them, and the separate sample set.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .config import EVERYDAY_CSV, REPORT_DIR, SEED, TRUNCATION_LENGTHS
from .data import chunk_frame, load_clean_dataset, split_rows
from .detector import LanguageDetector

warnings.filterwarnings("ignore")

ISO = {"ar": "Arabic", "da": "Danish", "nl": "Dutch", "en": "English", "fr": "French", "de": "German",
       "el": "Greek", "hi": "Hindi", "it": "Italian", "kn": "Kannada", "ml": "Malayalam", "pt": "Portuguese",
       "ru": "Russian", "es": "Spanish", "sv": "Swedish", "ta": "Tamil", "tr": "Turkish"}


def make_predictors() -> dict:
    import langdetect
    import langid

    langdetect.DetectorFactory.seed = SEED
    langid.set_languages(list(ISO))
    ours = LanguageDetector()

    def lang_detect(text):
        try:
            return ISO.get(langdetect.detect(text), "other")
        except Exception:               # raised for texts without letters
            return "other"

    def lang_id(text):
        return ISO[langid.classify(text)[0]]

    def lang_scout(text):
        return ours.detect(text).language or "other"

    return {"LangScout (this project)": lang_scout, "langid (restricted to 17)": lang_id, "langdetect": lang_detect}


def accuracy(predict, texts, labels) -> float:
    return float(np.mean([predict(t) == l for t, l in zip(texts, labels)]))


def main() -> None:
    df, _ = load_clean_dataset()
    _, _, test = split_rows(df)
    chunks = chunk_frame(test)
    texts, labels = chunks["text"].tolist(), chunks["language"].tolist()
    sample = pd.read_csv(EVERYDAY_CSV)
    predictors = make_predictors()

    rows = []
    for name, predict in predictors.items():
        start = time.perf_counter()
        acc = accuracy(predict, texts, labels)
        ms = (time.perf_counter() - start) / len(texts) * 1000
        row = {"tool": name, "test_accuracy": acc, "sample_set_accuracy": accuracy(predict, sample["text"], sample["language"]),
               "ms_per_text": ms}
        for k in (5, 20, 80):
            row[f"acc_{k}_chars"] = accuracy(predict, [t[:k] for t in texts], labels)
        rows.append(row)
        print(f"{name:28s} test {acc:.4f}  samples {row['sample_set_accuracy']:.3f}  {ms:.2f} ms/text")
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "benchmark_vs_libraries.csv", index=False, float_format="%.4f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = {n: [accuracy(p, [t[:k] for t in texts], labels) * 100 for k in TRUNCATION_LENGTHS] for n, p in predictors.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ys in curves.items():
        ax.plot(TRUNCATION_LENGTHS, ys, marker="o", label=name)
    ax.set_xscale("log"); ax.set_xticks(TRUNCATION_LENGTHS); ax.set_xticklabels(TRUNCATION_LENGTHS)
    ax.set_xlabel("Characters given to the detector"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("LangScout vs popular libraries (same 17 languages, same texts)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(REPORT_DIR / "benchmark_vs_libraries.png", dpi=140)
    print("saved reports/benchmark_vs_libraries.csv and .png")


if __name__ == "__main__":
    main()
