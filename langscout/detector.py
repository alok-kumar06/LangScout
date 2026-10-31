"""High-level inference API: script gating + statistical model + abstention.

    >>> from langscout import LanguageDetector
    >>> LanguageDetector().detect("Ceci est une belle journée").language
    'French'
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .config import DEFAULT_CONFIDENCE_THRESHOLD, METADATA_PATH, MODEL_PATH
from .explain import explain_pair
from .models import ngram_coverage
from .preprocess import clean_text
from .scripts import SCRIPT_TO_LANGUAGES, SHARED_SCRIPT_NOTES, dominant_script, script_profile

_SEGMENT_SPLIT = re.compile(r"(?<=[.!?\u0964\u0965\u061F\u2026])\s+|\n+")

# status values
OK, UNCERTAIN, NO_TEXT, UNSUPPORTED = "ok", "uncertain", "no_text", "unsupported_script"


@dataclass
class Prediction:
    text: str
    language: str | None          # best guess (None when nothing sensible can be said)
    confidence: float             # probability of ``language`` among the candidate languages
    status: str                   # ok | uncertain | no_text | unsupported_script
    script: str | None = None
    method: str = "model"         # "script" when the writing system alone decides, else "model"
    top_k: list[tuple[str, float]] = field(default_factory=list)
    note: str = ""

    @property
    def is_confident(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict:
        d = asdict(self)
        d["top_k"] = [(l, round(p, 4)) for l, p in self.top_k]
        d["confidence"] = round(self.confidence, 4)
        return d


class LanguageDetector:
    """Detect the language of a text among 17 supported languages."""

    def __init__(self, model_path: str | Path = MODEL_PATH, metadata_path: str | Path = METADATA_PATH,
                 threshold: float | None = None):
        self.model = joblib.load(model_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8")) if Path(metadata_path).exists() else {}
        self.classes = [str(c) for c in self.model.classes_]
        self._class_index = {c: i for i, c in enumerate(self.classes)}
        self.threshold = (threshold if threshold is not None
                          else self.metadata.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.min_coverage = self.metadata.get("min_ngram_coverage", 0.85)

    # ------------------------------------------------------------------ single text
    def detect(self, text: str, top_k: int = 3) -> Prediction:
        text = "" if text is None else str(text)
        script, share = dominant_script(text)
        if script is None:
            return Prediction(text, None, 0.0, NO_TEXT, None, note="No letters found - nothing to detect.")
        if script not in SCRIPT_TO_LANGUAGES:
            return Prediction(text, None, 0.0, UNSUPPORTED, script,
                              note=f"{script} script detected. This model supports only: "
                                   f"{', '.join(sorted(set(SCRIPT_TO_LANGUAGES)))} scripts.")

        candidates = SCRIPT_TO_LANGUAGES[script]
        probs = self.model.predict_proba([text])[0]
        mask = np.zeros_like(probs)
        mask[[self._class_index[c] for c in candidates]] = 1.0
        masked = probs * mask
        total = masked.sum()
        masked = masked / total if total > 0 else mask / mask.sum()

        order = np.argsort(masked)[::-1][: max(top_k, 1)]
        ranked = [(self.classes[i], float(masked[i])) for i in order]
        best, conf = ranked[0]
        method = "script" if len(candidates) == 1 else "model"

        notes = []
        if share < 0.7:
            notes.append(f"Mixed scripts ({share:.0%} {script}); predicting from the dominant one.")
        if script in SHARED_SCRIPT_NOTES:
            notes.append(SHARED_SCRIPT_NOTES[script])
        status = OK
        if conf < self.threshold:
            status = UNCERTAIN
            notes.append("Low confidence - the text may be too short, noisy or in an unsupported language.")
        elif len(clean_text(text)) < 4 and method == "model":
            status = UNCERTAIN
            notes.append("Very short text - treat this as a guess.")
        if (status == OK and method == "model" and len(clean_text(text)) >= 8
                and ngram_coverage(self.model, text) < self.min_coverage):
            status = UNCERTAIN
            notes.append("Unusual letter patterns for every supported language - this may not be real text.")
        return Prediction(text, best, conf, status, script, method, ranked, " ".join(notes))

    def detect_batch(self, texts, top_k: int = 3) -> list[Prediction]:
        return [self.detect(t, top_k) for t in texts]

    # ------------------------------------------------------------------ mixed-language text
    def detect_segments(self, text: str, merge: bool = True) -> list[Prediction]:
        """Detect the language of every sentence/line. Adjacent sentences in the same language are merged."""
        segments = [s.strip() for s in _SEGMENT_SPLIT.split(text or "") if s and s.strip()]
        preds = [self.detect(s) for s in segments]
        if not merge:
            return preds
        merged: list[Prediction] = []
        for p in preds:
            if merged and p.language and merged[-1].language == p.language and p.is_confident == merged[-1].is_confident:
                prev = merged[-1]
                prev.text = f"{prev.text} {p.text}"
                continue
            merged.append(p)
        return merged

    @property
    def languages(self) -> list[str]:
        return list(self.classes)

    # ------------------------------------------------------------------ explanation
    def explain(self, text: str, top_n: int = 8) -> dict | None:
        """Evidence for the predicted language versus the runner-up (None if there is no prediction)."""
        pred = self.detect(text, top_k=2)
        if pred.language is None or len(pred.top_k) < 2:
            return None
        return explain_pair(self.model, text, pred.top_k[0][0], pred.top_k[1][0], top_n=top_n)
