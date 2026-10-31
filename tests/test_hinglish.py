import random

import pandas as pd
import pytest

from langscout.config import HINGLISH_TEST_CSV, MODEL_PATH
from langscout.detector import LanguageDetector
from langscout.translit import hindi_to_hinglish


def test_transliteration_looks_like_hinglish():
    out = hindi_to_hinglish("मुझे एक कप चाय चाहिए, कृपया।", random.Random(0))
    assert out.isascii() and out.startswith("mujhe ek kap")
    assert "chahiye" in out


def test_spelling_varies_between_texts():
    outs = {hindi_to_hinglish("आज बहुत अच्छा दिन है और मौसम सुहावना है", random.Random(i)) for i in range(12)}
    assert len(outs) > 3


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")
def test_hinglish_sentences_are_recognised():
    detector = LanguageDetector()
    df = pd.read_csv(HINGLISH_TEST_CSV)
    hits = sum(detector.detect(t).language == "Hinglish" for t in df["text"])
    assert hits / len(df) >= 0.8


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")
def test_plain_english_is_not_mistaken_for_hinglish():
    detector = LanguageDetector()
    for text in ("Where is the nearest train station?", "I would like a cup of coffee, please.",
                 "The meeting has been moved to Thursday afternoon."):
        assert detector.detect(text).language == "English"
