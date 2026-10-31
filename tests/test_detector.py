import pandas as pd
import pytest

from langscout.config import EVERYDAY_CSV, MODEL_PATH
from langscout.detector import NO_TEXT, OK, UNCERTAIN, UNSUPPORTED, LanguageDetector

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")


@pytest.fixture(scope="module")
def detector():
    return LanguageDetector()


def test_everyday_out_of_domain_set(detector):
    """51 everyday sentences (3 per language) that are NOT from the Wikipedia training data."""
    df = pd.read_csv(EVERYDAY_CSV)
    preds = [detector.detect(t).language for t in df["text"]]
    accuracy = (pd.Series(preds) == df["language"]).mean()
    assert accuracy >= 0.95, f"accuracy {accuracy:.2%}"


@pytest.mark.parametrize("text,expected", [
    ("This is a beautiful day", "English"),
    ("Ceci est une belle journée", "French"),
    ("Este es un buen día", "Spanish"),
    ("यह एक अच्छा दिन है", "Hindi"),
    ("ഇത് ഒരു നല്ല ദിവസമാണ്", "Malayalam"),
    ("Hej, hvordan har du det i dag?", "Danish"),
    ("Jag älskar att programmera i Python", "Swedish"),
])
def test_known_sentences(detector, text, expected):
    p = detector.detect(text)
    assert p.language == expected and p.status == OK


def test_long_hindi_sentence_regression(detector):
    """The original word-level SimpleRNN labelled this Turkish."""
    assert detector.detect("मैं आज बाजार जा रहा हूँ और वहाँ से सब्जियाँ खरीदूँगा").language == "Hindi"


def test_unsupported_script_is_flagged(detector):
    for text in ("这是一个测试", "これはテストです", "안녕하세요 여러분"):
        p = detector.detect(text)
        assert p.status == UNSUPPORTED and p.language is None


def test_no_letters(detector):
    for text in ("", "   ", "12345 !!!", None):
        assert detector.detect(text).status == NO_TEXT


def test_gibberish_is_flagged(detector):
    for text in ("asdkjh qwe zxcv", "qzxv wkjh pfgd"):
        assert detector.detect(text).status == UNCERTAIN


def test_ambiguous_danish_swedish_greeting_is_never_confidently_wrong(detector):
    """Short Scandinavian greeting that is valid in both languages: right answer or flagged as uncertain."""
    p = detector.detect("Hej, hur mår du idag?")
    assert p.language == "Swedish" or p.status == UNCERTAIN


def test_very_short_text_is_flagged(detector):
    assert detector.detect("hi").status == UNCERTAIN


def test_confidence_is_a_distribution(detector):
    p = detector.detect("Das ist ein Test", top_k=5)
    assert abs(sum(prob for _, prob in p.top_k) - 1) < 0.25 and p.top_k[0][0] == p.language
    assert all(a[1] >= b[1] for a, b in zip(p.top_k, p.top_k[1:]))


def test_script_gating_never_returns_impossible_language(detector):
    assert detector.detect("இது ஒரு நல்ல நாள்").language == "Tamil"
    assert detector.detect("Привет, как дела?").language == "Russian"
    assert detector.detect("Καλημέρα σε όλους").language == "Greek"


def test_segments_split_mixed_text(detector):
    segs = detector.detect_segments("I love this city. Me encanta esta ciudad. Ich liebe diese Stadt.")
    assert [s.language for s in segs] == ["English", "Spanish", "German"]


def test_batch(detector):
    preds = detector.detect_batch(["Hello there my friend", "Bonjour tout le monde"])
    assert [p.language for p in preds] == ["English", "French"]
