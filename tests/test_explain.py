import numpy as np
import pytest

from langscout.config import MODEL_PATH
from langscout.detector import LanguageDetector
from langscout.explain import explain_pair, highlight_html

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")


@pytest.fixture(scope="module")
def detector():
    return LanguageDetector()


def test_word_scores_add_up_to_the_true_log_odds(detector):
    text = "Jeg elsker at programmere i Python"
    e = explain_pair(detector.model, text, "Danish", "Swedish")
    vec, clf = detector.model[0], detector.model[1]
    classes = list(clf.classes_)
    scores = np.asarray(vec.transform([text]) @ clf.feature_log_prob_.T).ravel()
    true_margin = scores[classes.index("Danish")] - scores[classes.index("Swedish")]
    assert e["margin"] == pytest.approx(true_margin, abs=1e-6)
    assert sum(s for _, s in e["words"]) == pytest.approx(true_margin, abs=1e-6)


def test_detector_explain_uses_prediction_and_runner_up(detector):
    info = detector.explain("Ceci est une belle journée")
    assert info["target"] == "French" and info["margin"] > 0
    assert info["favours_target"]


def test_nothing_to_explain_without_a_prediction(detector):
    assert detector.explain("12345") is None
    assert detector.explain("这是一个测试") is None


def test_highlight_escapes_html():
    out = highlight_html([("<script>", 1.0), ("ok", -0.5)])
    assert "<script>" not in out and "&lt;script&gt;" in out
