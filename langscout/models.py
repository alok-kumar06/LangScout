"""Classical (scikit-learn) model definitions. All pipelines accept RAW strings."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import SEED
from .preprocess import vectorizer_preprocessor


def make_vectorizer(ngram_range=(1, 3), min_df=2, max_features=None) -> TfidfVectorizer:
    """Character n-gram TF-IDF. ``char_wb`` pads words with spaces, so word boundaries become features."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        preprocessor=vectorizer_preprocessor,
        dtype="float32",
    )


def build_sklearn_models(**vec_kwargs) -> dict[str, Pipeline]:
    """The candidate models that can be deployed (lightweight, no deep-learning runtime needed)."""
    return {
        "Char n-gram TF-IDF + Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer(**vec_kwargs)),
            ("clf", MultinomialNB(alpha=0.01, fit_prior=False)),
        ]),
        "Char n-gram TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer(**vec_kwargs)),
            ("clf", LogisticRegression(C=30, max_iter=300, class_weight="balanced", random_state=SEED)),
        ]),
        "Char n-gram TF-IDF + Linear SVM (calibrated)": Pipeline([
            ("tfidf", make_vectorizer(**vec_kwargs)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3)),
        ]),
    }


def ngram_coverage(pipeline: Pipeline, text: str) -> float:
    """Share of the text's character n-grams that were seen in training (1.0 = fully familiar).

    Real text in a supported language scores ~0.95-1.0; random letters score much lower, which lets the
    detector refuse gibberish instead of returning a confident-looking guess.
    """
    vec = pipeline.named_steps["tfidf"]
    grams = [g for g in vec.build_analyzer()(text) if g.strip()]
    if not grams:
        return 0.0
    vocab = vec.vocabulary_
    return sum(g in vocab for g in grams) / len(grams)
