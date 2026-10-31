"""Explain a prediction of the Naive Bayes pipeline.

Naive Bayes scores a language by summing, over every character n-gram of the text, the n-gram's TF-IDF weight
times its log-probability under that language. The difference between two languages' scores is therefore a plain
sum of per-n-gram terms, so we can say exactly which parts of the text pushed the decision and by how much.
"""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from .preprocess import vectorizer_preprocessor


def explain_pair(pipeline, text: str, target: str, versus: str, top_n: int = 8) -> dict:
    """Why ``target`` rather than ``versus``?  Positive numbers favour ``target``."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    t_idx, v_idx = classes.index(target), classes.index(versus)
    diff = clf.feature_log_prob_[t_idx] - clf.feature_log_prob_[v_idx]       # per-feature evidence

    x = vec.transform([text])
    weight = dict(zip(x.indices, x.data))                                    # tf-idf weight of each known n-gram
    contrib = {i: weight[i] * diff[i] for i in weight}

    analyzer = vec.build_analyzer()
    words = vectorizer_preprocessor(text).split()
    per_word = [Counter(vec.vocabulary_[g] for g in analyzer(w) if g in vec.vocabulary_) for w in words]
    totals = Counter()
    for c in per_word:
        totals.update(c)
    word_scores = [(w, float(sum(contrib[i] * n / totals[i] for i, n in c.items()))) for w, c in zip(words, per_word)]

    inverse = {i: g for g, i in vec.vocabulary_.items()}
    ranked = sorted(((inverse[i].replace(" ", "_"), float(v)) for i, v in contrib.items()), key=lambda kv: kv[1])
    return {
        "target": target, "versus": versus,
        "margin": float(sum(contrib.values())),
        "words": word_scores,
        "favours_target": [kv for kv in ranked[::-1][:top_n] if kv[1] > 0],
        "favours_versus": [kv for kv in ranked[:top_n] if kv[1] < 0],
    }


def highlight_html(word_scores: list[tuple[str, float]]) -> str:
    """Words shaded green (towards the predicted language) or red (towards the runner-up)."""
    if not word_scores:
        return ""
    peak = max(abs(s) for _, s in word_scores) or 1.0
    spans = []
    for word, score in word_scores:
        alpha = min(abs(score) / peak, 1.0) * 0.65
        rgb = "46,160,67" if score >= 0 else "215,58,73"
        spans.append(f'<span style="background:rgba({rgb},{alpha:.2f});padding:1px 3px;border-radius:3px">'
                     f'{html.escape(word)}</span>')
    return " ".join(spans)
