"""Evaluation utilities: metrics, calibration, short-text robustness, selective prediction, plots."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def proba(model, texts) -> np.ndarray:
    return np.asarray(model.predict_proba(list(texts)))


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    """Average |confidence - accuracy| over confidence bins, weighted by bin size (lower is better)."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)


def evaluate(model, texts, labels, classes) -> dict:
    """Full metric bundle on one dataset."""
    classes = np.asarray(classes)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[l] for l in labels])
    p = proba(model, texts)
    pred = p.argmax(1)
    report = classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1), labels=range(len(classes)))),
        "ece": expected_calibration_error(p, y),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
        "_probs": p, "_y": y, "_pred": pred,
    }


def strip_arrays(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def short_text_curve(model, texts, labels, lengths) -> dict[int, float]:
    """Accuracy when every test text is truncated to its first ``k`` characters."""
    labels = np.asarray(labels)
    out = {}
    for k in lengths:
        cut = [t[:k] for t in texts]
        out[k] = float((np.asarray(model.predict(cut)) == labels).mean())
    return out


def selective_curve(probs, y_idx, thresholds=None) -> list[dict]:
    """Coverage / accuracy trade-off if the system abstains below a confidence threshold."""
    thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2) if thresholds is None else thresholds
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"threshold": float(t), "coverage": float(keep.mean()),
                     "accuracy": float((pred[keep] == y_idx[keep]).mean()) if keep.any() else float("nan")})
    return rows


def tune_threshold(curve: list[dict], target_accuracy: float = 0.99, fallback: float = 0.5) -> float:
    """Smallest threshold whose accepted predictions reach ``target_accuracy`` (keeps coverage high)."""
    ok = [r["threshold"] for r in curve if r["coverage"] > 0.5 and r["accuracy"] >= target_accuracy]
    return min(ok) if ok else fallback


# ------------------------------------------------------------------------------- plots
def plot_confusion(y, pred, classes, path, title):
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, fraction=0.04, label="Share of actual class")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_short_text(curves: dict[str, dict[int, float]], path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c in curves.items():
        ax.plot(list(c), [v * 100 for v in c.values()], marker="o", label=name)
    ax.set_xscale("log"); ax.set_xticks(list(next(iter(curves.values())))); ax.set_xticklabels(list(next(iter(curves.values()))))
    ax.set_xlabel("Characters given to the model (first k characters of each test text)")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("How much text does each model need?")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_per_class_f1(per_class: dict, path, title):
    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([k for k, _ in items], [v["f1-score"] * 100 for _, v in items], color="#3b82c4")
    ax.set_xlim(90, 100.5); ax.set_xlabel("F1 (%)"); ax.set_title(title); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_reliability(probs, y_idx, path, title, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append((pred[m] == y_idx[m]).mean())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.plot(xs, ys, marker="o", label="Model"); ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
