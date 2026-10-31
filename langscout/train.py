"""End-to-end training + benchmarking pipeline.

    python -m langscout.train            # sklearn models only (about 1 minute)
    python -m langscout.train --deep     # also benchmark Char-CNN and the original word-level SimpleRNN

Outputs: models/langscout_model.joblib, models/metadata.json and everything in reports/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from .config import (EVERYDAY_CSV, HINGLISH_TEST_CSV, METADATA_PATH, MODEL_DIR, MODEL_PATH, REPORT_DIR, SEED,
                     TRUNCATION_LENGTHS)
from .data import add_hinglish, chunk_frame, load_clean_dataset, split_rows
from .evaluate import (evaluate, plot_confusion, plot_per_class_f1, plot_reliability, plot_short_text,
                       selective_curve, short_text_curve, strip_arrays, tune_threshold)
from .models import build_sklearn_models, ngram_coverage

warnings.filterwarnings("ignore")
BEST_SKLEARN_PREFERENCE = "macro_f1"      # model selection uses the VALIDATION set only


def cross_validate_sklearn(df: pd.DataFrame, k: int = 5) -> dict[str, dict]:
    """Row-level stratified k-fold (chunking happens inside each fold -> no leakage)."""
    scores: dict[str, list[float]] = {}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df["language"]), 1):
        tr = add_hinglish(chunk_frame(df.iloc[tr_idx]), with_seed=True)
        te = add_hinglish(chunk_frame(df.iloc[te_idx]))
        for name, model in build_sklearn_models().items():
            model.fit(tr["text"], tr["language"])
            pred = model.predict(te["text"])
            from sklearn.metrics import f1_score
            scores.setdefault(name, []).append(f1_score(te["language"], pred, average="macro"))
        print(f"  CV fold {fold}/{k} done")
    return {n: {"macro_f1_mean": float(np.mean(s)), "macro_f1_std": float(np.std(s)), "folds": [float(x) for x in s]}
            for n, s in scores.items()}


def main(deep: bool = False, cv: bool = True) -> None:
    t0 = time.time()
    REPORT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

    print("1/7 Loading and auditing data ...")
    df, audit = load_clean_dataset()
    train, val, test = split_rows(df)
    ctr = add_hinglish(chunk_frame(train), with_seed=True)
    cva, cte = add_hinglish(chunk_frame(val)), add_hinglish(chunk_frame(test))
    assert not (set(train.row_id) & set(test.row_id)) and not (set(train.row_id) & set(val.row_id))
    audit.update({"train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                  "train_chunks": len(ctr), "val_chunks": len(cva), "test_chunks": len(cte),
                  "chunks_per_language_train": ctr["language"].value_counts().to_dict()})
    everyday = pd.read_csv(EVERYDAY_CSV)
    hinglish_hw = pd.read_csv(HINGLISH_TEST_CSV)
    classes = np.array(sorted(ctr["language"].unique()))

    print("2/7 Training scikit-learn candidates ...")
    fitted: dict = {}
    for name, model in build_sklearn_models().items():
        model.fit(ctr["text"], ctr["language"])
        fitted[name] = model
        print(f"  trained {name}")

    if deep:
        from .deep import CharCNN, WordSimpleRNN, tensorflow_available
        if not tensorflow_available():
            print("  TensorFlow not installed -> skipping deep models (pip install -r requirements-deep.txt)")
        else:
            print("3/7 Training deep-learning benchmarks (this takes a few minutes) ...")
            for cls in (WordSimpleRNN, CharCNN):
                m = cls().fit(ctr["text"].tolist(), ctr["language"].values, cva["text"].tolist(), cva["language"].values)
                fitted[m.name] = m
                print(f"  trained {m.name} ({m.epochs_trained} epochs)")

    print("4/7 Evaluating on validation, test and the everyday out-of-domain set ...")
    results, curves = {}, {}
    for name, model in fitted.items():
        r_val = evaluate(model, cva["text"], cva["language"], classes)
        r_test = evaluate(model, cte["text"], cte["language"], classes)
        r_hw = evaluate(model, everyday["text"], everyday["language"], classes)
        r_hg = evaluate(model, hinglish_hw["text"], hinglish_hw["language"], classes)
        deployable = hasattr(model, "named_steps")
        row_acc = None
        if deployable:   # original (un-chunked) test rows - includes the very long Wikipedia paragraphs
            row_acc = float((model.predict(test["text"]) == test["language"].values).mean())
        results[name] = {"deployable": deployable, "val": strip_arrays(r_val), "test": strip_arrays(r_test),
                         "everyday": strip_arrays(r_hw), "hinglish_everyday_accuracy": r_hg["accuracy"],
                         "test_row_level_accuracy": row_acc,
                         "_test_full": r_test}
        curves[name] = short_text_curve(model, cte["text"].tolist(), cte["language"].values, TRUNCATION_LENGTHS)

    best_name = max((n for n, r in results.items() if r["deployable"]),
                    key=lambda n: results[n]["val"][BEST_SKLEARN_PREFERENCE])
    best = fitted[best_name]
    best_test = results[best_name]["_test_full"]
    print(f"  selected (by validation macro-F1): {best_name}")

    print("5/7 Tuning the abstention threshold on the validation set ...")
    val_full = evaluate(best, cva["text"], cva["language"], classes)
    sel_val = selective_curve(val_full["_probs"], val_full["_y"])
    threshold = tune_threshold(sel_val, target_accuracy=0.995)
    cov_val = np.array([ngram_coverage(best, t) for t in cva["text"] if len(t) >= 8])
    min_coverage = float(np.floor(np.percentile(cov_val, 0.5) * 100) / 100)   # 99.5% of real validation text passes
    print(f"  n-gram coverage floor (0.5th percentile of validation text): {min_coverage:.2f}")
    sel_test = selective_curve(best_test["_probs"], best_test["_y"])
    at = min(sel_test, key=lambda r: abs(r["threshold"] - threshold))
    print(f"  threshold={threshold:.2f} -> test coverage {at['coverage']:.3f}, accuracy on accepted {at['accuracy']:.4f}")

    cv_results = {}
    if cv:
        print("6/7 5-fold cross-validation (sklearn models) ...")
        cv_results = cross_validate_sklearn(df)
    else:
        print("6/7 Skipping cross-validation")

    print("7/7 Saving model, metrics and plots ...")
    joblib.dump(best, MODEL_PATH, compress=3)
    for name, r in results.items():
        r.pop("_test_full")
    comparison = pd.DataFrame([{
        "model": n, "deployed": n == best_name,
        "val_macro_f1": r["val"]["macro_f1"], "test_accuracy": r["test"]["accuracy"],
        "test_macro_f1": r["test"]["macro_f1"], "test_log_loss": r["test"]["log_loss"], "test_ece": r["test"]["ece"],
        "everyday_accuracy": r["everyday"]["accuracy"],
        "hinglish_everyday_accuracy": r["hinglish_everyday_accuracy"],
        "acc_5_chars": curves[n][5], "acc_20_chars": curves[n][20], "acc_80_chars": curves[n][80],
        "cv_macro_f1_mean": cv_results.get(n, {}).get("macro_f1_mean"),
        "cv_macro_f1_std": cv_results.get(n, {}).get("macro_f1_std"),
    } for n, r in results.items()]).sort_values("test_macro_f1", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False, float_format="%.4f")

    (REPORT_DIR / "metrics.json").write_text(json.dumps({
        "selected_model": best_name, "confidence_threshold": threshold, "min_ngram_coverage": min_coverage,
        "results": results, "short_text_accuracy": {n: {str(k): v for k, v in c.items()} for n, c in curves.items()},
        "selective_prediction_test": sel_test, "cross_validation": cv_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_confusion(best_test["_y"], best_test["_pred"], classes, REPORT_DIR / "confusion_matrix.png",
                   f"Test-set confusion matrix - {best_name}")
    plot_per_class_f1(results[best_name]["test"]["per_class"], REPORT_DIR / "per_class_f1.png",
                      "Per-language F1 on the test set")
    plot_short_text(curves, REPORT_DIR / "short_text_robustness.png")
    plot_reliability(best_test["_probs"], best_test["_y"], REPORT_DIR / "calibration.png",
                     "Reliability diagram (test set)")

    METADATA_PATH.write_text(json.dumps({
        "model": best_name, "trained_on": date.today().isoformat(), "confidence_threshold": threshold,
        "min_ngram_coverage": min_coverage,
        "languages": classes.tolist(), "sklearn_version": sklearn.__version__, "python": platform.python_version(),
        "test_accuracy": results[best_name]["test"]["accuracy"], "test_macro_f1": results[best_name]["test"]["macro_f1"],
        "everyday_accuracy": results[best_name]["everyday"]["accuracy"],
        "hinglish_everyday_accuracy": results[best_name]["hinglish_everyday_accuracy"],
        "train_chunks": len(ctr), "seed": SEED,
    }, indent=2), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s.\n")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", action="store_true", help="also benchmark Keras models (needs TensorFlow)")
    ap.add_argument("--no-cv", action="store_true", help="skip 5-fold cross-validation")
    a = ap.parse_args()
    main(deep=a.deep, cv=not a.no_cv)
