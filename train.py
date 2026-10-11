"""Train and evaluate phishing-URL classifiers.

Usage:
    python train.py                         # data/urls.csv if present, else synthetic data
    python train.py --data my_urls.csv      # CSV with columns: url,label[,date]  (label 1 = phishing)
    python train.py --split random          # force a stratified random split
    python train.py --split time            # force a chronological split (needs a date column)

How the data is used (so the reported numbers are honest):
    train       fit the models
    validation  choose the strict alert threshold (nothing is fitted on it)
    test        the only data any reported score comes from; touched once, at the end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import load_dataset            # noqa: E402
from src.features import FEATURE_NAMES, featurize   # noqa: E402
from src.nn import TorchMLPClassifier        # noqa: E402
from src.split import make_split             # noqa: E402
from src.threshold import TARGET_PRECISION, pick_strict_threshold   # noqa: E402

def metrics(y, pred, proba):
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", choices=["auto", "time", "random"], default="auto",
                    help="auto = chronological when the data has dates, otherwise random")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models_dir, reports_dir = ROOT / "models", ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    df, source = load_dataset(args.data)
    print(f"Loaded {len(df):,} URLs from {source}  (phishing share: {df.label.mean():.1%})")

    split = make_split(df, args.split, args.val_size, args.test_size, args.seed)
    split_info = split.summary(df)
    print(f"Split mode: {split.mode}. {split.note}")
    for name in ("train", "val", "test"):
        i = split_info[name]
        span = f"  {i['date_from']} to {i['date_to']}" if "date_from" in i else ""
        print(f"  {name:<5} {i['n']:>7,} URLs, {i['phishing_share']:.1%} phishing{span}")

    t0 = time.time()
    X = featurize(df.url)
    y = df.label.values
    print(f"Extracted {X.shape[1]} features in {time.time() - t0:.1f}s")

    X_tr, y_tr = X.iloc[split.train], y[split.train]
    X_va, y_va = X.iloc[split.val], y[split.val]
    X_te, y_te = X.iloc[split.test], y[split.test]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, class_weight="balanced", random_state=args.seed),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=3, subsample=0.9, random_state=args.seed),
        "pytorch_mlp": TorchMLPClassifier(seed=args.seed),
    }

    results, probas, val_probas = {}, {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr.values, y_tr)                       # train only
        val_probas[name] = model.predict_proba(X_va.values)[:, 1]
        proba = model.predict_proba(X_te.values)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = metrics(y_te, pred, proba)
        probas[name] = proba
        print(f"\n=== {name}  ({time.time() - t0:.1f}s)  [test set] ===")
        print(classification_report(y_te, pred, target_names=["legitimate", "phishing"], digits=4))
        joblib.dump(model, models_dir / f"{name}.joblib")

    # 5-fold CV on the training part only (stability check; the test set stays untouched).
    # Folds are shuffled, so on a time split this is a stability check, not a time-aware score.
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    cv_f1 = {}
    for name in ("random_forest", "gradient_boosting"):
        pred = cross_val_predict(models[name].__class__(**models[name].get_params()),
                                 X_tr.values, y_tr, cv=cv, n_jobs=-1)
        cv_f1[name] = f1_score(y_tr, pred)
    print("\n5-fold CV F1 (train only):", {k: round(v, 4) for k, v in cv_f1.items()})

    # Anomaly detection view: Isolation Forest trained on legitimate training URLs only
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=args.seed)
    iso.fit(X_tr.values[y_tr == 0])
    iso_score = -iso.score_samples(X_te.values)            # higher = more anomalous
    iso_pred = (iso.predict(X_te.values) == -1).astype(int)
    results["isolation_forest_anomaly"] = metrics(y_te, iso_pred, iso_score)
    joblib.dump(iso, models_dir / "isolation_forest.joblib")

    # Ensemble: mean of the three supervised models
    ens = np.mean([probas[k] for k in models], axis=0)
    ens_val = np.mean([val_probas[k] for k in models], axis=0)
    results["ensemble_mean"] = metrics(y_te, (ens >= 0.5).astype(int), ens)
    val_results = {"ensemble_mean": metrics(y_va, (ens_val >= 0.5).astype(int), ens_val)}

    # Strict threshold: chosen on VALIDATION, then reported on TEST (never tuned on it).
    threshold, found = pick_strict_threshold(y_va, ens_val)
    strict_pred = (ens >= threshold).astype(int)
    strict_test = {
        "threshold": threshold,
        "found_on_validation": found,
        "precision": precision_score(y_te, strict_pred, zero_division=0),
        "recall": recall_score(y_te, strict_pred, zero_division=0),
        "alerts": int(strict_pred.sum()),
    }
    print(f"\nStrict threshold {threshold:.3f} (chosen on validation, target precision {TARGET_PRECISION:.0%}"
          f"{'' if found else '; NOT reached, fell back to 0.5'})")
    print(f"  on the test set it gives precision {strict_test['precision']:.3f}, recall {strict_test['recall']:.3f}")
    joblib.dump({"threshold": threshold, "models": list(models), "features": FEATURE_NAMES},
                models_dir / "ensemble_config.joblib")

    table = pd.DataFrame(results).T.round(4)
    print("\n=== Test-set summary ===")
    print(table.to_string())
    table.to_csv(reports_dir / "metrics.csv")
    with open(reports_dir / "summary.json", "w") as f:
        json.dump({"data_source": source, "n_urls": int(len(df)), "split": split_info,
                   "test_metrics": results, "validation_metrics": val_results,
                   "cv_f1": cv_f1, "cv_scope": "train split only",
                   "ensemble_threshold_98_precision": threshold,
                   "strict_threshold_on_test": strict_test}, f, indent=2)

    # ---- plots (test set) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(
            y_te, (probas[name] >= 0.5).astype(int), display_labels=["legit", "phish"], ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrices.png", dpi=140)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in {**probas, "ensemble": ens}.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves (test set)")
    ax.legend(); fig.tight_layout()
    fig.savefig(reports_dir / "precision_recall.png", dpi=140)

    rf = models["random_forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    imp.plot.barh(ax=ax); ax.set_title("Random Forest: top 15 features")
    fig.tight_layout()
    fig.savefig(reports_dir / "feature_importance.png", dpi=140)

    print(f"\nSaved models to {models_dir}/ and reports to {reports_dir}/")


if __name__ == "__main__":
    main()
