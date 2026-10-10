from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .explain import anatomy, defang, normalise, signals
from .features import FEATURE_NAMES, featurize
from .nn import TorchMLPClassifier  # noqa: F401  (required to unpickle the PyTorch model)

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "pytorch_mlp": "Neural network (PyTorch)",
}
SUSPICIOUS_FLOOR = 0.30


def verdict_for(p: float, threshold: float) -> str:
    if p >= threshold:
        return "phishing"
    if p >= min(SUSPICIOUS_FLOOR, threshold):
        return "suspicious"
    return "safe"


class Predictor:
    def __init__(self, models_dir: Path, reports_dir: Path):
        self.models_dir, self.reports_dir = Path(models_dir), Path(reports_dir)
        cfg = joblib.load(self.models_dir / "ensemble_config.joblib")
        self.names: list[str] = cfg["models"]
        self.strict_threshold: float = float(cfg["threshold"])
        self.models = [joblib.load(self.models_dir / f"{n}.joblib") for n in self.names]
        iso_path = self.models_dir / "isolation_forest.joblib"
        self.iso = joblib.load(iso_path) if iso_path.exists() else None

    # ---- single URL, full detail --------------------------------------------------------
    def analyse(self, url: str, threshold: float) -> dict:
        X = featurize([url])
        feats = {k: float(v) for k, v in X.iloc[0].items()}
        votes = {n: float(m.predict_proba(X.values)[0, 1]) for n, m in zip(self.names, self.models)}
        p = float(np.mean(list(votes.values())))
        anomaly = None
        if self.iso is not None:
            anomaly = bool(self.iso.predict(X.values)[0] == -1)
        return {
            "url": url,
            "normalised": normalise(url),
            "defanged": defang(url),
            "probability": p,
            "threshold": threshold,
            "verdict": verdict_for(p, threshold),
            "models": [{"id": n, "label": MODEL_LABELS.get(n, n), "probability": v} for n, v in votes.items()],
            "unusual_vs_legitimate": anomaly,
            "anatomy": anatomy(url),
            "signals": signals(url, feats),
            "features": feats,
        }

    # ---- many URLs, compact -------------------------------------------------------------
    def batch(self, urls: list[str], threshold: float) -> list[dict]:
        X = featurize(urls)
        probs = np.mean([m.predict_proba(X.values)[:, 1] for m in self.models], axis=0)
        rows = []
        for i, (u, p) in enumerate(zip(urls, probs)):
            feats = {k: float(v) for k, v in X.iloc[i].items()}
            sig = signals(u, feats)
            rows.append({
                "url": u,
                "probability": float(p),
                "verdict": verdict_for(float(p), threshold),
                "reasons": [s["title"] for s in sig[:3]],
            })
        return rows

    # ---- model report -------------------------------------------------------------------
    def report(self) -> dict:
        path = self.reports_dir / "summary.json"
        if not path.exists():
            return {"available": False}
        data = json.loads(path.read_text())
        data["available"] = True
        data["model_labels"] = {**MODEL_LABELS, "isolation_forest_anomaly": "Isolation Forest (anomaly)", "ensemble_mean": "Ensemble (average)"}
        data["images"] = [n for n in ("confusion_matrices.png", "precision_recall.png", "feature_importance.png")
                          if (self.reports_dir / n).exists()]
        return data

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES)
