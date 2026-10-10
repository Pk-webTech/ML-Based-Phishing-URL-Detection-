"""Automation script: flag suspicious URLs for security monitoring.

Examples:
    python scan_urls.py --input urls.txt                 # one URL per line (or CSV with a `url` column)
    python scan_urls.py --input urls.txt --output flagged.csv --threshold 0.6
    cat urls.txt | python scan_urls.py --input -         # read from stdin
    python scan_urls.py --input urls.txt --watch 60      # re-scan the file every 60 s, report only new URLs

Exit code is 1 if any URL was flagged (handy for cron / CI / SIEM hooks), else 0.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.features import featurize   # noqa: E402
from src.nn import TorchMLPClassifier   # noqa: E402,F401  (needed to unpickle)


def load_urls(src: str) -> list[str]:
    if src == "-":
        raw = [l.strip() for l in sys.stdin if l.strip()]
    else:
        p = Path(src)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            col = "url" if "url" in df.columns else df.columns[0]
            raw = df[col].astype(str).tolist()
        else:
            raw = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return list(dict.fromkeys(raw))      # de-duplicate, keep order


def load_models(models_dir: Path):
    cfg = joblib.load(models_dir / "ensemble_config.joblib")
    models = [joblib.load(models_dir / f"{n}.joblib") for n in cfg["models"]]
    return models, cfg["threshold"]   # high-precision threshold from training


def score(urls: list[str], models) -> np.ndarray:
    X = featurize(urls).values
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def severity(p: float) -> str:
    return "HIGH" if p >= 0.9 else "MEDIUM" if p >= 0.7 else "LOW"


def scan(urls, models, threshold, output: Path | None, quiet=False) -> int:
    probs = score(urls, models)
    now = dt.datetime.now().isoformat(timespec="seconds")
    flagged = [(u, p) for u, p in zip(urls, probs) if p >= threshold]
    flagged.sort(key=lambda t: -t[1])

    if not quiet:
        print(f"[{now}] scanned {len(urls)} URLs, flagged {len(flagged)} (threshold {threshold:.2f})")
        for u, p in flagged:
            print(f"  {severity(p):<6} {p:.3f}  {u}")

    if output and flagged:
        new = not output.exists()
        with open(output, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "url", "phishing_probability", "severity"])
            for u, p in flagged:
                w.writerow([now, u, f"{p:.4f}", severity(p)])
    return len(flagged)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="text file / CSV of URLs, or - for stdin")
    ap.add_argument("--output", default=None, help="append flagged URLs to this CSV log")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability cut-off (default 0.5, balanced precision/recall)")
    ap.add_argument("--strict", action="store_true",
                    help="use the >=98%% precision threshold chosen in training (fewer false alarms, lower recall)")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--watch", type=int, default=0, help="re-scan every N seconds, only new URLs")
    args = ap.parse_args()

    models, default_thr = load_models(Path(args.models_dir))
    thr = default_thr if args.strict else args.threshold
    out = Path(args.output) if args.output else None

    if not args.watch:
        n = scan(load_urls(args.input), models, thr, out)
        sys.exit(1 if n else 0)

    seen: set[str] = set()
    while True:
        try:
            urls = [u for u in load_urls(args.input) if u not in seen]
            if urls:
                scan(urls, models, thr, out)
                seen.update(urls)
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
