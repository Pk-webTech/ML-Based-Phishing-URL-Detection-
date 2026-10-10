# ML-Based Phishing URL Detection

Classifies URLs as **phishing** or **legitimate** from lexical and host-based features using
Random Forest, Gradient Boosting and a PyTorch neural network, with an Isolation Forest
anomaly-detection baseline and an automation script that flags suspicious URLs for security monitoring.

**Stack:** Python, scikit-learn, PyTorch, Pandas, NumPy (+ matplotlib, joblib)

## Project layout

```
train.py              train + evaluate all models, write metrics / plots / saved models
scan_urls.py          automation script: score a list of URLs and flag suspicious ones
app.py                PhishLens web app (FastAPI backend + static front end)
static/               front end: index.html, styles.css, app.js (no build step)
tests/test_api.py     API and security tests (python -m pytest -q)
run_app.bat           one-click Windows launcher
src/explain.py        URL anatomy and plain-language risk signals
src/predictor.py      model loading and scoring for the web app
src/features.py       38 lexical + host-based features (offline, no network calls)
src/data.py           dataset loader (real CSV or synthetic fallback)
src/nn.py             PyTorch MLP with a scikit-learn-style fit / predict_proba API
sample_urls.txt       example input for the scanner
models/               saved models (created by train.py)
reports/              metrics.csv, summary.json, confusion matrices, PR curves, feature importance
```

## Quick start

```bash
pip install -r requirements.txt
python train.py                                   # trains all models
python scan_urls.py --input sample_urls.txt --output reports/flagged_urls.csv
```

## Web app (PhishLens)

```bash
python app.py        # then open http://127.0.0.1:8000   (Windows: double-click run_app.bat)
```

- **Check:** paste one link and get a verdict, a phishing score against the alert threshold, the link split into
  parts (protocol, subdomain, main domain, path, query) with a note on each, ranked warning signs, and each model's vote.
- **Bulk scan:** paste or upload up to 1,000 links (.txt or .csv with a `url` column), then sort, filter, search and export CSV.
- **History:** links checked one at a time, kept in the browser only. **Model:** the latest evaluation report and charts.
- **Sensitivity:** Sensitive (alert at 30%), Balanced (50%), Strict (the threshold that kept precision at 98% or higher).
- Light and dark themes, keyboard accessible, responsive down to phone width.

Security design: the server never opens or resolves a submitted link; inputs are length-limited and validated;
responses carry a strict Content-Security-Policy and other security headers; API calls are rate limited
(`PHISHLENS_RATE_LIMIT`, default 120 per minute); CSV exports neutralise spreadsheet formulas; "Copy safe version"
defangs links (`hxxps://example[.]com`). It binds to 127.0.0.1 by default. To expose it, put it behind a TLS reverse proxy.
Settings: `PHISHLENS_HOST`, `PHISHLENS_PORT`.

## Using a real dataset (recommended)

Out of the box, `train.py` falls back to a **synthetic** dataset so the pipeline runs anywhere.
Metrics on synthetic data are only a pipeline check, **not** real-world performance.

For real results, create `data/urls.csv` with columns `url,label` (`1` = phishing, `0` = legitimate):

- Phishing: PhishTank or OpenPhish feeds
- Legitimate: Tranco top-sites list (prefix with `https://`), or Kaggle sets such as *Phishing Site URLs* / *PhiUSIIL*

Then run `python train.py` (or `python train.py --data path/to/file.csv`). Quote the numbers from
that run in your resume/report.

## Features (38)

| Group | Examples |
|---|---|
| Lexical (URL) | length, dots, hyphens, digits ratio, special chars, `@`, `%`, entropy, HTTPS, suspicious keywords (`login`, `verify`, ...) |
| Lexical (path/query) | path length and depth, query length, parameter count, executable extensions |
| Host-based | host length / entropy, subdomain count, IP-address host, non-standard port, URL shortener, suspicious TLD, punycode, brand name in subdomain, label lengths |

## Models and evaluation

- `RandomForestClassifier` (300 trees, balanced class weights)
- `GradientBoostingClassifier` (300 stages)
- PyTorch MLP (128-64, BatchNorm, dropout, AdamW, early stopping on a validation split)
- `IsolationForest` trained on legitimate URLs only (anomaly-detection view)
- Mean-probability ensemble of the three supervised models

Metrics: precision, recall, F1 (per class and for the phishing class), ROC-AUC, 5-fold CV F1.
Stratified 80/20 train/test split.

### Results on the bundled synthetic dataset (about 9.7k URLs, 3% label noise)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.968 | 0.980 | 0.974 |
| Gradient Boosting | 0.968 | 0.977 | 0.972 |
| PyTorch MLP | 0.967 | 0.979 | 0.973 |
| Isolation Forest (anomaly) | 0.890 | 0.899 | 0.894 |
| Ensemble | 0.967 | 0.980 | 0.973 |

The roughly 97% ceiling is expected: 3% of synthetic labels are deliberately flipped to mimic noisy feeds.

## Automation script

```bash
python scan_urls.py --input urls.txt                      # text file (one URL per line) or CSV with a `url` column
python scan_urls.py --input urls.txt --output flagged.csv # append flagged URLs to a CSV log
python scan_urls.py --input urls.txt --strict             # >=98% precision threshold, fewer false alarms
python scan_urls.py --input urls.txt --watch 60           # re-scan every 60 s, report only new URLs
cat urls.txt | python scan_urls.py --input -              # stdin
```

Each flagged URL gets a probability and a severity (HIGH >= 0.9, MEDIUM >= 0.7, LOW otherwise).
The exit code is `1` when anything is flagged, so it plugs into cron, CI or SIEM hooks.

## Limitations and next steps

- Purely URL-based: no page content, WHOIS age, DNS or TLS data. Adding live host lookups would improve accuracy.
- Trained on one dataset snapshot; phishing patterns drift, so retrain periodically.
- Flagged URLs are leads for analyst review, not verdicts.
