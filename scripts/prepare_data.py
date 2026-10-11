"""Build data/urls.csv from real feeds.

Inputs (download them yourself; this script never touches the network):
    PhishTank   https://phishtank.org/developer_info.php   -> verified_online.csv
                columns include: url, submission_time   (the date drives the time-based split)
    Tranco      https://tranco-list.eu/                    -> top-1m.csv   (rank,domain; no header)

Usage:
    python scripts/prepare_data.py --phishtank verified_online.csv --tranco top-1m.csv
    python scripts/prepare_data.py --phishtank verified_online.csv --tranco top-1m.csv --root-only

Output: data/urls.csv with columns  url,label,date   (label 1 = phishing, 0 = legitimate).

Legitimate links get no date. Ranking lists are a snapshot, not a timeline, so train.py splits
those rows at random across train/validation/test and splits the dated phishing rows by time.

IMPORTANT BIAS WARNING. Tranco lists domains, not full links, so legitimate rows become
"https://domain/" with no path, while phishing links usually carry long paths. A model can then
"cheat" by learning "has a path -> phishing". Two ways to keep the evaluation honest:
    --root-only   cut phishing links down to scheme + host too, so both classes look alike
    or train on legitimate links that carry real paths (for example from a web crawl or your
    own proxy logs) and leave --root-only off.
Run both ways and compare: a large score drop with --root-only shows how much the path leaked.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_phishtank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "url" not in df.columns:
        raise SystemExit(f"{path}: expected a 'url' column (PhishTank verified_online.csv).")
    date_col = next((c for c in ("submission_time", "verification_time", "date") if c in df.columns), None)
    out = pd.DataFrame({"url": df["url"].astype(str).str.strip()})
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None) if date_col else pd.NaT
    out["label"] = 1
    if date_col is None:
        print("Warning: no date column in the phishing file, so a time-based split will not be possible.")
    return out


def load_tranco(path: Path, top: int, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["rank", "domain"]).head(top)
    df = df.dropna(subset=["domain"]).sample(n=min(n, len(df)), random_state=seed)
    return pd.DataFrame({"url": "https://" + df["domain"].astype(str).str.strip() + "/", "date": pd.NaT, "label": 0})


def root_only(url: str) -> str:
    p = urlparse(url if "://" in url else "http://" + url)
    return f"{p.scheme}://{p.netloc}/" if p.netloc else url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phishtank", required=True, nargs="+", help="one or more PhishTank CSV files")
    ap.add_argument("--tranco", required=True, help="Tranco top-1m.csv")
    ap.add_argument("--tranco-top", type=int, default=100_000, help="sample legitimate domains from the top N ranks")
    ap.add_argument("--ratio", type=float, default=1.0, help="legitimate rows per phishing row (default 1.0)")
    ap.add_argument("--max-phish", type=int, default=None, help="keep only the newest N phishing rows")
    ap.add_argument("--root-only", action="store_true", help="reduce phishing links to scheme + host (see bias warning)")
    ap.add_argument("--output", default=str(ROOT / "data" / "urls.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    phish = pd.concat([load_phishtank(Path(p)) for p in args.phishtank], ignore_index=True)
    phish = phish[phish.url.str.len().between(8, 2048) & ~phish.url.str.contains(r"\s", regex=True)]
    phish = phish.sort_values("date", na_position="first").drop_duplicates("url", keep="last")
    if args.max_phish:
        phish = phish.tail(args.max_phish)
    if args.root_only:
        phish["url"] = phish.url.map(root_only)
        phish = phish.drop_duplicates("url", keep="last")

    legit = load_tranco(Path(args.tranco), args.tranco_top, int(len(phish) * args.ratio), args.seed)
    df = pd.concat([phish, legit], ignore_index=True)
    df = df.drop_duplicates("url").sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["url", "label", "date"]].to_csv(out, index=False)
    dated = df.date.dropna()
    print(f"Wrote {len(df):,} rows to {out}  (phishing {int(df.label.sum()):,}, legitimate {int((df.label == 0).sum()):,})")
    if len(dated):
        print(f"Phishing dates run from {dated.min().date()} to {dated.max().date()}")


if __name__ == "__main__":
    main()
