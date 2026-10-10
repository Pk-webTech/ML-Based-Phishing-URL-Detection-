from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "stackoverflow.com", "amazon.com",
    "microsoft.com", "apple.com", "nytimes.com", "bbc.co.uk", "linkedin.com",
    "reddit.com", "medium.com", "python.org", "mozilla.org", "cloudflare.com",
    "paypal.com", "netflix.com", "dropbox.com", "adobe.com", "ibm.com",
    "vit.ac.in", "coursera.org", "kaggle.com", "arxiv.org", "nature.com",
    "bankofamerica.com", "chase.com", "walmart.com", "ebay.com", "spotify.com",
]
LEGIT_PATHS = [
    "", "/", "/about", "/products", "/docs/getting-started", "/blog/2026/10/new-release",
    "/search?q=machine+learning", "/login", "/account/settings", "/support/contact",
    "/wiki/Phishing", "/questions/123456/how-to-parse-json", "/news/technology-67890",
    "/user/profile?id=4821&tab=activity", "/store/item/B09XYZ123?ref=nav_cart",
]
BRANDS = ["paypal", "apple", "microsoft", "amazon", "netflix", "google", "facebook", "chase", "dhl", "sbi"]
KEYWORDS = ["login", "verify", "secure", "update", "account", "confirm", "signin", "billing", "recover", "wallet"]
BAD_TLDS = ["xyz", "top", "tk", "ml", "click", "link", "work", "icu", "buzz", "com", "net", "info", "online", "site"]
GOOD_TLDS_FOR_PHISH = ["com", "net", "org", "info", "online", "site"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy"]
ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"


def _rand(rng, n):
    return "".join(rng.choice(ALNUM) for _ in range(n))


def _legit(rng: random.Random) -> str:
    dom = rng.choice(LEGIT_DOMAINS)
    sub = rng.choice(["", "", "www.", "www.", "mail.", "docs.", "support.", "blog.", "en."])
    scheme = "https://" if rng.random() < 0.93 else "http://"
    path = rng.choice(LEGIT_PATHS)
    # hard negatives: long tracking params, login pages, random ids
    if rng.random() < 0.15:
        path += f"?utm_source={_rand(rng, 6)}&utm_campaign={_rand(rng, 8)}&session={_rand(rng, 24)}"
    if rng.random() < 0.05:
        dom = f"{_rand(rng, rng.randint(6, 12))}-{rng.choice(['shop', 'cdn', 'app'])}.{rng.choice(['com', 'io', 'co'])}"
    return f"{scheme}{sub}{dom}{path}"


def _phish(rng: random.Random) -> str:
    kind = rng.random()
    brand, kw = rng.choice(BRANDS), rng.choice(KEYWORDS)
    scheme = "https://" if rng.random() < 0.55 else "http://"
    tld = rng.choice(BAD_TLDS if rng.random() < 0.6 else GOOD_TLDS_FOR_PHISH)
    if kind < 0.18:                                   # raw IP host
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        if rng.random() < 0.3:
            host += f":{rng.choice([8080, 8443, 8000])}"
        return f"{scheme}{host}/{kw}/{brand}/{_rand(rng, 6)}.php"
    if kind < 0.45:                                   # brand in subdomain of throwaway domain
        return f"{scheme}{brand}.{kw}.{_rand(rng, rng.randint(5, 10))}.{tld}/{kw}?id={_rand(rng, 10)}"
    if kind < 0.65:                                   # hyphenated lookalike
        return f"{scheme}{brand}-{kw}-{rng.choice(['center', 'help', 'team'])}.{tld}/{_rand(rng, 8)}"
    if kind < 0.75:                                   # @ trick
        return f"{scheme}{brand}.com@{_rand(rng, 8)}.{tld}/{kw}"
    if kind < 0.83:                                   # shortener
        return f"https://{rng.choice(SHORTENERS)}/{_rand(rng, 7)}"
    if kind < 0.92:                                   # hard positive: clean-looking
        return f"{scheme}{_rand(rng, rng.randint(6, 11))}.{tld}/"
    path = "/" + "/".join(_rand(rng, rng.randint(3, 8)) for _ in range(rng.randint(2, 4)))
    return f"{scheme}www.{brand}{rng.randint(1, 99)}.{tld}{path}"   # digits-in-brand lookalike


def make_synthetic(n: int = 12000, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        label = int(rng.random() < 0.5)
        url = _phish(rng) if label else _legit(rng)
        if rng.random() < label_noise:                # imperfect labels, as in real feeds
            label = 1 - label
        rows.append((url, label))
    return pd.DataFrame(rows, columns=["url", "label"])


def load_dataset(path: str | None = None, n_synthetic: int = 12000) -> tuple[pd.DataFrame, str]:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "urls.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[["url", "label"]].dropna().drop_duplicates("url")
        df["label"] = df["label"].astype(int)
        return df.reset_index(drop=True), f"real dataset: {p}"
    df = make_synthetic(n_synthetic).drop_duplicates("url").reset_index(drop=True)
    return df, "SYNTHETIC dataset (no data/urls.csv found)"
