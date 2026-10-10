from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work",
    "support", "country", "kim", "loan", "zip", "mov", "icu", "buzz", "rest",
}
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "wallet", "invoice", "billing", "suspend",
    "unlock", "recover", "support", "alert", "webscr", "paypal", "apple",
    "microsoft", "amazon", "netflix", "free", "bonus", "gift",
]
URL_SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy"}
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)

FEATURE_NAMES = [
    # lexical: whole URL
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_digits", "digit_ratio", "num_special_chars", "num_at", "num_question",
    "num_equals", "num_ampersand", "num_percent", "url_entropy", "has_https",
    "num_suspicious_words", "double_slash_in_path",
    # lexical: path / query
    "path_length", "query_length", "path_depth", "num_params", "has_fragment",
    "file_extension_exec",
    # host-based
    "host_length", "host_digits", "host_hyphens", "host_entropy", "num_subdomains",
    "has_ip_host", "has_port", "is_shortener", "tld_suspicious", "tld_length",
    "has_punycode", "brand_in_subdomain", "longest_host_label", "avg_host_label_len",
    "host_digit_ratio",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalise(url: str) -> str:
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url
    return url


def extract_features(url: str) -> dict:
    raw = str(url).strip()
    norm = _normalise(raw)
    try:
        p = urlparse(norm)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        p = urlparse("http://invalid")
        host, port = "", None
    path, query, frag = p.path or "", p.query or "", p.fragment or ""
    low = raw.lower()

    labels = [l for l in host.split(".") if l]
    tld = labels[-1] if len(labels) >= 2 else ""
    registered = ".".join(labels[-2:]) if len(labels) >= 2 else host
    subdomains = labels[:-2] if len(labels) > 2 else []
    n = max(len(raw), 1)
    digits = sum(c.isdigit() for c in raw)
    host_digits = sum(c.isdigit() for c in host)
    specials = sum(not c.isalnum() for c in raw)
    brands = ("paypal", "apple", "microsoft", "amazon", "netflix", "google", "facebook", "bank")

    return {
        "url_length": len(raw),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "num_special_chars": specials,
        "num_at": raw.count("@"),
        "num_question": raw.count("?"),
        "num_equals": raw.count("="),
        "num_ampersand": raw.count("&"),
        "num_percent": raw.count("%"),
        "url_entropy": _entropy(raw),
        "has_https": int(p.scheme == "https"),
        "num_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "double_slash_in_path": int("//" in path),
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": len([s for s in path.split("/") if s]),
        "num_params": len([q for q in query.split("&") if q]),
        "has_fragment": int(bool(frag)),
        "file_extension_exec": int(bool(re.search(r"\.(exe|zip|scr|apk|js|bat|msi|jar)$", path.lower()))),
        "host_length": len(host),
        "host_digits": host_digits,
        "host_hyphens": host.count("-"),
        "host_entropy": _entropy(host),
        "num_subdomains": len(subdomains),
        "has_ip_host": int(bool(IP_RE.match(host) or HEX_IP_RE.match(host))),
        "has_port": int(port is not None and port not in (80, 443)),
        "is_shortener": int(registered in URL_SHORTENERS),
        "tld_suspicious": int(tld in SUSPICIOUS_TLDS),
        "tld_length": len(tld),
        "has_punycode": int("xn--" in host),
        "brand_in_subdomain": int(any(b in ".".join(subdomains) for b in brands)),
        "longest_host_label": max((len(l) for l in labels), default=0),
        "avg_host_label_len": float(np.mean([len(l) for l in labels])) if labels else 0.0,
        "host_digit_ratio": host_digits / max(len(host), 1),
    }


def featurize(urls) -> pd.DataFrame:
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)
