"""Human-readable explanations: URL anatomy (segments with risk notes) and ranked risk signals.

Everything here works on the URL string only; the link is never opened or resolved.
"""
from __future__ import annotations

from urllib.parse import urlparse

from .features import SUSPICIOUS_TLDS, SUSPICIOUS_WORDS, URL_SHORTENERS, IP_RE, HEX_IP_RE

BRANDS = ("paypal", "apple", "microsoft", "amazon", "netflix", "google", "facebook", "chase", "dhl", "sbi", "bank")
SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu"}
RISK_ORDER = {"ok": 0, "warn": 1, "bad": 2}


def normalise(url: str) -> str:
    url = url.strip()
    return url if "://" in url else "http://" + url


def split_host(host: str) -> tuple[list[str], str]:
    """Return (subdomain labels, registered domain)."""
    labels = [l for l in host.split(".") if l]
    if len(labels) <= 2:
        return [], ".".join(labels)
    take = 3 if (len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL) else 2
    return labels[:-take], ".".join(labels[-take:])


def anatomy(url: str) -> list[dict]:
    """Split a URL into ordered segments. Joined, the `text` fields rebuild the normalised URL."""
    norm = normalise(url)
    try:
        p = urlparse(norm)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        return [{"part": "url", "text": norm, "risk": "bad", "note": "This link is malformed and could not be parsed."}]

    segs: list[dict] = []

    def add(part, text, risk="ok", note=""):
        if text:
            segs.append({"part": part, "text": text, "risk": risk, "note": note})

    if p.scheme == "https":
        add("scheme", "https://", "ok", "Encrypted connection. This does not prove the site is genuine.")
    else:
        add("scheme", p.scheme + "://", "warn", "Not encrypted. Sign-in pages should always use HTTPS.")

    netloc_prefix = ""
    if "@" in p.netloc:
        netloc_prefix = p.netloc.rsplit("@", 1)[0] + "@"
        add("userinfo", netloc_prefix, "bad",
            "Everything before the @ is ignored by the browser. The real site is the part after it.")

    is_ip = bool(IP_RE.match(host) or HEX_IP_RE.match(host))
    subs, registered = split_host(host)
    sub_text = ".".join(subs)
    if sub_text:
        brand = next((b for b in BRANDS if b in sub_text), None)
        if brand:
            add("subdomain", sub_text + ".", "bad",
                f'Uses "{brand}" in the subdomain, but the real site is {registered}.')
        elif len(subs) >= 3:
            add("subdomain", sub_text + ".", "warn", "Unusually many subdomain levels, a common way to hide the real domain.")
        else:
            add("subdomain", sub_text + ".", "ok", "")

    tld = registered.split(".")[-1] if registered else ""
    if is_ip:
        add("domain", registered, "bad", "The address is a raw IP number, not a registered domain name.")
    elif "xn--" in registered:
        add("domain", registered, "bad", "Contains punycode, which can imitate letters from a trusted brand.")
    elif registered in URL_SHORTENERS:
        add("domain", registered, "warn", "A link shortener hides the real destination.")
    elif tld in SUSPICIOUS_TLDS:
        add("domain", registered, "warn", f".{tld} is a domain ending often seen in throwaway phishing sites.")
    elif registered.count("-") >= 2:
        add("domain", registered, "warn", "Several hyphens in the domain name, a common lookalike pattern.")
    else:
        add("domain", registered, "ok", "")

    if port is not None and port not in (80, 443):
        add("port", f":{port}", "warn", "Non-standard port. Legitimate sites rarely use one.")

    low_path = (p.path or "").lower()
    words = [w for w in SUSPICIOUS_WORDS if w in low_path]
    if (p.path or "").lower().endswith((".exe", ".zip", ".scr", ".apk", ".bat", ".msi", ".jar", ".js")):
        add("path", p.path, "bad", "Points to a downloadable program or archive.")
    elif words:
        add("path", p.path, "warn", f'Contains words used to bait sign-ins: {", ".join(words[:3])}.')
    elif len(p.path or "") > 60:
        add("path", p.path, "warn", "A very long path.")
    else:
        add("path", p.path, "ok", "")

    if p.query:
        long_q = len(p.query) > 80
        add("query", "?" + p.query, "warn" if long_q else "ok",
            "A very long query string." if long_q else "")
    if p.fragment:
        add("fragment", "#" + p.fragment, "ok", "")
    return segs


def signals(url: str, feats: dict) -> list[dict]:
    """Ranked, plain-language reasons. severity: bad > warn."""
    out: list[dict] = []

    def add(sev, title, detail):
        out.append({"severity": sev, "title": title, "detail": detail})

    if feats["has_ip_host"]:
        add("bad", "Raw IP address instead of a domain", "Real services almost always use a registered domain name.")
    if feats["num_at"]:
        add("bad", "Contains an @ symbol", "Browsers ignore everything before the @, so it can disguise the real destination.")
    if feats["has_punycode"]:
        add("bad", "Possible lookalike characters", "Punycode (xn--) can imitate letters from a trusted brand name.")
    if feats["brand_in_subdomain"]:
        add("bad", "Brand name in a subdomain", "The brand appears before the real domain, not as the domain itself.")
    if feats["file_extension_exec"]:
        add("bad", "Links to a downloadable file", "The path ends in a program or archive extension.")
    if feats["is_shortener"]:
        add("warn", "Shortened link", "The final destination is hidden until opened.")
    if feats["tld_suspicious"]:
        add("warn", "Domain ending often used for abuse", "Cheap or free endings attract throwaway phishing sites.")
    if feats["num_suspicious_words"] >= 2:
        add("warn", "Sign-in or account bait words", f"{int(feats['num_suspicious_words'])} words such as login, verify or secure appear in the link.")
    elif feats["num_suspicious_words"] == 1:
        add("warn", "Sign-in or account bait word", "One word such as login or verify appears in the link.")
    if not feats["has_https"]:
        add("warn", "No HTTPS", "The connection is not encrypted.")
    if feats["has_port"]:
        add("warn", "Non-standard port", "Legitimate sites rarely use unusual ports.")
    if feats["num_subdomains"] >= 3:
        add("warn", "Many subdomain levels", f"{int(feats['num_subdomains'])} levels before the main domain.")
    if feats["host_hyphens"] >= 2:
        add("warn", "Several hyphens in the host name", "Common in lookalike names such as brand-secure-login.")
    if feats["host_digit_ratio"] > 0.25 and not feats["has_ip_host"]:
        add("warn", "Many digits in the host name", "Machine-generated names often mix in digits.")
    if feats["url_length"] > 100:
        add("warn", "Very long link", f"{int(feats['url_length'])} characters. Long links can bury the real domain.")
    if feats["host_entropy"] > 3.6 and not feats["has_ip_host"]:
        add("warn", "Random-looking host name", "The host name looks machine-generated rather than a real word or brand.")

    order = {"bad": 0, "warn": 1}
    out.sort(key=lambda s: order[s["severity"]])
    return out


def defang(url: str) -> str:
    """Make a URL safe to paste into chat or tickets (not clickable)."""
    s = url.strip().replace("http://", "hxxp://").replace("https://", "hxxps://")
    return s.replace(".", "[.]")
