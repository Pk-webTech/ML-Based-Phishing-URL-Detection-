"""API tests. Run:  python -m pytest -q   (models must exist: run train.py first)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as webapp  # noqa: E402
from src.explain import anatomy, defang, normalise  # noqa: E402

client = TestClient(webapp.app)
PHISH = "http://paypal.verify.k3j29dkx1a.xyz/login?id=9ak2l3jd0x"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    webapp._hits.clear()
    yield
    webapp._hits.clear()


def test_health_and_config():
    assert client.get("/api/health").json()["status"] == "ok"
    presets = {p["id"] for p in client.get("/api/config").json()["presets"]}
    assert presets == {"sensitive", "balanced", "strict"}


def test_phishing_link_is_flagged_with_explanations():
    r = client.post("/api/check", json={"url": PHISH}).json()
    assert r["verdict"] == "phishing" and r["probability"] > 0.5
    assert len(r["models"]) == 3 and r["signals"] and r["anatomy"]
    assert "".join(s["text"] for s in r["anatomy"]) == r["normalised"]
    assert r["defanged"].startswith("hxxp://") and "[.]" in r["defanged"]


def test_legitimate_link_is_low_risk():
    r = client.post("/api/check", json={"url": "https://en.wikipedia.org/wiki/Phishing"}).json()
    assert r["verdict"] == "safe"


@pytest.mark.parametrize("bad", ["", "   ", "has space.com", "a" * 2049, "x\x00y"])
def test_invalid_input_rejected(bad):
    assert client.post("/api/check", json={"url": bad}).status_code == 422


def test_threshold_must_be_a_probability():
    assert client.post("/api/check", json={"url": PHISH, "threshold": 1.5}).status_code == 422


def test_batch_counts_and_limits():
    urls = [PHISH, "https://github.com", PHISH, "bad url"]      # duplicate and invalid are dropped
    r = client.post("/api/batch", json={"urls": urls}).json()
    assert r["total"] == 2 and sum(r["counts"].values()) == 2
    assert client.post("/api/batch", json={"urls": []}).status_code == 422
    assert client.post("/api/batch", json={"urls": ["a.com"] * (webapp.MAX_BATCH + 1)}).status_code == 422


def test_security_headers_present():
    h = client.get("/").headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"


def test_report_image_whitelist():
    assert client.get("/api/report-image/summary.json").status_code == 404
    assert client.get("/api/report-image/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(webapp, "RATE", 3)
    codes = [client.get("/api/health").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429


def test_anatomy_edge_cases():
    assert any(s["part"] == "userinfo" and s["risk"] == "bad" for s in anatomy("http://apple.com@evil.tk/x"))
    assert any(s["part"] == "domain" and s["risk"] == "bad" for s in anatomy("http://10.0.0.5/login"))
    assert normalise("example.com") == "http://example.com"
    assert defang("https://a.com") == "hxxps://a[.]com"
