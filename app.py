from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.predictor import Predictor

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_URL_LEN = 2048
MAX_BATCH = 1000
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f\s]")
log = logging.getLogger("phishlens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="PhishLens", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global predictor
    if predictor is None:
        try:
            predictor = Predictor(ROOT / "models", ROOT / "reports")
        except FileNotFoundError:
            raise HTTPException(503, "Models not found. Run `python train.py` first.")
    return predictor


# ---------- validation ---------------------------------------------------------------------
def clean_url(raw: str) -> str:
    u = raw.strip()
    if not u:
        raise ValueError("Enter a link to check.")
    if len(u) > MAX_URL_LEN:
        raise ValueError(f"Links longer than {MAX_URL_LEN} characters are not supported.")
    if CONTROL_CHARS.search(u):
        raise ValueError("The link contains spaces or control characters.")
    return u


class CheckRequest(BaseModel):
    url: str
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        return clean_url(v)


class BatchRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=MAX_BATCH)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("urls")
    @classmethod
    def _urls(cls, v: list[str]) -> list[str]:
        out, seen = [], set()
        for raw in v:
            try:
                u = clean_url(raw)
            except ValueError:
                continue            # skip bad lines instead of failing the whole batch
            if u not in seen:
                seen.add(u)
                out.append(u)
        if not out:
            raise ValueError("No valid links found.")
        return out


# ---------- middleware: rate limit + security headers --------------------------------------
RATE = int(os.getenv("PHISHLENS_RATE_LIMIT", "120"))
_hits: dict[str, deque] = defaultdict(deque)

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        now = time.monotonic()
        q = _hits[request.client.host if request.client else "?"]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE:
            return JSONResponse({"detail": "Too many requests. Wait a moment and try again."}, status_code=429,
                                headers={"Retry-After": "30"})
        q.append(now)
    response = await call_next(request)
    response.headers.update({
        "Content-Security-Policy": CSP,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    })
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------- API ----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    p = get_predictor()
    return {"status": "ok", "models": p.names, "features": p.feature_count}


@app.get("/api/config")
def config():
    p = get_predictor()
    return {
        "max_batch": MAX_BATCH,
        "max_url_length": MAX_URL_LEN,
        "presets": [
            {"id": "sensitive", "label": "Sensitive", "threshold": 0.30, "hint": "Catches more, more false alarms"},
            {"id": "balanced", "label": "Balanced", "threshold": 0.50, "hint": "Default"},
            {"id": "strict", "label": "Strict", "threshold": round(p.strict_threshold, 3), "hint": "Fewer false alarms, may miss some"},
        ],
    }


@app.post("/api/check")
def check(req: CheckRequest):
    p = get_predictor()
    return p.analyse(req.url, req.threshold or 0.5)


@app.post("/api/batch")
def batch(req: BatchRequest):
    p = get_predictor()
    thr = req.threshold or 0.5
    rows = p.batch(req.urls, thr)
    counts = {"phishing": 0, "suspicious": 0, "safe": 0}
    for r in rows:
        counts[r["verdict"]] += 1
    return {"threshold": thr, "total": len(rows), "counts": counts, "results": rows}


@app.get("/api/report")
def report():
    return get_predictor().report()


@app.get("/api/report-image/{name}")
def report_image(name: str):
    if name not in {"confusion_matrices.png", "precision_recall.png", "feature_importance.png"}:
        raise HTTPException(404, "Not found")
    path = ROOT / "reports" / name
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/png")


# ---------- static frontend ----------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PHISHLENS_HOST", "127.0.0.1")
    port = int(os.getenv("PHISHLENS_PORT", "8000"))
    log.info("PhishLens running at http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
