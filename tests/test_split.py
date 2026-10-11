"""Tests for the train/validation/test split and the validation-chosen threshold.
Run:  python -m pytest -q tests/test_split.py   (no trained models needed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import make_split          # noqa: E402
from src.threshold import pick_strict_threshold   # noqa: E402


def frame(n_phish=600, n_legit=600, dated_legit=False, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.permutation(n_phish), unit="D")
    phish = pd.DataFrame({"url": [f"http://p{i}.xyz/" for i in range(n_phish)], "label": 1, "date": days})
    legit_dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.permutation(n_legit), unit="D") if dated_legit else pd.NaT
    legit = pd.DataFrame({"url": [f"https://l{i}.com/" for i in range(n_legit)], "label": 0, "date": legit_dates})
    return pd.concat([phish, legit], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


def test_parts_are_disjoint_and_cover_everything():
    df = frame()
    s = make_split(df, "time")
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(all_idx) == len(df) and len(set(all_idx)) == len(df)


def test_time_split_has_no_future_leakage():
    df = frame(dated_legit=True)
    s = make_split(df, "time")
    d = df.date.to_numpy()
    assert d[s.train].max() <= d[s.val].min() and d[s.val].max() <= d[s.test].min()


def test_undated_rows_spread_across_parts_with_similar_balance():
    df = frame()                                   # legitimate rows have no date
    s = make_split(df, "time")
    shares = [df.label.iloc[p].mean() for p in (s.train, s.val, s.test)]
    assert all(0.35 < x < 0.65 for x in shares)
    # phishing rows are chronological even though legit rows are random
    ph = df[df.label == 1].date
    assert ph.loc[ph.index.isin(s.train)].max() <= ph.loc[ph.index.isin(s.test)].min()


def test_auto_picks_time_with_dates_and_random_without():
    assert make_split(frame(), "auto").mode == "time"
    assert make_split(frame().drop(columns="date"), "auto").mode == "random"
    with pytest.raises(ValueError):
        make_split(frame().drop(columns="date"), "time")


def test_split_is_reproducible():
    df = frame()
    a, b = make_split(df, "time", seed=3), make_split(df, "time", seed=3)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.val, b.val)


def test_threshold_is_picked_on_validation_and_meets_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500), np.zeros(500)].astype(int)
    p = np.r_[rng.beta(6, 2, 500), rng.beta(2, 6, 500)]
    thr, found = pick_strict_threshold(y, p)
    assert found and 0.5 < thr < 1
    alerts = p >= thr
    assert y[alerts].mean() >= 0.98 and alerts.sum() >= 20


def test_threshold_falls_back_when_target_unreachable():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert pick_strict_threshold(y, p) == (0.5, False)
