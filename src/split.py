"""Train / validation / test splitting.

Two modes:

* ``time``   - the oldest rows go to train, the next to validation, the newest to test.
               This mimics real use: the model is trained on the past and judged on links
               that appeared later. Rows with no date (for example legitimate links taken
               from a ranking list) have no meaningful time, so they are split at random
               in the same proportions, keeping the class balance of every part similar.
* ``random`` - stratified random split. Optimistic, because near-duplicate links land on
               both sides, but it is the only option when the data has no dates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: np.ndarray      # positional indices into the dataframe
    val: np.ndarray
    test: np.ndarray
    mode: str              # "time" or "random"
    note: str              # one-line description for the report

    def summary(self, df: pd.DataFrame) -> dict:
        out = {"mode": self.mode, "note": self.note}
        for name in ("train", "val", "test"):
            idx = getattr(self, name)
            part = df.iloc[idx]
            info = {"n": int(len(idx)), "phishing_share": round(float(part.label.mean()), 4)}
            if "date" in df.columns and part.date.notna().any():
                info["date_from"] = str(part.date.min().date())
                info["date_to"] = str(part.date.max().date())
            out[name] = info
        return out


def _random_parts(idx: np.ndarray, y: np.ndarray, val_size: float, test_size: float, seed: int):
    """Stratified random three-way split of ``idx``."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for cls in np.unique(y[idx]):
        members = idx[y[idx] == cls]
        members = rng.permutation(members)
        n_test = int(round(len(members) * test_size))
        n_val = int(round(len(members) * val_size))
        parts["test"].append(members[:n_test])
        parts["val"].append(members[n_test:n_test + n_val])
        parts["train"].append(members[n_test + n_val:])
    return {k: np.concatenate(v) if v else np.array([], dtype=int) for k, v in parts.items()}


def make_split(df: pd.DataFrame, mode: str = "auto", val_size: float = 0.2,
               test_size: float = 0.2, seed: int = 42) -> Split:
    """Return positional indices for train / val / test.

    ``mode="auto"`` uses a time split when at least half the rows have a date and the
    phishing rows carry dates, otherwise it falls back to a random split.
    """
    if not 0 < val_size < 0.5 or not 0 < test_size < 0.5:
        raise ValueError("val_size and test_size must each be between 0 and 0.5")
    df = df.reset_index(drop=True)
    y = df.label.to_numpy()
    all_idx = np.arange(len(df))

    has_dates = "date" in df.columns and df.date.notna().mean() >= 0.5
    if mode == "auto":
        mode = "time" if has_dates else "random"
    if mode == "time" and not has_dates:
        raise ValueError("Time split needs a 'date' column filled in for at least half of the rows.")

    if mode == "random":
        p = _random_parts(all_idx, y, val_size, test_size, seed)
        return Split(p["train"], p["val"], p["test"], "random",
                     "Stratified random split (no dates available or requested).")

    dated = df.date.notna().to_numpy()
    d_idx = all_idx[dated]
    u_idx = all_idx[~dated]
    # Oldest first. Stable sort keeps ties in file order, so results are reproducible.
    d_idx = d_idx[np.argsort(df.date.to_numpy()[d_idx], kind="stable")]
    n = len(d_idx)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    train, val, test = d_idx[:n_train], d_idx[n_train:n_train + n_val], d_idx[n_train + n_val:]

    if len(u_idx):
        rp = _random_parts(u_idx, y, val_size, test_size, seed)
        train, val, test = (np.concatenate([a, rp[k]]) for a, k in ((train, "train"), (val, "val"), (test, "test")))

    # A time split is only meaningful when no future row leaks into an earlier part.
    dates = df.date.to_numpy()
    assert dates[d_idx[:n_train]].max() <= dates[d_idx[n_train:]].min() if n_train and n > n_train else True

    note = ("Chronological split: oldest rows train, next validation, newest test. "
            + ("Undated rows were split at random in the same proportions." if len(u_idx) else ""))
    return Split(np.sort(train), np.sort(val), np.sort(test), "time", note.strip())
