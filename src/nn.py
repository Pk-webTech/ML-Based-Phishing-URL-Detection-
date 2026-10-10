from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class _Net(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(128, 64), dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 epochs=60, batch_size=256, patience=8, val_fraction=0.1, seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mean_) / self.std_

        idx = rng.permutation(len(Xs))
        n_val = max(int(len(Xs) * self.val_fraction), 1)
        va, tr = idx[:n_val], idx[n_val:]
        Xtr, ytr = torch.tensor(Xs[tr]), torch.tensor(y[tr])
        Xva, yva = torch.tensor(Xs[va]), torch.tensor(y[va])

        self.model_ = _Net(Xs.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, bad = float("inf"), None, 0
        self.history_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xtr))
            running = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                running += loss.item() * len(b)
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xva), yva).item()
            self.history_.append((running / len(Xtr), val_loss))
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(X))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
