"""
mlp.py - multi-output neural network (multilayer perceptron) for the Phase 3 benchmark
(plan M4/M6, claude/Phase3.3_Execution_Plan.md E6).

One network predicts all targets at once. Scikit-learn-like interface, so it runs through the
same CV harness as the tree models (cv.cross_val_predict_multi):

    net = MLPRegressor(n_layers=3, width=128)
    net.fit(X, Y, groups=design_id)      # Y: (n,) or (n, k); groups: design ids of the rows
    Yhat = net.predict(X_new)            # same shape convention as Y

Training (all choices frozen in config.yaml benchmark_3_3 / mlp_fixed):
  - inputs and each target standardised with the training rows' mean and SD (stored in the model);
  - loss = mean squared error over the standardised targets (every target weighs the same);
  - AdamW optimiser, mini-batches, SiLU activations, optional dropout;
  - early stopping: `val_fraction` of the training *designs* (not rows) is held back; training stops
    after `patience` epochs without improvement on that set and the best weights are restored.
    Splitting by design keeps the same building out of both parts, as in the outer CV;
  - LightGBM/XGBoost loaded before PyTorch and one CPU thread (macOS OpenMP rules, see _torch);
  - CPU and fixed seeds -> identical results for identical inputs (MPS is not bit-reproducible);
    inputs are made C-contiguous so a DataFrame and an array give bit-identical predictions.
    Predictions computed in batches of different sizes may differ at float32 rounding (~1e-7 relative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

ACTIVATIONS = ("silu", "relu", "tanh")
# macOS OpenMP rules (found 5 Nov 2026 on the Mac; LightGBM/XGBoost use Homebrew libomp, PyTorch its
# bundled copy):
#   1. load order: LightGBM/XGBoost BEFORE PyTorch works; PyTorch first, then LightGBM crashes
#      ("OMP: Error #179 pthread_mutex_init failed"). _torch() therefore imports the boosting libraries
#      first, so any script that trains the MLP gets the safe order automatically.
#   2. threads: with both runtimes loaded, PyTorch's OpenMP pool can deadlock in fit(); one intra-op
#      thread keeps PyTorch out of OpenMP regions. The network is small, so the cost is minor, and
#      results do not depend on the machine's core count.
TORCH_THREADS = 1
_PRELOAD = ("lightgbm", "xgboost")


def _torch():
    import importlib
    import sys
    import warnings
    if "torch" not in sys.modules:
        for lib in _PRELOAD:
            try:
                importlib.import_module(lib)
            except ImportError:                      # library not installed: nothing to order
                pass
    elif any(lib not in sys.modules for lib in _PRELOAD):
        warnings.warn("torch was imported before lightgbm/xgboost; on macOS importing them now can crash "
                      "(OMP Error #179). Import surrogate.mlp (or lightgbm) before torch.", RuntimeWarning)
    import torch
    if torch.get_num_threads() != TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    return torch


def inner_split(n: int, groups, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx) of n rows; whole groups go to validation. groups None -> split rows."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        k = max(1, int(round(val_fraction * n)))
        return np.sort(perm[k:]), np.sort(perm[:k])
    groups = np.asarray(groups).astype(str)
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} entries for {n} rows")
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least 2 groups for the early-stopping split")
    k = max(1, int(round(val_fraction * len(uniq))))
    val_groups = set(rng.permutation(uniq)[:k])
    is_val = np.array([g in val_groups for g in groups])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=3, width=128, dropout=0.0, lr=1e-3, weight_decay=1e-4, batch_size=128,
                 max_epochs=1000, patience=50, val_fraction=0.10, activation="silu", optimizer="adamw",
                 device="cpu", seed=42):
        self.n_layers = n_layers
        self.width = width
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.activation = activation
        self.optimizer = optimizer
        self.device = device
        self.seed = seed
        self._validate()

    def _validate(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.optimizer != "adamw":
            raise ValueError("only the AdamW optimiser is supported")
        if self.device != "cpu":
            raise ValueError("device must be 'cpu' (plan E6: reproducibility)")
        if int(self.n_layers) < 1 or int(self.width) < 1 or int(self.batch_size) < 1:
            raise ValueError("n_layers, width and batch_size must be positive")
        if not 0.0 <= float(self.dropout) < 1.0 or not 0.0 < float(self.val_fraction) < 0.5:
            raise ValueError("dropout must be in [0, 1) and val_fraction in (0, 0.5)")

    # ── network ─────────────────────────────────────────────────────────────────────────────

    def _build(self, n_in: int, n_out: int):
        import torch.nn as nn
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[self.activation]
        layers, width_in = [], n_in
        for _ in range(int(self.n_layers)):
            layers += [nn.Linear(width_in, int(self.width)), act()]
            if self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            width_in = int(self.width)
        layers.append(nn.Linear(width_in, n_out))
        return nn.Sequential(*layers)

    # ── fit / predict ───────────────────────────────────────────────────────────────────────

    def fit(self, X, y, groups=None):
        torch = _torch()
        self._validate()
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X = np.ascontiguousarray(X, dtype=np.float64)       # one memory layout -> identical float32 maths
        Y = np.asarray(y, dtype=np.float64)
        self.single_output_ = Y.ndim == 1
        Y = Y.reshape(len(Y), -1)
        if len(X) != len(Y):
            raise ValueError(f"X has {len(X)} rows, y has {len(Y)}")
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            raise ValueError("X and y must be finite")
        self.x_mean_, self.x_sd_ = X.mean(0), X.std(0)
        self.x_sd_[self.x_sd_ == 0] = 1.0
        self.y_mean_, self.y_sd_ = Y.mean(0), Y.std(0)
        self.y_sd_[self.y_sd_ == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_sd_
        Ys = (Y - self.y_mean_) / self.y_sd_

        tr, va = inner_split(len(X), groups, float(self.val_fraction), int(self.seed))
        self.inner_train_idx_, self.inner_val_idx_ = tr, va

        torch.manual_seed(int(self.seed))
        gen = torch.Generator().manual_seed(int(self.seed))
        torch.use_deterministic_algorithms(True)
        xt = torch.tensor(Xs[tr], dtype=torch.float32)
        yt = torch.tensor(Ys[tr], dtype=torch.float32)
        xv = torch.tensor(Xs[va], dtype=torch.float32)
        yv = torch.tensor(Ys[va], dtype=torch.float32)
        net = self._build(X.shape[1], Y.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        loss_fn = torch.nn.MSELoss()
        bs = int(self.batch_size)
        best, best_state, best_epoch, wait = np.inf, None, 0, 0
        history = []
        for epoch in range(1, int(self.max_epochs) + 1):
            net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), bs):
                b = perm[i:i + bs]
                if len(b) < 2 and len(xt) > 1:           # skip a 1-row tail batch (unstable step)
                    continue
                opt.zero_grad()
                loss = loss_fn(net(xt[b]), yt[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(xv), yv))
            history.append(v)
            if v < best - 1e-7:
                best, best_epoch, wait = v, epoch, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= int(self.patience):
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.best_epoch_, self.n_epochs_, self.best_val_loss_ = best_epoch, epoch, best
        self.val_history_ = history
        return self

    def predict(self, X):
        torch = _torch()
        X = np.ascontiguousarray(X, dtype=np.float64)
        Z = np.ascontiguousarray((X - self.x_mean_) / self.x_sd_, dtype=np.float32)
        with torch.no_grad():
            out = self.net_(torch.from_numpy(Z)).numpy()
        Y = out.astype(np.float64) * self.y_sd_ + self.y_mean_
        return Y[:, 0] if self.single_output_ else Y
