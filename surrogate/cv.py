"""
cv.py - model-agnostic cross-validation harness for Phase 3 (plan M1, §4.2).

Two protocols on the TRAINING split only:
    grouped_folds(df)  development: GroupKFold(5, shuffle, seed 42) on design_id -
                       the exact folds of Sub-task 2.8, so every number is comparable.
    loco_folds(df)     diagnostic: leave-one-climate-out (10 folds), never used for tuning.

    df, cols = load_training()
    oof = cross_val_predict(df, df[cols["inputs"]], targets, factory, grouped_folds(df), "grouped")

Multi-output models (the MLP, Sub-task 3.3) use cross_val_predict_multi: one fit per fold for all
targets, same folds, guards and output frame.

`factory(target)` returns a fresh, unfitted estimator with fit(X, y) and predict(X), so any
model family (RF, XGBoost, LightGBM, GPR, MLP, a target-transform wrapper) runs through the same
code. The output is one long frame of out-of-fold predictions (same columns as eda_lib, plus
design_id and fit time), scored by surrogate.metrics.

Fit context (added 8 Nov 2026, Sub-task 3.4): if a model's fit() accepts `context`, it receives
the TRAINING rows of df for that fold (targets and metadata such as climate_key) - used by the
decomposed EUI and the physics MLP (models.py). Test rows are never passed.

Guards (plan §8): every entry point refuses non-training rows (eda_lib.HoldoutLeakError);
grouped folds never share a design_id; a LOCO fold never trains on its held-out climate;
every row is predicted exactly once per target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from eda import eda_lib as L
from surrogate.config import CONFIG

HoldoutLeakError = L.HoldoutLeakError
ModelFactory = Callable[[str], object]   # target name -> unfitted estimator


@dataclass(frozen=True)
class Fold:
    name: str               # "0".."4" for grouped CV, the held-out climate_key for LOCO
    train: np.ndarray       # row positions (iloc)
    test: np.ndarray


# ── data ──────────────────────────────────────────────────────────────────────

def load_training() -> tuple[pd.DataFrame, dict]:
    """Training split via the sha256-verified Phase 2 loader, checked against config.yaml."""
    df, cols, sha = L.load_split("training")
    expected = CONFIG["dataset_sha256"]["training"]
    if sha != expected:
        raise RuntimeError(f"training sha256 {sha[:12]}... differs from config.yaml {expected[:12]}...")
    L.require_training(df)
    return df, cols


# ── folds ─────────────────────────────────────────────────────────────────────

def _check_partition(folds: list[Fold], n: int) -> None:
    tested = np.concatenate([f.test for f in folds])
    if len(tested) != n or len(np.unique(tested)) != n:
        raise AssertionError("folds do not test every row exactly once")
    for f in folds:
        if np.intersect1d(f.train, f.test).size:
            raise AssertionError(f"fold {f.name}: a row is in both train and test")


def grouped_folds(df: pd.DataFrame, n_folds: int | None = None, seed: int | None = None,
                  group: str | None = None) -> list[Fold]:
    """GroupKFold on design_id, identical to eda_lib.cv_grouped (2.8)."""
    L.require_training(df)
    p = CONFIG["protocol"]["grouped_cv"]
    n_folds = n_folds or p["n_folds"]
    seed = CONFIG["protocol"]["seed"] if seed is None else seed
    groups = df[group or p["group"]].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_folds, shuffle=p["shuffle"], random_state=seed)
    folds = [Fold(str(k), tr, te) for k, (tr, te) in enumerate(splitter.split(np.zeros((len(df), 1)), groups=groups))]
    for f in folds:
        overlap = set(groups[f.train]) & set(groups[f.test])
        if overlap:
            raise AssertionError(f"design_id in both train and test: {sorted(overlap)[:5]}")
    _check_partition(folds, len(df))
    return folds


def loco_folds(df: pd.DataFrame, group: str | None = None) -> list[Fold]:
    """Leave-one-climate-out: one fold per climate, sorted by climate_key."""
    L.require_training(df)
    climate = df[group or CONFIG["protocol"]["loco"]["group"]].astype(str).to_numpy()
    folds = []
    for key in sorted(np.unique(climate)):
        te = np.flatnonzero(climate == key)
        tr = np.flatnonzero(climate != key)
        if key in set(climate[tr]):
            raise AssertionError(f"LOCO fold {key}: held-out climate present in training rows")
        folds.append(Fold(key, tr, te))
    _check_partition(folds, len(df))
    return folds


# ── out-of-fold prediction ────────────────────────────────────────────────────

def _rows(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def _fit_kwargs(model, df: pd.DataFrame, train: np.ndarray, design: np.ndarray) -> dict:
    """Optional fit arguments a model asks for: `groups` (design ids) and `context` (training rows)."""
    import inspect
    params = inspect.signature(model.fit).parameters
    kw = {}
    if "groups" in params:
        kw["groups"] = design[train]
    if "context" in params:
        kw["context"] = df.iloc[train].reset_index(drop=True)
    return kw


def cross_val_predict(df: pd.DataFrame, X, targets: list[str], factory: ModelFactory,
                      folds: list[Fold], scheme: str) -> pd.DataFrame:
    """Out-of-fold predictions for each target and fold.

    X: DataFrame (kept as a frame so models see feature names) or array, row-aligned with df.
    Returns long frame: run_id, design_id, climate_key, target, fold, y, yhat, fit_s, scheme.
    """
    L.require_training(df)
    if len(X) != len(df):
        raise ValueError(f"X has {len(X)} rows, df has {len(df)}")
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    else:
        X = np.asarray(X, dtype=float)
    run_id = df["run_id"].to_numpy()
    design = df["design_id"].astype(str).to_numpy()
    climate = df["climate_key"].astype(str).to_numpy()
    parts = []
    for t in targets:
        y = df[t].to_numpy(dtype=float, copy=True)
        for f in folds:
            model = factory(t)
            kw = _fit_kwargs(model, df, f.train, design)
            kw.pop("groups", None)                  # single-target models: groups only via cross_val_predict_multi
            t0 = time.perf_counter()
            model.fit(_rows(X, f.train), y[f.train], **kw)
            fit_s = time.perf_counter() - t0
            yhat = np.asarray(model.predict(_rows(X, f.test)), dtype=float).ravel()
            if len(yhat) != len(f.test):
                raise ValueError(f"{t}, fold {f.name}: model returned {len(yhat)} predictions for {len(f.test)} rows")
            parts.append(pd.DataFrame({"run_id": run_id[f.test], "design_id": design[f.test],
                                       "climate_key": climate[f.test], "target": t, "fold": f.name,
                                       "y": y[f.test], "yhat": yhat, "fit_s": fit_s}))
    out = pd.concat(parts, ignore_index=True)
    out["scheme"] = scheme
    return out


def cross_val_predict_multi(df: pd.DataFrame, X, targets: list[str], factory: Callable[[], object],
                            folds: list[Fold], scheme: str) -> pd.DataFrame:
    """Out-of-fold predictions from ONE multi-output model per fold (added 5 Nov 2026, Sub-task 3.3).

    factory() returns a fresh estimator with fit(X, Y[n, k]) and predict(X) -> (n, k), columns in
    `targets` order. If its fit accepts `groups`, it receives the training rows' design_id (the MLP
    uses it for a grouped early-stopping split). Same folds, guards and output frame as
    cross_val_predict; fit_s is the time of the one fit, repeated on every target of that fold.
    """
    L.require_training(df)
    if len(X) != len(df):
        raise ValueError(f"X has {len(X)} rows, df has {len(df)}")
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    else:
        X = np.asarray(X, dtype=float)
    run_id = df["run_id"].to_numpy()
    design = df["design_id"].astype(str).to_numpy()
    climate = df["climate_key"].astype(str).to_numpy()
    Y = df[list(targets)].to_numpy(dtype=float, copy=True)
    parts = []
    for f in folds:
        model = factory()
        kw = _fit_kwargs(model, df, f.train, design)
        t0 = time.perf_counter()
        model.fit(_rows(X, f.train), Y[f.train], **kw)
        fit_s = time.perf_counter() - t0
        yhat = np.asarray(model.predict(_rows(X, f.test)), dtype=float).reshape(len(f.test), -1)
        if yhat.shape != (len(f.test), len(targets)):
            raise ValueError(f"fold {f.name}: model returned shape {yhat.shape}, expected {(len(f.test), len(targets))}")
        for j, t in enumerate(targets):
            parts.append(pd.DataFrame({"run_id": run_id[f.test], "design_id": design[f.test],
                                       "climate_key": climate[f.test], "target": t, "fold": f.name,
                                       "y": Y[f.test, j], "yhat": yhat[:, j], "fit_s": fit_s}))
    out = pd.concat(parts, ignore_index=True)
    # same row order as cross_val_predict: target-major, then fold
    order = {t: i for i, t in enumerate(targets)}
    out = out.sort_values("target", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    out["scheme"] = scheme
    return out


def run_grouped(df: pd.DataFrame, X, targets: list[str], factory: ModelFactory) -> pd.DataFrame:
    return cross_val_predict(df, X, targets, factory, grouped_folds(df), "grouped")


def run_loco(df: pd.DataFrame, X, targets: list[str], factory: ModelFactory) -> pd.DataFrame:
    return cross_val_predict(df, X, targets, factory, loco_folds(df), "loco")
