"""
cv.py - model-agnostic cross-validation harness for Phase 3 (plan M1, §4.2).

Two protocols on the TRAINING split only:
    grouped_folds(df)  development: GroupKFold(5, shuffle, seed 42) on design_id -
                       the exact folds of Sub-task 2.8, so every number is comparable.
    loco_folds(df)     diagnostic: leave-one-climate-out (10 folds), never used for tuning.

    df, cols = load_training()
    oof = cross_val_predict(df, df[cols["inputs"]], targets, factory, grouped_folds(df), "grouped")

`factory(target)` returns a fresh, unfitted estimator with fit(X, y) and predict(X), so any
model family (RF, XGBoost, LightGBM, GPR, MLP, a target-transform wrapper) runs through the same
code. The output is one long frame of out-of-fold predictions (same columns as eda_lib, plus
design_id and fit time), scored by surrogate.metrics.

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
            t0 = time.perf_counter()
            model.fit(_rows(X, f.train), y[f.train])
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


def run_grouped(df: pd.DataFrame, X, targets: list[str], factory: ModelFactory) -> pd.DataFrame:
    return cross_val_predict(df, X, targets, factory, grouped_folds(df), "grouped")


def run_loco(df: pd.DataFrame, X, targets: list[str], factory: ModelFactory) -> pd.DataFrame:
    return cross_val_predict(df, X, targets, factory, loco_folds(df), "loco")
