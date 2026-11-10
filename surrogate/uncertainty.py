"""
uncertainty.py - prediction intervals and the applicability domain (Sub-task 3.5; master plan M8).

Plan: claude/Phase3.5_Execution_Plan.md (decisions G1-G7, agreed 10 Nov 2026).

Prediction intervals (normalised split-conformal, G2)
    Residual r = y - yhat in ORIGINAL units (after any inverse transform; handoff §7).
    A small LightGBM "difficulty model" predicts log(|r| + eps) from the model inputs plus yhat;
    sigma(x) = exp(prediction). Score s = |r| / sigma(x). With n calibration scores, the
    interval at level 1 - alpha is  yhat +/- q * sigma(x),  q = the ceil((n + 1)(1 - alpha))-th
    smallest score (finite-sample split-conformal quantile). Lower bound clipped at 0 (every
    target is an energy or a load). scale="none" gives plain conformal (constant width, reference);
    scale="given" uses a supplied sigma (e.g. the GPR predictive SD, G5 hybrid).

    The calibration scores are themselves cross-fitted: the difficulty model that scores a
    calibration row never saw that row (inner group split: design_id for standard intervals,
    climate_key for new-site intervals). The difficulty model used for NEW rows is then fitted on
    all calibration rows.

Honest evaluation (G3)
    nested_intervals(cal, test, outer=...) calibrates separately for each value v of the column
    `outer` on the calibration rows whose `outer` differs from v, and scores the test rows with
    `outer == v`. Used three ways:
        standard, grouped CV   cal = test = grouped OOF,     outer = "fold"
        new-site, LOCO          cal = test = LOCO OOF,        outer = "climate_key"
        standard applied in LOCO  cal = grouped OOF, test = LOCO OOF, outer = "climate_key"
    A scored row never enters its own calibration (checked in tests/test_uncertainty.py).

Applicability domain (G6, G7)
    DomainChecker: each of the 21 design inputs inside design_space.yaml [min, max] (tolerance =
    half the last kept decimal; n_floors integer); climate distance = Euclidean distance to the
    nearest training climate in the 7 Phase 2 climate inputs, z-scored over the 10 training
    climates (= climate_gen.climate_distances); range check on the 7 climate inputs and the 7 EPW
    indices. Levels: "training" (one of the 10 climates), "new" (distance <= D_max and all inputs in
    range), "extrapolation" (otherwise). D_max = largest nearest-neighbour distance among the training
    climates (the furthest case LOCO has tested; Phoenix, 3.56).

    from surrogate import uncertainty as U
    cal = U.Conformal(scale="lgbm").fit(X_cal, yhat_cal, y_cal, groups=design_ids)
    lo, hi = cal.interval(X_new, yhat_new)                  # 90 %, clipped at 0
    dom = U.DomainChecker.from_training()
    flags = dom.check(designs, climate)                     # one row per design
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from surrogate import climate_indices as CI
from surrogate import features as FE
from surrogate.paths import DESIGN_SPACE

# ── Settings: config.yaml block uncertainty_3_5 (frozen 10 Nov 2026, before any 3.5 run) ──────

from surrogate.config import CONFIG  # noqa: E402

CFG = CONFIG["uncertainty_3_5"]
_M = CFG["method"]
ALPHA = float(CFG["alpha"])                                  # 90 % intervals (M8, gate S7)
LEVELS = tuple(float(v) for v in CFG["levels"])              # reliability curve (G4, report only)
LOWER_BOUND = float(CFG["lower_bound"])                      # all targets >= 0
EPS_FRACTION = float(_M["difficulty_model"]["eps_fraction"])  # eps = share of the median |r|
SCALE_PARAMS = dict(_M["difficulty_model"]["params"])        # difficulty model, fixed (not tuned)
INNER_SPLITS = int(_M["inner_cross_fit"]["group_kfold_splits"])   # many groups (design_id)
LOGO_MAX_GROUPS = int(_M["inner_cross_fit"]["logo_max_groups"])   # <= this many groups: leave-one-out
INNER_SEED = int(_M["inner_cross_fit"]["seed"])
COVERAGE_BAND = tuple(float(v) for v in _M["selection"]["band"])  # G2 / G5 per-climate band (= gate S7)
HYBRID_MIN_WIDTH_GAIN = round(1.0 - float(CFG["gpr_comparison"]["deploy_rule"]["max_width_ratio"]), 12)
TRAINING_DISTANCE_TOL = float(CFG["domain"]["climate"]["training_tolerance"])
PICKS: dict = CFG["picks"]                                   # final model per target (3.3 / 3.4)
YHAT_COLUMN = "_yhat"                                        # yhat as an extra input of the difficulty model

SCALES = ("lgbm", "none", "given")


# ── Conformal intervals ────────────────────────────────────────────────────────────────────

def conformal_quantile(scores, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest score; +inf if that exceeds n (too few scores)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    if not np.isfinite(s).all():
        raise ValueError("non-finite calibration score")
    k = math.ceil((n + 1) * level - 1e-12)
    return float("inf") if k > n else float(s[k - 1])


def inner_splits(groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner cross-fitting split of the calibration rows by group.

    Few groups (climates): leave-one-group-out. Many groups (designs): GroupKFold(5, shuffle, seed 42).
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    g = np.asarray(groups).astype(str)
    n_groups = len(np.unique(g))
    if n_groups < 2:
        raise ValueError("cross-fitting needs at least 2 groups")
    if n_groups <= LOGO_MAX_GROUPS:
        splitter = LeaveOneGroupOut()
    else:
        splitter = GroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    out = list(splitter.split(np.zeros((len(g), 1)), groups=g))
    for tr, te in out:
        if set(g[tr]) & set(g[te]):
            raise AssertionError("inner split shares a group between fit and score rows")
    return out


def _design_matrix(X, yhat) -> pd.DataFrame:
    X = pd.DataFrame(X).reset_index(drop=True)
    if YHAT_COLUMN in X.columns:
        raise ValueError(f"input column name {YHAT_COLUMN!r} is reserved")
    out = X.copy()
    out[YHAT_COLUMN] = np.asarray(yhat, float)
    return out


class ScaleModel:
    """Difficulty model: LightGBM on log(|r| + eps); sigma(x) = exp(prediction)."""

    def __init__(self, params: dict | None = None, eps_fraction: float = EPS_FRACTION):
        self.params = dict(SCALE_PARAMS if params is None else params)
        self.eps_fraction = eps_fraction

    def fit(self, X, yhat, abs_resid) -> "ScaleModel":
        import lightgbm as lgb
        a = np.abs(np.asarray(abs_resid, float))
        med = float(np.median(a))
        self.eps_ = max(self.eps_fraction * med, 1e-12) if med > 0 else 1e-12
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.columns_ = list(_design_matrix(X, yhat).columns)
        self.model_.fit(_design_matrix(X, yhat), np.log(a + self.eps_))
        return self

    def predict(self, X, yhat) -> np.ndarray:
        Z = _design_matrix(X, yhat)
        if list(Z.columns) != self.columns_:
            raise ValueError("difficulty model: input columns differ from fit")
        return np.exp(self.model_.predict(Z))


@dataclass
class Conformal:
    """Split-conformal interval for one target and one model (fit on calibration rows only).

    scale: "lgbm" (normalised, G2), "none" (plain), "given" (sigma supplied by the caller, e.g. GPR SD).
    """
    alpha: float = ALPHA
    scale: str = "lgbm"
    lower: float | None = LOWER_BOUND
    params: dict | None = None
    scores_: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")

    # -- fit --
    def fit(self, X, yhat, y, groups=None, sigma=None) -> "Conformal":
        yhat, y = np.asarray(yhat, float), np.asarray(y, float)
        if yhat.shape != y.shape:
            raise ValueError("yhat and y differ in shape")
        a = np.abs(y - yhat)
        if self.scale == "none":
            s = a
        elif self.scale == "given":
            s = a / self._sigma(sigma, len(y))
        else:
            if groups is None:
                raise ValueError("scale='lgbm' needs groups for cross-fitting")
            X = pd.DataFrame(X).reset_index(drop=True)
            sig_cf = np.empty(len(y))
            for tr, te in inner_splits(groups):
                m = ScaleModel(self.params).fit(X.iloc[tr], yhat[tr], a[tr])
                sig_cf[te] = m.predict(X.iloc[te], yhat[te])
            s = a / sig_cf
            self.scale_model_ = ScaleModel(self.params).fit(X, yhat, a)
        self.scores_ = np.sort(s)
        self.n_ = len(s)
        return self

    @staticmethod
    def _sigma(sigma, n) -> np.ndarray:
        if sigma is None:
            raise ValueError("scale='given' needs sigma")
        sig = np.asarray(sigma, float)
        if sig.shape != (n,) or not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            raise ValueError("sigma must be finite, positive and one value per row")
        return sig

    # -- predict --
    def quantile(self, level: float | None = None) -> float:
        return conformal_quantile(self.scores_, 1 - self.alpha if level is None else level)

    def width_scale(self, X, yhat, sigma=None) -> np.ndarray:
        """sigma(x) for new rows (1 for plain conformal)."""
        yhat = np.asarray(yhat, float)
        if self.scale == "none":
            return np.ones(len(yhat))
        if self.scale == "given":
            return self._sigma(sigma, len(yhat))
        return self.scale_model_.predict(X, yhat)

    def interval(self, X, yhat, sigma=None, level: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.scores_ is None:
            raise RuntimeError("Conformal.fit first")
        yhat = np.asarray(yhat, float)
        half = self.quantile(level) * self.width_scale(X, yhat, sigma)
        lo, hi = yhat - half, yhat + half
        if self.lower is not None:
            lo = np.maximum(lo, self.lower)
        return lo, hi


def gpr_native_interval(mu_t, sd_t, level: float = 1 - ALPHA, inverse=np.expm1,
                        lower: float | None = LOWER_BOUND) -> tuple[np.ndarray, np.ndarray]:
    """GPR predictive interval in the model's target space mapped to original units (G5).

    For the heating pick (log1p form): expm1(mu +/- z * sd). Monotone inverse -> exact mapping.
    """
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    mu_t, sd_t = np.asarray(mu_t, float), np.asarray(sd_t, float)
    lo, hi = inverse(mu_t - z * sd_t), inverse(mu_t + z * sd_t)
    if lower is not None:
        lo = np.maximum(lo, lower)
    return lo, hi


def gpr_sigma_original(mu_t, sd_t) -> np.ndarray:
    """Delta-method SD in original units for the log1p form: d expm1(m)/dm = exp(m)."""
    return np.exp(np.asarray(mu_t, float)) * np.asarray(sd_t, float)


# ── Nested (honest) evaluation, G3 ─────────────────────────────────────────────────────────

def nested_intervals(cal: pd.DataFrame, test: pd.DataFrame, X: pd.DataFrame, outer: str,
                     inner_group: str, scale: str = "lgbm", levels=LEVELS,
                     sigma_col: str = "sigma", log=None) -> pd.DataFrame:
    """Intervals for every test row, each calibrated without that row's `outer` group.

    cal, test: out-of-fold frames for ONE target (columns run_id, design_id, climate_key, fold,
    y, yhat; plus `sigma_col` for scale="given"). X: inputs of the difficulty model indexed by
    run_id. Returns test[run_id, design_id, climate_key, fold, y, yhat] + width_scale + q_<level>
    + lo_<level> / hi_<level> per level, + n_cal.
    """
    for f, name in ((cal, "cal"), (test, "test")):
        if f["run_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate run_id (pass one target at a time)")
    keep = ["run_id", "design_id", "climate_key", "fold", "y", "yhat"]
    parts = []
    for v in sorted(test[outer].astype(str).unique()):
        c = cal[cal[outer].astype(str) != v]
        t = test[test[outer].astype(str) == v]
        if set(c["run_id"]) & set(t["run_id"]):
            raise AssertionError(f"outer {outer}={v}: a scored row is in its own calibration")
        sig_c = c[sigma_col].to_numpy(float) if scale == "given" else None
        sig_t = t[sigma_col].to_numpy(float) if scale == "given" else None
        Xc = X.loc[c["run_id"].to_numpy()] if scale == "lgbm" else None
        Xt = X.loc[t["run_id"].to_numpy()] if scale == "lgbm" else None
        conf = Conformal(scale=scale).fit(Xc, c["yhat"].to_numpy(float), c["y"].to_numpy(float),
                                          groups=c[inner_group].to_numpy(), sigma=sig_c)
        out = t[keep].copy().reset_index(drop=True)
        yh = out["yhat"].to_numpy(float)
        w = conf.width_scale(Xt, yh, sig_t)
        out["width_scale"] = w
        out["n_cal"] = len(c)
        for lev in levels:
            q = conf.quantile(lev)
            lo, hi = yh - q * w, yh + q * w
            if conf.lower is not None:
                lo = np.maximum(lo, conf.lower)
            tag = level_tag(lev)
            out[f"q_{tag}"], out[f"lo_{tag}"], out[f"hi_{tag}"] = q, lo, hi
        parts.append(out)
        if log:
            log(f"    {outer}={v}: calibrated on {len(c)} rows, scored {len(t)}")
    return pd.concat(parts, ignore_index=True)


def level_tag(level: float) -> str:
    return f"{int(round(level * 100))}"


def coverage_table(iv: pd.DataFrame, by: list[str] | None = None, levels=LEVELS) -> pd.DataFrame:
    """Coverage and width per group (or overall): share of rows with lo <= y <= hi, mean/median width,
    relative width = mean width / mean y."""
    def one(g: pd.DataFrame) -> dict:
        y = g["y"].to_numpy(float)
        rec = {"n": len(g), "mean_y": float(y.mean())}
        for lev in levels:
            t = level_tag(lev)
            lo, hi = g[f"lo_{t}"].to_numpy(float), g[f"hi_{t}"].to_numpy(float)
            w = hi - lo
            rec[f"coverage_{t}"] = float(np.mean((y >= lo) & (y <= hi)))
            rec[f"mean_width_{t}"] = float(w.mean())
            rec[f"median_width_{t}"] = float(np.median(w))
            rec[f"rel_width_{t}"] = float(w.mean() / y.mean()) if y.mean() > 0 else float("nan")
        return rec
    if not by:
        return pd.DataFrame([one(iv)])
    rows = [{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **one(g)}
            for k, g in iv.groupby(by, sort=True)]
    return pd.DataFrame(rows)


def n_climates_outside(per_climate: pd.DataFrame, level: float = 1 - ALPHA,
                       band: tuple[float, float] = COVERAGE_BAND) -> int:
    c = per_climate[f"coverage_{level_tag(level)}"]
    return int(((c < band[0]) | (c > band[1])).sum())


def choose_scale(per_climate_lgbm: pd.DataFrame, per_climate_plain: pd.DataFrame) -> str:
    """G2: normalised unless it has MORE climates outside 0.85-0.95 than plain (grouped CV)."""
    return "none" if n_climates_outside(per_climate_lgbm) > n_climates_outside(per_climate_plain) else "lgbm"


def choose_heating(overall_conformal: pd.DataFrame, overall_hybrid: pd.DataFrame,
                   per_climate_hybrid: pd.DataFrame) -> str:
    """G5: the hybrid (conformal scaled by the GPR SD) only if >= 10 % narrower on average AND
    every heating climate inside 0.85-0.95 in grouped CV; else the G2 conformal."""
    t = level_tag(1 - ALPHA)
    w_c = float(overall_conformal[f"mean_width_{t}"].iloc[0])
    w_h = float(overall_hybrid[f"mean_width_{t}"].iloc[0])
    narrower = w_h <= (1 - HYBRID_MIN_WIDTH_GAIN) * w_c
    return "given" if narrower and n_climates_outside(per_climate_hybrid) == 0 else "conformal"


# ── Applicability domain, G6 + G7 ──────────────────────────────────────────────────────────

class DomainError(ValueError):
    """A design outside design_space.yaml (refused by the strict API, G6)."""


def design_bounds(path=DESIGN_SPACE) -> pd.DataFrame:
    """min, max, tolerance, integer per design input, in FE.DESIGN_INPUTS order."""
    spec = yaml.safe_load(open(path, encoding="utf-8"))["continuous_parameters"]
    missing = [c for c in FE.DESIGN_INPUTS if c not in spec]
    if missing:
        raise KeyError(f"design_space.yaml lacks {missing}")
    rows = []
    for c in FE.DESIGN_INPUTS:
        p = spec[c]
        rows.append({"input": c, "min": float(p["min"]), "max": float(p["max"]),
                     "tol": 0.5 * 10.0 ** (-int(p["decimals"])),
                     "integer": p.get("type") == "integer"})
    return pd.DataFrame(rows).set_index("input")


CLIMATE_COLUMNS = FE.CLIMATE_INPUTS + CI.INDEX_COLUMNS    # 14 columns for the range check


@dataclass
class DomainChecker:
    bounds: pd.DataFrame                  # design_bounds()
    climates: pd.DataFrame                # index climate_key, CLIMATE_COLUMNS (10 training climates)
    holdout_keys: tuple[str, ...] = ()
    d_max_fixed: float | None = None      # LOCO evidence: keep the 10-climate D_max with 9 climates

    @classmethod
    def from_training(cls) -> "DomainChecker":
        from surrogate import cv
        from climate.climate_features import holdout_keys, load_climate_registry
        df, _ = cv.load_training()
        base = df.groupby("climate_key")[FE.CLIMATE_INPUTS].first()
        table = base.join(CI.training_indices(), how="inner")
        if len(table) != len(base):
            raise AssertionError("climate indices missing for a training climate")
        return cls(design_bounds(), table[CLIMATE_COLUMNS],
                   tuple(holdout_keys(load_climate_registry(check_files=False))))

    # -- climate geometry --
    @property
    def mean_(self) -> pd.Series:
        return self.climates[FE.CLIMATE_INPUTS].mean()

    @property
    def sd_(self) -> pd.Series:
        return self.climates[FE.CLIMATE_INPUTS].std(ddof=0)

    def _z(self, C: pd.DataFrame) -> np.ndarray:
        return ((C[FE.CLIMATE_INPUTS] - self.mean_) / self.sd_).to_numpy(float)

    def training_distances(self) -> pd.DataFrame:
        """Per training climate: distance to the nearest OTHER training climate (= climate_gen)."""
        z = self._z(self.climates)
        d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        j = d.argmin(axis=1)
        return pd.DataFrame({"distance_to_nearest": d.min(axis=1),
                             "nearest_climate": self.climates.index[j]}, index=self.climates.index)

    @property
    def d_max(self) -> float:
        if self.d_max_fixed is not None:
            return float(self.d_max_fixed)
        return float(self.training_distances()["distance_to_nearest"].max())

    def without(self, key: str) -> "DomainChecker":
        """The checker a LOCO fold would have had: climate `key` removed, scaling and ranges from the
        other climates, D_max kept at this checker's value (G7 evidence table)."""
        return DomainChecker(self.bounds, self.climates.drop(index=key), self.holdout_keys, self.d_max)

    def climate_distance(self, C: pd.DataFrame) -> pd.DataFrame:
        z_new, z_tr = self._z(C), self._z(self.climates)
        d = np.sqrt(((z_new[:, None, :] - z_tr[None, :, :]) ** 2).sum(-1))
        j = d.argmin(axis=1)
        return pd.DataFrame({"climate_distance": d.min(axis=1),
                             "nearest_climate": self.climates.index[j].to_numpy()}, index=C.index)

    # -- checks --
    def design_violations(self, designs: pd.DataFrame) -> pd.Series:
        """Per row: comma-separated violating inputs ('' if inside the design space)."""
        D = pd.DataFrame(designs)
        missing = [c for c in FE.DESIGN_INPUTS if c not in D.columns]
        if missing:
            raise KeyError(f"design inputs missing: {missing}")
        bad = pd.DataFrame(False, index=D.index, columns=FE.DESIGN_INPUTS)
        for c, b in self.bounds.iterrows():
            v = D[c].astype(float)
            out = ~np.isfinite(v) | (v < b["min"] - b["tol"]) | (v > b["max"] + b["tol"])
            if b["integer"]:
                out |= (v - np.round(v)).abs() > 1e-9
            bad[c] = out
        return bad.apply(lambda r: ",".join(r.index[r.to_numpy()]), axis=1)

    def climate_out_of_range(self, C: pd.DataFrame) -> pd.Series:
        lo, hi = self.climates[CLIMATE_COLUMNS].min(), self.climates[CLIMATE_COLUMNS].max()
        cols = [c for c in CLIMATE_COLUMNS if c in C.columns]
        if set(FE.CLIMATE_INPUTS) - set(cols):
            raise KeyError("the 7 climate inputs are required")
        span = (hi - lo).replace(0, 1.0)
        tol = 1e-9 * span
        bad = pd.DataFrame({c: (C[c] < lo[c] - tol[c]) | (C[c] > hi[c] + tol[c]) for c in cols}, index=C.index)
        return bad.apply(lambda r: ",".join(r.index[r.to_numpy()]), axis=1)

    def check(self, designs: pd.DataFrame, climate=None, climate_key: str | None = None,
              allow_holdout: bool = False) -> pd.DataFrame:
        """One row per design: design_in_domain, design_violations, climate_distance, nearest_climate,
        climate_out_of_range, climate_level ("training" / "new" / "extrapolation").

        climate: dict / Series (one site for all rows) or DataFrame aligned with `designs`, holding
        the 7 climate inputs and, when available, the 7 EPW indices. If None, taken from `designs`.
        """
        if climate_key is not None and climate_key in self.holdout_keys and not allow_holdout:
            raise CI.HoldoutClimateError(f"{climate_key!r} is the holdout climate: domain checks on it "
                                         "run only inside final_test.py (3.7)")
        D = pd.DataFrame(designs)
        if climate is None:
            C = D
        elif isinstance(climate, pd.DataFrame):
            C = climate.set_axis(D.index) if len(climate) == len(D) else None
            if C is None:
                raise ValueError("climate frame must have one row per design")
        else:
            C = pd.DataFrame([dict(climate)] * len(D), index=D.index)
        viol = self.design_violations(D)
        dist = self.climate_distance(C)
        oor = self.climate_out_of_range(C)
        level = np.where(dist["climate_distance"] < TRAINING_DISTANCE_TOL, "training",
                         np.where((dist["climate_distance"] <= self.d_max) & (oor == ""), "new",
                                  "extrapolation"))
        return pd.DataFrame({"design_in_domain": viol == "", "design_violations": viol,
                             "climate_distance": dist["climate_distance"],
                             "nearest_climate": dist["nearest_climate"],
                             "climate_out_of_range": oor, "climate_level": level}, index=D.index)

    def require_in_domain(self, designs: pd.DataFrame) -> None:
        """Strict mode (G6): refuse any design outside design_space.yaml."""
        v = self.design_violations(designs)
        if (v != "").any():
            first = v[v != ""]
            raise DomainError(f"{len(first)} design(s) outside design_space.yaml, e.g. row "
                              f"{first.index[0]}: {first.iloc[0]}")

    # -- persistence (3.8 metadata.json) --
    def to_dict(self) -> dict:
        return {"bounds": self.bounds.reset_index().to_dict(orient="records"),
                "climates": self.climates.reset_index().to_dict(orient="records"),
                "holdout_keys": list(self.holdout_keys), "d_max": self.d_max,
                "climate_distance_features": FE.CLIMATE_INPUTS, "range_features": CLIMATE_COLUMNS}

    @classmethod
    def from_dict(cls, d: dict) -> "DomainChecker":
        return cls(pd.DataFrame(d["bounds"]).set_index("input"),
                   pd.DataFrame(d["climates"]).set_index("climate_key")[CLIMATE_COLUMNS],
                   tuple(d["holdout_keys"]))


# ════════════════════════════════════════════════════════════════════════════════════════════
# CLI (Sub-task 3.5 steps 3-5): stages oof, calibrate, domain, figures
#
#   ./venv/bin/python surrogate/uncertainty.py --stage oof        # R1 + R2 (~30-35 min on the Mac)
#   ./venv/bin/python surrogate/uncertainty.py --stage calibrate  # coverage, widths, selection
#   ./venv/bin/python surrogate/uncertainty.py --stage domain     # D_max, LOCO evidence
#   ./venv/bin/python surrogate/uncertainty.py --stage figures    # report figures (saved tables only)
#   add --dev for a quick code check on 25 % of the designs (writes outputs/uncertainty_dev/)
# ════════════════════════════════════════════════════════════════════════════════════════════

import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from surrogate.paths import OUT  # noqa: E402

HEAT_T = "heating_demand_kwh_m2"
COOL_T = "cooling_demand_kwh_m2"
MODES = {   # mode: (cal scheme, test scheme, outer column, inner group)      G3
    "standard_grouped": ("grouped", "grouped", "fold", "design_id"),
    "new_site_loco": ("loco", "loco", "climate_key", "climate_key"),
    "standard_in_loco": ("grouped", "loco", "climate_key", "design_id"),
}
SHORT_T = {"heating_demand_kwh_m2": "Heating", "cooling_demand_kwh_m2": "Cooling", "eui_kwh_m2": "EUI",
           "peak_heating_w_m2": "Peak heat", "peak_cooling_w_m2": "Peak cool",
           "zone_cooling_core_kwh_m2": "Core", "zone_cooling_south_kwh_m2": "South",
           "zone_cooling_east_kwh_m2": "East", "zone_cooling_north_kwh_m2": "North",
           "zone_cooling_west_kwh_m2": "West"}
R1_FILE = "oof_mlp_cg_loco.csv.gz"
R2_FILE = "oof_gpr_heating_std_{scheme}.csv.gz"


def out_dir(dev: bool) -> Path:
    d = OUT / ("uncertainty_dev" if dev else "uncertainty")
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_stage(d: Path, stage: str, info: dict, dev: bool):
    from surrogate import benchmark as B
    path = d / "uncertainty_result.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    data["stages"][stage] = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
                             "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                             "environment": B.environment(), **info}
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def load(dev: bool):
    """(df, X, heat): training rows (dev: the fixed 25 % design sample of benchmark.load) and the C-G+I
    matrix indexed by run_id (inputs of the difficulty model)."""
    from surrogate import benchmark as B
    df, _, heat = B.load(dev)
    X = CI.feature_matrix(df)
    X.index = df["run_id"].to_numpy()
    return df, X, heat


# ── step 4: gap-filling runs R1, R2 ────────────────────────────────────────────────────────

class _GPRWithSD:
    """Wraps TargetForm(GPR, form): fit/predict unchanged; predict also records the GP mean and
    predictive SD in the model's target space (log1p for the heating pick)."""

    def __init__(self, est, store: list):
        self.est, self.store = est, store

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict(self, X):
        yhat = self.est.predict(X)
        mu, sd = self.est.base_.predict(X, return_std=True)
        mu = np.asarray(mu, float) / self.est.scale_
        sd = np.asarray(sd, float) / self.est.scale_
        self.store.append((mu, sd))
        return yhat


def run_r1(df, dev: bool, log=print) -> pd.DataFrame:
    """R1: the 3.3 multi-output MLP on C-G, LOCO folds (benchmark.oof_for, settings mlp__all)."""
    from surrogate import benchmark as B
    from surrogate import cv
    from surrogate.config import ALL_TARGETS
    r1 = CFG["gap_runs"]["r1_mlp_cg_loco"]
    X = FE.engineer(df)[FE.feature_set(r1["feature_set"])]
    pbt = B.tuned_params_for("mlp", ALL_TARGETS, dev, allow_missing=dev, feature_set=None if dev else r1["feature_set"])
    t0 = time.time()
    oof = B.oof_for("mlp", pbt, df, X, ALL_TARGETS, cv.loco_folds(df), "loco")
    log(f"  R1 MLP C-G LOCO: {len(oof)} rows, {time.time() - t0:.0f} s")
    return oof.assign(model="mlp", feature_set=r1["feature_set"], form="direct")


def run_r2(df, dev: bool, log=print) -> dict[str, pd.DataFrame]:
    """R2: the heating GPR / C-G+I / log1p refitted on the same folds with its predictive SD."""
    from surrogate import climate_gen as CG
    from surrogate import cv, models
    r2 = CFG["gap_runs"]["r2_gpr_heating_sd"]
    X = CI.feature_matrix(df)
    top_k = CG.resolve_gpr_top_k(dev)
    if not dev and top_k != int(r2["top_k"]):
        raise SystemExit(f"GPR top_k {top_k} differs from config ({r2['top_k']})")
    pbt = CG.params_for("gpr", [HEAT_T], dev, top_k)
    out = {}
    for scheme, folds in (("grouped", cv.grouped_folds(df)), ("loco", cv.loco_folds(df))):
        store: list = []
        base = models.make_form("gpr", pbt[HEAT_T], r2["form"])
        t0 = time.time()
        oof = cv.cross_val_predict(df, X, [HEAT_T], lambda t: _GPRWithSD(base(t), store), folds, scheme)
        mu = np.concatenate([m for m, _ in store])
        sd = np.concatenate([s for _, s in store])
        if len(mu) != len(oof):
            raise AssertionError("R2: recorded SD rows differ from the out-of-fold rows")
        out[scheme] = oof.assign(mu_t=mu, sd_t=sd, model="gpr", feature_set=r2["feature_set"], form=r2["form"])
        log(f"  R2 GPR heating {scheme:7s}: {len(folds)} fits, {time.time() - t0:.0f} s")
    return out


def stage_oof(dev: bool, log=print, only: list[str] | None = None):
    from surrogate import metrics as M
    d = out_dir(dev)
    df, _, heat = load(dev)
    info = {}
    if not only or "r1" in only:
        r1 = run_r1(df, dev, log)
        r1.to_csv(d / R1_FILE, index=False)
        summ, _ = M.loco_summary(r1, heat)
        mape = float(summ.set_index("target").loc[COOL_T, "mape_pct"])
        chk = CFG["gap_runs"]["r1_mlp_cg_loco"]["check"]
        ok = abs(mape - chk["cooling_loco_mape_pct"]) <= chk["tolerance_points"]
        info["r1"] = {"rows": len(r1), "cooling_loco_mape_pct": mape, "reproduces_3_3": bool(ok)}
        log(f"  R1 check: cooling LOCO MAPE {mape:.2f} % (3.3: {chk['cooling_loco_mape_pct']}) -> "
            + ("OK" if ok else "DIFFERENT" + (" (dev: expected)" if dev else "")))
        if not ok and not dev:     # recorded, not fatal: R1 is a valid LOCO run of the same model either way
            log("  WARNING: R1 differs from the 3.3 LOCO run by more than the tolerance - report it")
    if not only or "r2" in only:
        r2 = run_r2(df, dev, log)
        info["r2"] = {}
        for scheme, f in r2.items():
            f.to_csv(d / R2_FILE.format(scheme=scheme), index=False)
            saved = _grid_rows(scheme, HEAT_T, PICKS[HEAT_T]).set_index("run_id")["yhat"]
            common = f["run_id"].isin(saved.index)
            rel = np.abs(f.loc[common, "yhat"].to_numpy() - saved.loc[f.loc[common, "run_id"]].to_numpy())
            rel = rel / np.maximum(np.abs(saved.loc[f.loc[common, "run_id"]].to_numpy()), 1e-9)
            same_expm1 = float(np.max(np.abs(np.expm1(f["mu_t"]) - f["yhat"]) / np.maximum(np.abs(f["yhat"]), 1e-9)))
            info["r2"][scheme] = {"rows": len(f), "max_rel_diff_to_saved_yhat": float(rel.max()),
                                  "max_rel_expm1_mu_vs_yhat": same_expm1, "sd_median": float(f["sd_t"].median())}
            tol = CFG["gap_runs"]["r2_gpr_heating_sd"]["check"]["mean_equals_saved_yhat_rel"]
            ok = rel.max() <= tol
            log(f"  R2 check {scheme}: max rel. diff to saved yhat {rel.max():.2e} -> "
                + ("OK" if ok else "DIFFERENT" + (" (dev: expected)" if dev else "")))
            info["r2"][scheme]["reproduces_3_4"] = bool(ok)
            if not ok and not dev:  # recorded, not fatal: calibration uses the SAVED 3.4 yhat; R2 adds only the SD
                log("  WARNING: R2 mean differs from the saved 3.4 heating predictions beyond the tolerance - report it")
    record_stage(d, "oof", info, dev)
    return info


# ── OOF of the final picks ─────────────────────────────────────────────────────────────────

def _read_oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"fold": str, "design_id": str, "run_id": str, "climate_key": str})


_OOF_CACHE: dict = {}


def _cached(path: Path) -> pd.DataFrame:
    key = str(path)
    if key not in _OOF_CACHE:
        _OOF_CACHE[key] = _read_oof(path)
    return _OOF_CACHE[key]


def _grid_rows(scheme: str, target: str, pick: dict) -> pd.DataFrame:
    f = _cached(OUT / "climate_gen" / f"oof_grid_{scheme}.csv.gz")
    return f[(f["target"] == target) & (f["model"] == pick["model"]) & (f["feature_set"] == pick["feature_set"])
             & (f["form"] == pick["form"])]


def pick_oof(target: str, scheme: str, dev: bool, runs: set | None = None) -> pd.DataFrame:
    """Out-of-fold rows of the final pick of `target` (one row per run): official saved files for the
    grid/benchmark sources (read-only; dev: restricted to the dev designs), R1 for MLP C-G LOCO."""
    p = PICKS[target]
    if p["source"] == "climate_gen_grid":
        f = _grid_rows(scheme, target, p)
    elif scheme == "loco":
        f = _cached(out_dir(dev) / R1_FILE)
        f = f[f["target"] == target]
    else:
        name = "oof_tuned_grouped.csv.gz" if p["source"] == "benchmark_tuned" else "oof_secondary_grouped.csv.gz"
        f = _cached(OUT / "benchmark" / name)
        f = f[(f["target"] == target) & (f["model"] == p["model"])]
    if runs is not None:
        f = f[f["run_id"].isin(runs)]
    f = f[["run_id", "design_id", "climate_key", "fold", "y", "yhat"]].reset_index(drop=True)
    if f.empty or f["run_id"].duplicated().any():
        raise AssertionError(f"{target} {scheme}: {len(f)} rows / duplicated run_id in the pick's OOF")
    return f


def _attach_sigma(f: pd.DataFrame, scheme: str, dev: bool) -> pd.DataFrame:
    sd = _read_oof(out_dir(dev) / R2_FILE.format(scheme=scheme)).set_index("run_id")
    mu_t, sd_t = sd.loc[f["run_id"], "mu_t"].to_numpy(), sd.loc[f["run_id"], "sd_t"].to_numpy()
    return f.assign(mu_t=mu_t, sd_t=sd_t, sigma=gpr_sigma_original(mu_t, sd_t))


# ── step 5: calibrate ─────────────────────────────────────────────────────────────────────

def _native(test: pd.DataFrame) -> pd.DataFrame:
    out = test[["run_id", "design_id", "climate_key", "fold", "y", "yhat"]].copy()
    for lev in LEVELS:
        lo, hi = gpr_native_interval(test["mu_t"], test["sd_t"], lev)
        t = level_tag(lev)
        out[f"lo_{t}"], out[f"hi_{t}"] = lo, hi
    return out


def n_jobs_for(n_tasks: int) -> int:
    import os
    return max(1, min(n_tasks, (os.cpu_count() or 2) - 1))


def _calibrate_target(t: str, dev: bool, X: pd.DataFrame, heat, runs) -> tuple[str, pd.DataFrame, float]:
    """All interval variants of one target (3 modes x methods, + GPR native for heating)."""
    from surrogate import metrics as M
    t0 = time.time()
    frames = {s: pick_oof(t, s, dev, runs) for s in ("grouped", "loco")}
    if t == HEAT_T:
        frames = {s: _attach_sigma(f, s, dev) for s, f in frames.items()}
    scored = {s: M.scored_rows(f, t, heat) for s, f in frames.items()}
    methods = ["lgbm", "none"] + (["given"] if t == HEAT_T else [])
    ivs = []
    for mode, (cs, ts, outer, inner) in MODES.items():
        for m in methods:
            iv = nested_intervals(scored[cs], frames[ts], X, outer=outer, inner_group=inner, scale=m)
            ivs.append(iv.assign(target=t, mode=mode, method=m))
        if t == HEAT_T and cs == ts:                      # GPR native: no calibration needed
            ivs.append(_native(frames[ts]).assign(target=t, mode=mode, method="native"))
    return t, pd.concat(ivs, ignore_index=True), time.time() - t0


def stage_calibrate(dev: bool, log=print, targets: list[str] | None = None):
    from surrogate import metrics as M
    from surrogate.config import ALL_TARGETS
    d = out_dir(dev)
    df, X, heat = load(dev)
    runs = set(df["run_id"]) if dev else None
    t_all = targets or ALL_TARGETS
    overall, per_climate, selection = [], [], {}
    from joblib import Parallel, delayed
    n_jobs = n_jobs_for(len(t_all))
    log(f"  {len(t_all)} targets on {n_jobs} worker(s) (difficulty models single-threaded: same numbers)")
    results = Parallel(n_jobs=n_jobs)(delayed(_calibrate_target)(t, dev, X, heat, runs) for t in t_all)
    for t, secs in ((r[0], r[2]) for r in results):
        log(f"  {SHORT_T[t]:10s} {secs:6.0f} s")
    iv = pd.concat([r[1] for r in results], ignore_index=True)
    iv["scored"] = ~iv["target"].isin(M.HEATING_TARGETS) | iv["climate_key"].isin(heat)
    for (t, mode, m), g in iv.groupby(["target", "mode", "method"], sort=False):
        overall.append(coverage_table(g[g["scored"]]).assign(target=t, mode=mode, method=m))
        pc = coverage_table(g, by=["climate_key"]).assign(target=t, mode=mode, method=m)
        pc["scored"] = ~pc["target"].isin(M.HEATING_TARGETS) | pc["climate_key"].isin(heat)
        per_climate.append(pc)
    overall = pd.concat(overall, ignore_index=True)
    per_climate = pd.concat(per_climate, ignore_index=True)
    first = ["target", "mode", "method"]
    overall = overall[first + [c for c in overall.columns if c not in first]]
    per_climate = per_climate[first + ["climate_key", "scored"]
                              + [c for c in per_climate.columns if c not in first + ["climate_key", "scored"]]]

    # selection (G2, G5) on standard intervals in grouped CV, scored climates
    def pc_of(t, mode, m):
        return per_climate[(per_climate["target"] == t) & (per_climate["mode"] == mode)
                           & (per_climate["method"] == m) & per_climate["scored"]]

    def ov_of(t, mode, m):
        return overall[(overall["target"] == t) & (overall["mode"] == mode) & (overall["method"] == m)]

    for t in t_all:
        scale = choose_scale(pc_of(t, "standard_grouped", "lgbm"), pc_of(t, "standard_grouped", "none"))
        rec = {"g2_scale": scale,
               "n_climates_outside_band": {m: n_climates_outside(pc_of(t, "standard_grouped", m))
                                           for m in (["lgbm", "none"] + (["given"] if t == HEAT_T else []))}}
        if t == HEAT_T:
            h = choose_heating(ov_of(t, "standard_grouped", scale), ov_of(t, "standard_grouped", "given"),
                               pc_of(t, "standard_grouped", "given"))
            rec["g5_heating"] = h
            if h == "given":
                scale = "given"
        rec["deployed_method"] = scale
        rec["coverage_90"] = {mode: float(ov_of(t, mode, scale)["coverage_90"].iloc[0]) for mode in MODES}
        rec["mean_width_90"] = {mode: float(ov_of(t, mode, scale)["mean_width_90"].iloc[0]) for mode in MODES}
        rec["final_quantiles"] = final_quantiles(t, scale, X, heat, dev, runs)
        selection[t] = rec
        log(f"  {SHORT_T[t]:10s} deployed {scale:5s}  coverage 90 %: standard {rec['coverage_90']['standard_grouped']:.3f}"
            f" | new-site {rec['coverage_90']['new_site_loco']:.3f} | standard in LOCO "
            f"{rec['coverage_90']['standard_in_loco']:.3f}")
    iv.to_csv(d / "intervals_oof.csv.gz", index=False)
    overall.to_csv(d / "coverage_table.csv", index=False)
    per_climate.to_csv(d / "coverage_per_climate.csv", index=False)
    rel = overall.melt(id_vars=first, value_vars=[f"coverage_{level_tag(v)}" for v in LEVELS],
                       var_name="level", value_name="coverage")
    rel["level"] = rel["level"].str.replace("coverage_", "").astype(int) / 100
    rel.to_csv(d / "reliability.csv", index=False)
    if HEAT_T in t_all:
        overall[overall["target"] == HEAT_T].to_csv(d / "gpr_comparison.csv", index=False)
    sel = {"official": not dev, "created_utc": datetime.now(timezone.utc).isoformat(),
           "rule": {"g2": CFG["method"]["selection"], "g5": CFG["gpr_comparison"]["deploy_rule"]},
           "selection": selection}
    (d / "selection.json").write_text(json.dumps(sel, indent=2, default=float), encoding="utf-8")
    record_stage(d, "calibrate", {"targets": t_all, "n_interval_rows": len(iv)}, dev)
    return overall, per_climate, selection


def final_quantiles(t: str, scale: str, X: pd.DataFrame, heat, dev: bool, runs) -> dict:
    """q per level for the deployed method, calibrated on ALL scored rows (3.8 refits the same objects)."""
    from surrogate import metrics as M
    out = {}
    for typ, scheme, inner in (("standard", "grouped", "design_id"), ("new_site", "loco", "climate_key")):
        f = pick_oof(t, scheme, dev, runs)
        if t == HEAT_T:
            f = _attach_sigma(f, scheme, dev)
        f = M.scored_rows(f, t, heat)
        conf = Conformal(scale=scale).fit(X.loc[f["run_id"]] if scale == "lgbm" else None, f["yhat"], f["y"],
                                          groups=f[inner].to_numpy(),
                                          sigma=f["sigma"].to_numpy() if scale == "given" else None)
        out[typ] = {level_tag(v): conf.quantile(v) for v in LEVELS} | {"n": conf.n_}
    return out


# ── step 5: domain ─────────────────────────────────────────────────────────────────────────

def stage_domain(dev: bool, log=print):
    from surrogate import metrics as M
    from surrogate.config import ALL_TARGETS
    d = out_dir(dev)
    dom = DomainChecker.from_training()
    (d / "domain.json").write_text(json.dumps(dom.to_dict(), indent=2, default=float), encoding="utf-8")
    df, _, heat = load(dev)
    runs = set(df["run_id"]) if dev else None
    dist = dom.training_distances()
    pcs = pd.read_csv(d / "coverage_per_climate.csv")
    sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))["selection"]
    rows = []
    for c in dom.climates.index:
        loo = dom.without(c)
        f = loo.check(df[df["climate_key"] == c].iloc[:1], climate=dom.climates.loc[c]).iloc[0]
        rec = {"climate_key": c, "distance_to_nearest": float(dist.loc[c, "distance_to_nearest"]),
               "nearest_climate": dist.loc[c, "nearest_climate"], "out_of_range_vs_others": f["climate_out_of_range"],
               "n_out_of_range": 0 if not f["climate_out_of_range"] else len(f["climate_out_of_range"].split(",")),
               "level_if_unseen": f["climate_level"]}
        rows.append(rec)
    table = pd.DataFrame(rows).set_index("climate_key")
    long = []
    for t in ALL_TARGETS:
        loco = pick_oof(t, "loco", dev, runs)
        _, tab = M.loco_summary(loco.assign(target=t, scheme="loco"), heat)
        tab = tab.set_index("climate_key")
        m = sel[t]["deployed_method"]
        cov = pcs[(pcs["target"] == t) & (pcs["method"] == m)]
        for c in table.index:
            if c not in tab.index:
                continue
            r = {"target": t, "climate_key": c, **table.loc[c].to_dict(),
                 "loco_bias_pct": float(tab.loc[c, "bias_pct"]), "loco_mape_pct": float(tab.loc[c, "mape_pct"]),
                 "coverage_90_new_site": float(cov[(cov["mode"] == "new_site_loco") & (cov["climate_key"] == c)]["coverage_90"].iloc[0]),
                 "coverage_90_standard_in_loco": float(cov[(cov["mode"] == "standard_in_loco") & (cov["climate_key"] == c)]["coverage_90"].iloc[0])}
            long.append(r)
    ev = pd.DataFrame(long)
    from scipy.stats import spearmanr
    corr = {}
    for t, g in ev.groupby("target", sort=False):
        if len(g) >= 4:
            corr[t] = {"spearman_abs_bias_vs_distance": float(spearmanr(g["distance_to_nearest"], g["loco_bias_pct"].abs()).statistic),
                       "spearman_new_site_coverage_vs_distance": float(spearmanr(g["distance_to_nearest"], g["coverage_90_new_site"]).statistic)}
    table.to_csv(d / "distance_table.csv")
    ev.to_csv(d / "domain_evidence.csv", index=False)
    info = {"d_max": dom.d_max, "d_max_climate": dist["distance_to_nearest"].idxmax(),
            "levels_if_unseen": table["level_if_unseen"].to_dict(), "correlations": corr}
    record_stage(d, "domain", info, dev)
    log(f"  D_max {dom.d_max:.3f} ({info['d_max_climate']}); levels if unseen: "
        + ", ".join(f"{k} {v}" for k, v in info["levels_if_unseen"].items()))
    return table, ev, info


# ── report ────────────────────────────────────────────────────────────────────────────────

def write_report(d: Path, dev: bool):
    from surrogate.config import ALL_TARGETS
    ov = pd.read_csv(d / "coverage_table.csv")
    sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))["selection"]
    L = [f"# Sub-task 3.5 — uncertainty and applicability domain{' (DEV — not results)' if dev else ''}", "",
         "All widths in original units (kWh/m² or W/m²). Coverage = share of rows whose EnergyPlus value lies "
         "inside the interval; target 0.90. Heating targets scored on the 7 heating climates.", "",
         "## 1. Deployed method and 90 % coverage", "",
         "| Target | Method | Standard, grouped CV | New-site, LOCO | Standard used in LOCO | Mean width std / new-site |",
         "|---|---|---|---|---|---|"]
    for t in ALL_TARGETS:
        if t not in sel:
            continue
        r = sel[t]
        L.append(f"| {SHORT_T[t]} | {r['deployed_method']} | {r['coverage_90']['standard_grouped']:.3f} | "
                 f"{r['coverage_90']['new_site_loco']:.3f} | {r['coverage_90']['standard_in_loco']:.3f} | "
                 f"{r['mean_width_90']['standard_grouped']:.2f} / {r['mean_width_90']['new_site_loco']:.2f} |")
    L += ["", "## 2. All methods (90 %)", "", "| Target | Mode | Method | Coverage | Mean width | Rel. width |",
          "|---|---|---|---|---|---|"]
    for _, r in ov.iterrows():
        L.append(f"| {SHORT_T[r['target']]} | {r['mode']} | {r['method']} | {r['coverage_90']:.3f} | "
                 f"{r['mean_width_90']:.2f} | {r['rel_width_90']:.3f} |")
    L += ["", "## 3. Selection rules", ""]
    for t, r in sel.items():
        L.append(f"- {SHORT_T[t]}: G2 → {r['g2_scale']} (climates outside 0.85–0.95: {r['n_climates_outside_band']})"
                 + (f"; G5 → {r['g5_heating']}" if "g5_heating" in r else ""))
    if (d / "distance_table.csv").exists():
        dt = pd.read_csv(d / "distance_table.csv")
        L += ["", "## 4. Applicability domain — each training climate as if unseen", "",
              "| Climate | Distance to nearest | Nearest | Out of range (vs other 9) | Level |", "|---|---|---|---|---|"]
        for _, r in dt.iterrows():
            L.append(f"| {r['climate_key']} | {r['distance_to_nearest']:.2f} | {r['nearest_climate']} | "
                     f"{r['out_of_range_vs_others'] if isinstance(r['out_of_range_vs_others'], str) else '—'} | "
                     f"{r['level_if_unseen']} |")
    (d / "uncertainty_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Sub-task 3.5: prediction intervals and applicability domain")
    ap.add_argument("--stage", required=True, choices=["oof", "calibrate", "domain", "figures", "report"])
    ap.add_argument("--dev", action="store_true", help="quick code check on 25 %% of the designs")
    ap.add_argument("--only", nargs="+", choices=["r1", "r2"], help="oof stage: run only R1 or R2")
    ap.add_argument("--targets", nargs="+", help="calibrate: dev only, a subset of targets")
    a = ap.parse_args(argv)
    if a.targets and not a.dev:
        ap.error("--targets is for dev runs only (the official stage calibrates all 10 targets)")
    log = lambda s: print(s, flush=True)                  # noqa: E731
    d = out_dir(a.dev)
    log(f"Sub-task 3.5 — stage {a.stage}{' [DEV]' if a.dev else ''} -> {d}")
    t0 = time.time()
    if a.stage == "oof":
        stage_oof(a.dev, log, a.only)
    elif a.stage == "calibrate":
        stage_calibrate(a.dev, log, a.targets)
        write_report(d, a.dev)
    elif a.stage == "domain":
        stage_domain(a.dev, log)
        write_report(d, a.dev)
    elif a.stage == "figures":
        from surrogate import uncertainty_figures as UF
        UF.make_all(d, a.dev)
    else:
        write_report(d, a.dev)
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
