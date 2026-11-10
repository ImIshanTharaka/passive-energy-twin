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
        return float(self.training_distances()["distance_to_nearest"].max())

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
