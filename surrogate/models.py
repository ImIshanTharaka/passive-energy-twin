"""
models.py - model factories for the Phase 3 benchmark (plan M4/M5; claude/Phase3.3_Execution_Plan.md).

One place that knows how to build every model family, so tuning (3.3), climate tests (3.4),
uncertainty (3.5), the final fit (3.7) and Phase 4 all build models the same way.

    from surrogate import models
    p = models.default_params("lgbm")              # untuned settings frozen in config.yaml
    p = models.suggest(trial, "lgbm")              # one Optuna trial's settings (config search space)
    f = models.make("lgbm", p)                     # factory(target) -> fresh unfitted estimator
    oof = cv.run_grouped(df, X, targets, f)

    g = models.make_multi("mlp", models.default_params("mlp"))   # factory() -> one net for all targets
    oof = cv.cross_val_predict_multi(df, X, targets, g, cv.grouped_folds(df), "grouped")

Families: rf (scikit-learn RandomForestRegressor), xgb (XGBRegressor), lgbm (LGBMRegressor),
gpr (StandardScaler -> optional in-fold top-k selector -> GaussianProcessRegressor, plan E5),
mlp (surrogate.mlp.MLPRegressor, PyTorch, multi-output, plan E6).

Every setting comes from config.yaml `benchmark_3_3` (frozen 5 Nov 2026, before any 3.3 run):
untuned defaults (`defaults`), always-set values (`fixed`) and Optuna ranges (`search_spaces`).
Target transforms (log1p, / (HDD15 + 50), decomposed EUI) are added here in 3.4 as wrappers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from surrogate.config import CONFIG

CFG = CONFIG["benchmark_3_3"]
MODELS: tuple[str, ...] = tuple(CFG["models"])                 # rf, xgb, lgbm, gpr, mlp
TUNABLE: tuple[str, ...] = tuple(m for m in MODELS if CFG["tuning"]["trials"].get(m, 0) > 0)
MULTI_OUTPUT: tuple[str, ...] = ("mlp",)
LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "gpr": "Gaussian process", "mlp": "Neural network (MLP)"}


def _check(model: str) -> None:
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; expected one of {MODELS}")


# ── settings ────────────────────────────────────────────────────────────────────────────────

def default_params(model: str) -> dict:
    """Untuned settings (plan E2), exactly as frozen in config.yaml."""
    _check(model)
    d = CFG["defaults"][model]
    if d == "reference_2_8":
        return dict(CONFIG["reference_2_8"]["rf_params"])
    if d == "ablation_3_2":
        return dict(CONFIG["ablation_3_2"]["lgbm_params"])
    return dict(d)


def search_space(model: str) -> dict:
    _check(model)
    if model not in CFG["search_spaces"]:
        raise KeyError(f"{model} has no Optuna search space (plan E5: GPR uses marginal likelihood)")
    return CFG["search_spaces"][model]


def suggest(trial, model: str) -> dict:
    """One trial's settings from the frozen search space, merged with the always-set values."""
    params = {}
    for name, spec in search_space(model).items():
        kind = spec["type"]
        if kind == "fixed":
            params[name] = spec["value"]
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"{model}.{name}: unknown search-space type {kind!r}")
    return tuned_params(model, params)


def tuned_params(model: str, params: dict) -> dict:
    """Searched settings + fixed values of the space + the model's always-set values (seeds, threads)."""
    out = {n: s["value"] for n, s in search_space(model).items() if s["type"] == "fixed"}
    out.update(params)
    clash = set(CFG.get("fixed", {}).get(model, {})) & set(params)
    if clash:
        raise ValueError(f"{model}: {sorted(clash)} are fixed in config and cannot be tuned")
    out.update(CFG.get("fixed", {}).get(model, {}))
    return out


# ── factories ───────────────────────────────────────────────────────────────────────────────

def make(model: str, params: dict):
    """factory(target) -> fresh unfitted single-target estimator (fit/predict)."""
    _check(model)
    params = dict(params)
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        RandomForestRegressor(**params)                         # fail early on a bad key
        return lambda target: RandomForestRegressor(**params)
    if model == "lgbm":
        from lightgbm import LGBMRegressor
        return lambda target: LGBMRegressor(**params)
    if model == "xgb":
        from xgboost import XGBRegressor
        return lambda target: XGBRegressor(**params)
    if model == "gpr":
        GPR(**params)
        return lambda target: GPR(**params)
    return make_multi(model, params)                            # mlp: also works for one target


def make_multi(model: str, params: dict):
    """factory() -> fresh unfitted multi-output estimator: fit(X, Y[n, k], groups=...) / predict -> (n, k)."""
    _check(model)
    if model not in MULTI_OUTPUT:
        raise ValueError(f"{model} is single-output; use make()")
    from surrogate.mlp import MLPRegressor
    fixed = dict(CFG["mlp_fixed"])
    fixed.pop("targets", None)
    kw = {**fixed, **params}
    MLPRegressor(**kw)
    return lambda *_: MLPRegressor(**kw)


# ── Gaussian process (plan E5) ──────────────────────────────────────────────────────────────

class GPR(BaseEstimator, RegressorMixin):
    """Standardised inputs -> optional top-k feature selection -> GP with ARD Matérn + white noise.

    The kernel hyper-parameters (one length scale per feature, signal variance, noise) are
    fitted by maximising the marginal likelihood on the training rows only - this is the GPR's
    own tuning (plan E5). `top_k` (fallback, master plan §8): keep the k features with the
    highest LightGBM gain importance, ranked on the training rows of the fold (no leakage).
    predict(X, return_std=True) gives the native predictive standard deviation (3.5).
    """

    def __init__(self, nu=2.5, length_scale_init=1.0, length_scale_bounds=(0.01, 1000.0), constant_init=1.0,
                 constant_bounds=(0.001, 1000.0), noise_init=0.01, noise_bounds=(1e-6, 10.0), normalize_y=True,
                 n_restarts_optimizer=0, random_state=42, top_k=None):
        self.nu = nu
        self.length_scale_init = length_scale_init
        self.length_scale_bounds = length_scale_bounds
        self.constant_init = constant_init
        self.constant_bounds = constant_bounds
        self.noise_init = noise_init
        self.noise_bounds = noise_bounds
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.top_k = top_k

    def _select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.top_k or self.top_k >= X.shape[1]:
            return np.arange(X.shape[1])
        from lightgbm import LGBMRegressor
        ranker = LGBMRegressor(**default_params("lgbm"), importance_type="gain").fit(X, y)
        order = np.argsort(-ranker.feature_importances_, kind="stable")
        return np.sort(order[: self.top_k])

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.selected_ = self._select(Z, y)
        Z = Z[:, self.selected_]
        d = Z.shape[1]
        kernel = (ConstantKernel(self.constant_init, tuple(self.constant_bounds))
                  * Matern(length_scale=np.full(d, float(self.length_scale_init)),
                           length_scale_bounds=tuple(self.length_scale_bounds), nu=self.nu)
                  + WhiteKernel(self.noise_init, tuple(self.noise_bounds)))
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.gp_ = GaussianProcessRegressor(kernel=kernel, normalize_y=self.normalize_y,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                random_state=self.random_state).fit(Z, y)
        # A length scale at its upper bound = that feature switched off (ARD: expected, not a fault);
        # noise at its lower bound = near-deterministic simulator output. Kept for the report.
        self.bound_warnings_ = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        self.length_scales_ = np.atleast_1d(self.gp_.kernel_.k1.k2.length_scale)
        self.noise_level_ = float(self.gp_.kernel_.k2.noise_level)
        return self

    def _z(self, X):
        return self.scaler_.transform(np.asarray(X, dtype=float))[:, self.selected_]

    def predict(self, X, return_std: bool = False):
        return self.gp_.predict(self._z(X), return_std=return_std)

    @property
    def selected_features_(self) -> list | None:
        if self.feature_names_in_ is None:
            return None
        return [self.feature_names_in_[i] for i in self.selected_]


# ── target forms and decomposed EUI (Sub-task 3.4, claude/Phase3.4_Execution_Plan.md F2-F4) ─────
#
# Plain idea: the model learns an easier, physics-shaped quantity and the wrapper converts the
# answer back to kWh/m2 (or W/m2), so every caller still sees the original target.
#   direct      y                                 (as in 3.3)
#   log1p       log(1 + y)                        back: exp(z) - 1
#   ratio       y / (HDD15 + 50)                  back: z * (HDD15 + 50)   ("per unit of cold", fix 2)
#   decomposed  EUI = 48.87 + heating + cooling + OA level + OA deviation   (fix 3)
# Constants come from config.yaml `target_forms` (frozen in 3.1, M6). HDD15 is read from the
# `hdd15` column of X (feature set C-G+I, surrogate.climate_indices), so X must be a DataFrame.

TF = CONFIG["target_forms"]
HDD_COL = "hdd15"
HDD_OFFSET = float(TF["hdd_offset"])                  # 50
ELEC = float(TF["eui_constant_kwh_m2"])               # 48.87 kWh/m2, lights + equipment
FORMS: tuple[str, ...] = ("direct", "log1p", "ratio")
HEAT, COOL, EUI = "heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2"
PEAK_HEAT = "peak_heating_w_m2"


def climate_scale(X) -> np.ndarray:
    """HDD15 + 50 per row (the 'unit of cold'). Needs a DataFrame with an `hdd15` column."""
    if not isinstance(X, pd.DataFrame) or HDD_COL not in X.columns:
        raise ValueError(f"the ratio form needs X as a DataFrame with a '{HDD_COL}' column (feature set C-G+I)")
    s = X[HDD_COL].to_numpy(dtype=float) + HDD_OFFSET
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError("HDD15 + 50 must be finite and positive")
    return s


def to_form(y, form: str, X=None) -> np.ndarray:
    """Original target -> the quantity the model learns."""
    y = np.asarray(y, dtype=float)
    if form == "direct":
        return y.copy()
    if form == "log1p":
        if (y <= -1).any():
            raise ValueError("log1p form needs y > -1")
        return np.log1p(y)
    if form == "ratio":
        return y / climate_scale(X)
    raise ValueError(f"unknown target form {form!r}; expected one of {FORMS}")


def from_form(z, form: str, X=None) -> np.ndarray:
    """Model output -> original target units (inverse of to_form)."""
    z = np.asarray(z, dtype=float)
    if form == "direct":
        return z.copy()
    if form == "log1p":
        return np.expm1(z)
    if form == "ratio":
        return z * climate_scale(X)
    raise ValueError(f"unknown target form {form!r}; expected one of {FORMS}")


class TargetForm(BaseEstimator, RegressorMixin):
    """Wrap a single-target estimator so it learns `to_form(y)` and predicts in original units."""

    def __init__(self, base=None, form: str = "direct", scale_match: bool = False):
        self.base = base
        self.form = form
        self.scale_match = scale_match

    def fit(self, X, y):
        """scale_match (sensitivity_3_4 only): multiply the learned quantity by s = SD(y) / SD(form(y))
        on the training rows, so scale-dependent settings tuned on y (e.g. LightGBM reg_alpha) mean the
        same thing; predictions are divided by s before the inverse transform."""
        from sklearn.base import clone
        if self.form not in FORMS:
            raise ValueError(f"unknown target form {self.form!r}")
        z = to_form(y, self.form, X)
        self.scale_ = 1.0
        if self.scale_match:
            sz = float(np.std(z))
            if not sz > 0:
                raise ValueError("cannot scale-match a constant target")
            self.scale_ = float(np.std(np.asarray(y, dtype=float))) / sz
        self.base_ = clone(self.base)
        self.base_.fit(X, z * self.scale_)
        return self

    def predict(self, X):
        z = np.asarray(self.base_.predict(X), dtype=float).ravel() / self.scale_
        return from_form(z, self.form, X)


def _climate_index_table(X: pd.DataFrame, climate, cols) -> pd.DataFrame:
    """One row of climate-index values per climate; refuses values that vary within a climate."""
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"X lacks climate-index columns {missing} (feature set C-G+I)")
    frame = X[list(cols)].copy()
    frame["climate_key"] = np.asarray(climate).astype(str)
    g = frame.groupby("climate_key", sort=True)
    if (g.nunique() > 1).any().any():
        raise ValueError("climate-index values vary within a climate")
    return g.first()


class OALevel:
    """Straight line for the climate level of the outdoor-air (OA) term (F3).

    Ordinary least squares with intercept, one point per training climate:
        mean OA term of the climate = b0 + b1*occ_enth_excess + b2*occ_band_dh + b3*occ_heat_dh
    """

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, index_table: pd.DataFrame, oa_mean: pd.Series):
        idx = index_table.loc[oa_mean.index, self.cols].to_numpy(dtype=float)
        A = np.column_stack([np.ones(len(idx)), idx])
        if len(A) < A.shape[1]:
            raise ValueError(f"{len(A)} climates cannot fit {A.shape[1]} coefficients")
        self.coef_, *_ = np.linalg.lstsq(A, oa_mean.to_numpy(dtype=float), rcond=None)
        self.climates_ = list(oa_mean.index)
        self.fitted_ = pd.Series(A @ self.coef_, index=oa_mean.index)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        idx = X[self.cols].to_numpy(dtype=float)
        return self.coef_[0] + idx @ self.coef_[1:]


def oa_term(eui, heating, cooling) -> np.ndarray:
    """OA term = EUI - 48.87 - heating - cooling (exact by definition, Phase 2 reference §2.3)."""
    return np.asarray(eui, dtype=float) - ELEC - np.asarray(heating, dtype=float) - np.asarray(cooling, dtype=float)


def _context(context, n: int, need: list[str]) -> pd.DataFrame:
    if context is None:
        raise ValueError("this model needs fit(..., context=<training rows of the dataset>); "
                         "cv.cross_val_predict passes it automatically")
    context = context.reset_index(drop=True)
    if len(context) != n:
        raise ValueError(f"context has {len(context)} rows, X has {n}")
    missing = [c for c in need if c not in context.columns]
    if missing:
        raise ValueError(f"context lacks columns {missing}")
    return context


class OATermModel(BaseEstimator, RegressorMixin):
    """Outdoor-air (OA) term of EUI = OA level (straight line, climate) + OA deviation (design), F3.

    fit(X, y_eui, context): OA term = EUI - 48.87 - heating - cooling on the TRAINING rows (context
    gives heating, cooling, climate_key); its climate means (training rows only) fit the OALevel
    line; `dev` learns OA term - climate mean. predict(X) = line(X indices) + dev(X), i.e. the
    predicted OA term. Used inside DecomposedEUI and on its own by climate_gen.py, which adds the
    out-of-fold heating and cooling predictions instead of refitting them (identical result).
    """

    def __init__(self, dev=None, level_cols=None):
        self.dev = dev
        self.level_cols = level_cols

    def _level_cols(self):
        if self.level_cols is not None:
            return list(self.level_cols)
        from surrogate.climate_indices import OA_LEVEL_INDICES
        return list(OA_LEVEL_INDICES)

    def fit(self, X, y, context=None):
        from sklearn.base import clone
        ctx = _context(context, len(X), [HEAT, COOL, "climate_key"])
        climate = ctx["climate_key"].astype(str).to_numpy()
        oa = oa_term(y, ctx[HEAT].to_numpy(dtype=float), ctx[COOL].to_numpy(dtype=float))
        self.climate_mean_ = pd.Series(oa).groupby(climate).mean()
        table = _climate_index_table(X, climate, self._level_cols())
        self.level_ = OALevel(self._level_cols()).fit(table, self.climate_mean_)
        self.dev_ = clone(self.dev).fit(X, oa - self.climate_mean_.loc[climate].to_numpy())
        return self

    def predict_parts(self, X) -> pd.DataFrame:
        return pd.DataFrame({"oa_level": self.level_.predict(X),
                             "oa_dev": np.asarray(self.dev_.predict(X), dtype=float).ravel()})

    def predict(self, X):
        p = self.predict_parts(X)
        return (p["oa_level"] + p["oa_dev"]).to_numpy()


class DecomposedEUI(BaseEstimator, RegressorMixin):
    """EUI = 48.87 + heating + cooling + OA level + OA deviation (fix 3, plan F3).

    heat, cool, dev: unfitted single-target estimators (same family; F3). Heating is learned in the
    ratio form (heating / (HDD15 + 50), fix 2). OA level and deviation: OATermModel (line on the
    climate means of the OA term over the TRAINING rows only). fit() needs `context` = the training
    rows of the dataset (heating, cooling, climate_key); cv passes it. predict() needs only X.
    """

    def __init__(self, heat=None, cool=None, dev=None, level_cols=None):
        self.heat = heat
        self.cool = cool
        self.dev = dev
        self.level_cols = level_cols

    def fit(self, X, y, context=None):
        from sklearn.base import clone
        ctx = _context(context, len(X), [HEAT, COOL, "climate_key"])
        self.oa_ = OATermModel(self.dev, self.level_cols).fit(X, y, context=ctx)
        self.heat_ = TargetForm(self.heat, "ratio").fit(X, ctx[HEAT].to_numpy(dtype=float))
        self.cool_ = clone(self.cool).fit(X, ctx[COOL].to_numpy(dtype=float))
        return self

    @property
    def level_(self):
        return self.oa_.level_

    @property
    def oa_climate_mean_(self):
        return self.oa_.climate_mean_

    def predict_parts(self, X) -> pd.DataFrame:
        oa = self.oa_.predict_parts(X)
        parts = pd.DataFrame({
            "elec": np.full(len(X), ELEC),
            "heating": self.heat_.predict(X),
            "cooling": np.asarray(self.cool_.predict(X), dtype=float).ravel(),
            "oa_level": oa["oa_level"].to_numpy(),
            "oa_dev": oa["oa_dev"].to_numpy(),
        })
        parts["eui"] = parts[["elec", "heating", "cooling", "oa_level", "oa_dev"]].sum(axis=1)
        return parts

    def predict(self, X):
        return self.predict_parts(X)["eui"].to_numpy()


def make_oa_term(model: str, eui_params: dict):
    """factory(target) -> OATermModel whose deviation model is the family with its EUI settings (F3)."""
    if model in MULTI_OUTPUT:
        raise ValueError("the MLP learns the OA deviation as an extra output of MLPPhysics (F4)")
    dev = make(model, eui_params)

    def factory(target):
        if target != EUI:
            raise ValueError(f"the OA-term model is fitted against {EUI}, not {target}")
        return OATermModel(dev=dev(EUI))
    return factory


def make_form(model: str, params: dict, form: str, scale_match: bool = False):
    """factory(target) -> TargetForm(<family estimator>, form). Single-target families only."""
    base = make(model, params)
    if model in MULTI_OUTPUT:
        raise ValueError("the MLP gets its target forms through MLPPhysics (F4)")
    if form not in FORMS:
        raise ValueError(f"unknown target form {form!r}")
    return lambda target: TargetForm(base(target), form, scale_match)


def make_decomposed(model: str, params_by_target: dict):
    """factory(target) -> DecomposedEUI with heating / cooling / OA-deviation models of one family.

    params_by_target: {heating_demand_kwh_m2: ..., cooling_demand_kwh_m2: ..., eui_kwh_m2: ...}
    (the 3.3 tuned settings; the OA-deviation model uses the EUI settings, F3).
    """
    if model in MULTI_OUTPUT:
        raise ValueError("the MLP gets the decomposed EUI through MLPPhysics (F4)")
    h, c, e = (make(model, params_by_target[t]) for t in (HEAT, COOL, EUI))

    def factory(target):
        if target != EUI:
            raise ValueError(f"the decomposed form is for {EUI}, not {target}")
        return DecomposedEUI(heat=h(HEAT), cool=c(COOL), dev=e(EUI))
    return factory


# ── multi-output MLP with the physics forms (F4) ────────────────────────────────────────────

# Extra outputs learned by the one network next to the 10 original targets (F4: 14 outputs).
MLP_EXTRA = (("heating_ratio", HEAT, "ratio"), ("heating_log1p", HEAT, "log1p"),
             ("peak_heating_ratio", PEAK_HEAT, "ratio"), ("oa_dev", EUI, "oa_dev"))


class MLPPhysics(BaseEstimator, RegressorMixin):
    """One multi-output MLP (3.3 settings) on 14 outputs: the 10 targets + MLP_EXTRA (F4).

    fit(X, Y, groups, context): Y has the 10 targets in `targets` order; context = training rows
    (climate_key) for the OA climate means and line. predict(X) -> the 10 direct outputs (same as a
    3.3 net, so cv.cross_val_predict_multi works). predict_forms(X) -> DataFrame with columns
    (target, form) for every candidate, all in original units:
        direct (10), heating ratio, heating log1p, peak heating ratio,
        EUI decomposed = 48.87 + heating (ratio) + cooling (direct) + OA level + OA deviation.
    """

    def __init__(self, mlp_params=None, targets=None, level_cols=None):
        self.mlp_params = mlp_params
        self.targets = targets
        self.level_cols = level_cols

    def _targets(self):
        if self.targets is not None:
            return list(self.targets)
        from surrogate.config import ALL_TARGETS
        return list(ALL_TARGETS)

    def _level_cols(self):
        if self.level_cols is not None:
            return list(self.level_cols)
        from surrogate.climate_indices import OA_LEVEL_INDICES
        return list(OA_LEVEL_INDICES)

    def extra_outputs(self, X, Y: pd.DataFrame, climate) -> np.ndarray:
        """The 4 extra training columns (in MLP_EXTRA order) for rows with known targets."""
        oa = oa_term(Y[EUI], Y[HEAT], Y[COOL])
        dev = oa - self.oa_climate_mean_.loc[np.asarray(climate).astype(str)].to_numpy()
        return np.column_stack([to_form(Y[HEAT], "ratio", X), to_form(Y[HEAT], "log1p"),
                                to_form(Y[PEAK_HEAT], "ratio", X), dev])

    def fit(self, X, Y, groups=None, context=None):
        targets = self._targets()
        Y = pd.DataFrame(np.asarray(Y, dtype=float).reshape(len(X), -1), columns=targets)
        for t in (HEAT, COOL, EUI, PEAK_HEAT):
            if t not in targets:
                raise ValueError(f"MLPPhysics needs target {t}")
        ctx = _context(context, len(X), ["climate_key"])
        climate = ctx["climate_key"].astype(str).to_numpy()
        oa = oa_term(Y[EUI], Y[HEAT], Y[COOL])
        self.oa_climate_mean_ = pd.Series(oa).groupby(climate).mean()
        table = _climate_index_table(X, climate, self._level_cols())
        self.level_ = OALevel(self._level_cols()).fit(table, self.oa_climate_mean_)
        Y14 = np.column_stack([Y.to_numpy(), self.extra_outputs(X, Y, climate)])
        self.net_ = make_multi("mlp", dict(self.mlp_params or {}))()
        self.net_.fit(X, Y14, groups=groups)
        self.output_names_ = targets + [name for name, _, _ in MLP_EXTRA]
        return self

    def predict_raw(self, X) -> pd.DataFrame:
        """The 14 network outputs as learned (extra columns in their transformed units)."""
        return pd.DataFrame(self.net_.predict(X), columns=self.output_names_)

    def predict(self, X):
        return self.predict_raw(X)[self._targets()].to_numpy()

    def predict_forms(self, X) -> pd.DataFrame:
        raw = self.predict_raw(X)
        out = {(t, "direct"): raw[t].to_numpy() for t in self._targets()}
        heat_ratio = from_form(raw["heating_ratio"], "ratio", X)
        out[(HEAT, "ratio")] = heat_ratio
        out[(HEAT, "log1p")] = from_form(raw["heating_log1p"], "log1p")
        out[(PEAK_HEAT, "ratio")] = from_form(raw["peak_heating_ratio"], "ratio", X)
        out[(EUI, "decomposed")] = ELEC + heat_ratio + raw[COOL].to_numpy() + self.level_.predict(X) \
            + raw["oa_dev"].to_numpy()
        frame = pd.DataFrame(out)
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["target", "form"])
        return frame
