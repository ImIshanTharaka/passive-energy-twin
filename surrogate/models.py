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
