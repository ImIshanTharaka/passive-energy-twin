"""
registry.py - Sub-task 3.8: the deployed surrogate behind one front door.

Plan: claude/Phase3.8_Execution_Plan.md (decisions J1-J7, agreed 18 Nov 2026). Every setting is read
from config.yaml block `registry_3_8` (appended 18 Nov 2026, before any 3.8 run).

In plain words
    Phase 4 (optimiser) and Phase 5 (web service) use only this module:

        from surrogate.registry import Surrogate, Site
        s = Surrogate.load()                               # checks the SHA-256 of every file
        site = s.site("colombo")                           # a training climate ...
        site = Site.from_epw("kandy.epw", name="Kandy")    # ... or any site from its EPW file
        res = s.predict(designs, site)                     # 10 outputs + ranges + warnings
        Y = s.predict_values(designs, site)                # fast path: NumPy array n x 10

    Nothing is refitted (J1): the four objects fitted in 3.6 serve the 10 targets exactly as tested
    in 3.7, through the shared-feature path (J5: features engineered once per chunk). The heating GPR
    is loaded from a slim copy (J2: the 72 MB Cholesky factor, needed only for the GPR's own standard
    deviation, removed - predictions identical). Ranges are the 3.7 conformal interval models (J3),
    standard for a training climate and new-site otherwise (J4). Messages are the agreed text (J7).

CLI (Mac for official numbers; --dev writes to outputs/registry_dev/, never a result)
    ./venv/bin/python surrogate/registry.py --stage build [--dev]   # slim GPR + registry.json
    ./venv/bin/python surrogate/registry.py --stage check [--dev]   # acceptance: load, self-checks, timing
    ./venv/bin/python surrogate/registry.py --stage report [--dev]  # registry_report.md

Guards: this module never imports final_test.py or explain.py and never reads a holdout file
(test_registry checks it). `load()` needs no dataset and no EPW: registry.json carries the climate
table, the domain checker and the messages, so a Phase 5 container needs only the files it lists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import features as FE
from surrogate import uncertainty as U
from surrogate.config import ALL_TARGETS, CONFIG, CONFIG_PATH
from surrogate.paths import OUT, PROJECT_ROOT

from sklearn.gaussian_process import GaussianProcessRegressor

# ── Settings: config.yaml block registry_3_8 (frozen 18 Nov 2026, before any 3.8 run) ──────────

CFG = CONFIG["registry_3_8"]
_ST, _IV, _DOM, _PR, _MSG, _ACC = (CFG["storage"], CFG["intervals"], CFG["domain"], CFG["prediction"],
                                   CFG["messages"], CFG["acceptance"])
MODELS_DIR = OUT / "models"                                # the 3.6 / 3.7 files (+ slim GPR, registry.json)
DEV_DIR = OUT / Path(CFG["dev"]["out_dir"]).name           # outputs/registry_dev
REGISTRY_FILE = _ST["manifest"]                            # registry.json
SLIM_FILE = _ST["gpr_slim"]["file"]                        # gpr_heating_slim.pkl
LEVELS = tuple(float(v) for v in _IV["levels"])            # 0.5, 0.8, 0.9, 0.95
DEFAULT_LEVEL = float(_IV["default_level"])
TYPE_BY_LEVEL: dict = dict(_IV["type_by_climate_level"])   # training -> standard, new/extrapolation -> new_site
CLIMATE_LEVELS = tuple(_DOM["levels"])
READABLE: dict = dict(_DOM["readable_names"])
CHUNK = int(CONFIG["explain_3_6"]["final_models"]["predict_chunk_rows"])   # 10,000 (same as 3.6 / 3.7)
EXACT_TOL = float(_PR["equality_with_3_7_path"]["exact_rel_tol"])
MLP_TOL = float(_PR["equality_with_3_7_path"]["mlp_rel_tol"])
SLIM_TOL = float(_ST["gpr_slim"]["equality_rel_tol"])
TARGETS: list[str] = list(ALL_TARGETS)
CLIMATE_COLUMNS = list(FE.CLIMATE_INPUTS) + list(CI.INDEX_COLUMNS)     # 7 Phase 2 inputs + 7 EPW indices
UNITS = {t: ("W/m²" if t.startswith("peak_") else "kWh/m²") for t in TARGETS}

# Files a Phase 5 image needs besides the model files (J2 risk row; handoff §7 3.2 row).
RUNTIME_MODULES = {
    "surrogate": ["__init__.py", "registry.py", "models.py", "mlp.py", "features.py", "climate_indices.py",
                  "uncertainty.py", "config.py", "paths.py", "config.yaml"],
    "simulation": ["climate/", "idf_generator/ (construction, fenestration, internal_loads and their packages)",
                   "config/design_space.yaml", "config/climate_config.yaml"],
}


class RegistryError(RuntimeError):
    """A registry file is missing, changed or inconsistent."""


class DesignError(U.DomainError):
    """A design outside design_space.yaml (strict mode, 3.5 G6) - message is the agreed J7 text."""


# ── small helpers ───────────────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Path stored in registry.json: relative to surrogate/outputs (portable between machines)."""
    return Path(path).resolve().relative_to(OUT.resolve()).as_posix()


def _abs(rel: str) -> Path:
    return OUT / rel


def environment() -> dict:
    import importlib
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for lib, mod in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("scikit-learn", "sklearn"),
                     ("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("torch", "torch")]:
        try:
            env[lib] = importlib.import_module(mod).__version__
        except ImportError:
            env[lib] = None
    return env


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    mod = [ln[3:] for ln in run("status", "--porcelain", "--", "surrogate", "simulation").splitlines()
           if ln and not ln.startswith("??")]
    return {"commit": run("rev-parse", "HEAD") or None, "modified_tracked": mod}


def readable(columns: str | list[str]) -> list[str]:
    """Column names -> the agreed readable names (J4)."""
    cols = [c for c in columns.split(",") if c] if isinstance(columns, str) else list(columns)
    return [READABLE.get(c, c) for c in cols]


def _display(key: str, names: dict | None = None) -> str:
    return (names or {}).get(key) or key.replace("_", " ").title()


# ── J2: slim GPR ────────────────────────────────────────────────────────────────────────────────

class SlimGaussianProcess(GaussianProcessRegressor):
    """A fitted scikit-learn GPR without its Cholesky factor `L_` (72 MB for 3,000 rows).

    The mean prediction uses only `alpha_` and `X_train_`, so it is unchanged (same arithmetic,
    bit-identical - checked in the build and in test_registry). The GPR's own standard deviation
    needs `L_`; it is not used in deployment (intervals are conformal, 3.5 G5) and is refused here."""

    def predict(self, X, return_std=False, return_cov=False):
        if return_std or return_cov:
            raise RuntimeError("the deployed (slim) heating GPR has no Cholesky factor: its own standard "
                               "deviation is not available - use the conformal ranges of Surrogate.predict, "
                               "or load the archived gpr_heating.pkl for research use")
        return super().predict(X)


def slim_gpr(full) -> object:
    """Copy of the 3.6 heating object (TargetForm(GPR)) with the inner GPR made slim."""
    obj = copy.deepcopy(full)
    gp = obj.base_.gp_
    if not isinstance(gp, GaussianProcessRegressor) or not hasattr(gp, "alpha_"):
        raise TypeError("expected a fitted scikit-learn GaussianProcessRegressor at .base_.gp_")
    import importlib
    cls = importlib.import_module("surrogate.registry").SlimGaussianProcess   # pickled as surrogate.registry.*
    slim = cls.__new__(cls)                                                  # (not __main__ when run as a script)
    state = {k: v for k, v in gp.__dict__.items() if k != "L_"}
    slim.__dict__.update(state)
    obj.base_.gp_ = slim
    return obj


# ── site ────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Site:
    """One site: the 7 Phase 2 climate inputs + the 7 EPW indices (3.4). Model inputs need both.

    key: a training climate key (then the 3.6/3.7 table values are used), else None.
    """
    name: str
    climate: dict                       # 14 values, CLIMATE_COLUMNS
    key: str | None = None
    source: str = "values"              # training_key | epw_file | values
    ashrae_zone: str | None = None

    def __post_init__(self):
        missing = [c for c in CLIMATE_COLUMNS if c not in self.climate]
        if missing:
            raise KeyError(f"site {self.name!r}: climate values missing {missing}")
        bad = [c for c in CLIMATE_COLUMNS if not np.isfinite(float(self.climate[c]))]
        if bad:
            raise ValueError(f"site {self.name!r}: non-finite climate values {bad}")
        object.__setattr__(self, "climate", {c: float(self.climate[c]) for c in CLIMATE_COLUMNS})

    @classmethod
    def from_epw(cls, path: str | Path, name: str | None = None, ashrae_zone: str | None = None) -> "Site":
        """Any site from its EPW file: the Phase 2 climate features and the 3.4 indices, same code as training.

        Refuses an EPW the parser rejects, or one that declares holidays / daylight saving (the
        occupied-hour mask of the indices assumes none; climate_indices._check_calendar)."""
        from climate.climate_features import compute_features
        from climate.epw_parser import parse_epw
        path = Path(path)
        epw = parse_epw(path)
        zone = ashrae_zone or _PR["epw_default_ashrae_zone"]
        feat = compute_features(epw, zone)
        vals = {c: float(getattr(feat, c)) for c in FE.CLIMATE_INPUTS}
        vals.update(CI.indices_from_epw(epw))
        return cls(name=name or path.stem, climate=vals, key=None, source="epw_file", ashrae_zone=zone)

    @classmethod
    def from_values(cls, name: str, values: dict) -> "Site":
        return cls(name=name, climate=dict(values), key=None, source="values")

    def to_dict(self) -> dict:
        return {"name": self.name, "key": self.key, "source": self.source, "ashrae_zone": self.ashrae_zone,
                "climate": dict(self.climate)}


# ── result ──────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """What Surrogate.predict returns. All values in original units (kWh/m², W/m²)."""
    values: pd.DataFrame                 # n x 10
    lower: pd.DataFrame                  # n x 10, clipped at 0
    upper: pd.DataFrame
    level: float
    interval_type: str                   # standard | new_site
    flags: pd.DataFrame                  # per design: design_in_domain, design_violations
    climate: dict                        # site, level, distance, nearest, out_of_range (readable)
    message: str                         # agreed J7 text for the site's level
    heating_note: str
    units: dict = field(default_factory=lambda: dict(UNITS))

    def to_frame(self) -> pd.DataFrame:
        """One wide table: <target>, <target>_lo_<lvl>, <target>_hi_<lvl>, then the design flags."""
        tag = U.level_tag(self.level)
        cols = {}
        for t in self.values.columns:
            cols[t] = self.values[t]
            cols[f"{t}_lo_{tag}"] = self.lower[t]
            cols[f"{t}_hi_{tag}"] = self.upper[t]
        out = pd.DataFrame(cols, index=self.values.index)
        return pd.concat([out, self.flags], axis=1)

    def to_dict(self) -> dict:
        """JSON-ready (Phase 5 response body)."""
        return {"level": self.level, "interval_type": self.interval_type, "units": self.units,
                "climate": self.climate, "message": self.message, "heating_note": self.heating_note,
                "designs": json.loads(self.to_frame().reset_index(drop=True).to_json(orient="records"))}


# ── the surrogate ───────────────────────────────────────────────────────────────────────────────

class Surrogate:
    """The deployed surrogate (J1 option A). Create with Surrogate.load()."""

    TARGETS = TARGETS

    def __init__(self, objects: dict, intervals: dict, info: dict, checker: U.DomainChecker,
                 climates: pd.DataFrame, path: Path | None = None):
        self.objects, self.intervals, self.registry = objects, intervals, info
        self.checker, self.climates, self.path = checker, climates, path
        self.target_object: dict = dict(info["target_object"])
        self.object_targets = {n: [t for t in TARGETS if self.target_object[t] == n] for n in objects}
        self.feature_sets = {k: list(v) for k, v in info["feature_sets"].items()}
        self.names: dict = dict(info["climates"]["display_names"])
        self.messages: dict = dict(info["messages"])
        self.readable: dict = dict(info["readable_names"])

    # -- loading --------------------------------------------------------------------------------
    @classmethod
    def load(cls, registry_dir: str | Path | None = None, check_hash: bool = True) -> "Surrogate":
        """Load registry.json and every file it lists (SHA-256 checked). No dataset or EPW is read."""
        d = Path(registry_dir) if registry_dir is not None else MODELS_DIR
        rpath = d / REGISTRY_FILE
        if not rpath.exists():
            raise RegistryError(f"no {REGISTRY_FILE} in {d}: run `registry.py --stage build`")
        info = json.loads(rpath.read_text(encoding="utf-8"))
        files = info["files"]
        if check_hash:
            for name, rec in files.items():
                p = _abs(rec["path"])
                if not p.exists():
                    raise RegistryError(f"{name}: file {p} is missing")
                if sha256(p) != rec["sha256"]:
                    raise RegistryError(f"{name}: {p.name} differs from {REGISTRY_FILE} (SHA-256) - file changed?")
        _check_versions(info.get("environment", {}))
        from surrogate.mlp import _torch
        _torch()                                         # OpenMP-safe import order + 1 thread (handoff §2.4)
        objects = {}
        for name in info["objects"]:
            with open(_abs(files[name]["path"]), "rb") as fh:
                objects[name] = pickle.load(fh)
        with open(_abs(files["intervals"]["path"]), "rb") as fh:
            intervals = pickle.load(fh)
        dom = info["domain"]
        climates = pd.DataFrame(info["climates"]["table"]).set_index("climate_key")[CLIMATE_COLUMNS]
        checker = U.DomainChecker(pd.DataFrame(dom["bounds"]).set_index("input"), climates,
                                  tuple(dom["holdout_keys"]), d_max_fixed=float(dom["d_max"]))   # J4 fix
        return cls(objects, intervals, info, checker, climates, d)

    # -- sites ----------------------------------------------------------------------------------
    def site(self, key: str) -> Site:
        """A training climate by key (values exactly as in training)."""
        if key not in self.climates.index:
            raise KeyError(f"{key!r} is not one of the 10 training climates {sorted(self.climates.index)}; "
                           "use Site.from_epw(path) for any other site")
        return Site(name=self.names.get(key, key), climate=self.climates.loc[key].to_dict(), key=key,
                    source="training_key")

    @property
    def training_climates(self) -> list[str]:
        return list(self.climates.index)

    # -- designs --------------------------------------------------------------------------------
    @staticmethod
    def designs_frame(designs) -> pd.DataFrame:
        """DataFrame / list of dicts / one dict -> the 21 design inputs (float), index kept."""
        if isinstance(designs, pd.Series):
            designs = designs.to_dict()
        if isinstance(designs, dict):
            D = pd.DataFrame([designs])
        elif isinstance(designs, pd.DataFrame):
            D = designs
        else:
            D = pd.DataFrame(list(designs))
        missing = [c for c in FE.DESIGN_INPUTS if c not in D.columns]
        if missing:
            raise KeyError(f"design inputs missing: {missing}")
        try:
            out = D[FE.DESIGN_INPUTS].astype(float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"design inputs must be numbers: {e}") from None
        return out

    def bounds(self) -> pd.DataFrame:
        """21 design inputs: min, max, integer (design_space.yaml) - the optimiser's search box."""
        return self.checker.bounds[["min", "max", "integer"]].copy()

    def design_violations(self, D: pd.DataFrame) -> pd.Series:
        """Same rule and output as uncertainty.DomainChecker.design_violations (3.5 G6), vectorised
        (the pandas row loop there costs ~5 ms per call - too slow for one-design calls)."""
        b = self.checker.bounds.loc[FE.DESIGN_INPUTS]
        A = D[FE.DESIGN_INPUTS].to_numpy(float)
        lo, hi, tol = (b[c].to_numpy(float) for c in ("min", "max", "tol"))
        bad = ~np.isfinite(A) | (A < lo - tol) | (A > hi + tol)
        integer = b["integer"].to_numpy(bool)
        if integer.any():
            bad[:, integer] |= np.abs(A[:, integer] - np.round(A[:, integer])) > 1e-9
        out = np.full(len(A), "", dtype=object)
        cols = np.asarray(FE.DESIGN_INPUTS)
        for i in np.flatnonzero(bad.any(axis=1)):
            out[i] = ",".join(cols[bad[i]])
        return pd.Series(out, index=D.index, dtype=object)

    def design_flags(self, D: pd.DataFrame, strict: bool) -> pd.DataFrame:
        viol = self.design_violations(D)
        if strict and (viol != "").any():
            bad = viol[viol != ""]
            raise DesignError(self._refusal(D, bad.index[0], bad.iloc[0].split(",")[0], n_bad=len(bad)))
        return pd.DataFrame({"design_in_domain": viol == "", "design_violations": viol}, index=D.index)

    def _refusal(self, D: pd.DataFrame, row, col: str, n_bad: int = 1) -> str:
        b = self.checker.bounds.loc[col]
        fmt = (lambda v: f"{v:g}")
        msg = self.messages["design_refused"].format(input=col, value=fmt(D.at[row, col]),
                                                     min=fmt(b["min"]), max=fmt(b["max"]))
        if b["integer"]:
            msg = msg.rstrip(".") + " (whole numbers only)."
        if n_bad > 1:
            msg += f" ({n_bad} designs refused; first: row {row!r})"
        return msg

    # -- climate check (J4) ---------------------------------------------------------------------
    def climate_check(self, site: Site) -> dict:
        level, d, nearest, oor_cols = climate_level(self.checker, site.climate)
        if site.key is not None and site.key in self.climates.index:
            level = "training"
        return {"site": site.name, "source": site.source, "level": level, "distance": d,
                "d_max": float(self.checker.d_max), "nearest_climate": nearest,
                "nearest_name": self.names.get(nearest, nearest), "out_of_range": oor_cols,
                "out_of_range_readable": [self.readable.get(c, c) for c in oor_cols]}

    def message_for(self, chk: dict) -> str:
        txt = self.messages[chk["level"]]
        return txt.format(nearest=chk["nearest_name"], inputs=", ".join(chk["out_of_range_readable"]) or "none")

    # -- model inputs (J5: features once per chunk, shared by the four objects) ----------------
    def matrices(self, D: pd.DataFrame, site: Site) -> dict[str, pd.DataFrame]:
        raw = D.reset_index(drop=True).copy()
        for c in FE.CLIMATE_INPUTS:
            raw[c] = site.climate[c]
        cg = FE.engineer(raw)[self.feature_sets["C-G"]]
        idx = pd.DataFrame({c: np.full(len(raw), site.climate[c]) for c in CI.INDEX_COLUMNS})
        cgi = pd.concat([cg, idx], axis=1)
        if list(cgi.columns) != self.feature_sets["C-G+I"]:
            raise RegistryError("feature columns differ from registry.json (C-G+I)")
        return {"C-G": cg, "C-G+I": cgi}

    def _predict_object(self, name: str, Xc: dict) -> pd.DataFrame:
        est, tg = self.objects[name], self.object_targets[name]
        if name == "gpr_heating":
            return pd.DataFrame({tg[0]: np.asarray(est.predict(Xc["C-G+I"]), float)})
        if name == "xgb_eui":
            return pd.DataFrame({tg[0]: np.asarray(est.predict(Xc["C-G+I"]), float)})
        if name == "mlp_cg":
            out = pd.DataFrame(np.asarray(est.predict(Xc["C-G"]), dtype=float).reshape(len(Xc["C-G"]), -1),
                               columns=TARGETS)
            return out[tg]
        if name == "mlp_physics":
            forms = est.predict_forms(Xc["C-G+I"])
            return pd.DataFrame({t: forms[(t, "direct")].to_numpy(float) for t in tg})
        raise RegistryError(f"unknown object {name!r}")

    def _values_chunks(self, D: pd.DataFrame, site: Site, with_x: bool = False):
        for i in range(0, len(D), CHUNK):
            Xc = self.matrices(D.iloc[i:i + CHUNK], site)
            parts = [self._predict_object(n, Xc) for n in self.objects]
            yield pd.concat(parts, axis=1)[TARGETS], (Xc["C-G+I"] if with_x else None)

    # -- public prediction ------------------------------------------------------------------------
    def predict_values(self, designs, site: Site | str, strict: bool | None = None) -> np.ndarray:
        """Fast path (Phase 4): the 10 outputs as a NumPy array (n x 10, Surrogate.TARGETS order)."""
        site = self.site(site) if isinstance(site, str) else site
        D = self.designs_frame(designs)
        self.design_flags(D, _DOM["strict_default"] if strict is None else strict)
        if len(D) == 0:
            return np.empty((0, len(TARGETS)))
        return np.vstack([v.to_numpy(float) for v, _ in self._values_chunks(D, site)])

    def predict(self, designs, site: Site | str, level: float = DEFAULT_LEVEL,
                strict: bool | None = None) -> Prediction:
        """The 10 outputs, a range at `level` for each, the design flags and the site warning (J6)."""
        level = float(level)
        if not any(abs(level - l) < 1e-12 for l in LEVELS):
            raise ValueError(f"level must be one of {LEVELS}")
        site = self.site(site) if isinstance(site, str) else site
        D = self.designs_frame(designs)
        flags = self.design_flags(D, _DOM["strict_default"] if strict is None else strict)
        chk = self.climate_check(site)
        typ = TYPE_BY_LEVEL[chk["level"]]
        vals, los, his = [], [], []
        for v, Xi in self._values_chunks(D, site, with_x=True):
            lo, hi = {}, {}
            for t in TARGETS:
                lo[t], hi[t] = self.intervals[t][typ].interval(Xi, v[t].to_numpy(float), level=level)
            vals.append(v)
            los.append(pd.DataFrame(lo)[TARGETS])
            his.append(pd.DataFrame(hi)[TARGETS])
        empty = pd.DataFrame(columns=TARGETS, dtype=float)
        cat = (lambda xs: pd.concat(xs, ignore_index=True).set_axis(D.index) if xs else empty.copy())
        return Prediction(values=cat(vals), lower=cat(los), upper=cat(his), level=level, interval_type=typ,
                          flags=flags, climate=chk, message=self.message_for(chk),
                          heating_note=self.messages["heating_note"])

    def info(self) -> dict:
        """What is deployed (registry.json without the bulky tables)."""
        skip = {"climates", "domain", "feature_sets"}
        return {k: v for k, v in self.registry.items() if k not in skip}


def climate_level(checker: U.DomainChecker, climate: dict) -> tuple[str, float, str, list[str]]:
    """3.5 G7 rule (J4, unchanged): training if distance ~ 0; new if distance <= D_max and all 14 values
    inside the training range; else extrapolation. Returns (level, distance, nearest, out-of-range columns)."""
    C = pd.DataFrame([climate])
    dist = checker.climate_distance(C).iloc[0]
    oor = checker.climate_out_of_range(C).iloc[0]
    d = float(dist["climate_distance"])
    if d < U.TRAINING_DISTANCE_TOL:
        level = "training"
    elif d <= checker.d_max and oor == "":
        level = "new"
    else:
        level = "extrapolation"
    return level, d, str(dist["nearest_climate"]), [c for c in oor.split(",") if c]


def _check_versions(recorded: dict) -> None:
    """Pickled models depend on library versions: warn (not fail) if they differ from the build."""
    if not recorded:
        return
    now = environment()
    diff = {k: (recorded.get(k), now.get(k)) for k in ("numpy", "scikit-learn", "xgboost", "lightgbm", "torch")
            if recorded.get(k) and now.get(k) and str(now[k]).split("+")[0] != str(recorded[k]).split("+")[0]}
    if diff:
        warnings.warn(f"library versions differ from the registry build {diff}: predictions may change",
                      RuntimeWarning)


# ════════════════════════════════════════════════════════════════════════════════════════════
# Build (registry.json + slim GPR)
# ════════════════════════════════════════════════════════════════════════════════════════════

def out_dir(dev: bool) -> Path:
    """Where build writes the slim GPR + registry.json (official: outputs/models)."""
    d = DEV_DIR / "models" if dev else MODELS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(dev: bool) -> Path:
    d = DEV_DIR / "registry" if dev else OUT / "registry"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_models_manifest() -> dict:
    """The 3.6 manifest (outputs/models/manifest.json), dataset hash checked."""
    m = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    if m["dataset_sha256_training"] != CONFIG["dataset_sha256"]["training"]:
        raise RegistryError("models/manifest.json: fitted on another dataset")
    return m


def load_full_objects(manifest: dict) -> dict:
    """The four 3.6 objects from outputs/models (SHA-256 checked) - build only."""
    from surrogate.mlp import _torch
    _torch()
    objs = {}
    for name, rec in manifest["objects"].items():
        p = MODELS_DIR / rec["file"]
        if sha256(p) != rec["sha256"]:
            raise RegistryError(f"{p.name}: SHA-256 differs from models/manifest.json")
        with open(p, "rb") as fh:
            objs[name] = pickle.load(fh)
    return objs


def lhs_designs(n: int, seed: int) -> pd.DataFrame:
    """n random designs inside design_space.yaml (scipy LHS; n_floors = floor of uniform [1, 6))."""
    from scipy.stats import qmc
    b = U.design_bounds()
    u = qmc.LatinHypercube(d=len(FE.DESIGN_INPUTS), seed=seed).random(n)
    D = pd.DataFrame({c: b.loc[c, "min"] + u[:, k] * (b.loc[c, "max"] - b.loc[c, "min"])
                      for k, c in enumerate(FE.DESIGN_INPUTS)})
    D["n_floors"] = np.floor(1 + u[:, FE.DESIGN_INPUTS.index("n_floors")] * 5).clip(1, 5)
    return D


def training_climate_table() -> tuple[pd.DataFrame, dict]:
    """10 x 14 climate table exactly as used in training (inputs from the training rows, indices from
    the EPWs) + display names from the climate registry."""
    from climate.climate_features import load_climate_registry, training_keys
    from surrogate import cv
    df, _ = cv.load_training()
    g = df.groupby("climate_key")[FE.CLIMATE_INPUTS]
    if (g.nunique() > 1).any().any():
        raise AssertionError("climate inputs vary within a climate")
    tab = g.first().join(CI.training_indices(), how="inner")[CLIMATE_COLUMNS]
    reg = load_climate_registry(check_files=False)
    keys = training_keys(reg)
    if sorted(keys) != sorted(tab.index):
        raise AssertionError("training climates differ between dataset and climate registry")
    names = {k: reg[k].city for k in keys}
    return tab.loc[sorted(tab.index)], names


def slim_equality(full, slim, matrices: list[pd.DataFrame]) -> dict:
    out = {"rows": 0, "max_rel_diff": 0.0, "identical": True}
    for M in matrices:
        a, b = np.asarray(full.predict(M), float), np.asarray(slim.predict(M), float)
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
        out["rows"] += len(M)
        out["max_rel_diff"] = max(out["max_rel_diff"], float(rel.max()) if len(rel) else 0.0)
        out["identical"] &= bool(np.array_equal(a, b))
    out["pass"] = out["max_rel_diff"] <= SLIM_TOL
    return out


def _matrices_for(D: pd.DataFrame, climate: dict, feature_sets: dict) -> pd.DataFrame:
    raw = D.reset_index(drop=True).copy()
    for c in FE.CLIMATE_INPUTS:
        raw[c] = climate[c]
    cg = FE.engineer(raw)[feature_sets["C-G"]]
    idx = pd.DataFrame({c: np.full(len(raw), climate[c]) for c in CI.INDEX_COLUMNS})
    return pd.concat([cg, idx], axis=1)


def cv_summary() -> dict:
    """Grouped-CV reference of the deployed picks (3.7 cv_reference.csv, primary rows) - report only."""
    p = OUT / "final_test" / "cv_reference.csv"
    if not p.exists():
        return {}
    ref = pd.read_csv(p)
    ref = ref[ref["variant"] == "primary"].set_index("target")
    return {t: {k: float(ref.at[t, k]) for k in ("design_skill", "r2", "mae", "rmse")} for t in ref.index}


def holdout_summary() -> dict:
    """3.7 gate results (final_test_result.json, timing_result.json) - report only, never a choice (I1)."""
    out = {}
    p = OUT / "final_test" / "final_test_result.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        out["final_test"] = {"finished_utc": r.get("finished_utc"), "git_commit": r.get("git", {}).get("commit"),
                             "gate": {k: v.get("status") for k, v in r.get("gate", {}).items()
                                      if isinstance(v, dict) and "status" in v},
                             "primary_s3": {t: {"design_skill": v["design_skill"], "r2": v["r2"]}
                                            for t, v in r["gate"]["S3"]["targets"].items()}
                             if "S3" in r.get("gate", {}) else {}}
    p = OUT / "timing" / "timing_result.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        out["timing_3_7"] = {"single_median_ms": r["variants"]["primary"]["single"]["median_ms"],
                             "batch_designs_per_s": r["s8"]["batch_designs_per_s"], "s8": r["s8"]["status"]}
    return out


def interval_quantiles(iv: dict) -> dict:
    return {t: {typ: {U.level_tag(l): float(iv[t][typ].quantile(l)) for l in LEVELS} for typ in iv[t]}
            for t in iv}


def check_quantiles_against_selection(q: dict) -> float:
    """Max |dq| vs uncertainty/selection.json final_quantiles (3.5) - must be <= 1e-9 (3.7 I2)."""
    sel = json.loads((OUT / "uncertainty" / "selection.json").read_text(encoding="utf-8"))["selection"]
    worst = 0.0
    for t, by in q.items():
        for typ, levels in by.items():
            ref = sel[t]["final_quantiles"][typ]
            for tag, v in levels.items():
                worst = max(worst, abs(float(ref[tag]) - v))
    return worst


def build(dev: bool, log=print, target_dir: Path | None = None) -> dict:
    """Slim GPR + registry.json (J2, J6). Reads the training rows only (climate table, equality rows)."""
    t0 = time.perf_counter()
    d = Path(target_dir) if target_dir is not None else out_dir(dev)
    d.mkdir(parents=True, exist_ok=True)
    man = load_models_manifest()
    objs = load_full_objects(man)
    target_object = dict(man["target_object"])
    if set(target_object) != set(TARGETS):
        raise RegistryError("models/manifest.json does not cover the 10 targets")
    feature_sets = {"C-G": list(FE.feature_set("C-G")),
                    "C-G+I": list(FE.feature_set("C-G")) + list(CI.INDEX_COLUMNS)}
    tab, names = training_climate_table()

    # J2 - slim GPR, equality on the configured rows
    slim = slim_gpr(objs["gpr_heating"])
    from surrogate import cv
    df, _ = cv.load_training()
    M_train = pd.concat([_matrices_for(g[FE.DESIGN_INPUTS], tab.loc[k].to_dict(), feature_sets)
                         for k, g in df.groupby("climate_key", sort=True)], ignore_index=True)
    lhs = lhs_designs(500, int(_ST["gpr_slim"]["lhs_seed"]))
    M_lhs = pd.concat([_matrices_for(g, tab.loc[k].to_dict(), feature_sets)
                       for k, g in zip(tab.index, [lhs.iloc[i::len(tab)] for i in range(len(tab))])],
                      ignore_index=True)
    centre = tab.mean().to_dict()                                    # a site that is none of the 10
    M_new = _matrices_for(lhs, centre, feature_sets)
    eq = slim_equality(objs["gpr_heating"], slim, [M_train, M_lhs, M_new])
    log(f"  slim GPR vs full: {eq['rows']:,} rows, max rel. diff {eq['max_rel_diff']:.1e}, "
        f"bit-identical {eq['identical']}")
    if not eq["pass"]:
        raise RegistryError(f"slim GPR differs from the full GPR by {eq['max_rel_diff']:.2e} > {SLIM_TOL}")
    slim_path = d / SLIM_FILE
    with open(slim_path, "wb") as fh:
        pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # intervals (3.7) - hash and quantiles
    ivman = json.loads((MODELS_DIR / "intervals_manifest.json").read_text(encoding="utf-8"))
    iv_path = MODELS_DIR / "intervals.pkl"
    if sha256(iv_path) != ivman["sha256"]:
        raise RegistryError("intervals.pkl: SHA-256 differs from intervals_manifest.json")
    with open(iv_path, "rb") as fh:
        iv = pickle.load(fh)
    q = interval_quantiles(iv)
    dq = check_quantiles_against_selection(q)
    if dq > 1e-9:
        raise RegistryError(f"interval quantiles differ from uncertainty/selection.json by {dq:.2e}")

    # domain checker (3.5 G6/G7) with the training table; D_max stored explicitly (J4 fix)
    dom = U.DomainChecker.from_training()
    if abs(dom.d_max - float(_DOM["d_max"])) > float(_DOM["d_max_tol"]):
        raise RegistryError(f"D_max {dom.d_max:.4f} != config {_DOM['d_max']}")
    if not np.allclose(dom.climates.loc[tab.index, CLIMATE_COLUMNS].to_numpy(float), tab.to_numpy(float),
                       rtol=0, atol=0):
        raise RegistryError("domain checker climate table differs from the model-input table")

    files = {"gpr_heating": {"path": _rel(slim_path), "sha256": sha256(slim_path),
                             "bytes": slim_path.stat().st_size, "slim_of": man["objects"]["gpr_heating"]["file"]}}
    for name in ("xgb_eui", "mlp_cg", "mlp_physics"):
        p = MODELS_DIR / man["objects"][name]["file"]
        files[name] = {"path": _rel(p), "sha256": man["objects"][name]["sha256"], "bytes": p.stat().st_size}
    files["intervals"] = {"path": _rel(iv_path), "sha256": ivman["sha256"], "bytes": iv_path.stat().st_size}

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "official": not dev, "dev": dev,
        "plan": "claude/Phase3.8_Execution_Plan.md (J1-J7, agreed 18 Nov 2026)",
        "config_block": "registry_3_8", "config_md5": md5(CONFIG_PATH),
        "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
        "deployed_variant": CFG["deployed"]["variant"],
        "objects": list(objs), "target_object": target_object, "picks": man["picks"],
        "units": UNITS, "files": files,
        "archive": {"gpr_heating_full": {"file": man["objects"]["gpr_heating"]["file"],
                                         "sha256": man["objects"]["gpr_heating"]["sha256"],
                                         "note": "3.6 artefact with the Cholesky factor; not loaded"}},
        "slim_gpr_check": eq, "feature_sets": feature_sets, "predict_chunk_rows": CHUNK,
        "intervals": {"method": "scaled split-conformal (3.5 G2), both types", "levels": list(LEVELS),
                      "default_level": DEFAULT_LEVEL, "type_by_climate_level": TYPE_BY_LEVEL,
                      "quantiles": q, "max_abs_diff_vs_selection_json": dq},
        "climates": {"table": tab.reset_index().to_dict(orient="records"), "display_names": names},
        "domain": {"bounds": dom.bounds.reset_index().to_dict(orient="records"), "d_max": float(dom.d_max),
                   "holdout_keys": list(dom.holdout_keys), "levels": list(CLIMATE_LEVELS),
                   "training_distance_tol": U.TRAINING_DISTANCE_TOL, "strict_default": bool(_DOM["strict_default"])},
        "readable_names": READABLE,
        "messages": {k: _MSG[k] for k in ("training", "new", "extrapolation", "heating_note", "design_refused")},
        "messages_version": int(_MSG["version"]),
        "cv_reference": cv_summary(), "results_3_7_report_only": holdout_summary(),
        "runtime_modules": RUNTIME_MODULES, "environment": environment(), "git": git_state(),
        "build_s": None,
    }
    info["build_s"] = time.perf_counter() - t0
    (d / REGISTRY_FILE).write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    log(f"  wrote {slim_path.name} ({files['gpr_heating']['bytes'] / 1e6:.2f} MB) and {REGISTRY_FILE} in "
        f"{info['build_s']:.0f} s; interval quantiles = selection.json (max |dq| {dq:.1e})")
    return info


# ════════════════════════════════════════════════════════════════════════════════════════════
# Check (acceptance) and report
# ════════════════════════════════════════════════════════════════════════════════════════════

def timing_settings(dev: bool) -> dict:
    t = CONFIG["final_test_3_7"]["timing"]
    s = {"seed": int(t["seed"]), "warmup_calls": int(t["warmup_calls"]), "single_calls": int(t["single_calls"]),
         "batch_sizes": [int(v) for v in t["batch_sizes"]], "batch_repeats": int(t["batch_repeats"])}
    if dev:
        s.update({"single_calls": 40, "batch_sizes": [2000, 10000], "batch_repeats": 2})
    return s


def design_pool() -> pd.DataFrame:
    from surrogate import cv
    df, _ = cv.load_training()
    return df[["climate_key", *FE.DESIGN_INPUTS]].reset_index(drop=True)


def _time_single(fn, pool, sites, s) -> dict:
    rng = np.random.default_rng(s["seed"])
    order = rng.permutation(len(pool))
    for i in range(s["warmup_calls"]):
        r = pool.iloc[[order[i]]]
        fn(r[FE.DESIGN_INPUTS], sites[r["climate_key"].iat[0]])
    ms = []
    for k in range(s["single_calls"]):
        r = pool.iloc[[order[(s["warmup_calls"] + k) % len(pool)]]]
        site = sites[r["climate_key"].iat[0]]
        D = r[FE.DESIGN_INPUTS]
        t0 = time.perf_counter()
        fn(D, site)
        ms.append(1e3 * (time.perf_counter() - t0))
    ms = np.asarray(ms)
    return {"calls": len(ms), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "max_ms": float(ms.max())}


def _time_batch(fn, pool, sites, size, s) -> dict:
    """Same rows as the 3.7 batch (seed + size); one call per site (one site per call, J6)."""
    rng = np.random.default_rng(s["seed"] + size)
    batch = pool.iloc[rng.integers(0, len(pool), size)].reset_index(drop=True)
    groups = [(sites[k], g[FE.DESIGN_INPUTS]) for k, g in batch.groupby("climate_key", sort=True)]
    fn(groups[0][1].iloc[:100], groups[0][0])                      # warm-up
    secs = []
    for _ in range(s["batch_repeats"]):
        t0 = time.perf_counter()
        n = 0
        for site, D in groups:
            out = fn(D, site)
            n += len(out.values) if isinstance(out, Prediction) else len(out)
        secs.append(time.perf_counter() - t0)
        if n != size:
            raise AssertionError(f"batch of {size} returned {n} rows")
    med = float(np.median(secs))
    return {"designs": size, "calls": len(groups), "repeats": s["batch_repeats"], "median_s": med,
            "designs_per_s": size / med, "all_s": [float(x) for x in secs]}


def check(dev: bool, log=print) -> dict:
    """Acceptance (plan §3): load through registry.json, self-checks, timing through the public API."""
    s = timing_settings(dev)
    t0 = time.perf_counter()
    sur = Surrogate.load(out_dir(dev))
    load_s = time.perf_counter() - t0
    log(f"  load: {load_s:.2f} s (all hashes verified)")
    sites = {k: sur.site(k) for k in sur.training_climates}
    pool = design_pool()
    vals = lambda D, site: sur.predict_values(D, site)                       # noqa: E731
    full = lambda D, site: sur.predict(D, site)                              # noqa: E731
    # timing first, straight after loading - the 3.7 timing order (single, then batches, then with ranges).
    # Found 21 Nov: a one-design values timing taken just after a call WITH ranges (LightGBM difficulty
    # models) ran ~2x slower on the Mac, while batches and with-range calls were unaffected; the
    # "after a ranges call" row below records that effect (report only, not the acceptance number).
    tm = {"values": {"single": _time_single(vals, pool, sites, s),
                     "batch": [_time_batch(vals, pool, sites, n, s) for n in s["batch_sizes"]]},
          "with_intervals": {"single": _time_single(full, pool, sites, s),
                             "batch": [_time_batch(full, pool, sites, s["batch_sizes"][0], s)]}}
    sur.predict(pool[FE.DESIGN_INPUTS].iloc[:200], sites[sorted(sites)[0]])
    tm["values_after_ranges_call"] = {"single": _time_single(vals, pool, sites, s)}
    # self-checks (after timing, so they cannot change the timed state)
    q = interval_quantiles(sur.intervals)
    dq = check_quantiles_against_selection(q)
    lvl = {k: sur.climate_check(sites[k])["level"] for k in sites}
    probe = sur.predict(pool[FE.DESIGN_INPUTS].iloc[:200], sites[sorted(sites)[0]])
    self_checks = {"quantiles_max_abs_diff": dq, "training_levels": lvl,
                   "all_training_level": all(v == "training" for v in lvl.values()),
                   "probe_finite": bool(np.isfinite(probe.values.to_numpy()).all()),
                   "probe_lower_le_value_le_upper": bool(((probe.lower <= probe.values + 1e-9)
                                                          & (probe.values <= probe.upper + 1e-9)).all().all())}
    ok_self = dq <= 1e-9 and self_checks["all_training_level"] and self_checks["probe_finite"] \
        and self_checks["probe_lower_le_value_le_upper"]
    gate_b = tm["values"]["batch"][-1]
    single = tm["values"]["single"]["median_ms"]
    ok_t = single < float(_ACC["single_max_ms"]) and gate_b["designs_per_s"] >= float(_ACC["batch_min_per_s"])
    res = {"official": not dev, "dev": dev, "created_utc": datetime.now(timezone.utc).isoformat(),
           "config_md5": md5(CONFIG_PATH), "git": git_state(), "environment": environment(),
           "machine": {"platform": platform.platform(), "processor": platform.processor()},
           "load_s": load_s, "self_checks": self_checks, "timing_settings": s, "timing": tm,
           "acceptance": {"self_checks": "PASS" if ok_self else "FAIL",
                          "speed": "PASS" if ok_t else "FAIL",
                          "single_median_ms": single, "single_max_ms": float(_ACC["single_max_ms"]),
                          "batch_designs": gate_b["designs"], "batch_designs_per_s": gate_b["designs_per_s"],
                          "batch_min_per_s": float(_ACC["batch_min_per_s"])}}
    d = report_dir(dev)
    (d / "registry_check.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log(f"  self-checks {res['acceptance']['self_checks']} | speed {res['acceptance']['speed']}: one design "
        f"{single:.1f} ms (< {_ACC['single_max_ms']}), {gate_b['designs']:,} designs "
        f"{gate_b['designs_per_s']:,.0f}/s (>= {_ACC['batch_min_per_s']:,}); with ranges "
        f"{tm['with_intervals']['single']['median_ms']:.1f} ms / "
        f"{tm['with_intervals']['batch'][0]['designs_per_s']:,.0f}/s; values after a ranges call "
        f"{tm['values_after_ranges_call']['single']['median_ms']:.1f} ms (report only)")
    write_report(dev)
    return res


def write_report(dev: bool) -> Path:
    d = report_dir(dev)
    reg = json.loads((out_dir(dev) / REGISTRY_FILE).read_text(encoding="utf-8"))
    p = d / "registry_check.json"
    chk = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    L = [f"# Deployed surrogate (registry){' — DEV (code check only)' if dev else ''}", "",
         f"Built {reg['created_utc']} · config MD5 {reg['config_md5']} · git {reg['git'].get('commit')} · "
         f"variant **{reg['deployed_variant']}**", "", "## What serves each output", "",
         "| Output | Object | Model / features / form | File | SHA-256 |", "|---|---|---|---|---|"]
    for t in TARGETS:
        o = reg["target_object"][t]
        pk = reg["picks"][t]
        f = reg["files"][o]
        L.append(f"| {t} | {o} | {pk['model']} / {pk['feature_set']} / {pk['form']} | `{f['path']}` "
                 f"({f['bytes'] / 1e6:.2f} MB) | {f['sha256'][:12]}… |")
    f = reg["files"]["intervals"]
    L += ["", f"Intervals: `{f['path']}` ({f['bytes'] / 1e6:.1f} MB, {f['sha256'][:12]}…) — "
          f"{reg['intervals']['method']}; quantiles = 3.5 selection.json (max |Δq| "
          f"{reg['intervals']['max_abs_diff_vs_selection_json']:.1e}).", "",
          f"Slim heating GPR: {reg['slim_gpr_check']['rows']:,} rows compared with the full 3.6 file, max relative "
          f"difference {reg['slim_gpr_check']['max_rel_diff']:.1e}, bit-identical "
          f"{reg['slim_gpr_check']['identical']}. Full file kept as archive "
          f"({reg['archive']['gpr_heating_full']['sha256'][:12]}…).", "",
          f"Climate levels: training / new (distance ≤ {reg['domain']['d_max']:.3f}, all values in range) / "
          f"extrapolation. Design check strict by default.", ""]
    if chk:
        a = chk["acceptance"]
        tm = chk["timing"]
        L += ["## Acceptance", "", f"Load {chk['load_s']:.2f} s (all hashes verified) · self-checks **{a['self_checks']}**"
              f" · speed **{a['speed']}**", "",
              "| Path | One design (median) | p95 | Batch | Designs/s |", "|---|---|---|---|---|"]
        for lab, key in (("values only", "values"), ("with ranges + flags", "with_intervals")):
            sg = tm[key]["single"]
            for b in tm[key]["batch"]:
                L.append(f"| {lab} | {sg['median_ms']:.1f} ms | {sg['p95_ms']:.1f} ms | {b['designs']:,} "
                         f"({b['calls']} calls) | {b['designs_per_s']:,.0f} |")
        if "values_after_ranges_call" in tm:
            sg = tm["values_after_ranges_call"]["single"]
            L.append(f"| values only, just after one call with ranges (report only) | {sg['median_ms']:.1f} ms | "
                     f"{sg['p95_ms']:.1f} ms | - | - |")
        L += ["", f"Limits (same as the 3.7 speed check): one design < {a['single_max_ms']:.0f} ms; "
              f"{a['batch_designs']:,} designs ≥ {a['batch_min_per_s']:,.0f} designs/s.", ""]
        r37 = reg.get("results_3_7_report_only", {}).get("timing_3_7")
        if r37:
            L.append(f"3.7 path for comparison: {r37['single_median_ms']:.1f} ms, {r37['batch_designs_per_s']:,.0f} "
                     "designs/s (mixed climates in one call).")
    out = d / "registry_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["build", "check", "report"])
    ap.add_argument("--dev", action="store_true", help="write to outputs/registry_dev (never a result)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    print(f"registry.py --stage {a.stage}{' --dev' if a.dev else ''}")
    if a.stage == "build":
        build(a.dev)
    elif a.stage == "check":
        check(a.dev)
    else:
        print(f"  wrote {write_report(a.dev)}")
    print(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
