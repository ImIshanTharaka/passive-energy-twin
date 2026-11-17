"""
final_test.py - Sub-task 3.7: the final holdout test (gates S3, S4, S7) - scored ONCE.

Plan: claude/Phase3.7_Execution_Plan.md (decisions I1-I6, agreed 17 Nov 2026). Every setting is read
from config.yaml block `final_test_3_7` (appended 17 Nov 2026, before any 3.7 run).

In plain words
    1. intervals  Fit the interval "difficulty" models for all 10 targets from the training
                  out-of-fold rows only (I2) and save them next to the final models. They must
                  reproduce the 3.5 quantiles exactly. Never reads the holdout.
    2. score      Load the 3.6 final models (SHA-256 checked, no refit), predict the 400 holdout rows
                  once - 300 designs in the 10 training climates + 100 designs in London - for the
                  primary picks and the all-MLP alternative (I3), and evaluate:
                      S3  in-distribution holdout vs grouped CV (primary targets, FAIL)
                      S4  London: bias, MAPE, ranking (WARN, report)
                      S7  90 % interval coverage on the 300 in-distribution rows (WARN)
    3. figures    Parity plots, CV vs holdout, coverage (reads the saved tables only).
    4. report     Gate table S0-S8 and the Phase 4 go/no-go (reads final_test + timing results).

    ./venv/bin/python surrogate/final_test.py --stage intervals [--dev]
    ./venv/bin/python surrogate/final_test.py --stage score --dev        # training rows pose as holdout
    ./venv/bin/python surrogate/final_test.py --stage score --confirm    # THE run: once only
    ./venv/bin/python surrogate/final_test.py --stage figures | report [--dev]

Write-once (I1): `score --confirm` refuses to start unless the git tree is clean under surrogate/ and
simulation/ and outputs/final_test/ holds neither RUN_STARTED nor the result file. RUN_STARTED is
written BEFORE the holdout is read and no code ever removes it: if a run crashes after that point,
the holdout has been seen, and any re-run is a deliberate, documented decision (move the folder to
_archive/ by hand and record why in the plan). Result files are made read-only.

Dev (`--dev`): 30 training designs per climate pose as the in-distribution holdout and the training
climate San Francisco poses as London (its indices go through the EPW path a new site takes). The
models were fitted on these rows, so the numbers are optimistic and are never results; output goes
to outputs/final_test_dev/ (overwritten freely).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import climate_indices as CI
from surrogate import explain as X
from surrogate import features as FE
from surrogate import metrics as M
from surrogate import uncertainty as U
from surrogate.config import ALL_TARGETS, CONFIG, CONFIG_PATH, PRIMARY_TARGETS, SECONDARY_TARGETS
from surrogate.paths import OUT, PROJECT_ROOT

# ── Settings: config.yaml block final_test_3_7 (frozen 17 Nov 2026, before any 3.7 run) ─────────

CFG = CONFIG["final_test_3_7"]
GATE = CONFIG["gate"]
_W, _H, _I, _V = CFG["write_once"], CFG["holdout"], CFG["intervals"], CFG["variants"]
_S3, _S4, _S7, _DEV = CFG["s3"], CFG["s4"], CFG["s7"], CFG["dev"]

HEAT, COOL, EUI = X.HEAT, X.COOL, X.EUI
HEATING_TARGETS = tuple(CONFIG["targets"]["heating"])
LEVELS = U.LEVELS
LEVEL = float(_S7["level"])
TAG = U.level_tag(LEVEL)
BAND = tuple(float(v) for v in _S7["coverage_band"])
LONDON = _H["london"]["climate_key"]
LOCK, RESULT = _W["lock_file"], _W["result_file"]
VARIANTS = ("primary", "all_mlp")
ALT_FORMS = {t: (v["object"], v["form"]) for t, v in _V["all_mlp"].items() if t != "others"}
Q_TOL = 1e-9
EPS = 1e-12                                                 # float tolerance on the inclusive limits
SHORT_T = U.SHORT_T


class FinalTestRefused(RuntimeError):
    """The write-once rules (I1) do not allow this run."""


# ── paths, hashes, bookkeeping ──────────────────────────────────────────────────────────────────

def out_dir(dev: bool) -> Path:
    return OUT / ("final_test_dev" if dev else "final_test")


def intervals_dir(dev: bool) -> Path:
    return X.models_dir(dev)


def sha256(path: Path) -> str:
    return X.sha256(path)


def md5_text(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def git_state(paths=tuple(_W["clean_tree_paths"])) -> dict:
    """Commit and modified tracked files under `paths` (None values if git is unavailable)."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    try:
        commit = run("rev-parse", "HEAD").strip()
        dirty = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no", "--", *paths).splitlines()
                 if ln.strip()]
        return {"commit": commit, "modified_tracked": dirty}
    except (OSError, subprocess.CalledProcessError) as e:
        return {"commit": None, "modified_tracked": None, "error": str(e)}


def check_write_once(d: Path, git: dict) -> None:
    """I1 preconditions for the official run (pure apart from the folder listing)."""
    if (d / RESULT).exists():
        raise FinalTestRefused(f"{d / RESULT} exists: the holdout has been scored. It is scored once only.")
    if (d / LOCK).exists():
        raise FinalTestRefused(f"{d / LOCK} exists: an earlier run started (the holdout may have been read). "
                               "A re-run is a deliberate decision: move the folder to _archive/ by hand and "
                               "record why in claude/Phase3.7_Execution_Plan.md.")
    if git.get("commit") is None:
        raise FinalTestRefused(f"git state unknown ({git.get('error')}): the run must be tied to a commit")
    if git["modified_tracked"]:
        raise FinalTestRefused("commit first - modified tracked files: " + ", ".join(git["modified_tracked"][:8]))


def write_lock(d: Path, info: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)


def make_read_only(d: Path) -> None:
    for p in d.iterdir():
        if p.is_file():
            p.chmod(0o444)


def environment() -> dict:
    from surrogate import benchmark as B
    return B.environment()


# ── 1. interval models (I2) ─────────────────────────────────────────────────────────────────────

def fit_intervals(log=print) -> dict:
    """{target: {"standard": Conformal, "new_site": Conformal}} from the training OOF rows (same calls
    as uncertainty.final_quantiles). Stops if any quantile differs from selection.json."""
    sel = json.loads((OUT / "uncertainty" / "selection.json").read_text(encoding="utf-8"))["selection"]
    df, Xm, heat = U.load(dev=False)                      # full training rows (the calibration set)
    out = {}
    for t in ALL_TARGETS:
        method = sel[t]["deployed_method"]
        if method != "lgbm":
            raise NotImplementedError(f"{t}: deployed method {method!r} (3.7 expects scaled conformal)")
        out[t] = {}
        for typ, (scheme, inner) in _I["calibration"].items():
            f = M.scored_rows(U.pick_oof(t, scheme, dev=False), t, heat)
            conf = U.Conformal(scale="lgbm").fit(Xm.loc[f["run_id"]], f["yhat"], f["y"], groups=f[inner].to_numpy())
            ref = sel[t]["final_quantiles"][typ]
            for lev in LEVELS:
                q, qr = conf.quantile(lev), float(ref[U.level_tag(lev)])
                if abs(q - qr) > Q_TOL:
                    raise AssertionError(f"{t} {typ} q{U.level_tag(lev)} = {q} differs from selection.json {qr}")
            if conf.n_ != int(ref["n"]):
                raise AssertionError(f"{t} {typ}: {conf.n_} calibration rows, selection.json {ref['n']}")
            out[t][typ] = conf
        log(f"  {SHORT_T[t]:10s} standard q90 {out[t]['standard'].quantile(LEVEL):.4f} | "
            f"new-site q90 {out[t]['new_site'].quantile(LEVEL):.4f}  (= selection.json)")
    return out


def stage_intervals(dev: bool, log=print) -> dict:
    t0 = time.perf_counter()
    iv = fit_intervals(log)
    d = intervals_dir(dev)
    path = d / _I["store"]["file"]
    with open(path, "wb") as fh:
        pickle.dump(iv, fh, protocol=pickle.HIGHEST_PROTOCOL)
    back = load_intervals(dev, check_hash=False, objects_only=True)
    probe = U.load(dev=False)[1].iloc[:50]
    for t in ALL_TARGETS:                                  # round trip: identical widths
        for typ in iv[t]:
            a = iv[t][typ].width_scale(probe, np.ones(len(probe)))
            b = back[t][typ].width_scale(probe, np.ones(len(probe)))
            if not np.array_equal(a, b):
                raise AssertionError(f"saved -> loaded interval model differs ({t} {typ})")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "official": not dev,
                "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
                "config_block": "final_test_3_7.intervals", "file": path.name, "sha256": sha256(path),
                "bytes": path.stat().st_size, "fit_s": round(time.perf_counter() - t0, 1),
                "quantiles": {t: {typ: {U.level_tag(v): iv[t][typ].quantile(v) for v in LEVELS} | {"n": iv[t][typ].n_}
                                  for typ in iv[t]} for t in ALL_TARGETS},
                "reproduces": "outputs/uncertainty/selection.json final_quantiles (|dq| <= 1e-9)",
                "environment": environment()}
    (d / _I["store"]["manifest"]).write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    log(f"  saved {path.relative_to(OUT.parent)} ({manifest['bytes'] / 1e6:.1f} MB, sha256 {manifest['sha256'][:12]}…)"
        f" in {manifest['fit_s']:.0f} s")
    return manifest


def load_intervals(dev: bool, check_hash: bool = True, objects_only: bool = False):
    d = intervals_dir(dev)
    path = d / _I["store"]["file"]
    if not path.exists():
        raise SystemExit(f"no interval models in {d}: run `final_test.py --stage intervals{' --dev' if dev else ''}`")
    manifest = None
    if check_hash:
        manifest = json.loads((d / _I["store"]["manifest"]).read_text(encoding="utf-8"))
        if sha256(path) != manifest["sha256"]:
            raise RuntimeError(f"{path.name}: SHA-256 differs from {_I['store']['manifest']}")
    with open(path, "rb") as fh:
        iv = pickle.load(fh)
    return iv if objects_only else (iv, manifest)


# ── model inputs for any climate (London included) ──────────────────────────────────────────────

def index_table_with(extra: dict[str, dict] | None = None) -> pd.DataFrame:
    """Training index table plus extra sites {key: 7 indices} (London in the official run)."""
    tab = X.index_table().copy()
    for k, v in (extra or {}).items():
        if k in tab.index:
            raise ValueError(f"{k!r} is already a training climate")
        tab.loc[k] = [float(v[c]) for c in CI.INDEX_COLUMNS]
    return tab[CI.INDEX_COLUMNS]


def matrices_with(raw: pd.DataFrame, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """As explain.matrices, but the climate indices come from `table` (so a new site works)."""
    raw = raw.reset_index(drop=True)
    cg = FE.engineer(raw)[FE.feature_set("C-G")]
    keys = raw["climate_key"].astype(str).to_numpy()
    missing = sorted(set(keys) - set(table.index))
    if missing:
        raise KeyError(f"no climate indices for {missing}")
    idx = table.loc[keys, CI.INDEX_COLUMNS].reset_index(drop=True)
    return {"C-G": cg, "C-G+I": pd.concat([cg, idx], axis=1)}


def _primary_chunk(fm: X.FinalModels, Xc: dict) -> pd.DataFrame:
    names = list(dict.fromkeys(X.TARGET_OBJECT.values()))
    return pd.concat([fm._predict_chunk(n, Xc) for n in names], axis=1)[ALL_TARGETS]


def _all_mlp_chunk(fm: X.FinalModels, Xc: dict) -> pd.DataFrame:
    """The all-MLP alternative (I3): only the two MLP objects are evaluated."""
    if any(obj != "mlp_physics" for obj, _ in ALT_FORMS.values()):
        raise ValueError("all_mlp variant: only mlp_physics forms are defined")
    forms = fm.objects["mlp_physics"].predict_forms(Xc["C-G+I"])
    parts = {}
    for t in ALL_TARGETS:
        if t in ALT_FORMS:
            parts[t] = forms[(t, ALT_FORMS[t][1])].to_numpy()
        elif X.TARGET_OBJECT[t] == "mlp_physics":
            parts[t] = forms[(t, "direct")].to_numpy()
    cg = fm._predict_chunk("mlp_cg", Xc)
    for t in cg.columns:
        parts[t] = cg[t].to_numpy()
    missing = set(ALL_TARGETS) - set(parts)
    if missing:
        raise AssertionError(f"all_mlp variant misses {sorted(missing)}")
    return pd.DataFrame(parts)[ALL_TARGETS]


VARIANT_CHUNK = {"primary": _primary_chunk, "all_mlp": _all_mlp_chunk}


def predict_variant(fm: X.FinalModels, raw: pd.DataFrame, table: pd.DataFrame, variant: str) -> pd.DataFrame:
    """10 targets of one variant for raw rows (climate_key + 28 inputs), original units, chunked."""
    f = VARIANT_CHUNK[variant]
    parts = [f(fm, matrices_with(raw.iloc[i:i + X.CHUNK], table)) for i in range(0, len(raw), X.CHUNK)]
    out = pd.concat(parts, ignore_index=True)
    out.index = raw.index
    return out


def predict_variants(fm: X.FinalModels, raw: pd.DataFrame, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{"primary": 10 targets, "all_mlp": 10 targets}."""
    return {v: predict_variant(fm, raw, table, v) for v in VARIANTS}


# ── reference: grouped CV of the same models (I4) ──────────────────────────────────────────────

def cv_reference(heat: list[str]) -> dict[str, pd.DataFrame]:
    """{variant: summary per target} from the grouped-CV OOF rows (same metric code as the holdout)."""
    prim = pd.concat([U.pick_oof(t, "grouped", dev=False).assign(target=t) for t in ALL_TARGETS], ignore_index=True)
    alt = [prim[~prim["target"].isin(ALT_FORMS)]]
    for t in ALT_FORMS:
        g = U._grid_rows("grouped", t, {"model": "mlp", "feature_set": "C-G+I", "form": ALT_FORMS[t][1]})
        if len(g) != 3000:
            raise AssertionError(f"all_mlp CV reference {t}: {len(g)} rows")
        alt.append(g[["run_id", "design_id", "climate_key", "fold", "y", "yhat"]].assign(target=t))
    alt = pd.concat(alt, ignore_index=True)
    return {"primary": M.summarise(prim, heat)[0], "all_mlp": M.summarise(alt, heat)[0]}


# ── holdout data ────────────────────────────────────────────────────────────────────────────────

def load_holdout() -> tuple[pd.DataFrame, str]:
    """The 400 holdout rows via the verified Phase 2 loader. Called only by the confirmed official run."""
    from eda import eda_lib as L
    df, _, sha = L.load_split("holdout")
    if sha != CONFIG["dataset_sha256"]["holdout"]:
        raise RuntimeError(f"holdout sha256 {sha[:12]}… differs from config.yaml")
    return df, sha


def check_holdout_shape(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(in-distribution 300 rows, London 100 rows) with the counts the protocol fixes."""
    ind = df[df["design_set"] == _H["in_distribution"]["design_set"]]
    lon = df[df["design_set"] == _H["london"]["design_set"]]
    if len(ind) != _H["in_distribution"]["rows"] or len(lon) != _H["london"]["rows"] or len(ind) + len(lon) != len(df):
        raise AssertionError(f"holdout rows: {len(ind)} in-distribution, {len(lon)} London, {len(df)} total")
    per = ind.groupby("climate_key").size()
    if (per != _H["in_distribution"]["per_climate"]).any() or set(per.index) != set(X.climate_table().index):
        raise AssertionError(f"in-distribution rows per climate: {per.to_dict()}")
    if set(lon["climate_key"]) != {LONDON}:
        raise AssertionError(f"London rows have climates {sorted(set(lon['climate_key']))}")
    if ind["design_id"].duplicated().any() or lon["design_id"].duplicated().any():
        raise AssertionError("duplicate design in a holdout part")
    return ind.reset_index(drop=True), lon.reset_index(drop=True)


def check_training_climate_inputs(ind: pd.DataFrame) -> float:
    """The in-distribution rows carry exactly the training climates' 7 climate inputs."""
    tab = X.climate_table().loc[ind["climate_key"].to_numpy(), FE.CLIMATE_INPUTS].to_numpy(float)
    gap = float(np.max(np.abs(ind[FE.CLIMATE_INPUTS].to_numpy(float) - tab)))
    if gap > 1e-9:
        raise AssertionError(f"in-distribution climate inputs differ from the training climates by {gap}")
    return gap


def site_climate(rows: pd.DataFrame, key: str) -> pd.Series:
    """The 7 climate inputs of one site (constant within the site)."""
    g = rows[rows["climate_key"] == key][FE.CLIMATE_INPUTS]
    if (g.nunique() > 1).any():
        raise AssertionError(f"{key}: climate inputs vary within the site")
    return g.iloc[0].astype(float)


def dev_holdout(log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Training rows posing as the holdout (code check). Returns (ind, new_site, indices, site_key)."""
    from surrogate import cv
    from climate.climate_features import load_climate_registry
    from climate.epw_parser import parse_epw
    from surrogate.paths import EPW_DIR
    df, _ = cv.load_training()
    rng = np.random.default_rng(int(_DEV["seed"]))
    site = _DEV["new_site_climate"]
    parts = []
    for c, g in df.groupby("climate_key"):
        pick = rng.choice(np.sort(g["design_id"].astype(str).unique()), int(_DEV["in_distribution_per_climate"]),
                          replace=False)
        parts.append(g[g["design_id"].astype(str).isin(pick)])
    ind = pd.concat(parts, ignore_index=True)
    g = df[df["climate_key"] == site]
    lon = g.loc[np.sort(rng.choice(g.index.to_numpy(), _H["london"]["rows"], replace=False))].reset_index(drop=True)
    registry = load_climate_registry(check_files=False)
    indices = CI.indices_from_epw(parse_epw(EPW_DIR / registry[site].epw_filename))   # the new-site path
    ref = X.index_table().loc[site, CI.INDEX_COLUMNS]
    if not np.allclose([indices[c] for c in CI.INDEX_COLUMNS], ref.to_numpy(float), rtol=0, atol=1e-9):
        raise AssertionError("EPW path indices differ from the training table")
    log(f"  DEV: {len(ind)} training rows pose as the in-distribution holdout; {len(lon)} {site} rows pose as London")
    return ind, lon, indices, site


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────

def long_frame(rows: pd.DataFrame, pred: pd.DataFrame, targets=ALL_TARGETS) -> pd.DataFrame:
    parts = []
    for t in targets:
        parts.append(pd.DataFrame({"run_id": rows["run_id"].to_numpy(), "design_id": rows["design_id"].to_numpy(),
                                   "climate_key": rows["climate_key"].to_numpy(), "target": t,
                                   "y": rows[t].to_numpy(float), "yhat": pred[t].to_numpy(float)}))
    return pd.concat(parts, ignore_index=True)


def add_intervals(lf: pd.DataFrame, Xm: pd.DataFrame, iv: dict, typ: str) -> pd.DataFrame:
    """lo_/hi_ per level for each row of a long frame (Xm: C-G+I rows in the order of `rows`)."""
    out = []
    for t, g in lf.groupby("target", sort=False):
        conf = iv[t][typ]
        g = g.copy()
        if len(g) != len(Xm) or not np.array_equal(g["run_id"].to_numpy(), lf["run_id"].iloc[:len(Xm)].to_numpy()):
            raise AssertionError(f"{t}: interval rows not aligned with the model inputs")
        Xt = Xm.reset_index(drop=True)                     # long_frame keeps the row order within each target
        yh = g["yhat"].to_numpy(float)
        w = conf.width_scale(Xt, yh)
        g["width_scale"] = w
        for lev in LEVELS:
            lo, hi = conf.interval(Xt, yh, level=lev)
            tag = U.level_tag(lev)
            g[f"lo_{tag}"], g[f"hi_{tag}"] = lo, hi
        g["interval"] = typ
        out.append(g)
    return pd.concat(out, ignore_index=True)


def london_metrics(lf: pd.DataFrame) -> pd.DataFrame:
    """One row per target for one site (I5)."""
    rows = []
    for t, g in lf.groupby("target", sort=False):
        y, yh = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
        reg = M.regression_metrics(y, yh)
        corrected = yh - yh.mean() + y.mean()
        from eda import eda_lib as L
        rows.append({"target": t, "n": len(g), "mean_y": float(y.mean()), "median_y": float(np.median(y)),
                     "bias_pct": M.bias_pct(y, yh), "mape_pct": M.mape_pct(y, yh), "mae": reg["mae"],
                     "rmse": reg["rmse"], "r2": reg["r2"], "r2_bias_removed": L.r2(y, corrected),
                     "spearman": M.spearman(y, yh), "top_hit": M.top_hit_rate(y, yh)})
    return pd.DataFrame(rows)


def coverage(iv_frame: pd.DataFrame, heat: list[str], by_climate: bool = False) -> pd.DataFrame:
    f = iv_frame[~iv_frame["target"].isin(HEATING_TARGETS) | iv_frame["climate_key"].isin(heat)]
    by = ["target", "climate_key"] if by_climate else ["target"]
    return U.coverage_table(f, by=by)


# ── gate rules (pure; tested on hand-made tables) ───────────────────────────────────────────────

def evaluate_s3(hold: pd.DataFrame, cv: pd.DataFrame) -> dict:
    """hold, cv: summaries (target, r2, design_skill). FAIL if any primary target misses a rule."""
    h, c = hold.set_index("target"), cv.set_index("target")
    rows, fail = {}, False
    for t in ALL_TARGETS:
        skill, r2, r2_cv = float(h.loc[t, "design_skill"]), float(h.loc[t, "r2"]), float(c.loc[t, "r2"])
        if t in PRIMARY_TARGETS:
            ok = skill >= float(_S3["design_skill_min"]) and (r2_cv - r2) <= float(_S3["pooled_r2_max_drop"]) + EPS
            fail |= not ok
            status = "PASS" if ok else "FAIL"
        else:
            status = "ok" if skill >= float(_S3["secondary_report"]["design_skill_min"]) else "flag"
        rows[t] = {"design_skill": skill, "design_skill_cv": float(c.loc[t, "design_skill"]), "r2": r2,
                   "r2_cv": r2_cv, "r2_drop": r2_cv - r2, "status": status}
    return {"status": "FAIL" if fail else "PASS", "severity": GATE["S3"]["severity"], "targets": rows}


def evaluate_s4(lon: pd.DataFrame) -> dict:
    """lon: london_metrics of the primary variant. WARN if a MAPE or Spearman rule is missed."""
    m = lon.set_index("target")
    rows, warn = {}, False
    for t in PRIMARY_TARGETS:
        mape, rho = float(m.loc[t, "mape_pct"]), float(m.loc[t, "spearman"])
        lim = float(_S4["mape_max_pct"][t])
        ok_m, ok_r = mape <= lim, rho >= float(_S4["spearman_min"])
        warn |= not (ok_m and ok_r)
        rows[t] = {"mape_pct": mape, "mape_max_pct": lim, "spearman": rho, "bias_pct": float(m.loc[t, "bias_pct"]),
                   "status": "PASS" if ok_m and ok_r else "WARN"}
    return {"status": "WARN" if warn else "PASS", "severity": GATE["S4"]["severity"], "targets": rows}


def evaluate_s7(cov: pd.DataFrame) -> dict:
    """cov: coverage per target (standard intervals, in-distribution, scored rows)."""
    c = cov.set_index("target")
    rows, warn = {}, False
    for t in ALL_TARGETS:
        v = float(c.loc[t, f"coverage_{TAG}"])
        ok = BAND[0] <= v <= BAND[1]
        warn |= not ok
        rows[t] = {"coverage": v, "n": int(c.loc[t, "n"]), "mean_width": float(c.loc[t, f"mean_width_{TAG}"]),
                   "status": "PASS" if ok else "WARN"}
    return {"status": "WARN" if warn else "PASS", "severity": GATE["S7"]["severity"], "targets": rows}


def phase4_decision(statuses: dict[str, str]) -> str:
    """NO-GO on any FAIL, REVIEW on any WARN, else GO (master plan §6). Pending checks block a decision."""
    vals = [str(v).upper() for v in statuses.values()]
    if any(v.startswith("PENDING") for v in vals):
        return "PENDING"
    if any(v.startswith("FAIL") for v in vals):
        return "NO-GO"
    if any(v.startswith("WARN") for v in vals):
        return "REVIEW"
    return "GO"


# ── stage score ─────────────────────────────────────────────────────────────────────────────────

def stage_score(dev: bool, confirm: bool, log=print) -> dict:
    if not dev and not confirm:
        raise FinalTestRefused("the official run needs --confirm (it reads the holdout once); use --dev to check code")
    if dev and confirm:
        raise FinalTestRefused("--dev and --confirm together make no sense")
    d = out_dir(dev)
    git = git_state()
    if not dev:
        check_write_once(d, git)
    d.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # everything that does not need the holdout first (a crash here costs nothing)
    log("1. models, interval models, CV reference (no holdout)")
    fm = X.final_models(dev=False)                        # SHA-256 checked against manifest.json
    iv, iv_manifest = load_intervals(dev)
    _, _, heat = U.load(dev=False)
    heat = list(heat)
    cv_ref = cv_reference(heat)
    for t in PRIMARY_TARGETS:
        r = cv_ref["primary"].set_index("target").loc[t]
        log(f"   CV reference {SHORT_T[t]:8s} design skill {r['design_skill']:.3f}, pooled R² {r['r2']:.3f}")
    dom = U.DomainChecker.from_training()
    info = {"started_utc": datetime.now(timezone.utc).isoformat(), "git": git,
            "config_md5": md5_text(CONFIG_PATH), "dataset_sha256_training": CONFIG["dataset_sha256"]["training"],
            "models": {k: v["sha256"] for k, v in fm.manifest["objects"].items()},
            "intervals_sha256": iv_manifest["sha256"], "environment": environment()}

    log("2. holdout rows")
    if dev:
        ind, lon, site_idx, site = dev_holdout(log)
        hold_sha = None
    else:
        site = LONDON
        site_idx = CI.indices_for_key(LONDON, allow_holdout=True)       # computed only here
        write_lock(d, info)                                             # the holdout is read after this line
        log(f"   {LOCK} written - reading the holdout (once)")
        hold, hold_sha = load_holdout()
        ind, lon = check_holdout_shape(hold)
    info["dataset_sha256_holdout"] = hold_sha
    check_training_climate_inputs(ind)
    table = index_table_with({} if dev else {LONDON: site_idx})
    lon_raw = lon[["climate_key", *FE.RAW_INPUTS]]
    ind_raw = ind[["climate_key", *FE.RAW_INPUTS]]

    log(f"3. predict ({len(ind)} in-distribution + {len(lon)} {'London' if not dev else site + ' (as London)'})")
    t0 = time.perf_counter()
    p_ind = predict_variants(fm, ind_raw, table)
    p_lon = predict_variants(fm, lon_raw, table)
    predict_s = time.perf_counter() - t0
    X_ind = matrices_with(ind_raw, table)["C-G+I"]
    X_lon = matrices_with(lon_raw, table)["C-G+I"]

    log("4. score")
    site_clim = {**site_climate(lon, lon["climate_key"].iloc[0]).to_dict(), **site_idx}
    dom_lon = dom.check(lon[FE.DESIGN_INPUTS], climate=site_clim, climate_key=None if dev else LONDON,
                        allow_holdout=not dev)
    dom_ind = dom.check(ind[FE.DESIGN_INPUTS + FE.CLIMATE_INPUTS])
    summaries, per_climate, lon_tables, frames = {}, {}, {}, []
    for v in VARIANTS:
        lf_i = long_frame(ind, p_ind[v]).assign(variant=v, part="in_distribution")
        lf_l = long_frame(lon, p_lon[v]).assign(variant=v, part="london")
        s, pc = M.summarise(lf_i, heat)
        summaries[v], per_climate[v] = s, pc.assign(variant=v)
        lon_tables[v] = london_metrics(lf_l).assign(variant=v)
        frames += [lf_i, lf_l]
    preds = pd.concat(frames, ignore_index=True)

    lf_i = long_frame(ind, p_ind["primary"])
    lf_l = long_frame(lon, p_lon["primary"])
    iv_ind = add_intervals(lf_i, X_ind, iv, "standard").assign(part="in_distribution")
    iv_lon = pd.concat([add_intervals(lf_l, X_lon, iv, "new_site"),
                        add_intervals(lf_l, X_lon, iv, "standard")], ignore_index=True).assign(part="london")
    cov_ind = coverage(iv_ind, heat)
    cov_ind_pc = coverage(iv_ind, heat, by_climate=True)
    cov_lon = pd.concat([U.coverage_table(g, by=["target"]).assign(interval=typ)
                         for typ, g in iv_lon.groupby("interval")], ignore_index=True)

    s3 = evaluate_s3(summaries["primary"], cv_ref["primary"])
    s4 = evaluate_s4(lon_tables["primary"])
    s7 = evaluate_s7(cov_ind)
    alt_s3 = evaluate_s3(summaries["all_mlp"], cv_ref["all_mlp"])        # reported only (I3)

    log("5. write")
    tables = {
        "holdout_summary.csv": pd.concat([summaries[v].assign(variant=v) for v in VARIANTS], ignore_index=True),
        "cv_reference.csv": pd.concat([cv_ref[v].assign(variant=v) for v in VARIANTS], ignore_index=True),
        "holdout_per_climate.csv": pd.concat(per_climate.values(), ignore_index=True),
        "london_metrics.csv": pd.concat(lon_tables.values(), ignore_index=True),
        "coverage_in_distribution.csv": cov_ind,
        "coverage_in_distribution_per_climate.csv": cov_ind_pc,
        "coverage_london.csv": cov_lon,
        "domain_check.csv": pd.concat([dom_ind.assign(part="in_distribution", run_id=ind["run_id"].to_numpy()),
                                       dom_lon.assign(part="london", run_id=lon["run_id"].to_numpy())],
                                      ignore_index=True),
    }
    for name, tab in tables.items():
        tab.to_csv(d / name, index=False)
    preds.to_csv(d / "predictions.csv.gz", index=False)
    pd.concat([iv_ind, iv_lon], ignore_index=True).to_csv(d / "intervals.csv.gz", index=False)

    lon_heat = lon_tables["primary"].set_index("target").loc[HEAT]
    result = {**info, "official": not dev, "dev": dev, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "runtime_s": round(time.perf_counter() - t_start, 1), "predict_s": round(predict_s, 2),
              "heating_climates": heat, "new_site": site, "new_site_indices": site_idx,
              "new_site_domain": {"climate_level": str(dom_lon["climate_level"].iloc[0]),
                                  "climate_distance": float(dom_lon["climate_distance"].iloc[0]),
                                  "nearest_climate": str(dom_lon["nearest_climate"].iloc[0]),
                                  "climate_out_of_range": str(dom_lon["climate_out_of_range"].iloc[0])},
              "designs_outside_bounds": int((~pd.concat([dom_ind, dom_lon])["design_in_domain"]).sum()),
              "new_site_median_heating": float(lon_heat["median_y"]),
              "gate": {"S3": s3, "S4": s4, "S7": s7},
              "all_mlp_reported_only": {"s3_rules": alt_s3,
                                        "london": lon_tables["all_mlp"].set_index("target")
                                        .loc[list(ALT_FORMS)].to_dict(orient="index")}}
    (d / RESULT).write_text(json.dumps(result, indent=2, default=X._json_default), encoding="utf-8")
    write_report(d, dev)
    if not dev:
        make_read_only(d)
    log(f"   S3 {s3['status']} | S4 {s4['status']} | S7 {s7['status']}   ({result['runtime_s']:.0f} s)")
    return result


# ── report ──────────────────────────────────────────────────────────────────────────────────────

def _fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def gate_statuses(dev: bool) -> dict[str, str]:
    st = {k: str(v) for k, v in CFG["prior_gate"].items()}
    res_path = out_dir(dev) / RESULT
    if res_path.exists():
        g = json.loads(res_path.read_text(encoding="utf-8"))["gate"]
        st.update({k: g[k]["status"] for k in ("S3", "S4", "S7")})
    else:
        st.update({k: "PENDING" for k in ("S3", "S4", "S7")})
    tpath = OUT / ("timing_dev" if dev else "timing") / "timing_result.json"
    st["S8"] = json.loads(tpath.read_text(encoding="utf-8"))["s8"]["status"] if tpath.exists() else "PENDING"
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    return {k: st[k] for k in order}


def write_report(d: Path, dev: bool) -> Path:
    r = json.loads((d / RESULT).read_text(encoding="utf-8"))
    hs = pd.read_csv(d / "holdout_summary.csv")
    cvr = pd.read_csv(d / "cv_reference.csv")
    lm = pd.read_csv(d / "london_metrics.csv")
    cl = pd.read_csv(d / "coverage_london.csv")
    L = [f"# Final holdout test (Sub-task 3.7){' — DEV (training rows posing as the holdout; not a result)' if dev else ''}",
         "", f"Run {r['finished_utc']} · commit `{(r['git'] or {}).get('commit')}` · config MD5 `{r['config_md5']}` · "
         f"holdout sha256 `{r.get('dataset_sha256_holdout')}` · runtime {r['runtime_s']} s", "",
         "All errors in original units. Heating targets scored over the heating climates only.", "",
         "## In-distribution holdout (300 designs, 10 training climates) vs grouped CV", "",
         "| Variant | Target | Holdout design skill | CV design skill | Holdout pooled R² | CV pooled R² | Drop | MAE | RMSE | Status |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, key in (("primary", "S3"), ("all_mlp", None)):
        rules = r["gate"]["S3"] if key else r["all_mlp_reported_only"]["s3_rules"]
        for t in ALL_TARGETS:
            h = hs[(hs["variant"] == v) & (hs["target"] == t)].iloc[0]
            c = cvr[(cvr["variant"] == v) & (cvr["target"] == t)].iloc[0]
            s = rules["targets"][t]["status"] + ("" if key else " (report)")
            L.append(f"| {v} | {SHORT_T[t]} | {h['design_skill']:.3f} | {c['design_skill']:.3f} | {h['r2']:.4f} | "
                     f"{c['r2']:.4f} | {c['r2'] - h['r2']:+.4f} | {h['mae']:.3f} | {h['rmse']:.3f} | {s} |")
    L += ["", f"**S3 {r['gate']['S3']['status']}** (primary targets: holdout design skill ≥ {_S3['design_skill_min']} "
          f"and pooled R² ≤ {_S3['pooled_r2_max_drop']} below CV).", "",
          f"## {'London' if not dev else r['new_site'] + ' (posing as London)'} (100 designs, unseen climate)", "",
          f"Climate level `{r['new_site_domain']['climate_level']}`, distance {r['new_site_domain']['climate_distance']:.3f} "
          f"(nearest {r['new_site_domain']['nearest_climate']}); median heating {r['new_site_median_heating']:.2f} kWh/m².", "",
          "| Variant | Target | Bias % | MAPE % | MAE | RMSE | R² | R² bias removed | Spearman | Top-10 % hit |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in lm.iterrows():
        L.append(f"| {x['variant']} | {SHORT_T[x['target']]} | {x['bias_pct']:+.1f} | {x['mape_pct']:.1f} | {x['mae']:.3f} | "
                 f"{x['rmse']:.3f} | {x['r2']:.3f} | {x['r2_bias_removed']:.3f} | {x['spearman']:.3f} | {x['top_hit']:.2f} |")
    L += ["", f"**S4 {r['gate']['S4']['status']}** (MAPE ≤ 10 % cooling/EUI, ≤ 25 % heating; Spearman ≥ 0.90; primary variant).",
          "", "## 90 % intervals", "", "| Target | In-distribution coverage (standard) | Mean width | "
          "London coverage (new-site) | London coverage (standard, for comparison) | Status |", "|---|---|---|---|---|---|"]
    for t in ALL_TARGETS:
        s7 = r["gate"]["S7"]["targets"][t]
        ln = cl[(cl["target"] == t) & (cl["interval"] == "new_site")].iloc[0]
        ls = cl[(cl["target"] == t) & (cl["interval"] == "standard")].iloc[0]
        L.append(f"| {SHORT_T[t]} | {s7['coverage']:.3f} (n {s7['n']}) | {s7['mean_width']:.3f} | "
                 f"{ln[f'coverage_{TAG}']:.3f} | {ls[f'coverage_{TAG}']:.3f} | {s7['status']} |")
    L += ["", f"**S7 {r['gate']['S7']['status']}** (in-distribution coverage {BAND[0]}–{BAND[1]}, all 10 targets).", "",
          "## Gate and Phase 4 decision", "", "| Check | Status |", "|---|---|"]
    st = gate_statuses(dev)
    L += [f"| {k} | {v} |" for k, v in st.items()]
    L += ["", f"**Phase 4: {phase4_decision(st)}** (NO-GO on any FAIL, REVIEW on any WARN, else GO).", ""]
    path = d / "final_test_report.md"
    if path.exists() and not os.access(path, os.W_OK):
        path = d / "final_test_report_gate.md"            # the official report is read-only; gate update beside it
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def stage_report(dev: bool, log=print):
    d = out_dir(dev)
    if not (d / RESULT).exists():
        raise SystemExit(f"no {RESULT} in {d}")
    p = write_report(d, dev)
    st = gate_statuses(dev)
    log("  " + " | ".join(f"{k} {v}" for k, v in st.items()))
    log(f"  Phase 4: {phase4_decision(st)}  -> {p.relative_to(OUT.parent)}")


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sub-task 3.7 final holdout test")
    ap.add_argument("--stage", required=True, choices=["intervals", "score", "figures", "report"])
    ap.add_argument("--dev", action="store_true", help="code check: training rows pose as the holdout")
    ap.add_argument("--confirm", action="store_true", help="official scoring of the holdout (once only)")
    a = ap.parse_args(argv)
    if a.confirm and a.stage != "score":
        ap.error("--confirm applies to --stage score only")
    print(f"final_test.py --stage {a.stage}{' --dev' if a.dev else ''}{' --confirm' if a.confirm else ''}")
    t0 = time.time()
    try:
        if a.stage == "intervals":
            stage_intervals(a.dev)
        elif a.stage == "score":
            stage_score(a.dev, a.confirm)
        elif a.stage == "figures":
            from surrogate import final_figures
            final_figures.make_all(out_dir(a.dev), a.dev)
        else:
            stage_report(a.dev)
    except FinalTestRefused as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
