#!/usr/bin/env python3
"""
assemble.py - Build the surrogate-model dataset from the 2.6 campaign (Sub-task 2.7).

Joins, for every simulated run,
    sampling/samples_<main|holdout>.csv      what the building was   (21 design inputs)
    climate/climate_metadata.csv             where it was simulated  (7 continuous climate features)
    outputs/<main|holdout>/results.csv       how it performed        (10 TARGETS)
on run_id / climate_key, validates the result, and writes

    outputs/dataset/dataset_training.csv     3,000 rows  (batch main: 10 training climates)
    outputs/dataset/dataset_holdout.csv        400 rows  (holdout_id 300 + holdout_ood London 100)
    outputs/dataset/dataset_diagnostics.csv  3,400 rows  (non-target results + simulation QA, per run_id)
    outputs/dataset/dataset_manifest.json    column roles, units, sources + sha256, output sha256, checks
    outputs/dataset/data_quality_report.md   checks, counts, per-climate target summary, gate

Column order of both dataset files (fixed; recorded in the manifest):
    id       run_id, batch, design_set, design_id, climate_key
    meta     ashrae_zone                      (NOT a model input - Decision D2)
    design   samples_manifest.json -> parameters.model_inputs (21; glazing_vt excluded, S2)
    climate  CONTINUOUS_FEATURES (7)
    target   output_parser.TARGETS keys (10)

Decisions (2.7 plan, claude/Phase2.7_Execution_Plan.md):
    DS1  nothing is dropped or imputed: any missing, failed or non-finite run is a FAIL
    DS2  no feature engineering here - the dataset is the raw record; Phase 3 owns transforms
    DS3  diagnostics live in a separate file so they can never be mistaken for inputs
    DS4  outliers (> 4 sigma from the climate mean) are flagged, never removed
    DS5  group column for cross-validation = design_id (anchors repeat across climates)
    DS6  values are written exactly as read (no rounding); CSVs are byte-deterministic

USAGE (from the repository root)
    python dataset/assemble.py            # check, and write outputs/dataset/ unless the gate is NO-GO
    python dataset/assemble.py --check    # checks + report to stdout only, write nothing
    python dataset/assemble.py --verify   # re-hash written files against dataset_manifest.json

Phase 3 should load data with load_dataset() - it verifies the file hash and column order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from climate.climate_features import (CONTINUOUS_FEATURES, holdout_keys,  # noqa: E402
                                      load_climate_registry, training_keys)
from runner.output_parser import COLUMNS as RESULT_COLUMNS                 # noqa: E402
from runner.output_parser import TARGETS                                   # noqa: E402

SCHEMA_VERSION = 1
SAMPLING = ROOT / "sampling"
OUTPUTS = ROOT / "outputs"
DESIGN_SPACE = ROOT / "config" / "design_space.yaml"
SAMPLES_MANIFEST = SAMPLING / "samples_manifest.json"
METADATA_CSV = ROOT / "climate" / "climate_metadata.csv"
ACCEPTED = OUTPUTS / "accepted_exceptions.csv"
OUT_DIR = OUTPUTS / "dataset"
MANIFEST_NAME = "dataset_manifest.json"
REPORT_NAME = "data_quality_report.md"
DIAGNOSTICS_NAME = "dataset_diagnostics.csv"

# split -> (samples file, batch output folder, dataset file)
SPLITS: dict[str, tuple[str, str, str]] = {
    "training": ("samples_main.csv", "main", "dataset_training.csv"),
    "holdout": ("samples_holdout.csv", "holdout", "dataset_holdout.csv"),
}
# batch values allowed in each split, and which climate role each batch may use
SPLIT_BATCHES = {"training": {"main": "training"},
                 "holdout": {"holdout_id": "training", "holdout_ood": "holdout"}}

ID_COLUMNS = ["run_id", "batch", "design_set", "design_id", "climate_key"]
META_COLUMNS = ["ashrae_zone"]
GROUP_COLUMN = "design_id"
TARGET_NAMES = list(TARGETS)
TARGET_SOURCES = list(TARGETS.values())

CLIMATE_UNITS = {"hdd18": "°C·day (base 18 °C)", "cdd10": "°C·day (base 10 °C)", "annual_ghi": "kWh/m²·yr",
                 "mean_dbt": "°C", "mean_rh": "%", "latitude": "° N", "mean_wind": "m/s"}
TARGET_UNITS = {t: ("W/m²" if t.startswith("peak_") else "kWh/m²·yr") for t in TARGET_NAMES}

HOT_HUMID = ("chennai", "colombo", "miami")       # heating expected ~ 0
HOT = HOT_HUMID + ("phoenix",)                    # where the cooling maximum may lie
COLDEST = "minneapolis"                           # highest heating expected
HEATING_NEAR_ZERO = 1.0                           # kWh/m2, climate MEDIAN
OUTLIER_SIGMA = 4.0
EUI_LITERATURE = (50.0, 350.0)                    # kWh/m2 office range (2.7 plan; INFO only)
AREA_TOL = 0.001                                  # cond_area vs floor_area * n_floors
SUMMARY_TARGETS = ["heating_demand_kwh_m2", "cooling_demand_kwh_m2", "eui_kwh_m2",
                   "peak_heating_w_m2", "peak_cooling_w_m2"]

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


class AssemblyError(Exception):
    pass


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SplitSource:
    name: str
    samples_file: str
    batch_dir: str
    samples: pd.DataFrame
    results: pd.DataFrame
    run_log: pd.DataFrame
    err_summary: pd.DataFrame
    run_meta: list = field(default_factory=list)
    samples_sha: str = ""
    expected_samples_sha: str | None = None
    results_fresh: bool = True


@dataclass
class Sources:
    splits: dict[str, SplitSource]
    climate: pd.DataFrame                          # key, role, ashrae_zone, 7 features (from the registry)
    inputs: list[str]                              # the 21 design inputs, manifest order
    bounds: dict[str, tuple[float, float]]
    input_units: dict[str, str]
    integer_inputs: set[str]
    accepted: dict[str, str]                       # run_id -> category
    climate_sha: str = ""
    expected_climate_sha: str | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def training_climates(self) -> list[str]:
        return sorted(self.climate.loc[self.climate.role == "training", "key"])

    @property
    def holdout_climates(self) -> list[str]:
        return sorted(self.climate.loc[self.climate.role == "holdout", "key"])


def climate_frame(registry) -> pd.DataFrame:
    rows = [{"key": k, "role": r.role, "ashrae_zone": r.features.ashrae_zone,
             **{n: getattr(r.features, n) for n in CONTINUOUS_FEATURES}} for k, r in sorted(registry.items())]
    return pd.DataFrame(rows)


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def load_sources(root: Path = ROOT) -> Sources:
    """Read every input file. Pure I/O - all judgement is in the checks."""
    root = Path(root)
    smanifest = json.loads((root / "sampling" / "samples_manifest.json").read_text())
    inputs = list(smanifest["parameters"]["model_inputs"])
    space = yaml.safe_load((root / "config" / "design_space.yaml").read_text())["continuous_parameters"]
    missing = [n for n in inputs if n not in space]
    if missing:
        raise AssemblyError(f"model inputs not in design_space.yaml: {missing}")
    bounds = {n: (float(space[n]["min"]), float(space[n]["max"])) for n in inputs}
    units = {n: str(space[n].get("unit", "")) for n in inputs}
    ints = {n for n in inputs if space[n].get("type") == "integer"}

    meta_csv = root / "climate" / "climate_metadata.csv"
    registry = load_climate_registry(meta_csv, root / "climate" / "epw_files")
    hashes = {}

    splits = {}
    for name, (sfile, bdir, _) in SPLITS.items():
        spath, bpath = root / "sampling" / sfile, root / "outputs" / bdir
        rpath, lpath = bpath / "results.csv", bpath / "run_log.csv"
        epath, mpath = bpath / "err_summary.csv", bpath / "run_meta.json"
        for p in (spath, rpath, lpath, epath):
            if not p.exists():
                raise AssemblyError(f"missing input: {p}")
            hashes[_rel(p, root)] = sha256_file(p)
        meta = json.loads(mpath.read_text()) if mpath.exists() else []
        splits[name] = SplitSource(
            name=name, samples_file=sfile, batch_dir=bdir,
            samples=pd.read_csv(spath, dtype={c: str for c in ID_COLUMNS}),
            results=pd.read_csv(rpath, dtype={"run_id": str}),
            run_log=pd.read_csv(lpath, dtype={"run_id": str}),
            err_summary=pd.read_csv(epath, dtype={"run_id": str}),
            run_meta=meta if isinstance(meta, list) else [meta],
            samples_sha=hashes[_rel(spath, root)],
            expected_samples_sha=smanifest.get("outputs", {}).get(sfile, {}).get("sha256"),
            results_fresh=rpath.stat().st_mtime >= lpath.stat().st_mtime,
        )

    accepted = {}
    apath = root / "outputs" / "accepted_exceptions.csv"
    if apath.exists():
        a = pd.read_csv(apath, dtype=str)
        accepted = dict(zip(a.run_id, a.category))
        hashes[_rel(apath, root)] = sha256_file(apath)
    for p in (meta_csv, root / "config" / "design_space.yaml", root / "sampling" / "samples_manifest.json"):
        hashes[_rel(p, root)] = sha256_file(p)

    return Sources(splits=splits, climate=climate_frame(registry), inputs=inputs, bounds=bounds,
                   input_units=units, integer_inputs=ints, accepted=accepted,
                   climate_sha=hashes[_rel(meta_csv, root)],
                   expected_climate_sha=smanifest.get("inputs_sha256", {}).get("climate/climate_metadata.csv"),
                   file_hashes=hashes)


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def column_order(inputs: list[str]) -> list[str]:
    return ID_COLUMNS + META_COLUMNS + list(inputs) + list(CONTINUOUS_FEATURES) + TARGET_NAMES


def join_split(samples: pd.DataFrame, results: pd.DataFrame, climate: pd.DataFrame,
               inputs: list[str]) -> pd.DataFrame:
    """samples (left) + climate (on climate_key) + targets (on run_id). Left joins, so a
    missing climate or result shows up as NaN for the checks instead of a silently lost row."""
    s = samples[ID_COLUMNS + list(inputs)].copy()
    c = climate[["key", "ashrae_zone", *CONTINUOUS_FEATURES]].rename(columns={"key": "climate_key"})
    r = results[["run_id", *TARGET_SOURCES]].drop_duplicates("run_id")
    r = r.rename(columns={v: k for k, v in TARGETS.items()})
    out = s.merge(c, on="climate_key", how="left", validate="many_to_one")
    out = out.merge(r, on="run_id", how="left", validate="one_to_one")
    return out[column_order(inputs)].reset_index(drop=True)


def outlier_flags(df: pd.DataFrame, sigma: float = OUTLIER_SIGMA) -> pd.Series:
    """Per row, the targets more than `sigma` standard deviations from their climate mean (DS4)."""
    flags = pd.Series([""] * len(df), index=df.index, dtype=object)
    for _, g in df.groupby("climate_key"):
        for t in TARGET_NAMES:
            x = g[t].astype(float)
            sd = x.std(ddof=1)
            if not np.isfinite(sd) or sd == 0 or len(x) < 3:
                continue
            hit = ((x - x.mean()).abs() / sd) > sigma
            for i in hit[hit].index:
                flags[i] = f"{flags[i]};{t}" if flags[i] else t
    return flags


def diagnostics_frame(split: SplitSource, data: pd.DataFrame, accepted: dict[str, str]) -> pd.DataFrame:
    """Everything about a run that is NOT a model column (DS3)."""
    keep = [c for c in RESULT_COLUMNS if c not in TARGET_SOURCES and c != "run_id"]
    r = split.results[["run_id", *[c for c in keep if c in split.results.columns]]]
    r = r.rename(columns={"n_floors": "sim_n_floors"})
    e = split.err_summary[["run_id", "clean", "reported_warnings", "reported_severe", "categories"]]
    e = e.rename(columns={"clean": "err_clean", "reported_warnings": "err_warnings",
                          "reported_severe": "err_severe", "categories": "err_categories"})
    d = data[["run_id", "batch", "climate_key"]].merge(r, on="run_id", how="left").merge(e, on="run_id", how="left")
    d["accepted_exception"] = d.run_id.map(accepted).fillna("")
    d["outlier_targets"] = outlier_flags(data).values
    return d


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str


def _fmt_list(xs, n=5) -> str:
    xs = list(xs)
    return ", ".join(map(str, xs[:n])) + (f" (+{len(xs) - n} more)" if len(xs) > n else "")


def check_provenance(src: Sources) -> Check:
    problems = []
    if src.expected_climate_sha and src.climate_sha != src.expected_climate_sha:
        problems.append("climate_metadata.csv differs from the version the samples were built with")
    for sp in src.splits.values():
        if sp.expected_samples_sha and sp.samples_sha != sp.expected_samples_sha:
            problems.append(f"{sp.samples_file} sha256 != samples_manifest")
        run_shas = {m.get("samples_sha256") for m in sp.run_meta}
        if not sp.run_meta:
            problems.append(f"outputs/{sp.batch_dir}/run_meta.json missing")
        elif run_shas != {sp.samples_sha}:
            problems.append(f"outputs/{sp.batch_dir} was run from a different {sp.samples_file}")
        if not sp.results_fresh:
            problems.append(f"outputs/{sp.batch_dir}/results.csv older than run_log.csv - re-run output_parser.py")
    if problems:
        return Check("P1", "provenance", FAIL, "; ".join(problems))
    return Check("P1", "provenance", PASS,
                 "samples sha256 = samples_manifest; every run_meta entry ran these samples; "
                 "climate_metadata unchanged since sampling; results.csv newer than run_log.csv")


def check_completeness(src: Sources) -> Check:
    problems, parts = [], []
    for sp in src.splits.values():
        ids = sp.samples.run_id
        sset = set(ids)
        if ids.duplicated().any():
            problems.append(f"{sp.name}: duplicate run_id in samples")
        log = sp.run_log
        if log.run_id.duplicated().any():
            problems.append(f"{sp.name}: duplicate run_id in run_log ({_fmt_list(log.run_id[log.run_id.duplicated()])})")
        not_ok = log.loc[log.run_id.isin(sset) & (log.status != "ok"), "run_id"]
        if len(not_ok):
            problems.append(f"{sp.name}: {len(not_ok)} runs not ok ({_fmt_list(not_ok)})")
        no_log = sset - set(log.run_id)
        if no_log:
            problems.append(f"{sp.name}: {len(no_log)} runs never logged ({_fmt_list(sorted(no_log))})")
        res = sp.results.run_id
        if res.duplicated().any():
            problems.append(f"{sp.name}: duplicate run_id in results ({_fmt_list(res[res.duplicated()])})")
        missing = sset - set(res)
        extra = set(res) - sset
        if missing:
            problems.append(f"{sp.name}: {len(missing)} runs without results ({_fmt_list(sorted(missing))})")
        if extra:
            problems.append(f"{sp.name}: {len(extra)} results not in samples ({_fmt_list(sorted(extra))})")
        parts.append(f"{sp.name} {len(sset)}/{len(sset)}")
    if problems:
        return Check("C1", "completeness", FAIL, "; ".join(problems))
    return Check("C1", "completeness", PASS,
                 "every sampled run logged ok once and parsed once; no extra results (" + ", ".join(parts) + ")")


def check_schema(src: Sources, data: dict[str, pd.DataFrame]) -> Check:
    want = column_order(src.inputs)
    problems = []
    if "glazing_vt" in want:
        problems.append("glazing_vt is a model column (S2)")
    leaked = set(want) & (set(RESULT_COLUMNS) - {"run_id", "n_floors"})
    if leaked:
        problems.append(f"results diagnostics among model columns: {sorted(leaked)}")
    for name, df in data.items():
        if list(df.columns) != want:
            problems.append(f"{name}: column order differs from the schema")
            continue
        if df.run_id.duplicated().any():
            problems.append(f"{name}: duplicate run_id")
        num = df[list(src.inputs) + list(CONTINUOUS_FEATURES) + TARGET_NAMES].apply(pd.to_numeric, errors="coerce")
        bad = ~np.isfinite(num.to_numpy(dtype=float))
        if bad.any():
            cols = sorted({num.columns[j] for j in np.where(bad)[1]})
            problems.append(f"{name}: {int(bad.sum())} missing/non-finite values in {_fmt_list(cols)}")
        if df[ID_COLUMNS + META_COLUMNS].isna().any().any():
            problems.append(f"{name}: missing id/meta values (unknown climate_key?)")
    if problems:
        return Check("S1", "schema", FAIL, "; ".join(problems))
    n = len(want)
    return Check("S1", "schema", PASS,
                 f"{n} columns = 5 id + 1 meta + {len(src.inputs)} design + {len(CONTINUOUS_FEATURES)} climate "
                 f"+ {len(TARGET_NAMES)} targets; no glazing_vt, no diagnostics; all values finite")


def check_split_isolation(src: Sources, data: dict[str, pd.DataFrame]) -> Check:
    problems, notes = [], []
    roles = dict(zip(src.climate.key, src.climate.role))
    for name, df in data.items():
        allowed = SPLIT_BATCHES[name]
        bad_batch = set(df.batch) - set(allowed)
        if bad_batch:
            problems.append(f"{name}: unexpected batch {sorted(bad_batch)}")
        for batch, role in allowed.items():
            ck = set(df.loc[df.batch == batch, "climate_key"])
            wrong = sorted(k for k in ck if roles.get(k) != role)
            if wrong:
                problems.append(f"{name}/{batch}: climates {wrong} are not '{role}' climates")
    if "training" in data and "holdout" in data:
        tr, ho = data["training"], data["holdout"]
        tr_designs = set(tr.design_id)
        hid = set(ho.loc[ho.batch == "holdout_id", "design_id"])
        shared = hid & tr_designs
        if shared:
            problems.append(f"holdout_id designs also in training: {_fmt_list(sorted(shared))}")
        if set(tr.run_id) & set(ho.run_id):
            problems.append("run_id in both files")
        ood = set(ho.loc[ho.batch == "holdout_ood", "design_id"])
        anchors = set(tr.loc[tr.design_set == "anchor", "design_id"])
        if ood and not ood <= anchors:
            problems.append("holdout_ood designs are not the training anchors (S4)")
        elif ood:
            notes.append(f"holdout_ood reuses the {len(ood)} anchors by design (S4) - unseen climate, seen designs")
        if set(tr.climate_key) & set(src.holdout_climates):
            problems.append("held-out climate rows in training")
    if problems:
        return Check("L1", "split isolation", FAIL, "; ".join(problems))
    return Check("L1", "split isolation", PASS,
                 "training = training climates only; holdout_id designs disjoint from training; "
                 "London only in holdout_ood. " + " ".join(notes))


def check_input_bounds(src: Sources, data: dict[str, pd.DataFrame]) -> Check:
    problems = []
    for name, df in data.items():
        for n in src.inputs:
            lo, hi = src.bounds[n]
            x = pd.to_numeric(df[n], errors="coerce")
            out = df.run_id[(x < lo) | (x > hi)]
            if len(out):
                problems.append(f"{name}.{n}: {len(out)} outside [{lo:g}, {hi:g}]")
            if n in src.integer_inputs and ((x % 1) != 0).any():
                problems.append(f"{name}.{n}: non-integer values")
    if problems:
        return Check("I1", "input bounds", FAIL, "; ".join(problems))
    return Check("I1", "input bounds", PASS,
                 f"all {len(src.inputs)} design inputs inside design_space.yaml bounds; integers integral")


def check_geometry(src: Sources, data: dict[str, pd.DataFrame]) -> Check:
    """The simulated building is the sampled building: n_floors, zones, conditioned area."""
    problems, worst = [], 0.0
    for name, df in data.items():
        r = src.splits[name].results.set_index("run_id")
        d = df.set_index("run_id")
        common = d.index.intersection(r.index)
        if len(common) == 0:
            continue
        nf_s, nf_r = d.loc[common, "n_floors"].astype(int), r.loc[common, "n_floors"].astype(int)
        if (nf_s != nf_r).any():
            problems.append(f"{name}: n_floors differs for {_fmt_list(common[nf_s != nf_r])}")
        if (r.loc[common, "n_zones"].astype(int) != 5 * nf_s).any():
            problems.append(f"{name}: n_zones != 5 x n_floors")
        expect = d.loc[common, "floor_area"].astype(float) * nf_s
        rel = (r.loc[common, "cond_area_m2"].astype(float) / expect - 1).abs()
        worst = max(worst, float(rel.max()))
        if (rel > AREA_TOL).any():
            problems.append(f"{name}: conditioned area != floor_area x n_floors for {_fmt_list(common[rel > AREA_TOL])}")
    if problems:
        return Check("G1", "geometry consistency", FAIL, "; ".join(problems))
    return Check("G1", "geometry consistency", PASS,
                 f"simulated n_floors = sampled; n_zones = 5 x n_floors; conditioned area = floor_area x n_floors "
                 f"(max deviation {worst * 100:.3f} %)")


def check_simulation_quality(src: Sources) -> Check:
    problems, accepted_used = [], []
    for sp in src.splits.values():
        e = sp.err_summary[sp.err_summary.run_id.isin(set(sp.samples.run_id))]
        missing = set(sp.samples.run_id) - set(e.run_id)
        if missing:
            problems.append(f"{sp.name}: {len(missing)} runs without err_summary")
        bad = e[(e.clean != 1) | (e.reported_severe > 0) | (e.get("terminated_fatal", 0) != 0)]
        for rid in bad.run_id:
            (accepted_used if rid in src.accepted else problems).append(rid)
    if problems:
        return Check("Q1", "simulation quality", FAIL,
                     f"runs with severe/non-benign messages not in accepted_exceptions.csv: {_fmt_list(problems)}")
    n = sum(len(sp.samples) for sp in src.splits.values())
    txt = f"{n - len(accepted_used)}/{n} runs clean (only benign W1/unused-construction warnings)"
    if accepted_used:
        txt += f"; {len(accepted_used)} reviewed exceptions kept (W2): {_fmt_list(accepted_used)}"
    return Check("Q1", "simulation quality", PASS, txt)


def check_targets_valid(data: dict[str, pd.DataFrame]) -> Check:
    problems = []
    for name, df in data.items():
        for t in TARGET_NAMES:
            x = pd.to_numeric(df[t], errors="coerce")
            if (x < 0).any():
                problems.append(f"{name}.{t}: {int((x < 0).sum())} negative")
    if problems:
        return Check("T1", "targets non-negative", FAIL, "; ".join(problems))
    return Check("T1", "targets non-negative", PASS, f"all {len(TARGET_NAMES)} targets >= 0 in every row")


def check_physics(data: dict[str, pd.DataFrame]) -> Check:
    """Climate-level direction checks on the training set (WARN, not FAIL: they test physics, not plumbing)."""
    df = data.get("training")
    if df is None or df.empty:
        return Check("T2", "climate physics", INFO, "no training data")
    med = df.groupby("climate_key")[["heating_demand_kwh_m2", "cooling_demand_kwh_m2"]].median()
    problems, facts = [], []
    hot = [k for k in HOT_HUMID if k in med.index]
    for k in hot:
        v = med.loc[k, "heating_demand_kwh_m2"]
        if v >= HEATING_NEAR_ZERO:
            problems.append(f"{k} median heating {v:.2f} kWh/m² (expected < {HEATING_NEAR_ZERO})")
    if hot:
        facts.append("median heating " + ", ".join(f"{k} {med.loc[k, 'heating_demand_kwh_m2']:.2f}" for k in hot))
    top_h = med.heating_demand_kwh_m2.idxmax()
    if COLDEST in med.index and top_h != COLDEST:
        problems.append(f"highest median heating is {top_h}, not {COLDEST}")
    facts.append(f"highest median heating {top_h} ({med.heating_demand_kwh_m2.max():.1f})")
    top_c = med.cooling_demand_kwh_m2.idxmax()
    if top_c not in HOT:
        problems.append(f"highest median cooling is {top_c}, not a hot climate")
    facts.append(f"highest median cooling {top_c} ({med.cooling_demand_kwh_m2.max():.1f})")
    if problems:
        return Check("T2", "climate physics", WARN, "; ".join(problems) + ". " + "; ".join(facts))
    return Check("T2", "climate physics", PASS, "; ".join(facts))


def check_eui_range(data: dict[str, pd.DataFrame]) -> Check:
    lo, hi = EUI_LITERATURE
    parts = []
    for name, df in data.items():
        x = df.eui_kwh_m2.astype(float)
        inside = ((x >= lo) & (x <= hi)).mean() * 100
        parts.append(f"{name}: {inside:.1f} % within {lo:g}-{hi:g} (range {x.min():.1f}-{x.max():.1f})")
    return Check("T3", "EUI vs literature", INFO,
                 "; ".join(parts) + ". Site EUI with ideal loads (no plant efficiency, Decision A) - "
                 "not directly comparable to metered office EUI")


def check_outliers(diag: pd.DataFrame) -> Check:
    flagged = diag[diag.outlier_targets != ""]
    if flagged.empty:
        return Check("T4", f"outliers > {OUTLIER_SIGMA:g} sigma", INFO, "none")
    counts = pd.Series([t for s in flagged.outlier_targets for t in s.split(";")]).value_counts()
    return Check("T4", f"outliers > {OUTLIER_SIGMA:g} sigma", INFO,
                 f"{len(flagged)} runs flagged, kept (DS4); by target: "
                 + ", ".join(f"{k} {v}" for k, v in counts.items()) + " - see dataset_diagnostics.csv")


def gate(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    return "NO-GO" if FAIL in statuses else "REVIEW" if WARN in statuses else "GO"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclass
class Build:
    data: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame
    checks: list[Check]
    gate: str


def build(src: Sources) -> Build:
    data = {name: join_split(sp.samples, sp.results, src.climate, src.inputs) for name, sp in src.splits.items()}
    diag = pd.concat([diagnostics_frame(src.splits[n], d, src.accepted) for n, d in data.items()], ignore_index=True)
    checks = [check_provenance(src), check_completeness(src), check_schema(src, data),
              check_split_isolation(src, data), check_input_bounds(src, data), check_geometry(src, data),
              check_simulation_quality(src), check_targets_valid(data), check_physics(data),
              check_eui_range(data), check_outliers(diag)]
    return Build(data=data, diagnostics=diag, checks=checks, gate=gate(checks))


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.write_text(df.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return sha256_file(path)


def roles(inputs: list[str]) -> dict:
    return {"id": ID_COLUMNS, "meta": META_COLUMNS, "design_inputs": list(inputs),
            "climate_inputs": list(CONTINUOUS_FEATURES), "inputs": list(inputs) + list(CONTINUOUS_FEATURES),
            "targets": TARGET_NAMES, "group": GROUP_COLUMN, "all": column_order(inputs)}


def manifest_dict(src: Sources, b: Build, outputs: dict[str, dict], root: Path = ROOT) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "dataset/assemble.py",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": b.gate,
        "columns": roles(src.inputs),
        "units": {**{n: src.input_units[n] for n in src.inputs}, **CLIMATE_UNITS, **TARGET_UNITS},
        "target_sources": {k: f"outputs/<batch>/results.csv:{v}" for k, v in TARGETS.items()},
        "splits": {name: {"rows": len(df),
                          "by_batch": df.batch.value_counts().sort_index().to_dict(),
                          "by_design_set": df.design_set.value_counts().sort_index().to_dict(),
                          "climates": sorted(df.climate_key.unique()),
                          "n_designs": int(df.design_id.nunique())} for name, df in b.data.items()},
        "files": outputs,
        "sources_sha256": src.file_hashes,
        "checks": [c.__dict__ for c in b.checks],
        "notes": {
            "cv": "split and cross-validate by design_id (GroupKFold): anchors appear in all 10 training climates",
            "ashrae_zone": "metadata only, not a model input (D2)",
            "glazing_vt": "excluded: identical to glazing_shgc (S2)",
            "london": "holdout_ood = extrapolation test (CDD10, GHI, latitude outside the training range)",
            "heating": "near zero in chennai/colombo/miami - report heating error per climate group or in absolute units",
        },
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }


def report_md(src: Sources, b: Build) -> str:
    L = ["# Phase 2.7 — Data Quality Report", "",
         f"Generated by `dataset/assemble.py` · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
         f"**Gate 2.7 → 2.8: {b.gate}**", "",
         "## Checks", "", "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    L += [f"| {c.id} | {c.name} | {c.status} | {c.detail} |" for c in b.checks]
    L += ["", "## Rows per climate", ""]
    frames = []
    for name, df in b.data.items():
        t = df.groupby(["climate_key", "batch", "design_set"]).size().rename("rows").reset_index()
        t.insert(0, "split", name)
        frames.append(t)
    counts = pd.concat(frames)
    pv = counts.pivot_table(index="climate_key", columns=["split", "batch", "design_set"], values="rows",
                            fill_value=0, aggfunc="sum")
    hdr = ["climate"] + [f"{s}/{bt}/{ds}" for s, bt, ds in pv.columns] + ["total"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for k, row in pv.iterrows():
        L.append(f"| {k} | " + " | ".join(str(int(v)) for v in row) + f" | {int(row.sum())} |")
    L.append("| **total** | " + " | ".join(str(int(v)) for v in pv.sum()) + f" | {int(pv.values.sum())} |")
    for name, df in b.data.items():
        L += ["", f"## Target summary — {name} (median [min – max] per climate)", "",
              "| climate | " + " | ".join(SUMMARY_TARGETS) + " |", "|" + "---|" * (len(SUMMARY_TARGETS) + 1)]
        for k, g in df.groupby("climate_key"):
            cells = [f"{g[t].median():.1f} [{g[t].min():.1f} – {g[t].max():.1f}]" for t in SUMMARY_TARGETS]
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## Column dictionary", "", "| # | column | role | unit |", "|---|---|---|---|"]
    r = roles(src.inputs)
    role_of = {**{c: "id" for c in r["id"]}, **{c: "meta (not an input)" for c in r["meta"]},
               **{c: "design input" for c in r["design_inputs"]}, **{c: "climate input" for c in r["climate_inputs"]},
               **{c: "target" for c in r["targets"]}}
    units = {**src.input_units, **CLIMATE_UNITS, **TARGET_UNITS}
    for i, c in enumerate(r["all"], 1):
        L.append(f"| {i} | `{c}` | {role_of[c]} | {units.get(c, '')} |")
    L += ["", "Diagnostics (non-target results columns, simulation messages, accepted exceptions, outlier flags) "
          "are in `dataset_diagnostics.csv`, keyed by `run_id`.", ""]
    return "\n".join(L)


def write_outputs(src: Sources, b: Build, out_dir: Path = OUT_DIR, root: Path = ROOT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in b.data.items():
        fname = SPLITS[name][2]
        files[name] = {"path": fname, "rows": len(df), "sha256": _write_csv(df, out_dir / fname)}
    files["diagnostics"] = {"path": DIAGNOSTICS_NAME, "rows": len(b.diagnostics),
                            "sha256": _write_csv(b.diagnostics, out_dir / DIAGNOSTICS_NAME)}
    (out_dir / REPORT_NAME).write_text(report_md(src, b), encoding="utf-8")
    manifest = manifest_dict(src, b, files, root)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Consumer interface (Phase 3)
# ---------------------------------------------------------------------------

def load_dataset(split: str = "training", out_dir: str | Path = OUT_DIR,
                 verify: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load dataset_<split>.csv; returns (frame, column roles).

    roles: id, meta, design_inputs, climate_inputs, inputs (28), targets (10), group ('design_id'), all.
    With verify=True the file must match the sha256 and column order in dataset_manifest.json, so a
    stale or hand-edited file can never be trained on silently.
        df, cols = load_dataset("training")
        X, y, groups = df[cols["inputs"]], df[cols["targets"]], df[cols["group"]]
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if split not in manifest["files"]:
        raise AssemblyError(f"unknown split {split!r}; have {sorted(manifest['files'])}")
    entry = manifest["files"][split]
    path = out_dir / entry["path"]
    if verify and sha256_file(path) != entry["sha256"]:
        raise AssemblyError(f"{path.name} does not match dataset_manifest.json - rebuild with dataset/assemble.py")
    df = pd.read_csv(path, dtype={c: str for c in ID_COLUMNS + META_COLUMNS})
    if split != "diagnostics" and list(df.columns) != manifest["columns"]["all"]:
        raise AssemblyError(f"{path.name}: columns differ from the manifest")
    return df, manifest["columns"]


def verify(out_dir: Path = OUT_DIR) -> int:
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    ok = True
    for name, e in manifest["files"].items():
        good = sha256_file(out_dir / e["path"]) == e["sha256"]
        ok &= good
        print(f"  {'OK ' if good else 'BAD'}  {e['path']}  ({e['rows']} rows)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="run checks only; write nothing")
    ap.add_argument("--verify", action="store_true", help="re-hash written files against the manifest")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.out)
    src = load_sources()
    b = build(src)
    for c in b.checks:
        print(f"  {c.status:5} {c.id:3} {c.name:24} {c.detail}")
    for name, df in b.data.items():
        print(f"  {name}: {len(df)} rows x {df.shape[1]} columns")
    print(f"GATE 2.7 -> 2.8: {b.gate}")
    if a.check:
        return 0 if b.gate != "NO-GO" else 1
    if b.gate == "NO-GO":
        print("Nothing written (NO-GO).")
        return 1
    m = write_outputs(src, b, a.out)
    print(f"Wrote {a.out}/: " + ", ".join(f"{v['path']} ({v['rows']})" for v in m["files"].values())
          + f", {REPORT_NAME}, {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
