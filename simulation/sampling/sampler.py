"""
sampler.py - Build the simulation sample files for Phase 2 (Sub-task 2.4).

Every row of a sample file is ONE EnergyPlus run: a building design (the 22
values build_idf() needs) paired with a climate key. The batch runner (2.6)
reads a row, builds the IDF and runs it with registry[climate_key].epw_path.
Climate FEATURES are not copied here: 2.7 joins them from
climate/climate_metadata.csv on climate_key, so they live in one place.

DESIGN SETS (Decisions S1-S4, see config/climate_config.yaml -> sampling)
-----------
    A  anchors      100 designs, plain LHS     -> every training climate + London
    B  specific     10 x 200, sliced LHS       -> slice s runs in training climate s
    C  holdout_id   10 x 30, sliced LHS        -> slice s runs in training climate s
    P  pilot LHS    42 designs, plain LHS      -> the 4 pilot climates
    E  edge cases   8 fixed designs            -> the 4 pilot climates

OUTPUT FILES (sampling/)
------------
    samples_main.csv      batch=main         A x 10 training + B      = 3,000 rows
    samples_holdout.csv   batch=holdout_id   C                        =   300
                          batch=holdout_ood  A x London               =   100
    samples_pilot.csv     batch=pilot        (P + E) x 4 pilot        =   200
    samples_manifest.json seeds, chosen candidates, input/output SHA-256s

Columns: run_id, batch, design_set, design_id, climate_key, then the 22 IDF
parameters in design_space.yaml order (glazing_vt right after glazing_shgc).
The surrogate's design inputs are the 21 SAMPLED parameters
(manifest["parameters"]["model_inputs"]); glazing_vt is a copy of
glazing_shgc and must not be a model input.

Phase 3 WARNING: an anchor design appears in 10 training climates. Split and
cross-validate by design_id (GroupKFold), never by row, or the same building
lands in both train and test.

USAGE (from the repository root)
-----
    python sampling/sampler.py --build        write the sample files
    python sampling/sampler.py --validate     rebuild in memory; compare; check invariants
    python sampling/sampler.py --report       sampling_report.md + figures
    python sampling/sampler.py --preflight [--workers N]
                                              build_idf() on every unique design
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate.climate_features import (   # noqa: E402
    load_climate_registry, training_keys, holdout_keys,
)
from sampling import lhs as L            # noqa: E402
from sampling.transform import DesignSpace, load_design_space  # noqa: E402

SAMPLING_DIR = ROOT / "sampling"
CLIMATE_CONFIG = ROOT / "config" / "climate_config.yaml"
METADATA_CSV = ROOT / "climate" / "climate_metadata.csv"

MAIN_CSV = "samples_main.csv"
HOLDOUT_CSV = "samples_holdout.csv"
PILOT_CSV = "samples_pilot.csv"
MANIFEST = "samples_manifest.json"
PREFLIGHT_JSON = "preflight.json"

ID_COLUMNS = ("run_id", "batch", "design_set", "design_id", "climate_key")
MANIFEST_VERSION = 1

# Child-seed index of each LHS design set. Never renumber: it would change
# every sample. Add new sets with new numbers.
SET_SEED_INDEX = {"A": 1, "B": 2, "C": 3, "P": 4}
SET_LABEL = {"A": "anchor", "B": "specific", "C": "holdout", "P": "pilot_lhs", "E": "edge"}


# ---------------------------------------------------------------------------
# Edge cases for the pilot (2.5). Values not listed stay at mid-range.
# ---------------------------------------------------------------------------

def edge_cases(space: DesignSpace) -> dict[str, tuple[str, dict[str, float]]]:
    """E01-E08: id -> (what it stresses, full sampled-parameter values)."""
    lo = {p.name: p.lo for p in space.params}
    hi = {p.name: p.hi for p in space.params}
    mid = space.midpoint()
    wwr = ("wwr_north", "wwr_east", "wwr_south", "wwr_west")
    ovh = ("overhang_depth_n", "overhang_depth_e", "overhang_depth_s", "overhang_depth_w")

    def mid_with(**kw: float) -> dict[str, float]:
        return {**mid, **kw}

    return {
        "E01": ("every parameter at its minimum", dict(lo)),
        "E02": ("every parameter at its maximum", dict(hi)),
        "E03": ("most slender: smallest footprint, aspect 3, 5 tall floors (narrowest core)",
                mid_with(floor_area=lo["floor_area"], aspect_ratio=hi["aspect_ratio"],
                         n_floors=hi["n_floors"], floor_height=hi["floor_height"])),
        "E04": ("largest: biggest square footprint, 5 floors",
                mid_with(floor_area=hi["floor_area"], aspect_ratio=lo["aspect_ratio"],
                         n_floors=hi["n_floors"])),
        "E05": ("single storey (roof and ground coupling dominate)",
                mid_with(n_floors=lo["n_floors"])),
        "E06": ("max glazing + deepest overhangs and fins at the lowest floor height",
                mid_with(floor_height=lo["floor_height"], sidefin_depth=hi["sidefin_depth"],
                         **{w: hi[w] for w in wwr}, **{o: hi[o] for o in ovh})),
        "E07": ("min glazing, no shading",
                mid_with(sidefin_depth=lo["sidefin_depth"],
                         **{w: lo[w] for w in wwr}, **{o: lo[o] for o in ovh})),
        "E08": ("reference: every parameter mid-range", dict(mid)),
    }


# ---------------------------------------------------------------------------
# Building the samples
# ---------------------------------------------------------------------------

@dataclass
class Samples:
    space: DesignSpace
    config: dict
    training: list[str]
    holdout: list[str]
    designs: dict[str, dict] = field(default_factory=dict)       # design_id -> params
    unit: dict[str, np.ndarray] = field(default_factory=dict)    # set -> unit-cube points
    chosen: dict[str, L.Design] = field(default_factory=dict)    # set -> Design
    main: list[dict] = field(default_factory=list)
    holdout_rows: list[dict] = field(default_factory=list)
    pilot: list[dict] = field(default_factory=list)

    def unique_design_ids(self) -> list[str]:
        return list(self.designs)


def load_sampling_config(path: Path = CLIMATE_CONFIG) -> dict:
    import yaml

    cfg = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("sampling")
    if not cfg:
        raise ValueError(f"{path.name}: no 'sampling' section")
    required = ("replication_strategy", "master_seed", "n_anchor_designs",
                "n_specific_per_climate", "n_holdout_id_per_climate",
                "holdout_ood_designs", "n_candidates", "decorrelate_iterations",
                "n_pilot_lhs", "pilot_climates")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"sampling config missing {missing}")
    if cfg["replication_strategy"] != "hybrid":
        raise ValueError("this sampler implements replication_strategy 'hybrid' (Decision S1)")
    if cfg["holdout_ood_designs"] != "anchors":
        raise ValueError("this sampler implements holdout_ood_designs 'anchors' (Decision S4)")
    return cfg


def _ids(prefix: str, n: int) -> list[str]:
    width = max(2, len(str(n)))
    return [f"{prefix}{i:0{width}d}" for i in range(1, n + 1)]


def _row(batch: str, design_set: str, design_id: str, climate: str, params: dict) -> dict:
    return {"run_id": f"{batch}_{design_id}_{climate}", "batch": batch,
            "design_set": design_set, "design_id": design_id, "climate_key": climate, **params}


def build_samples(space: DesignSpace | None = None, config: dict | None = None,
                  registry: dict | None = None) -> Samples:
    space = space or load_design_space()
    config = config or load_sampling_config()
    registry = registry if registry is not None else load_climate_registry(METADATA_CSV)
    train, hold = training_keys(registry), holdout_keys(registry)

    pilot_climates = list(config["pilot_climates"])
    bad = [c for c in pilot_climates if c not in train]
    if bad:
        raise ValueError(f"pilot climates must be training climates; not training: {bad}")

    s = Samples(space, config, train, hold)
    seed0 = int(config["master_seed"])
    k = int(config["n_candidates"])
    t = len(train)

    plan = {
        "A": (1, int(config["n_anchor_designs"])),
        "B": (t, int(config["n_specific_per_climate"])),
        "C": (t, int(config["n_holdout_id_per_climate"])),
        "P": (1, int(config["n_pilot_lhs"])),
    }
    for name, (slices, per) in plan.items():
        design = L.best_design(slices, per, space.dim, (seed0, SET_SEED_INDEX[name]), k,
                               int(config["decorrelate_iterations"]))
        s.chosen[name] = design
        s.unit[name] = design.points
        ids = _ids(name, slices * per)
        for did, params in zip(ids, space.to_physical(design.points)):
            s.designs[did] = params

    edges = edge_cases(space)
    for did, (_, values) in edges.items():
        s.designs[did] = space.from_values(values)

    A = [d for d in s.designs if d.startswith("A")]
    B = [d for d in s.designs if d.startswith("B")]
    C = [d for d in s.designs if d.startswith("C")]
    P = [d for d in s.designs if d.startswith("P")]
    E = list(edges)
    nb, nc = plan["B"][1], plan["C"][1]

    # main: anchors in every training climate, then each specific design in its slice's climate
    for did in A:
        for c in train:
            s.main.append(_row("main", SET_LABEL["A"], did, c, s.designs[did]))
    for i, did in enumerate(B):
        s.main.append(_row("main", SET_LABEL["B"], did, train[i // nb], s.designs[did]))

    # holdout: unseen designs in seen climates, then seen (anchor) designs in unseen climates
    for i, did in enumerate(C):
        s.holdout_rows.append(_row("holdout_id", SET_LABEL["C"], did, train[i // nc], s.designs[did]))
    for c in hold:
        for did in A:
            s.holdout_rows.append(_row("holdout_ood", SET_LABEL["A"], did, c, s.designs[did]))

    # pilot: LHS designs and edge cases in each pilot climate
    for c in pilot_climates:
        for did in P:
            s.pilot.append(_row("pilot", SET_LABEL["P"], did, c, s.designs[did]))
        for did in E:
            s.pilot.append(_row("pilot", SET_LABEL["E"], did, c, s.designs[did]))
    return s


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _fmt(space: DesignSpace, name: str, value) -> str:
    src = dict(space.derived).get(name, name)
    p = space.spec(src)
    return str(int(value)) if p.is_integer else f"{float(value):.{p.decimals}f}"


def csv_text(rows: list[dict], space: DesignSpace) -> str:
    """Deterministic CSV text (fixed decimals, '\\n' line endings)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    cols = list(ID_COLUMNS) + space.idf_names
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in ID_COLUMNS] + [_fmt(space, n, r[n]) for n in space.idf_names])
    return buf.getvalue()


def load_samples(path: str | Path, space: DesignSpace | None = None) -> list[dict]:
    """Read a sample CSV back into typed rows - what the 2.6 runner should call.

    Each row: the 5 id columns as str, every IDF parameter as float (integer
    parameters as int), ready for build_idf(row_params, idd).
    """
    space = space or load_design_space()
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        expected = list(ID_COLUMNS) + space.idf_names
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} != expected {expected}")
        ints = {p.name for p in space.params if p.is_integer}
        rows = []
        for r in reader:
            row = {c: r[c] for c in ID_COLUMNS}
            for n in space.idf_names:
                row[n] = int(r[n]) if n in ints else float(r[n])
            rows.append(row)
    return rows


def idf_params(row: dict, space: DesignSpace) -> dict:
    """The build_idf() argument for a sample row."""
    return {n: row[n] for n in space.idf_names}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_dict(s: Samples, texts: dict[str, str]) -> dict:
    import scipy

    space, cfg = s.space, s.config
    sets = {}
    for name, d in s.chosen.items():
        sets[name] = {
            "label": SET_LABEL[name],
            "n_designs": int(len(d.points)),
            "n_slices": d.n_slices,
            "per_slice": d.per_slice,
            "seed": [int(cfg["master_seed"]), SET_SEED_INDEX[name]],
            "n_candidates": int(cfg["n_candidates"]),
            "chosen_candidate": d.candidate,
            "centred_discrepancy_whole": round(d.discrepancy_whole, 6),
            "centred_discrepancy_slices_mean": round(float(np.mean(d.discrepancy_slices)), 6),
            "decorrelate_iterations": int(cfg["decorrelate_iterations"]),
            "max_abs_corr_before_decorrelation": round(d.max_abs_corr_before, 4),
            "max_abs_corr_after_decorrelation": round(d.max_abs_corr_after, 4),
            "design_ids": f"{_ids(name, len(d.points))[0]}..{_ids(name, len(d.points))[-1]}",
        }
    sets["E"] = {"label": "edge", "n_designs": 8,
                 "cases": {k: v[0] for k, v in edge_cases(space).items()}}
    nb, nc = int(cfg["n_specific_per_climate"]), int(cfg["n_holdout_id_per_climate"])
    b_ids, c_ids = _ids("B", nb * len(s.training)), _ids("C", nc * len(s.training))
    slice_map = {c: {"B": f"{b_ids[i * nb]}..{b_ids[(i + 1) * nb - 1]}",
                     "C": f"{c_ids[i * nc]}..{c_ids[(i + 1) * nc - 1]}"}
                 for i, c in enumerate(s.training)}
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_by": "sampling/sampler.py --build",
        "decisions": {
            "S1": "hybrid replication: anchors in every training climate + sliced-LHS climate-specific designs",
            "S2": "glazing_vt not sampled; glazing_vt = glazing_shgc",
            "S3": "n_floors uniform over integers 1..5 (floor of U[1, 6))",
            "S4": "out-of-distribution holdout (London) runs the anchor designs",
        },
        "master_seed": int(cfg["master_seed"]),
        "inputs_sha256": {
            "config/design_space.yaml": _sha256(space.source.read_bytes()),
            "config/climate_config.yaml": _sha256(CLIMATE_CONFIG.read_bytes()),
            "climate/climate_metadata.csv": _sha256(METADATA_CSV.read_bytes()),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "parameters": {
            "model_inputs": space.sampled_names,
            "idf_columns": space.idf_names,
            "derived": {n: f"= {src}" for n, src in space.derived},
        },
        "climates": {
            "training": s.training,
            "holdout_ood": s.holdout,
            "pilot": list(cfg["pilot_climates"]),
            "slice_to_climate": slice_map,
        },
        "design_sets": sets,
        "outputs": {name: {"rows": text.count("\n") - 1, "sha256": _sha256(text.encode())}
                    for name, text in texts.items()},
    }


def render(s: Samples) -> dict[str, str]:
    """All output files as text, keyed by file name."""
    texts = {MAIN_CSV: csv_text(s.main, s.space),
             HOLDOUT_CSV: csv_text(s.holdout_rows, s.space),
             PILOT_CSV: csv_text(s.pilot, s.space)}
    texts[MANIFEST] = json.dumps(manifest_dict(s, texts), indent=2) + "\n"
    return texts


# ---------------------------------------------------------------------------
# Invariants - used by --validate and by the test suite
# ---------------------------------------------------------------------------

def check_invariants(main: list[dict], holdout: list[dict], pilot: list[dict],
                     space: DesignSpace, training: list[str], hold: list[str],
                     config: dict) -> list[str]:
    """Every property the campaign relies on. Returns a list of problems."""
    problems: list[str] = []
    t = len(training)
    nA, nb, nc = (int(config[k]) for k in
                  ("n_anchor_designs", "n_specific_per_climate", "n_holdout_id_per_climate"))
    all_rows = main + holdout + pilot

    # run_id unique everywhere
    dup = [k for k, v in Counter(r["run_id"] for r in all_rows).items() if v > 1]
    if dup:
        problems.append(f"duplicate run_id: {dup[:5]}")

    # --- leakage ---
    main_climates = {r["climate_key"] for r in main}
    if main_climates != set(training):
        problems.append(f"main climates {sorted(main_climates)} != training {training}")
    for h in hold:
        if any(r["climate_key"] == h for r in main):
            problems.append(f"holdout climate {h!r} appears in main")
    main_ids = {r["design_id"] for r in main}
    hid = [r for r in holdout if r["batch"] == "holdout_id"]
    hood = [r for r in holdout if r["batch"] == "holdout_ood"]
    if {r["design_id"] for r in hid} & main_ids:
        problems.append("holdout_id design ids also in main")
    names = space.sampled_names
    main_vecs = {tuple(r[n] for n in names) for r in main}
    if any(tuple(r[n] for n in names) in main_vecs for r in hid):
        problems.append("a holdout_id design has the same parameter values as a main design")
    if {r["climate_key"] for r in hid} - set(training):
        problems.append("holdout_id uses a non-training climate")
    if {r["climate_key"] for r in hood} != set(hold):
        problems.append(f"holdout_ood climates {sorted({r['climate_key'] for r in hood})} != {hold}")
    pilot_ids = {r["design_id"] for r in pilot}
    if pilot_ids & (main_ids | {r["design_id"] for r in holdout}):
        problems.append("pilot design ids reused in main/holdout")
    if {r["climate_key"] for r in pilot} & set(hold):
        problems.append("a holdout climate appears in the pilot")

    # --- counts and replication ---
    anchors = sorted({r["design_id"] for r in main if r["design_set"] == "anchor"})
    specific = [r for r in main if r["design_set"] == "specific"]
    if len(anchors) != nA:
        problems.append(f"{len(anchors)} anchors in main, expected {nA}")
    per_anchor = Counter(r["design_id"] for r in main if r["design_set"] == "anchor")
    if any(per_anchor[a] != t for a in anchors):
        problems.append("an anchor is not in every training climate")
    for a in anchors:
        climates = {r["climate_key"] for r in main if r["design_id"] == a}
        if climates != set(training):
            problems.append(f"anchor {a} climates {sorted(climates)}")
            break
    if sorted({r["design_id"] for r in hood}) != anchors or len(hood) != nA * len(hold):
        problems.append("holdout_ood is not exactly the anchors in each holdout climate")
    if len(specific) != nb * t or len({r["design_id"] for r in specific}) != nb * t:
        problems.append("specific designs are not each used exactly once")
    spc = Counter(r["climate_key"] for r in specific)
    if any(spc[c] != nb for c in training):
        problems.append(f"specific designs per climate {dict(spc)} != {nb}")
    hpc = Counter(r["climate_key"] for r in hid)
    if any(hpc[c] != nc for c in training) or len(hid) != nc * t:
        problems.append(f"holdout_id per climate {dict(hpc)} != {nc}")

    # --- values: bounds, integers, derived ---
    for r in all_rows:
        for p in space.params:
            if not p.contains(r[p.name]):
                problems.append(f"{r['run_id']}: {p.name}={r[p.name]} outside [{p.lo}, {p.hi}]")
        for name, src in space.derived:
            if r[name] != r[src]:
                problems.append(f"{r['run_id']}: {name} != {src}")
        if len(problems) > 50:
            problems.append("... (stopping)")
            return problems

    # --- every row of a design carries identical values ---
    seen: dict[str, tuple] = {}
    for r in all_rows:
        vec = tuple(r[n] for n in space.idf_names)
        if seen.setdefault(r["design_id"], vec) != vec:
            problems.append(f"design {r['design_id']} has different values in different rows")
            break
    return problems


# ---------------------------------------------------------------------------
# Pre-flight: build_idf on every unique design (no EnergyPlus needed)
# ---------------------------------------------------------------------------

def resolve_idd() -> str:
    """$EPLUS_IDD, else Energy+.idd beside the binary named in energyplus_path.txt."""
    override = os.environ.get("EPLUS_IDD")
    if override:
        return override
    exe_file = ROOT / "energyplus_path.txt"
    if exe_file.exists():
        return str(Path(exe_file.read_text().strip()).parent / "Energy+.idd")
    return "/Applications/EnergyPlus-24-1-0/Energy+.idd"


def _preflight_one(args: tuple[str, dict, str]) -> tuple[str, str | None, float]:
    did, params, idd = args
    start = time.perf_counter()
    try:
        from idf_generator import build_idf
        idf = build_idf(params, idd)
        n_zones = len(idf.idfobjects["Zone"])
        n_win = len(idf.idfobjects["FenestrationSurface:Detailed"])
        nf = params["n_floors"]
        if n_zones != 5 * nf:
            return did, f"{n_zones} zones, expected {5 * nf}", time.perf_counter() - start
        if n_win != 4 * nf:
            return did, f"{n_win} windows, expected {4 * nf}", time.perf_counter() - start
        return did, None, time.perf_counter() - start
    except Exception as e:                                   # noqa: BLE001
        return did, f"{type(e).__name__}: {e}"[:300], time.perf_counter() - start


def preflight(designs: dict[str, dict], idd: str, workers: int = 1) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    if not Path(idd).exists():
        raise FileNotFoundError(f"IDD not found: {idd} (set EPLUS_IDD)")
    jobs = [(d, p, idd) for d, p in designs.items()]
    start = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_preflight_one, jobs, chunksize=16))
    else:
        results = [_preflight_one(j) for j in jobs]
    times = [r[2] for r in results]
    failures = [{"design_id": d, "error": e} for d, e, _ in results if e]
    return {
        "n_designs": len(results),
        "n_passed": len(results) - len(failures),
        "n_failed": len(failures),
        "failures": failures,
        "idd": idd,
        "idd_version": Path(idd).read_text(errors="replace").splitlines()[0].strip(),
        "build_seconds_mean": round(float(np.mean(times)), 3),
        "wall_seconds": round(time.time() - start, 1),
        "workers": workers,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_build(out_dir: Path = SAMPLING_DIR) -> int:
    s = build_samples()
    texts = render(s)
    problems = check_invariants(s.main, s.holdout_rows, s.pilot, s.space,
                                s.training, s.holdout, s.config)
    if problems:
        print("NOT WRITTEN - invariant violations:\n  " + "\n  ".join(problems))
        return 1
    for name, text in texts.items():
        (out_dir / name).write_text(text, encoding="utf-8", newline="")
        print(f"wrote {name:24} {text.count(chr(10)) - 1 if name.endswith('.csv') else '':>6}")
    return 0


def cmd_validate(out_dir: Path = SAMPLING_DIR) -> int:
    s = build_samples()
    texts = render(s)
    ok = True
    for name, text in texts.items():
        path = out_dir / name
        if not path.exists():
            print(f"MISSING  {name}")
            ok = False
            continue
        on_disk = path.read_text(encoding="utf-8")
        if name == MANIFEST:   # environment versions may legitimately differ
            a, b = json.loads(on_disk), json.loads(text)
            a.pop("environment", None), b.pop("environment", None)
            same = a == b
        else:
            same = on_disk == text
        print(f"{'OK      ' if same else 'DIFFERS '} {name}")
        ok &= same
    space = s.space
    rows = [load_samples(out_dir / n, space) for n in (MAIN_CSV, HOLDOUT_CSV, PILOT_CSV)]
    problems = check_invariants(*rows, space, s.training, s.holdout, s.config)
    for p in problems:
        print(f"PROBLEM  {p}")
    ok &= not problems
    print("VALID" if ok else "INVALID")
    return 0 if ok else 1


def cmd_preflight(workers: int, out_dir: Path = SAMPLING_DIR) -> int:
    space = load_design_space()
    designs: dict[str, dict] = {}
    for name in (MAIN_CSV, HOLDOUT_CSV, PILOT_CSV):
        for r in load_samples(out_dir / name, space):
            designs.setdefault(r["design_id"], idf_params(r, space))
    result = preflight(designs, resolve_idd(), workers)
    (out_dir / PREFLIGHT_JSON).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"pre-flight: {result['n_passed']}/{result['n_designs']} designs built "
          f"({result['wall_seconds']} s, {workers} workers)")
    for f in result["failures"][:20]:
        print(f"  FAIL {f['design_id']}: {f['error']}")
    return 0 if result["n_failed"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--preflight", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    if a.build:
        return cmd_build()
    if a.validate:
        return cmd_validate()
    if a.preflight:
        return cmd_preflight(a.workers)
    from sampling.report import write_report
    return write_report()


if __name__ == "__main__":
    sys.exit(main())
