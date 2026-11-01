"""
test_sampling.py - Checks for Sub-task 2.4 (sampling).

Tests verify:
  1. lhs.py       plain and sliced Latin hypercubes have the stratification
                  they claim; decorrelation keeps it; seeds are deterministic
  2. transform.py integer levels equally likely; rounding; VT derived;
                  design_space.yaml validation; parameters match build_idf()
  3. sampler.py   the real samples: counts, replication, per-climate slices,
                  LEAKAGE (each guard is shown to catch an injected violation),
                  reproducibility (files on disk == fresh rebuild), CSV round trip
  4. pre-flight   the 8 edge-case designs build valid IDFs (EnergyPlus IDD)

Run:  python sampling/test_sampling.py
The IDD is found like the generator suite: $EPLUS_IDD, else energyplus_path.txt.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sampling import lhs as L                       # noqa: E402
from sampling import sampler as S                   # noqa: E402
from sampling.transform import (                    # noqa: E402
    DesignSpaceError, ParamSpec, load_design_space,
)

SPACE = load_design_space()
_BUILT: S.Samples | None = None


def built() -> S.Samples:
    """The real samples, built once in memory."""
    global _BUILT
    if _BUILT is None:
        _BUILT = S.build_samples()
    return _BUILT


def is_sliced_lhs(points: np.ndarray, t: int, m: int) -> bool:
    return L.is_latin_hypercube(points) and all(
        L.is_latin_hypercube(points[s * m:(s + 1) * m]) for s in range(t))


# ---------------------------------------------------------------------------
# 1. lhs.py
# ---------------------------------------------------------------------------

def check_plain_lhs_is_latin():
    for seed in range(5):
        p = L.lhs(37, 6, np.random.default_rng(seed))
        assert p.shape == (37, 6) and p.min() >= 0 and p.max() < 1
        assert L.is_latin_hypercube(p)


def check_sliced_lhs_whole_and_every_slice_latin():
    for t, m, d in ((2, 3, 2), (2, 4, 2), (10, 30, 21), (4, 7, 5)):
        for seed in range(3):
            p = L.sliced_lhs(t, m, d, np.random.default_rng(seed))
            assert is_sliced_lhs(p, t, m), (t, m, d, seed)


def check_random_split_is_not_sliced():
    """Why sliced LHS: an ordinary LHS cut into slices usually loses the per-slice property."""
    failures = 0
    for seed in range(20):
        p = L.lhs(40, 3, np.random.default_rng(seed))
        failures += not all(L.is_latin_hypercube(p[s * 10:(s + 1) * 10]) for s in range(4))
    assert failures >= 18, failures


def check_is_latin_hypercube_detects_violation():
    p = L.lhs(10, 3, np.random.default_rng(0))
    p[0, 1] = p[1, 1]
    assert not L.is_latin_hypercube(p)


def check_decorrelate_keeps_sliced_lhs_and_values():
    t, m = 5, 20
    p = L.sliced_lhs(t, m, 8, np.random.default_rng(1))
    q = L.decorrelate(p, t, m, np.random.default_rng(2), 20000)
    assert is_sliced_lhs(q, t, m)
    for s in range(t):                    # every slice keeps its exact values per column
        a, b = p[s * m:(s + 1) * m], q[s * m:(s + 1) * m]
        assert np.array_equal(np.sort(a, 0), np.sort(b, 0))
    assert L.max_abs_corr(q, t, m) < L.max_abs_corr(p, t, m)


def check_best_design_deterministic():
    a = L.best_design(3, 10, 4, (7, 1), 5, 2000)
    b = L.best_design(3, 10, 4, (7, 1), 5, 2000)
    c = L.best_design(3, 10, 4, (7, 2), 5, 2000)
    assert np.array_equal(a.points, b.points) and a.candidate == b.candidate
    assert not np.array_equal(a.points, c.points)


def check_best_design_picks_lowest_discrepancy_candidate():
    d = L.best_design(1, 30, 5, (3, 3), 8)
    scores = []
    for k in range(8):
        pts = L.sliced_lhs(1, 30, 5, np.random.default_rng(np.random.SeedSequence([3, 3, k])))
        scores.append(L.centred_discrepancy(pts))
    assert d.candidate == int(np.argmin(scores))


# ---------------------------------------------------------------------------
# 2. transform.py
# ---------------------------------------------------------------------------

def check_space_shape():
    assert SPACE.dim == 21
    assert "glazing_vt" not in SPACE.sampled_names
    names = SPACE.idf_names
    assert len(names) == 22 and names.index("glazing_vt") == names.index("glazing_shgc") + 1


def check_parameters_match_build_idf():
    from idf_generator.generator.generator import _REQUIRED_PARAMS
    assert set(SPACE.idf_names) == set(_REQUIRED_PARAMS)


def check_integer_levels_equally_likely():
    p = SPACE.spec("n_floors")
    assert p.is_integer and (p.lo, p.hi) == (1, 5)
    u = (np.arange(1000) + 0.5) / 1000
    counts = Counter(p.to_physical(u).astype(int))
    assert counts == {1: 200, 2: 200, 3: 200, 4: 200, 5: 200}, counts
    assert p.to_physical(np.array([0.0, 0.1999, 0.2, 0.9999]))[[0, 1, 2, 3]].tolist() == [1, 1, 2, 5]


def check_continuous_mapping_and_rounding():
    p = ParamSpec("x", 0.1, 0.8, 4, False, "")
    out = p.to_physical(np.array([0.0, 0.5, 0.123456789]))
    assert out[0] == 0.1 and abs(out[1] - 0.45) < 1e-12
    assert out[2] == round(0.1 + 0.123456789 * 0.7, 4)
    for bad in (-0.01, 1.0):
        try:
            p.to_physical(np.array([bad]))
        except ValueError:
            continue
        raise AssertionError(f"u={bad} accepted")


def check_vt_derived_equals_shgc():
    rows = SPACE.to_physical(np.random.default_rng(0).random((50, SPACE.dim)))
    assert all(r["glazing_vt"] == r["glazing_shgc"] for r in rows)
    assert all(isinstance(r["n_floors"], int) for r in rows)


def check_from_values_rejects_out_of_bounds():
    v = SPACE.midpoint()
    v["wwr_north"] = 0.95
    try:
        SPACE.from_values(v)
    except DesignSpaceError:
        return
    raise AssertionError("out-of-bounds edge value accepted")


def _yaml_with(mutate) -> Path:
    import yaml
    cfg = yaml.safe_load(SPACE.source.read_text())
    mutate(cfg)
    tmp = Path(tempfile.mkdtemp()) / "ds.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    return tmp


def check_design_space_validation_rejects_bad_files():
    cases = {
        "min >= max": lambda c: c["continuous_parameters"]["wwr_north"].update(min=0.9),
        "missing decimals": lambda c: c["continuous_parameters"]["glazing_u"].pop("decimals"),
        "wrong schema": lambda c: c["metadata"].update(schema_version=1),
        "bad derived source": lambda c: c["derived_parameters"]["glazing_vt"].update(equal_to="nope"),
        "non-integer bounds": lambda c: c["continuous_parameters"]["n_floors"].update(max=5.5),
    }
    for label, mutate in cases.items():
        try:
            load_design_space(_yaml_with(mutate))
        except DesignSpaceError:
            continue
        raise AssertionError(f"accepted: {label}")


# ---------------------------------------------------------------------------
# 3. sampler.py - the real samples
# ---------------------------------------------------------------------------

def check_counts():
    s = built()
    assert len(s.main) == 3000, len(s.main)
    assert Counter(r["batch"] for r in s.holdout_rows) == {"holdout_id": 300, "holdout_ood": 100}
    assert len(s.pilot) == 200
    assert len(s.designs) == 100 + 2000 + 300 + 42 + 8


def check_invariants_hold():
    s = built()
    problems = S.check_invariants(s.main, s.holdout_rows, s.pilot, s.space,
                                  s.training, s.holdout, s.config)
    assert not problems, problems[:5]


def check_training_and_holdout_climates():
    s = built()
    assert len(s.training) == 10 and s.holdout == ["london"]
    assert {r["climate_key"] for r in s.main} == set(s.training)
    assert {r["climate_key"] for r in s.holdout_rows if r["batch"] == "holdout_ood"} == {"london"}


def check_every_set_is_lhs_and_every_climate_slice_too():
    s = built()
    t = len(s.training)
    for name, (slices, per) in {"A": (1, 100), "B": (t, 200), "C": (t, 30), "P": (1, 42)}.items():
        assert is_sliced_lhs(s.unit[name], slices, per), name


def check_slices_land_in_their_climate():
    """Slice s of B (and C) must be exactly the designs run in training climate s."""
    s = built()
    for name, per, rows in (("B", 200, s.main), ("C", 30, s.holdout_rows)):
        for i, c in enumerate(s.training):
            ids = {r["design_id"] for r in rows if r["climate_key"] == c and r["design_id"].startswith(name)}
            expected = set(S._ids(name, per * len(s.training))[i * per:(i + 1) * per])
            assert ids == expected, (name, c)


def check_design_correlation_removed():
    s = built()
    for name, d in s.chosen.items():
        limit = 0.1 if name == "C" else 0.02      # C slices have only 30 points
        assert d.max_abs_corr_after < limit, (name, d.max_abs_corr_after)
        assert d.max_abs_corr_after < d.max_abs_corr_before


def check_floor_counts_balanced():
    s = built()
    for prefix, per_level in (("A", 20), ("B", 400), ("C", 60)):
        cnt = Counter(p["n_floors"] for d, p in s.designs.items() if d.startswith(prefix))
        assert set(cnt.values()) == {per_level}, (prefix, cnt)
    for c in s.training:                           # and within every climate's slice
        cnt = Counter(r["n_floors"] for r in s.main if r["climate_key"] == c and r["design_set"] == "specific")
        assert set(cnt.values()) == {40}, (c, cnt)
    cnt = Counter(p["n_floors"] for d, p in s.designs.items() if d.startswith("P"))
    assert max(cnt.values()) - min(cnt.values()) <= 1, cnt


def check_edge_cases():
    s = built()
    lo = {p.name: p.lo for p in SPACE.params}
    hi = {p.name: p.hi for p in SPACE.params}
    e1, e2 = s.designs["E01"], s.designs["E02"]
    assert all(e1[n] == lo[n] for n in lo) and all(e2[n] == hi[n] for n in hi)
    assert s.designs["E03"]["floor_area"] == 1500 and s.designs["E03"]["n_floors"] == 5
    assert s.designs["E05"]["n_floors"] == 1
    for c in s.config["pilot_climates"]:
        ids = {r["design_id"] for r in s.pilot if r["climate_key"] == c}
        assert {f"E0{i}" for i in range(1, 9)} <= ids and len(ids) == 50


def check_run_id_format():
    s = built()
    for r in s.main[:5] + s.holdout_rows[-5:] + s.pilot[:5]:
        assert r["run_id"] == f"{r['batch']}_{r['design_id']}_{r['climate_key']}"
    assert s.holdout_rows[-1]["run_id"] == "holdout_ood_A100_london"


def _violations(mutate) -> list[str]:
    s = built()
    main, hold, pilot = copy.deepcopy(s.main), copy.deepcopy(s.holdout_rows), copy.deepcopy(s.pilot)
    mutate(main, hold, pilot)
    return S.check_invariants(main, hold, pilot, s.space, s.training, s.holdout, s.config)


def check_leakage_guards_catch_injected_violations():
    s = built()

    def london_in_main(m, h, p):
        r = dict(m[0], climate_key="london", run_id="main_A001_london")
        m.append(r)

    def holdout_equals_training(m, h, p):
        hid = next(i for i, r in enumerate(h) if r["batch"] == "holdout_id")
        donor = next(r for r in m if r["design_set"] == "specific")
        for n in SPACE.idf_names:
            h[hid][n] = donor[n]

    def holdout_id_reuses_main_id(m, h, p):
        h[0]["design_id"] = m[-1]["design_id"]

    def pilot_id_in_main(m, h, p):
        m[-1]["design_id"] = p[0]["design_id"]

    def anchor_missing_a_climate(m, h, p):
        m.pop(0)

    def london_in_pilot(m, h, p):
        p[0]["climate_key"] = "london"

    def out_of_bounds(m, h, p):
        m[5]["wwr_south"] = 0.95

    def vt_not_shgc(m, h, p):
        m[5]["glazing_vt"] = m[5]["glazing_shgc"] + 0.01

    def duplicate_run(m, h, p):
        m.append(dict(m[0]))

    for case in (london_in_main, holdout_equals_training, holdout_id_reuses_main_id,
                 pilot_id_in_main, anchor_missing_a_climate, london_in_pilot,
                 out_of_bounds, vt_not_shgc, duplicate_run):
        assert _violations(case), f"not caught: {case.__name__}"
    assert not _violations(lambda m, h, p: None)
    assert s.holdout == ["london"]


def check_files_on_disk_match_fresh_build():
    """Reproducibility: the committed files are exactly what the seed produces."""
    texts = S.render(built())
    for name in (S.MAIN_CSV, S.HOLDOUT_CSV, S.PILOT_CSV):
        on_disk = (S.SAMPLING_DIR / name).read_text(encoding="utf-8")
        assert on_disk == texts[name], f"{name} differs from a fresh build - rebuild with --build"


def check_different_seed_gives_different_samples():
    cfg = dict(built().config, master_seed=1, n_candidates=2, decorrelate_iterations=0)
    other = S.build_samples(config=cfg)
    assert S.csv_text(other.main, SPACE) != S.csv_text(built().main, SPACE)


def check_manifest_records_outputs():
    import json
    m = json.loads((S.SAMPLING_DIR / S.MANIFEST).read_text())
    texts = S.render(built())
    for name in (S.MAIN_CSV, S.HOLDOUT_CSV, S.PILOT_CSV):
        assert m["outputs"][name]["sha256"] == S._sha256(texts[name].encode())
    assert m["parameters"]["model_inputs"] == SPACE.sampled_names
    assert m["master_seed"] == built().config["master_seed"]


def check_csv_round_trip():
    rows = S.load_samples(S.SAMPLING_DIR / S.PILOT_CSV, SPACE)
    assert len(rows) == 200
    assert all(isinstance(r["n_floors"], int) for r in rows)
    s = built()
    for r in rows[:60]:
        assert S.idf_params(r, SPACE) == s.designs[r["design_id"]], r["design_id"]


# ---------------------------------------------------------------------------
# 4. Pre-flight on the edge cases (needs the EnergyPlus IDD)
# ---------------------------------------------------------------------------

def check_edge_cases_build_valid_idfs():
    idd = S.resolve_idd()
    if not Path(idd).exists():
        raise AssertionError(f"IDD not found at {idd} - set EPLUS_IDD")
    s = built()
    designs = {d: s.designs[d] for d in s.designs if d.startswith("E")}
    result = S.preflight(designs, idd, workers=1)
    assert result["n_failed"] == 0, result["failures"]


ALL_CHECKS = [
    check_plain_lhs_is_latin,
    check_sliced_lhs_whole_and_every_slice_latin,
    check_random_split_is_not_sliced,
    check_is_latin_hypercube_detects_violation,
    check_decorrelate_keeps_sliced_lhs_and_values,
    check_best_design_deterministic,
    check_best_design_picks_lowest_discrepancy_candidate,
    check_space_shape,
    check_parameters_match_build_idf,
    check_integer_levels_equally_likely,
    check_continuous_mapping_and_rounding,
    check_vt_derived_equals_shgc,
    check_from_values_rejects_out_of_bounds,
    check_design_space_validation_rejects_bad_files,
    check_counts,
    check_invariants_hold,
    check_training_and_holdout_climates,
    check_every_set_is_lhs_and_every_climate_slice_too,
    check_slices_land_in_their_climate,
    check_design_correlation_removed,
    check_floor_counts_balanced,
    check_edge_cases,
    check_run_id_format,
    check_leakage_guards_catch_injected_violations,
    check_files_on_disk_match_fresh_build,
    check_different_seed_gives_different_samples,
    check_manifest_records_outputs,
    check_csv_round_trip,
    check_edge_cases_build_valid_idfs,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("SAMPLING (2.4) - CHECKS")
    print("=" * 72)
    for chk in ALL_CHECKS:
        total += 1
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {chk.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {chk.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
