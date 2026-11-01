"""
test_coverage_report.py - Checks for the climate coverage analysis.

Tests verify:
  1. The building blocks (rank, standardisation, sign fixing, neighbours,
     envelope) on inputs with hand-computable answers
  2. The holdout never influences the scaling - the same rule the
     surrogate must obey in Phase 3
  3. PCA properties: orthonormal components, perfectly collinear data has
     effective rank 1
  4. On the real 11-climate set: the reported findings (rank 5, London's
     envelope, nearest neighbours, near-duplicate pair)
  5. The report and figure are produced, complete and deterministic

Run:  python climate/test_coverage_report.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import climate_features as C   # type: ignore
import coverage_report as R    # type: ignore


# ---------------------------------------------------------------------------
# Synthetic registries
# ---------------------------------------------------------------------------

def make_record(key: str, values: list[float], role: str = "training",
                zone: str = "4A") -> C.ClimateRecord:
    return C.ClimateRecord(
        key=key, city=key.title(), country="Testland", role=role, koppen="Cfa",
        wmo="000000", epw_filename=f"{key}.epw", epw_path=Path(f"{key}.epw"),
        features=C.ClimateFeatures(*values, ashrae_zone=zone),
        diagnostics=C.ClimateDiagnostics(
            longitude=0.0, elevation=0.0, dbt_min=0.0, dbt_max=0.0, dbt_std=0.0,
            hdd10=0.0, cdd18=0.0, ghi_summer=0.0, ghi_winter=0.0, n_hours=8760,
            n_substituted=0, data_source="SYN", period_of_record="2011-2025"),
    )


def random_registry(n_train: int = 8, seed: int = 0, with_holdout: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    reg = {f"c{i}": make_record(f"c{i}", list(rng.normal(size=7) * 10 + 50)) for i in range(n_train)}
    if with_holdout:
        reg["h"] = make_record("h", list(rng.normal(size=7) * 40 + 50), role="holdout")
    return reg


# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------

def check_effective_rank():
    assert R.effective_rank(np.array([0.7, 0.2, 0.09, 0.01]), 0.99) == 3
    assert R.effective_rank(np.array([1.0, 0.0, 0.0]), 0.99) == 1
    assert R.effective_rank(np.array([0.5, 0.5]), 0.99) == 2


def check_standardise_rejects_constant_feature():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    try:
        R.standardise(X, X.mean(0), X.std(0))
    except ValueError:
        return
    raise AssertionError("constant feature was accepted")


def check_sign_fixing_is_deterministic():
    comps = np.array([[0.1, -0.9, 0.2], [0.6, 0.1, -0.2]])
    fixed = R._fix_signs(comps)
    assert np.allclose(fixed[0], [-0.1, 0.9, -0.2]), fixed[0]
    assert np.allclose(fixed[1], comps[1]), fixed[1]
    assert np.allclose(R._fix_signs(fixed), fixed), "not idempotent"
    assert np.allclose(R._fix_signs(-comps), fixed), "depends on input sign"


def check_nearest_neighbours():
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    nn = R.nearest_neighbours(Z, ["a", "b", "c"], Z, ["a", "b", "c"], exclude_self=True)
    assert nn == [("a", "b", 1.0), ("b", "a", 1.0), ("c", "b", 4.0)], nn
    q = R.nearest_neighbours(Z, ["a", "b", "c"], np.array([[4.0, 0.0]]), ["q"], exclude_self=False)
    assert q == [("q", "c", 1.0)], q


def check_envelope_table():
    X_train = np.array([[0.0, 10.0], [10.0, 20.0]])
    rows = R.envelope_table(X_train, np.array([[5.0, 25.0]]), ["h"], ("f1", "f2"))
    assert rows[0][5] == "inside" and rows[0][6] == 0.0, rows[0]
    assert rows[1][5] == "above" and abs(rows[1][6] - 50.0) < 1e-9, rows[1]
    below = R.envelope_table(X_train, np.array([[-2.0, 15.0]]), ["h"], ("f1", "f2"))
    assert below[0][5] == "below" and abs(below[0][6] - 20.0) < 1e-9, below[0]


# ---------------------------------------------------------------------------
# 2-3. Analysis properties
# ---------------------------------------------------------------------------

def check_holdout_never_influences_scaling():
    """Mean/std, correlation and components come from training climates only."""
    with_h = R.analyse(random_registry(with_holdout=True))
    without_h = R.analyse(random_registry(with_holdout=False))
    assert np.allclose(with_h.mean, without_h.mean)
    assert np.allclose(with_h.std, without_h.std)
    assert np.allclose(with_h.correlation, without_h.correlation)
    assert np.allclose(with_h.components, without_h.components)


def check_components_are_orthonormal():
    a = R.analyse(random_registry())
    gram = a.components @ a.components.T
    assert np.allclose(gram, np.eye(len(gram)), atol=1e-9), gram
    assert abs(a.explained.sum() - 1.0) < 1e-9


def check_collinear_features_have_rank_one():
    """Seven features that are all linear in one variable carry one dimension."""
    reg = {}
    for i, t in enumerate(np.linspace(0, 1, 6)):
        reg[f"c{i}"] = make_record(f"c{i}", [1 + 2 * t, 3 - t, 5 + t, 2 * t, 9 - 3 * t, 4 + t, 1 + 0.5 * t])
    a = R.analyse(reg)
    assert a.effective_rank == 1, a.explained


def check_too_few_training_climates_rejected():
    reg = {f"c{i}": make_record(f"c{i}", [float(i + j) for j in range(7)]) for i in range(2)}
    try:
        R.analyse(reg)
    except ValueError:
        return
    raise AssertionError("2 training climates accepted")


# ---------------------------------------------------------------------------
# 4. The real climate set
# ---------------------------------------------------------------------------

_REAL = None


def real():
    global _REAL
    if _REAL is None:
        if not C.DEFAULT_METADATA_CSV.exists():
            print("      (skipped - climate_metadata.csv not built)")
            return None
        reg = C.load_climate_registry(check_files=False)
        _REAL = (reg, R.analyse(reg))
    return _REAL


def check_real_effective_rank():
    r = real()
    if r is None:
        return
    assert r[1].effective_rank == 5, r[1].explained


def check_real_london_envelope():
    r = real()
    if r is None:
        return
    outside = {row[1] for row in r[1].envelope if row[0] == "london" and row[5] != "inside"}
    assert outside == {"annual_ghi", "latitude", "cdd10"}, outside
    ghi = next(row for row in r[1].envelope if row[0] == "london" and row[1] == "annual_ghi")
    assert ghi[5] == "below" and 45 < ghi[6] < 55, ghi


def check_real_london_nearest_is_chicago():
    r = real()
    if r is None:
        return
    assert r[1].nn_holdout[0][:2] == ("london", "chicago"), r[1].nn_holdout


def check_real_near_duplicate_pair():
    r = real()
    if r is None:
        return
    a = r[1]
    flagged = {k for k, _, d in a.nn_train if d < R.NEAR_DUPLICATE_FRACTION * a.nn_median}
    assert flagged == {"chicago", "minneapolis"}, flagged


def check_real_mean_dbt_is_redundant():
    r = real()
    if r is None:
        return
    a = r[1]
    i, j = a.features.index("mean_dbt"), a.features.index("cdd10")
    assert abs(a.correlation[i, j]) >= R.HIGH_CORRELATION, a.correlation[i, j]


# ---------------------------------------------------------------------------
# 5. Outputs
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["## 1. Climate features", "## 2. Feature correlation", "## 3. Dimensionality",
                     "## 4. Principal components", "## 5. Nearest neighbours",
                     "## 6. Held-out climate", "## 7. ASHRAE zone labels", "## 8. Methodology"]


def check_report_and_figure_written_and_deterministic():
    r = real()
    if r is None:
        return
    reg, a = r
    with tempfile.TemporaryDirectory() as td:
        md, png = Path(td) / "report.md", Path(td) / "fig" / "pca.png"
        R.plot_pca(a, reg, png)
        R.write_report(a, reg, path=md, figure_path=png)
        text = md.read_text()
        for section in REQUIRED_SECTIONS:
            assert section in text, f"missing section {section!r}"
        assert "partial extrapolation" in text
        assert "![Climate coverage: PCA](fig/pca.png)" in text, "figure link wrong"
        assert png.stat().st_size > 20_000, "figure suspiciously small"
        first = md.read_bytes()
        R.write_report(a, reg, path=md, figure_path=png)
        assert md.read_bytes() == first, "report not deterministic"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_effective_rank,
    check_standardise_rejects_constant_feature,
    check_sign_fixing_is_deterministic,
    check_nearest_neighbours,
    check_envelope_table,
    check_holdout_never_influences_scaling,
    check_components_are_orthonormal,
    check_collinear_features_have_rank_one,
    check_too_few_training_climates_rejected,
    check_real_effective_rank,
    check_real_london_envelope,
    check_real_london_nearest_is_chicago,
    check_real_near_duplicate_pair,
    check_real_mean_dbt_is_redundant,
    check_report_and_figure_written_and_deterministic,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("CLIMATE COVERAGE REPORT - CHECKS")
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
