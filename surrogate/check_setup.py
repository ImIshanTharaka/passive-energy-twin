"""
check_setup.py - Sub-task 3.1 step 1 check: shared venv, path files, Phase 2 imports.

Run from anywhere on the Mac:
    ./venv/bin/python surrogate/check_setup.py
Exit code 0 = all checks pass. Read-only: writes nothing.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((name, True, str(detail)))
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        results.append((name, False, f"{type(exc).__name__}: {exc}"))


def python_version():
    assert sys.version_info[:2] == (3, 13), f"need Python 3.13, got {sys.version.split()[0]}"
    return sys.version.split()[0]


def shared_venv():
    expected = (PROJECT / "venv").resolve()
    actual = Path(sys.prefix).resolve()
    assert actual == expected, f"running in {actual}, expected {expected} (use ./venv/bin/python)"
    return str(actual)


def path_files():
    site = Path(sysconfig.get_paths()["purelib"])
    wanted = {"simulation.pth": PROJECT / "simulation", "project.pth": PROJECT}
    for name, target in wanted.items():
        f = site / name
        assert f.exists(), f"{f} missing - re-run setup_venv.sh"
        content = Path(f.read_text(encoding="utf-8").strip()).resolve()
        assert content == target.resolve(), f"{name} points to {content}, expected {target}"
    return "simulation.pth, project.pth"


def phase2_imports():
    os.chdir(PROJECT / "surrogate")        # prove it works from the new folder, not only from simulation/
    import climate.epw_parser             # noqa: F401
    import dataset.assemble               # noqa: F401
    import runner.output_parser           # noqa: F401
    import sampling                       # noqa: F401
    import idf_generator                  # noqa: F401
    return "dataset, climate, runner, sampling, idf_generator"


def eda_namespace_import():
    from eda import eda_lib                # eda/ has no __init__.py -> namespace package (plan P-L4)
    from surrogate.paths import SIM_ROOT
    assert Path(eda_lib.ROOT).resolve() == SIM_ROOT.resolve(), f"eda_lib.ROOT = {eda_lib.ROOT}"
    return f"eda_lib.ROOT = {eda_lib.ROOT}"


def surrogate_import():
    import surrogate.paths as p
    for name in ("DATASET_DIR", "DESIGN_SPACE", "CLIMATE_CONFIG", "EPW_DIR", "OAT_DIR"):
        assert getattr(p, name).exists(), f"{name} not found: {getattr(p, name)}"
    return "all Phase 2 paths exist"


def dataset_loads():
    from dataset.assemble import load_dataset
    df, cols = load_dataset("training")    # sha256-verified against the manifest
    assert len(cols["inputs"]) == 28 and len(cols["targets"]) == 10, "unexpected column roles"
    return f"{len(df)} training rows, 28 inputs, 10 targets, sha256 OK"


def versions_match_phase2():
    import numpy, pandas, sklearn, scipy, shap  # noqa: E401
    want = {"numpy": "2.5.3", "pandas": "3.0.6", "sklearn": "1.9.1", "scipy": "1.18.1", "shap": "0.52.0"}
    got = {"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
           "scipy": scipy.__version__, "shap": shap.__version__}
    diff = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    assert not diff, f"differs from simulation/venv: {diff}"
    return ", ".join(f"{k} {v}" for k, v in got.items())


for name, fn in [("Python 3.13", python_version), ("Shared venv in use", shared_venv),
                 ("Path files in venv", path_files), ("Phase 2 imports", phase2_imports),
                 ("eda namespace import", eda_namespace_import), ("surrogate package + paths", surrogate_import),
                 ("Frozen dataset loads", dataset_loads), ("Same versions as Phase 2", versions_match_phase2)]:
    check(name, fn)

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
n_fail = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
