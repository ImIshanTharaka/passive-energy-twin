#!/usr/bin/env python3
"""
run_tests.py - Run every test suite in the project and summarise.

Discovers idf_generator/*/test_*.py, climate/test_*.py, sampling/test_*.py,
runner/test_*.py, dataset/test_*.py and eda/test_*.py,
runs each in its own interpreter (so one crash cannot hide another suite's
result), and prints a single pass/fail table.

    python run_tests.py              # all suites
    python run_tests.py climate      # only suites whose path contains "climate"

The generator and sampling suites need the EnergyPlus IDD. It is found via
energyplus_path.txt, or override with:  EPLUS_IDD=/path/to/Energy+.idd
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = (sorted(ROOT.glob("idf_generator/*/test_*.py")) + sorted(ROOT.glob("climate/test_*.py"))
          + sorted(ROOT.glob("sampling/test_*.py")) + sorted(ROOT.glob("runner/test_*.py"))
          + sorted(ROOT.glob("dataset/test_*.py")) + sorted(ROOT.glob("eda/test_*.py")))


def main() -> int:
    wanted = sys.argv[1:]
    suites = [s for s in SUITES if not wanted or any(w in str(s) for w in wanted)]
    if not suites:
        print("no matching test suites")
        return 1

    print(f"Interpreter: {sys.executable}\n")
    print(f"{'suite':52}{'result':>16}{'time':>8}")
    print("-" * 76)
    all_ok = True
    for suite in suites:
        start = time.time()
        proc = subprocess.run([sys.executable, str(suite)], capture_output=True, text=True)
        elapsed = time.time() - start
        out = proc.stdout + proc.stderr
        match = re.search(r"RESULTS:\s*(\d+)/(\d+) passed", out)
        if match and proc.returncode == 0:
            status = f"{match.group(1)}/{match.group(2)} pass"
        elif match:
            status = f"{match.group(1)}/{match.group(2)} FAIL"
        else:
            errors = [ln for ln in out.splitlines() if "Error" in ln or "not found" in ln]
            status = "CRASH"
            if errors:
                status += f"  ({errors[-1].strip()[:60]})"
        ok = proc.returncode == 0 and bool(match)
        all_ok &= ok
        print(f"{str(suite.relative_to(ROOT)):52}{status:>16}{elapsed:7.1f}s")
        if not ok:
            for line in out.splitlines():
                if line.strip().startswith(("FAIL", "ERROR")) or "Error" in line:
                    print(f"      {line.strip()[:110]}")
    print("-" * 76)
    print("ALL SUITES PASSED" if all_ok else "SOME SUITES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
