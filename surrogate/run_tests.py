#!/usr/bin/env python3
"""
run_tests.py - run every surrogate test suite (surrogate/tests/test_*.py) and summarise.

Same format as simulation/run_tests.py; each suite runs in its own interpreter.
Run with the shared venv (the suites import `surrogate` and the Phase 2 packages):

    ./venv/bin/python surrogate/run_tests.py          # all suites
    ./venv/bin/python surrogate/run_tests.py cv       # suites whose name contains "cv"
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
SUITES = sorted(TESTS.glob("test_*.py"))


def main() -> int:
    wanted = sys.argv[1:]
    suites = [s for s in SUITES if not wanted or any(w in s.name for w in wanted)]
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
            status = "CRASH" + (f"  ({errors[-1].strip()[:60]})" if errors else "")
        ok = proc.returncode == 0 and bool(match)
        all_ok &= ok
        print(f"{'surrogate/tests/' + suite.name:52}{status:>16}{elapsed:7.1f}s")
        if not ok:
            for line in out.splitlines():
                if line.strip().startswith(("FAIL", "ERROR")) or "Error" in line:
                    print(f"      {line.strip()[:110]}")
    print("-" * 76)
    print("ALL SUITES PASSED" if all_ok else "SOME SUITES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
