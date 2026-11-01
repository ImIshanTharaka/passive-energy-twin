"""
test_batch_runner.py - Checks for runner/batch_runner.py (Sub-task 2.5.1).

EnergyPlus is replaced by a FAKE executable (a small Python script) whose
behaviour is chosen by a MODE line in the IDF text:
    ok         writes eplusout.sql/.err/.end + junk files, "Completed Successfully"
    fail_once  fails on the first attempt of a run, succeeds on the retry
    fail       always exits 1
    fatal      exits 0 but eplusout.end reports a Fatal error
    hang       sleeps longer than the timeout
    nosql      completes but writes no eplusout.sql
The IDF builder is also faked (no eppy needed). So these checks cover the
runner's own logic: parallelism, isolation, retry, timeout, success criteria,
file retention, checkpoint/resume and row selection - not EnergyPlus itself.

Run:  python runner/test_batch_runner.py
"""

from __future__ import annotations

import csv
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import batch_runner as B          # noqa: E402

FAKE_EP = r'''#!{python}
import os, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "-x", "runner must pass -x (ExpandObjects)"
d = Path(args[args.index("-d") + 1]); idf = Path(args[-1])
assert Path(args[args.index("-w") + 1]).name.endswith(".epw")
mode = [l.split("=", 1)[1].strip() for l in idf.read_text().splitlines() if l.startswith("MODE=")][0]
run_id = d.name
state = Path(os.environ["FAKE_EP_STATE"]); state.mkdir(exist_ok=True)
(state / f"{{run_id}}.{{os.getpid()}}").write_text("")            # one file per attempt
def ok(warn=3):
    (d / "eplusout.sql").write_bytes(b"SQLite format 3\x00")
    (d / "eplusout.err").write_text("** Warning ** fake\n")
    (d / "eplusout.eso").write_text("junk")
    (d / "eplusout.end").write_text(f"EnergyPlus Completed Successfully-- {{warn}} Warning; 0 Severe Errors; Elapsed Time=00hr 00min  1.00sec\n")
    sys.exit(0)
if mode == "ok": ok()
if mode == "fail_once":
    if len(list(state.glob(f"{{run_id}}.*"))) >= 2: ok(1)
    sys.exit(1)
if mode == "fail":
    (d / "eplusout.end").write_text("EnergyPlus Terminated--Fatal Error Detected. 0 Warning; 2 Severe Errors; Elapsed Time=0\n")
    sys.exit(1)
if mode == "fatal":
    (d / "eplusout.sql").write_bytes(b"x")
    (d / "eplusout.end").write_text("EnergyPlus Completed Successfully-- 0 Warning; 1 Severe Errors; Fatal\n")
    sys.exit(0)
if mode == "hang": time.sleep(30)
if mode == "nosql":
    (d / "eplusout.end").write_text("EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;\n")
    sys.exit(0)
'''


def fake_build(params: dict, idd: str) -> str:
    """Stand-in for build_idf: the IDF text carries the fake EnergyPlus MODE."""
    if params.get("mode") == "build_error":
        raise ValueError("bad parameter combination")
    return f"! fake idf\nMODE={params['mode']}\nVersion,24.1;\n"


class Env:
    """Temp folder with a fake energyplus, fake EPWs and a RunnerConfig."""

    def __init__(self, timeout=20.0, keep="sql", retries=1):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        exe = t / "energyplus"
        exe.write_text(FAKE_EP.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        os.environ["FAKE_EP_STATE"] = str(t / "state")
        epws = {}
        for k in ("colombo", "phoenix", "minneapolis"):
            (t / f"{k}.epw").write_text("fake")
            epws[k] = str(t / f"{k}.epw")
        self.out = t / "out"
        self.cfg = B.RunnerConfig(energyplus=str(exe), idd="unused.idd", out_dir=str(self.out),
                                  epw_paths=epws, timeout_s=timeout, retries=retries, keep=keep,
                                  builder="runner.test_batch_runner:fake_build")

    def attempts(self, run_id):
        return len(list((Path(self.tmp.name) / "state").glob(f"{run_id}.*")))

    def log(self):
        with (self.out / B.LOG_NAME).open(newline="") as fh:
            return list(csv.DictReader(fh))


def rows_for(modes: dict[str, str], climate="colombo") -> list[dict]:
    return [{"run_id": rid, "batch": "pilot", "design_id": rid, "climate_key": climate,
             "mode": m} for rid, m in modes.items()]


def params_of(r):
    return {"mode": r["mode"]}


def run(env, rows, workers=4):
    return {r.run_id: r for r in B.run_batch(rows, params_of, env.cfg, workers=workers,
                                              progress=False)}


# ---------------------------------------------------------------------------

def check_successful_runs_are_logged_ok():
    env = Env()
    res = run(env, rows_for({f"r{i}": "ok" for i in range(12)}))
    assert all(r.status == "ok" for r in res.values()), res
    assert all(r.attempts == 1 and r.warnings == 3 and r.severe == 0 for r in res.values())
    log = env.log()
    assert len(log) == 12 and {r["status"] for r in log} == {"ok"}
    assert list(log[0]) == B.LOG_COLUMNS


def check_runs_are_parallel_and_isolated():
    env = Env()
    res = run(env, rows_for({f"r{i}": "ok" for i in range(16)}), workers=4)
    assert len({r.worker_pid for r in res.values()}) > 1, "only one worker process used"
    dirs = {p.name for p in env.out.iterdir() if p.is_dir()}
    assert dirs == {f"r{i}" for i in range(16)}, "each run must get its own folder"


def check_keep_sql_prunes_successful_runs():
    env = Env(keep="sql")
    run(env, rows_for({"a": "ok"}))
    assert {p.name for p in (env.out / "a").iterdir()} == B.KEEP_SQL


def check_keep_all_and_none():
    env = Env(keep="all")
    run(env, rows_for({"a": "ok"}))
    names = {p.name for p in (env.out / "a").iterdir()}
    assert {"in.idf", "eplusout.eso", "stdout.log"} <= names
    env = Env(keep="none")
    run(env, rows_for({"a": "ok"}))
    assert {p.name for p in (env.out / "a").iterdir()} == B.KEEP_NONE


def check_retry_recovers_a_transient_failure():
    env = Env(retries=1)
    r = run(env, rows_for({"a": "fail_once"}))["a"]
    assert r.status == "ok" and r.attempts == 2 and env.attempts("a") == 2


def check_persistent_failure_keeps_files_for_debugging():
    env = Env(retries=1)
    r = run(env, rows_for({"a": "fail"}))["a"]
    assert r.status == "failed" and r.attempts == 2 and r.severe == 2 and r.fatal
    assert (env.out / "a" / "in.idf").exists() and (env.out / "a" / "stdout.log").exists()


def check_no_retry_when_retries_zero():
    env = Env(retries=0)
    r = run(env, rows_for({"a": "fail"}))["a"]
    assert r.status == "failed" and r.attempts == 1 and env.attempts("a") == 1


def check_fatal_and_missing_sql_are_failures():
    env = Env(retries=0)
    res = run(env, rows_for({"f": "fatal", "n": "nosql"}))
    assert res["f"].status == "failed" and "fatal" in res["f"].message
    assert res["n"].status == "failed" and "sql" in res["n"].message


def check_timeout_kills_run():
    env = Env(timeout=1.5, retries=0)
    r = run(env, rows_for({"h": "hang"}))["h"]
    assert r.status == "failed" and "timeout" in r.message and r.wall_s < 10


def check_build_error_is_not_retried_and_does_not_stop_batch():
    env = Env()
    res = run(env, rows_for({"b": "build_error", "a": "ok"}))
    assert res["b"].status == "build_error" and res["b"].attempts == 0
    assert "bad parameter" in res["b"].message and env.attempts("b") == 0
    assert res["a"].status == "ok"


def check_missing_executable_is_reported():
    env = Env()
    cfg = B.RunnerConfig(**{**env.cfg.__dict__, "energyplus": "/nonexistent/energyplus"})
    r = B.run_one(rows_for({"a": "ok"})[0], {"mode": "ok"}, cfg)
    assert r.status == "failed" and "cannot start" in r.message


def check_resume_skips_completed_runs():
    env = Env(retries=0)
    rows = rows_for({"a": "ok", "b": "ok", "c": "fail"})
    run(env, rows[:2])                              # first session: a, b done
    todo, skipped = B.select_rows(rows, B.read_log(env.out / B.LOG_NAME))
    assert [r["run_id"] for r in todo] == ["c"] and skipped == 2
    run(env, todo)                                  # c fails
    todo, skipped = B.select_rows(rows, B.read_log(env.out / B.LOG_NAME))
    assert todo == [] and skipped == 3, "failed runs are not rerun by default"
    todo, _ = B.select_rows(rows, B.read_log(env.out / B.LOG_NAME), rerun_failed=True)
    assert [r["run_id"] for r in todo] == ["c"]
    assert len(env.log()) == 3 and env.attempts("a") == 1, "log is append-only, no rework"


def check_rerun_appends_and_latest_line_wins():
    env = Env(retries=0)
    rows = rows_for({"c": "fail"})
    run(env, rows)
    rows[0]["mode"] = "ok"                          # e.g. generator fixed
    todo, _ = B.select_rows(rows, B.read_log(env.out / B.LOG_NAME), rerun_failed=True)
    run(env, todo)
    assert [r["status"] for r in env.log()] == ["failed", "ok"]
    assert B.read_log(env.out / B.LOG_NAME)["c"]["status"] == "ok"


def check_row_selection():
    rows = (rows_for({"c1": "ok", "c2": "ok", "c3": "ok"}, "colombo")
            + rows_for({"p1": "ok", "p2": "ok"}, "phoenix"))
    todo, _ = B.select_rows(rows, {}, per_climate=1)
    assert [r["run_id"] for r in todo] == ["c1", "p1"]
    todo, _ = B.select_rows(rows, {}, run_ids=["p2", "c3"])
    assert [r["run_id"] for r in todo] == ["c3", "p2"], "file order is kept"
    todo, _ = B.select_rows(rows, {}, limit=2)
    assert len(todo) == 2
    for bad in (dict(run_ids=["nope"]),):
        try:
            B.select_rows(rows, {}, **bad)
            raise AssertionError("unknown run_id accepted")
        except ValueError:
            pass
    try:
        B.select_rows(rows + rows[:1], {})
        raise AssertionError("duplicate run_id accepted")
    except ValueError:
        pass


def check_end_file_parsing_real_format():
    """Against a real EnergyPlus 24.1 eplusout.end from the smoke runs, if present."""
    real = ROOT / "outputs" / "smoke_orient_000"
    if not (real / "eplusout.end").exists():
        return
    done, w, s, fatal = B.parse_end_file(real)
    assert done and w is not None and s == 0 and not fatal, (done, w, s, fatal)


def check_meta_is_appended():
    env = Env()
    samples = Path(env.tmp.name) / "s.csv"
    samples.write_text("x")
    for _ in range(2):
        env.out.mkdir(exist_ok=True)
        B.write_meta(env.cfg, samples, 4, 1, "t0", "t1", [])
    import json
    meta = json.loads((env.out / B.META_NAME).read_text())
    assert len(meta) == 2 and meta[0]["samples_sha256"] and meta[0]["keep"] == "sql"


ALL_CHECKS = [
    check_successful_runs_are_logged_ok,
    check_runs_are_parallel_and_isolated,
    check_keep_sql_prunes_successful_runs,
    check_keep_all_and_none,
    check_retry_recovers_a_transient_failure,
    check_persistent_failure_keeps_files_for_debugging,
    check_no_retry_when_retries_zero,
    check_fatal_and_missing_sql_are_failures,
    check_timeout_kills_run,
    check_build_error_is_not_retried_and_does_not_stop_batch,
    check_missing_executable_is_reported,
    check_resume_skips_completed_runs,
    check_rerun_appends_and_latest_line_wins,
    check_row_selection,
    check_end_file_parsing_real_format,
    check_meta_is_appended,
]


def main():
    total = failed = 0
    print("=" * 72)
    print("BATCH RUNNER (2.5.1) - CHECKS")
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
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
