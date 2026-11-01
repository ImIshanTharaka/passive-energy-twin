#!/usr/bin/env python3
"""
batch_runner.py - Run a sample file through EnergyPlus in parallel (2.5 pilot, 2.6 campaign).

One sample row = one run:
    row -> build_idf(idf_params(row)) -> <out>/<run_id>/in.idf
        -> energyplus -x -w <epw> -d <out>/<run_id> in.idf
        -> status line in <out>/run_log.csv

FEATURES
--------
  Parallel     ProcessPoolExecutor; each worker builds its own IDFs (eppy's IDD
               is a class attribute, set once per worker process).
  Isolation    every run gets its own directory, so runs never share files.
  Checkpoint   run_log.csv is append-only and flushed after every run. On a
               restart, run_ids already logged as "ok" are skipped (resume).
  Retry        a failed EnergyPlus run is retried once in a clean directory.
               A failed IDF build is NOT retried (it is deterministic).
  Timeout      a run exceeding --timeout seconds (default 300, Decision T3) is killed
               and counted as failed.
  Retention    --keep sql  (default) keep eplusout.sql/.err/.end, delete the rest
               --keep all  keep everything
               --keep none keep only .err/.end
               Failed runs always keep everything, for debugging.

USAGE (from the repository root)
-----
  python runner/batch_runner.py --samples sampling/samples_pilot.csv --per-climate 1   # smoke: 1 run per climate
  python runner/batch_runner.py --samples sampling/samples_pilot.csv                   # whole pilot
  python runner/batch_runner.py --samples sampling/samples_pilot.csv --run-ids pilot_P01_colombo
  python runner/batch_runner.py --samples ... --rerun-failed       # retry rows logged as failed

Output folder defaults to outputs/<batch>/ (batch = the CSV's batch column).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_NAME = "run_log.csv"
META_NAME = "run_meta.json"
LOG_COLUMNS = [
    "run_id", "batch", "design_id", "climate_key", "status", "attempts", "returncode",
    "warnings", "severe", "fatal", "build_s", "sim_s", "wall_s", "worker_pid",
    "finished_at", "run_dir", "message",
]
KEEP_SQL = {"eplusout.sql", "eplusout.err", "eplusout.end"}
KEEP_NONE = {"eplusout.err", "eplusout.end"}
DEFAULT_BUILDER = "runner.batch_runner:eppy_build"

# eplusout.end, e.g. "EnergyPlus Completed Successfully-- 7 Warning; 0 Severe Errors; Elapsed Time=..."
_END_RE = re.compile(r"(\d+)\s+Warning;\s*(\d+)\s+Severe Errors", re.I)


# ---------------------------------------------------------------------------
# Configuration and per-run result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunnerConfig:
    energyplus: str                       # path to the energyplus executable
    idd: str                              # Energy+.idd (for eppy)
    out_dir: str                          # batch output folder
    epw_paths: dict                       # climate_key -> EPW path
    timeout_s: float = 300.0             # T3: pilot max 65.8 s (8 workers) -> 4.5x margin
    retries: int = 1                      # extra attempts after a failed simulation
    keep: str = "sql"                     # sql | all | none
    builder: str = DEFAULT_BUILDER        # "module:function"(params, idd) -> IDF text


@dataclass
class RunResult:
    run_id: str
    batch: str
    design_id: str
    climate_key: str
    status: str = "failed"                # ok | failed | build_error
    attempts: int = 0
    returncode: int | None = None
    warnings: int | None = None
    severe: int | None = None
    fatal: bool = False
    build_s: float = 0.0
    sim_s: float = 0.0
    wall_s: float = 0.0
    worker_pid: int = 0
    finished_at: str = ""
    run_dir: str = ""
    message: str = ""

    def log_row(self) -> dict:
        d = asdict(self)
        for k in ("build_s", "sim_s", "wall_s"):
            d[k] = f"{d[k]:.2f}"
        d["fatal"] = int(self.fatal)
        d["message"] = " ".join(str(d["message"]).split())[:300]
        return {k: ("" if d[k] is None else d[k]) for k in LOG_COLUMNS}


# ---------------------------------------------------------------------------
# IDF building (runs inside the worker)
# ---------------------------------------------------------------------------

def eppy_build(params: dict, idd: str) -> str:
    """Default builder: the 2.2 generator. Returns the IDF as text.

    Written WITHOUT eppy's '!- field name' comments: eppy's default writer looks
    each field up with a linear list.index (~145,000 calls per IDF), which took
    8-9 s of the ~11 s build time in the 2.5 pilot. The comment-free text has the
    same objects, order and values (checked field by field on 3 designs, numbers
    compared as numbers: '200' vs '200.0'); EnergyPlus ignores comments.
    """
    from idf_generator import build_idf
    idf = build_idf(params, idd)
    idf.outputtype = "nocomment"
    return idf.idfstr()


_BUILDER_CACHE: dict[str, object] = {}


def _resolve_builder(spec: str):
    fn = _BUILDER_CACHE.get(spec)
    if fn is None:
        module, _, name = spec.partition(":")
        fn = getattr(importlib.import_module(module), name)
        _BUILDER_CACHE[spec] = fn
    return fn


def _worker_init(root: str) -> None:
    """Each worker: make the repo importable. eppy's IDD is then set on the
    worker's first build_idf() call and reused (same path) for every later one."""
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def parse_end_file(run_dir: Path) -> tuple[bool, int | None, int | None, bool]:
    """(completed_successfully, warnings, severe, fatal) from eplusout.end."""
    end = run_dir / "eplusout.end"
    if not end.exists():
        return False, None, None, False
    text = end.read_text(errors="replace")
    m = _END_RE.search(text)
    warnings, severe = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return "Completed Successfully" in text, warnings, severe, "Fatal" in text


def _prune(run_dir: Path, keep: str) -> None:
    if keep == "all":
        return
    wanted = KEEP_SQL if keep == "sql" else KEEP_NONE
    for p in run_dir.iterdir():
        if p.name in wanted:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def run_one(row: dict, params: dict, cfg: RunnerConfig) -> RunResult:
    """Build, simulate, check and tidy one run. Never raises."""
    t0 = time.perf_counter()
    res = RunResult(row["run_id"], row["batch"], row["design_id"], row["climate_key"],
                    worker_pid=os.getpid())
    run_dir = Path(cfg.out_dir) / row["run_id"]
    res.run_dir = str(run_dir.relative_to(cfg.out_dir))
    try:
        epw = cfg.epw_paths[row["climate_key"]]
        tb = time.perf_counter()
        idf_text = _resolve_builder(cfg.builder)(params, cfg.idd)
        res.build_s = time.perf_counter() - tb
    except Exception as e:                                    # noqa: BLE001
        res.status, res.message = "build_error", f"{type(e).__name__}: {e}"
        return _finish(res, t0)

    for attempt in range(1, cfg.retries + 2):
        res.attempts = attempt
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        (run_dir / "in.idf").write_text(idf_text)
        cmd = [cfg.energyplus, "-x", "-w", str(epw), "-d", str(run_dir), str(run_dir / "in.idf")]
        ts = time.perf_counter()
        try:
            with (run_dir / "stdout.log").open("w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=cfg.timeout_s, cwd=run_dir)
            res.returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            res.returncode, timed_out = None, True
        except OSError as e:                                  # e.g. executable missing
            res.sim_s = time.perf_counter() - ts
            res.message = f"cannot start EnergyPlus: {e}"
            return _finish(res, t0)
        res.sim_s = time.perf_counter() - ts

        done, res.warnings, res.severe, res.fatal = parse_end_file(run_dir)
        sql_ok = (run_dir / "eplusout.sql").is_file()
        if res.returncode == 0 and done and sql_ok and not res.fatal:
            res.status, res.message = "ok", ""
            _prune(run_dir, cfg.keep)
            return _finish(res, t0)
        if timed_out:
            res.message = f"timeout after {cfg.timeout_s:.0f}s"
        elif not done:
            res.message = f"returncode={res.returncode}; not 'Completed Successfully'"
        elif not sql_ok:
            res.message = "eplusout.sql missing"
        else:
            res.message = "fatal error reported"
    res.status = "failed"                                     # keep the folder for debugging
    return _finish(res, t0)


def _finish(res: RunResult, t0: float) -> RunResult:
    res.wall_s = time.perf_counter() - t0
    res.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return res


# ---------------------------------------------------------------------------
# Run log (checkpoint)
# ---------------------------------------------------------------------------

def read_log(path: Path) -> dict[str, dict]:
    """Latest log line per run_id (a rerun appends a newer line)."""
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


class RunLog:
    """Append-only CSV, flushed and fsynced after every line."""

    def __init__(self, path: Path):
        new = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=LOG_COLUMNS)
        if new:
            self._w.writeheader()
            self._flush()

    def write(self, res: RunResult) -> None:
        self._w.writerow(res.log_row())
        self._flush()

    def _flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Selecting rows
# ---------------------------------------------------------------------------

def select_rows(rows: list[dict], done: dict[str, dict], *, run_ids=None, per_climate=None,
                limit=None, rerun_failed=False) -> tuple[list[dict], int]:
    """Rows still to run, in file order. Returns (todo, n_skipped_as_done)."""
    ids = [r["run_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run_id in samples file")
    if run_ids:
        unknown = set(run_ids) - set(ids)
        if unknown:
            raise ValueError(f"unknown run_ids: {sorted(unknown)}")
        rows = [r for r in rows if r["run_id"] in set(run_ids)]
    if per_climate:
        seen: dict[str, int] = {}
        picked = []
        for r in rows:
            if seen.get(r["climate_key"], 0) < per_climate:
                seen[r["climate_key"]] = seen.get(r["climate_key"], 0) + 1
                picked.append(r)
        rows = picked
    todo, skipped = [], 0
    for r in rows:
        prev = done.get(r["run_id"], {}).get("status")
        if prev == "ok" or (prev in ("failed", "build_error") and not rerun_failed):
            skipped += 1
        else:
            todo.append(r)
    if limit is not None:
        todo = todo[:limit]
    return todo, skipped


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def run_batch(rows: list[dict], params_of, cfg: RunnerConfig, *, workers: int = 8,
              progress: bool = True) -> list[RunResult]:
    """Run rows in parallel; append each result to run_log.csv as it finishes."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / LOG_NAME)
    results: list[RunResult] = []
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(rows), unit="run", dynamic_ncols=True)
        except ImportError:
            pass
    n_fail = 0
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                               initargs=(str(ROOT),))
    try:
        futures = {pool.submit(run_one, r, params_of(r), cfg): r for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            log.write(res)
            results.append(res)
            n_fail += res.status != "ok"
            if bar is not None:
                bar.update(1)
                bar.set_postfix(failed=n_fail, last=f"{res.sim_s:.0f}s")
            elif progress:
                print(f"[{len(results)}/{len(rows)}] {res.status:11} {res.run_id} "
                      f"({res.wall_s:.1f}s) {res.message}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted - finished runs are logged; rerun the same command to resume.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        log.close()
        if bar is not None:
            bar.close()
    return results


def summarise(results: list[RunResult]) -> str:
    if not results:
        return "nothing to run"
    ok = [r for r in results if r.status == "ok"]
    sims = sorted(r.sim_s for r in ok)
    lines = [f"runs: {len(results)}   ok: {len(ok)}   failed: "
             f"{sum(r.status == 'failed' for r in results)}   build_error: "
             f"{sum(r.status == 'build_error' for r in results)}"]
    if sims:
        lines.append(f"EnergyPlus time per run: median {sims[len(sims) // 2]:.1f}s, "
                     f"max {sims[-1]:.1f}s")
        lines.append(f"warnings per run: max {max(r.warnings or 0 for r in ok)};  "
                     f"severe in ok runs: {sum(r.severe or 0 for r in ok)}")
    for r in results:
        if r.status != "ok":
            lines.append(f"  {r.status:11} {r.run_id}: {r.message}")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_meta(cfg: RunnerConfig, samples: Path, workers: int, n_todo: int,
               started: str, finished: str, results: list[RunResult]) -> None:
    """run_meta.json: one entry per invocation (appended), for provenance."""
    path = Path(cfg.out_dir) / META_NAME
    history = json.loads(path.read_text()) if path.exists() else []
    history.append({
        "started": started, "finished": finished,
        "samples": str(samples), "samples_sha256": _sha256(samples),
        "energyplus": cfg.energyplus, "idd": cfg.idd, "workers": workers,
        "timeout_s": cfg.timeout_s, "retries": cfg.retries, "keep": cfg.keep,
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(),
        "n_submitted": n_todo, "n_ok": sum(r.status == "ok" for r in results),
    })
    path.write_text(json.dumps(history, indent=2) + "\n")


def energyplus_exe() -> str:
    override = os.environ.get("ENERGYPLUS_EXE")
    if override:
        return override
    f = ROOT / "energyplus_path.txt"
    return f.read_text().strip() if f.exists() else "/Applications/EnergyPlus-24-1-0/energyplus"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="output folder (default outputs/<batch>)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--keep", choices=("sql", "all", "none"), default="sql")
    ap.add_argument("--limit", type=int, help="run at most N rows (after skipping done ones)")
    ap.add_argument("--per-climate", type=int, help="first N rows of each climate")
    ap.add_argument("--run-ids", nargs="+")
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    a = ap.parse_args(argv)

    from climate.climate_features import load_climate_registry
    from sampling.sampler import idf_params, load_samples, resolve_idd
    from sampling.transform import load_design_space

    space = load_design_space()
    rows = load_samples(a.samples, space)
    batches = {r["batch"] for r in rows}
    out = a.out or ROOT / "outputs" / (batches.pop() if len(batches) == 1 else a.samples.stem)
    registry = load_climate_registry()
    missing = {r["climate_key"] for r in rows} - set(registry)
    if missing:
        print(f"climates not in registry: {sorted(missing)}")
        return 2

    cfg = RunnerConfig(
        energyplus=energyplus_exe(), idd=resolve_idd(), out_dir=str(out.resolve()),
        epw_paths={k: str(v.epw_path) for k, v in registry.items()},
        timeout_s=a.timeout, retries=a.retries, keep=a.keep)
    for label, p in (("EnergyPlus", cfg.energyplus), ("IDD", cfg.idd)):
        if not Path(p).exists():
            print(f"{label} not found: {p}")
            return 2

    done = read_log(out / LOG_NAME)
    todo, skipped = select_rows(rows, done, run_ids=a.run_ids, per_climate=a.per_climate,
                                limit=a.limit, rerun_failed=a.rerun_failed)
    print(f"samples: {a.samples}  ({len(rows)} rows)\noutput:  {out}\n"
          f"already logged (skipped): {skipped}   to run: {len(todo)}   workers: {a.workers}")
    if a.dry_run or not todo:
        for r in todo:
            print("  ", r["run_id"])
        return 0

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    results = run_batch(todo, lambda r: idf_params(r, space), cfg, workers=a.workers)
    wall = time.perf_counter() - t0
    write_meta(cfg, a.samples, a.workers, len(todo), started,
               datetime.now(timezone.utc).isoformat(timespec="seconds"), results)
    print(summarise(results))
    print(f"wall-clock: {wall / 60:.1f} min  ->  {wall / max(len(todo), 1):.1f}s per run "
          f"with {a.workers} workers")
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
