#!/usr/bin/env python3
"""
err_parser.py - Parse EnergyPlus eplusout.err files into structured messages (Sub-task 2.5.2).

eplusout.end only gives totals ("3 Warning; 0 Severe Errors"). eplusout.err has
the messages themselves. This module turns one .err into an ErrReport:

    level      Warning | Severe | Fatal
    text       the first line of the message
    details    its "~~~" continuation lines
    category   a short, stable label from CATEGORY_RULES (e.g. "interzone_construction_reverse")
    signature  the text with names and numbers masked - groups identical message
               types across runs even when zone/surface names differ
    recurring  True if it came from the "Recurring Error Summary" block
    count      occurrences (recurring messages: "This error occurred N total times")

and aggregates a whole batch folder into two CSVs, so warnings can be joined to
the sample parameters (step 2.5.6: "does this warning only appear with 5 floors?"):

    <batch>/err_summary.csv    one row per run: totals + category counts
    <batch>/err_messages.csv   one row per message

USAGE (from the repository root)
-----
    python runner/err_parser.py outputs/pilot                 # aggregate a batch folder
    python runner/err_parser.py outputs/pilot/<run_id>/eplusout.err   # inspect one file
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Categories. First match wins; order specific -> general. "benign" = expected
# for this model and accepted (documented); anything else needs a look.
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, str, bool]] = [
    # (category, regex on the message text + details, benign)
    ("world_coordinate_system", r"World Coordinate System selected", True),
    ("interzone_construction_reverse", r"does not have the same materials in the reverse order", True),
    ("weather_location_mismatch", r"Weather file location will be used|Location .* does not match", True),
    # 1-storey designs never use the interior floor/ceiling constructions the generator
    # always defines; EnergyPlus notes them as unused (2.5 pilot, 11 designs). No effect.
    ("unused_construction", r"CheckUsedConstructions|nominally unused constructions", True),
    # 2.6 smoke: EnergyPlus' outdoor wet-bulb solver (PsyTwbFnTdbWPb -> PsyPsatFnTemp) fails to
    # converge at one timestep per climate (Denver 10/14 01:00, New York 01/07 14:00; near 0 C).
    # Same timestep and same bad value in every design -> weather-driven, not building-driven.
    # Nothing in this model uses outdoor wet-bulb (ideal loads; no evaporative/tower equipment),
    # so the targets are unaffected. Matched on the routine name in the details.
    ("outdoor_wetbulb_psychrometric", r"PsyTwbFnTdbWPb", True),
    ("input_enum", r"Failed to match against any enum", False),
    ("input_schema", r"<root>\[|Missing required property|Value type .* for input|"
                     r"not a valid|Object type .* unknown|invalid", False),
    ("inside_surface_temperature", r"CalcHeatBalanceInsideSurf|Temperature \(?(low|high)\)? out of bounds|"
                                   r"inside surface temperature", False),
    ("outside_surface_temperature", r"CalcHeatBalanceOutsideSurf|outside surface temperature", False),
    ("shading_geometry", r"CHKSBS|DetermineShadowingCombinations|shadow|Shading Surface|"
                         r"not.*(convex|planar)|Too many figures", False),
    ("subsurface_geometry", r"subsurface|SubSurface|base surface|window.*(outside|extends)|"
                            r"Fenestration.*(outside|area)", False),
    ("surface_geometry", r"GetSurfaceData|GetVertices|surface.*(collinear|degenerate|area)|"
                         r"Zone volume|zone.*volume|CalculateZoneVolume|enclosure", False),
    ("convergence", r"did not converge|Maximum iterations|exceeded.*iterations|convergence", False),
    ("ideal_loads", r"IdealLoadsAirSystem|Ideal Loads", False),
    ("hvac_template", r"HVACTemplate|ExpandObjects", False),
    ("internal_gains", r"GetInternalHeatGains|People|Lights|ElectricEquipment", False),
    ("infiltration", r"Infiltration", False),
    ("window_glazing", r"WindowMaterial|Glazing|SimpleGlazingSystem|Window", False),
    ("construction_material", r"Construction|Material|thermal resistance|conductivity", False),
    ("schedule", r"Schedule", False),
    ("output_request", r"Output:Variable|Output:Meter|Output:Table|requested.*(variable|meter)|"
                       r"not found.*(variable|meter)", False),
    ("sizing", r"Sizing", False),
    ("timestep", r"Timestep|time step", False),
]
_COMPILED = [(c, re.compile(p, re.I), b) for c, p, b in CATEGORY_RULES]
# Boilerplate in recurring-summary entries; must not drive categorisation
# (before 2.6, "during Sizing 0 times." made every unmatched recurring message "sizing").
_RECUR_BOILERPLATE = re.compile(r"^during (Warmup|Sizing) \d+ times|^Max=.*Min=", re.I)
BENIGN = {c for c, _, b in CATEGORY_RULES if b}
FATAL_CATEGORY = "fatal_termination"

_MSG_RE = re.compile(r"^\s*(?:\*{13}\s+)?\*\*\s*(Warning|Severe|Fatal)\s*\*\*\s?(.*)$")
_CONT_RE = re.compile(r"^\s*(?:\*{13}\s+)?\*\*\s+~~~\s+\*\*\s?(.*)$")
_RECUR_HDR = re.compile(r"Recurring Error Summary", re.I)
_RECUR_LINE = re.compile(r"^\s*\*{13}\s+\*\*\s*(Warning|Severe|Fatal)\s*\*\*")
_OCCURRED = re.compile(r"This error occurred\s+(\d+)\s+total times", re.I)
_FINAL = re.compile(r"EnergyPlus (Completed Successfully|Terminated--Fatal Error Detected)"
                    r".*?(\d+)\s+Warning;\s*(\d+)\s+Severe Errors"
                    r"(?:;\s*Elapsed Time=(\d+)hr\s+(\d+)min\s+([\d.]+)sec)?")
_VERSION = re.compile(r"Version\s+([\d.]+)")


@dataclass
class ErrMessage:
    level: str
    text: str
    details: list[str] = field(default_factory=list)
    recurring: bool = False
    count: int = 1
    category: str = ""
    signature: str = ""

    @property
    def benign(self) -> bool:
        return self.category in BENIGN


@dataclass
class ErrReport:
    path: str = ""
    version: str | None = None
    completed: bool = False
    terminated_fatal: bool = False
    reported_warnings: int | None = None      # from the final summary line
    reported_severe: int | None = None
    elapsed_s: float | None = None
    messages: list[ErrMessage] = field(default_factory=list)

    def by_level(self, level: str) -> list[ErrMessage]:
        return [m for m in self.messages if m.level == level]

    @property
    def n_fatal(self) -> int:
        return len(self.by_level("Fatal"))

    def category_counts(self, *, include_benign: bool = True) -> Counter:
        c: Counter = Counter()
        for m in self.messages:
            if include_benign or not m.benign:
                c[m.category] += 1
        return c

    @property
    def clean(self) -> bool:
        """Completed, no severe/fatal, and every warning in a benign category."""
        return (self.completed and not self.terminated_fatal and not self.reported_severe
                and all(m.benign for m in self.messages))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def signature(text: str) -> str:
    """Mask the run-specific parts so the same message type groups across runs."""
    s = re.sub(r'"[^"]*"', '"<s>"', text)                         # quoted values
    s = re.sub(r"\[[^\]]*\]", "[<s>]", s)                          # [paths], [units]
    s = re.sub(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", "<NAME>", s)   # ZONE_F1_CORE_FLOOR
    s = re.sub(r"\b[A-Z]{3,}[A-Z0-9]*\b", "<NAME>", s)              # INTERIORCEILINGFLOOR
    s = re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", "<n>", s)    # numbers
    s = re.sub(r"(?:<NAME>[\s,]*){2,}", "<NAME> ", s)
    return " ".join(s.split())[:200]


def categorise(msg: ErrMessage) -> str:
    if msg.level == "Fatal":
        return FATAL_CATEGORY
    hay = " ".join([msg.text, *[d for d in msg.details if not _RECUR_BOILERPLATE.search(d)]])
    for cat, rx, _ in _COMPILED:
        if rx.search(msg.text):                 # the headline decides first ...
            return cat
    for cat, rx, _ in _COMPILED:
        if rx.search(hay):                      # ... then the details
            return cat
    return "other"


def parse_err_text(text: str, path: str = "") -> ErrReport:
    rep = ErrReport(path=path)
    in_recurring = False
    current: ErrMessage | None = None
    for line in text.splitlines():
        if rep.version is None and line.startswith("Program Version"):
            m = _VERSION.search(line)
            rep.version = m.group(1) if m else None
            continue
        if _RECUR_HDR.search(line):
            in_recurring, current = True, None
            continue
        fm = _FINAL.search(line)
        if fm:
            rep.completed = fm.group(1).startswith("Completed")
            rep.terminated_fatal = not rep.completed
            rep.reported_warnings, rep.reported_severe = int(fm.group(2)), int(fm.group(3))
            if fm.group(4) is not None:
                rep.elapsed_s = int(fm.group(4)) * 3600 + int(fm.group(5)) * 60 + float(fm.group(6))
            current = None
            continue
        cm = _CONT_RE.match(line)
        if cm and current is not None:
            detail = cm.group(1).strip()
            occ = _OCCURRED.search(detail)
            if occ and current.recurring:
                current.count = int(occ.group(1))
            elif detail:
                current.details.append(detail)
            continue
        mm = _MSG_RE.match(line)
        if mm:
            current = ErrMessage(level=mm.group(1), text=mm.group(2).strip(),
                                 recurring=in_recurring and bool(_RECUR_LINE.match(line)))
            rep.messages.append(current)
            continue
        if line.strip().startswith("*************"):      # banner/status lines end a message
            current = None
    for m in rep.messages:
        m.category = categorise(m)
        m.signature = signature(m.text)
    # A recurring-summary entry repeats a message printed earlier in full (with its routine and
    # timestamp details): it inherits that message's category.
    first = {}
    for m in rep.messages:
        if not m.recurring:
            first.setdefault((m.level, m.signature), m.category)
    for m in rep.messages:
        if m.recurring and (m.level, m.signature) in first:
            m.category = first[(m.level, m.signature)]
    return rep


def parse_err(path: str | Path) -> ErrReport:
    p = Path(path)
    return parse_err_text(p.read_text(errors="replace"), str(p))


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = ["run_id", "version", "completed", "terminated_fatal", "reported_warnings",
                   "reported_severe", "n_fatal", "n_nonbenign_warnings", "clean", "elapsed_s",
                   "categories"]
MESSAGE_COLUMNS = ["run_id", "level", "category", "benign", "recurring", "count",
                   "signature", "text", "details"]


def summary_row(run_id: str, rep: ErrReport) -> dict:
    cats = rep.category_counts()
    return {
        "run_id": run_id, "version": rep.version or "", "completed": int(rep.completed),
        "terminated_fatal": int(rep.terminated_fatal),
        "reported_warnings": "" if rep.reported_warnings is None else rep.reported_warnings,
        "reported_severe": "" if rep.reported_severe is None else rep.reported_severe,
        "n_fatal": rep.n_fatal,
        "n_nonbenign_warnings": sum(1 for m in rep.by_level("Warning") if not m.benign),
        "clean": int(rep.clean),
        "elapsed_s": "" if rep.elapsed_s is None else f"{rep.elapsed_s:.2f}",
        "categories": ";".join(f"{k}:{v}" for k, v in sorted(cats.items())),
    }


def message_rows(run_id: str, rep: ErrReport) -> list[dict]:
    return [{"run_id": run_id, "level": m.level, "category": m.category, "benign": int(m.benign),
             "recurring": int(m.recurring), "count": m.count, "signature": m.signature,
             "text": m.text[:300], "details": " | ".join(m.details)[:500]}
            for m in rep.messages]


def aggregate(batch_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Parse <batch_dir>/<run_id>/eplusout.err for every run; write the two CSVs."""
    batch_dir = Path(batch_dir)
    summaries, messages = [], []
    for err in sorted(batch_dir.glob("*/eplusout.err")):
        run_id = err.parent.name
        rep = parse_err(err)
        summaries.append(summary_row(run_id, rep))
        messages.extend(message_rows(run_id, rep))
    for name, cols, rows in (("err_summary.csv", SUMMARY_COLUMNS, summaries),
                             ("err_messages.csv", MESSAGE_COLUMNS, messages)):
        with (batch_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return summaries, messages


def category_table(messages: list[dict], n_runs: int) -> str:
    """Text table: category -> level, benign, #runs affected, #messages."""
    runs: dict[tuple, set] = {}
    msgs: Counter = Counter()
    for m in messages:
        key = (m["category"], m["level"], m["benign"])
        runs.setdefault(key, set()).add(m["run_id"])
        msgs[key] += 1
    lines = [f"{'category':34}{'level':9}{'benign':>7}{'runs':>8}{'msgs':>7}", "-" * 65]
    for key in sorted(runs, key=lambda k: (-len(runs[k]), k)):
        cat, lvl, ben = key
        lines.append(f"{cat:34}{lvl:9}{'yes' if int(ben) else 'NO':>7}"
                     f"{len(runs[key]):>5}/{n_runs:<3}{msgs[key]:>6}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    target = Path(argv[0])
    if target.is_file():
        rep = parse_err(target)
        print(f"{target}\n  version {rep.version}  completed={rep.completed}  "
              f"warnings={rep.reported_warnings} severe={rep.reported_severe} "
              f"fatal={rep.n_fatal}  clean={rep.clean}")
        for m in rep.messages:
            tag = "benign" if m.benign else "CHECK"
            rec = f" (recurring x{m.count})" if m.recurring else ""
            print(f"  [{m.level:7}] {m.category:30} {tag:6}{rec}\n      {m.text[:150]}")
        return 0
    summaries, messages = aggregate(target)
    n = len(summaries)
    print(f"{target}: {n} runs parsed -> err_summary.csv, err_messages.csv")
    print(f"clean runs: {sum(int(s['clean']) for s in summaries)}/{n}   "
          f"with severe: {sum(1 for s in summaries if s['reported_severe'] not in ('', 0, '0'))}   "
          f"fatal: {sum(int(s['terminated_fatal']) for s in summaries)}\n")
    print(category_table(messages, n) if messages else "no warnings or errors in any run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
