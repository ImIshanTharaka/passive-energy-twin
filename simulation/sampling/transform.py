"""
transform.py - Unit-cube points -> physical design-parameter values (2.4).

The ONLY reader of config/design_space.yaml. Everything about a parameter -
its bounds, its rounding, whether it is an integer, whether it is derived -
comes from that file, so the sampler, the batch runner and (later) the web
backend cannot disagree about the design space.

    space = load_design_space()
    space.sampled_names     # 21 names, unit-cube column order = model inputs
    space.idf_names         # 22 names, what build_idf() needs (adds glazing_vt)
    rows  = space.to_physical(unit_points)   # list of dicts, rounded, + derived

Mapping (Decisions S2, S3):
    continuous   x = min + u * (max - min), rounded to `decimals`
    integer      x = min + floor(u * (max - min + 1))  -> each integer equally likely
    derived      x = value of the parameter named in `equal_to`

Rounding happens here, before anything is written, so the sample CSV holds
exactly the values EnergyPlus receives and the surrogate will be trained on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESIGN_SPACE = ROOT / "config" / "design_space.yaml"
SUPPORTED_SCHEMA = 2


class DesignSpaceError(Exception):
    """design_space.yaml is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    decimals: int
    is_integer: bool
    unit: str

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if np.any((u < 0) | (u >= 1)):
            raise ValueError(f"{self.name}: unit values must lie in [0, 1)")
        if self.is_integer:
            n_levels = int(self.hi) - int(self.lo) + 1
            return np.minimum(int(self.lo) + np.floor(u * n_levels), int(self.hi))
        return np.round(self.lo + u * (self.hi - self.lo), self.decimals)

    def contains(self, x: float) -> bool:
        if self.is_integer:
            return float(x).is_integer() and self.lo <= x <= self.hi
        return self.lo <= x <= self.hi


@dataclass(frozen=True)
class DesignSpace:
    params: tuple[ParamSpec, ...]           # sampled, in unit-cube column order
    derived: tuple[tuple[str, str], ...]    # (name, equal_to)
    source: Path

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def sampled_names(self) -> list[str]:
        """Sampled parameters = the design part of the surrogate's input."""
        return [p.name for p in self.params]

    @property
    def idf_names(self) -> list[str]:
        """Every parameter build_idf() needs, each derived one placed right
        after its source (glazing_vt after glazing_shgc)."""
        names = self.sampled_names
        for name, src in self.derived:
            names.insert(names.index(src) + 1, name)
        return names

    def spec(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def to_physical(self, unit_points: np.ndarray) -> list[dict[str, float | int]]:
        """Map an (n, dim) unit-cube array to n rounded parameter dicts."""
        unit_points = np.asarray(unit_points, dtype=float)
        if unit_points.ndim != 2 or unit_points.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {unit_points.shape}")
        cols = {p.name: p.to_physical(unit_points[:, j]) for j, p in enumerate(self.params)}
        return [self._finish({n: cols[n][i] for n in cols}) for i in range(len(unit_points))]

    def from_values(self, values: dict[str, float]) -> dict[str, float | int]:
        """Build a full parameter dict from explicit sampled values (edge cases).
        Values are checked against the bounds and rounded like sampled ones."""
        unknown = set(values) - set(self.sampled_names)
        if unknown:
            raise DesignSpaceError(f"unknown or non-sampled parameters: {sorted(unknown)}")
        missing = set(self.sampled_names) - set(values)
        if missing:
            raise DesignSpaceError(f"missing parameters: {sorted(missing)}")
        out = {}
        for p in self.params:
            x = float(values[p.name])
            x = float(int(x)) if p.is_integer else round(x, p.decimals)
            if not p.contains(x):
                raise DesignSpaceError(f"{p.name}={values[p.name]} outside [{p.lo}, {p.hi}]")
            out[p.name] = x
        return self._finish(out)

    def midpoint(self) -> dict[str, float]:
        """Centre of every range (integers: the middle integer)."""
        return {p.name: (math.floor((p.lo + p.hi) / 2) if p.is_integer else (p.lo + p.hi) / 2)
                for p in self.params}

    def _finish(self, sampled: dict[str, Any]) -> dict[str, float | int]:
        row: dict[str, float | int] = {}
        for p in self.params:
            v = sampled[p.name]
            row[p.name] = int(v) if p.is_integer else float(v)
        for name, src in self.derived:
            row[name] = row[src]
        return {n: row[n] for n in self.idf_names}


def load_design_space(path: str | Path = DEFAULT_DESIGN_SPACE) -> DesignSpace:
    import yaml

    path = Path(path)
    if not path.exists():
        raise DesignSpaceError(f"{path} not found")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    schema = (cfg.get("metadata") or {}).get("schema_version")
    if schema != SUPPORTED_SCHEMA:
        raise DesignSpaceError(
            f"{path.name}: schema_version {schema!r}, this sampler reads {SUPPORTED_SCHEMA}")

    raw = cfg.get("continuous_parameters") or {}
    if not raw:
        raise DesignSpaceError(f"{path.name}: no continuous_parameters")

    params = []
    for name, entry in raw.items():
        for key in ("min", "max", "decimals"):
            if key not in entry:
                raise DesignSpaceError(f"{name}: missing {key!r}")
        lo, hi, dec = float(entry["min"]), float(entry["max"]), int(entry["decimals"])
        if not lo < hi:
            raise DesignSpaceError(f"{name}: min {lo} must be < max {hi}")
        if dec < 0:
            raise DesignSpaceError(f"{name}: decimals must be >= 0")
        kind = entry.get("type", "continuous")
        if kind not in ("continuous", "integer"):
            raise DesignSpaceError(f"{name}: unknown type {kind!r}")
        is_int = kind == "integer"
        if is_int and not (lo.is_integer() and hi.is_integer()):
            raise DesignSpaceError(f"{name}: integer bounds must be whole numbers")
        params.append(ParamSpec(name, lo, hi, dec, is_int, str(entry.get("unit", ""))))

    derived = []
    for name, entry in (cfg.get("derived_parameters") or {}).items():
        src = (entry or {}).get("equal_to")
        if src not in raw:
            raise DesignSpaceError(f"derived {name}: equal_to {src!r} is not a sampled parameter")
        if name in raw:
            raise DesignSpaceError(f"{name} is both sampled and derived")
        derived.append((name, src))

    return DesignSpace(tuple(params), tuple(derived), path)
