"""
idf_generator - Parametric EnergyPlus IDF generator (Sub-task 2.2).

Public API
----------
    from idf_generator import build_idf
    idf = build_idf(params, idd_path)

Layout
------
    geometry/, construction/, fenestration/,
    internal_loads/, hvac/, outputs/     pure-data spec modules (no eppy)
    generator/                           orchestrator - the ONLY eppy user

build_idf is imported lazily (PEP 562), so importing a spec sub-package
does not pull in eppy. The "only generator.py touches eppy" rule holds.

Imports require the repository root on sys.path. Test files add it
themselves; entry points (batch runner) do the same or run with
`python -m` from the repository root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["build_idf"]

if TYPE_CHECKING:
    from .generator.generator import build_idf


def __getattr__(name: str) -> Any:
    if name == "build_idf":
        from .generator.generator import build_idf
        return build_idf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
