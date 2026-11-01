"""
lhs.py - Latin hypercube designs on the unit cube (Sub-task 2.4).

Pure numpy + scipy (discrepancy only). Knows nothing about buildings or
climates: it produces points in [0, 1)^d. transform.py maps them to physical
parameter values.

WHAT A LATIN HYPERCUBE IS
-------------------------
Cut every axis of [0, 1)^d into N equal strata. An N-point design is a Latin
hypercube (LHS) when, on every axis, each stratum holds exactly one point.

SLICED LATIN HYPERCUBE (Qian 2012, JASA 107:393-402)
----------------------------------------------------
N = t * m points split into t slices of m points such that:
  - the whole N-point set is an LHS (N fine strata per axis), AND
  - every m-point slice is itself an LHS (m coarse strata per axis).
Used for the climate-specific designs: t = 10 climates, m = 200 per climate.
A random split of an ordinary LHS would lose the per-slice guarantee - a
climate could, by chance, receive no high-WWR building.

Construction, independently for each axis:
  1. In each slice, give its m points the coarse strata 0..m-1 in random
     order.
  2. Coarse stratum k contains the t fine strata k*t .. k*t+t-1. The t points
     (one per slice) that sit in coarse stratum k get those fine strata in
     random order.
  3. Place each point uniformly at random inside its fine stratum.
A plain LHS is the special case t = 1.

Implemented here rather than with scipy.stats.qmc.LatinHypercube so that the
plain and sliced designs share one code path, and so the designs depend only
on numpy's Generator stream (scipy is used only to score candidates).

CANDIDATE SELECTION, THEN DECORRELATION
---------------------------------------
Any random LHS is valid, but LHS only controls the 1-D projections. Two
further steps improve the design in the full d-dimensional space:

1. best_design() draws K candidates from deterministic child seeds and keeps
   the one with the lowest centred L2 discrepancy (scipy.stats.qmc.discrepancy,
   method "CD"). For sliced designs the score is CD(whole) + mean CD(slices).

2. decorrelate() then removes spurious correlation between parameters. A
   random 100-point LHS in 21 dimensions typically has some parameter pair
   correlated at |r| ~ 0.3 by chance (0.5+ within a 30-point climate slice),
   which would confound the effects of those two parameters in the surrogate
   and in SHAP. The search swaps two values of one column within one slice
   and keeps the swap if it lowers sum(r^2) over all parameter pairs (whole
   set + that slice). A within-slice swap leaves every column's set of values
   untouched, so the whole design AND every slice remain Latin hypercubes.
   Correlation-reduction by restricted pairing follows Iman & Conover (1982);
   the pairwise-exchange search is the one used for optimal LHDs (Morris &
   Mitchell 1995). It also lowered the discrepancy in every set tried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Design:
    """A unit-cube design and how it was chosen."""
    points: np.ndarray          # shape (n_slices * per_slice, d), values in [0, 1)
    n_slices: int               # t (1 for a plain LHS)
    per_slice: int              # m; slice s = points[s*m:(s+1)*m]
    candidate: int              # index of the chosen candidate
    score: float                # selection score of the chosen candidate
    discrepancy_whole: float    # CD of the whole set
    discrepancy_slices: tuple[float, ...]  # CD of each slice (len 1 for plain LHS)
    max_abs_corr_before: float = float("nan")  # largest |r| (whole or any slice) before decorrelate()
    max_abs_corr_after: float = float("nan")   # ... and after

    def slice(self, s: int) -> np.ndarray:
        if not 0 <= s < self.n_slices:
            raise IndexError(f"slice {s} out of range 0..{self.n_slices - 1}")
        return self.points[s * self.per_slice:(s + 1) * self.per_slice]


def sliced_lhs(n_slices: int, per_slice: int, dim: int,
               rng: np.random.Generator) -> np.ndarray:
    """One sliced Latin hypercube, rows ordered slice by slice.

    Returns an array of shape (n_slices * per_slice, dim) in [0, 1).
    """
    if n_slices < 1 or per_slice < 1 or dim < 1:
        raise ValueError(f"need n_slices, per_slice, dim >= 1; got {n_slices}, {per_slice}, {dim}")
    t, m = n_slices, per_slice
    n = t * m
    points = np.empty((n, dim))
    for j in range(dim):
        # Step 1: coarse stratum of every point, a permutation within each slice
        coarse = np.stack([rng.permutation(m) for _ in range(t)])        # (t, m)
        # Step 2: within each coarse stratum, hand the t fine strata to the slices
        offset = np.stack([rng.permutation(t) for _ in range(m)])        # (m, t): offset[k, s]
        fine = coarse * t + offset[coarse, np.arange(t)[:, None]]        # (t, m)
        # Step 3: jitter inside the fine stratum
        u = rng.random((t, m))
        points[:, j] = ((fine + u) / n).reshape(n)
    return points


def lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Latin hypercube: sliced_lhs with one slice."""
    return sliced_lhs(1, n, dim, rng)


def centred_discrepancy(points: np.ndarray) -> float:
    """Centred L2 discrepancy (lower = more uniform). Needs >= 2 points."""
    return float(qmc.discrepancy(points, method="CD"))


def _score(points: np.ndarray, t: int, m: int) -> tuple[float, float, tuple[float, ...]]:
    whole = centred_discrepancy(points)
    if t == 1:
        return whole, whole, (whole,)
    slices = tuple(centred_discrepancy(points[s * m:(s + 1) * m]) for s in range(t))
    return whole + float(np.mean(slices)), whole, slices


# Seed-sequence key of the decorrelation stream (candidates use 0..K-1).
DECORRELATE_KEY = 1_000_000


def _standardise(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(0)) / a.std(0)


def max_abs_corr(points: np.ndarray, n_slices: int = 1, per_slice: int | None = None) -> float:
    """Largest |r| between two columns, over the whole set and every slice."""
    def worst(a: np.ndarray) -> float:
        c = np.corrcoef(a, rowvar=False)
        np.fill_diagonal(c, 0.0)
        return float(np.abs(c).max())
    m = per_slice or len(points)
    parts = [points] + ([points[s * m:(s + 1) * m] for s in range(n_slices)] if n_slices > 1 else [])
    return max(worst(p) for p in parts)


def decorrelate(points: np.ndarray, n_slices: int, per_slice: int,
                rng: np.random.Generator, n_iter: int) -> np.ndarray:
    """Reduce correlation between columns by within-slice value swaps.

    Objective: sum of squared off-diagonal correlations of the whole set plus
    those of the slice being changed. Swaps never change a column's set of
    values inside a slice, so the (sliced) Latin hypercube property, means and
    variances are all preserved; correlations are updated in O(d) per swap.
    """
    t, m = n_slices, per_slice
    x = points.copy()
    n, d = x.shape
    z = _standardise(x)
    C = z.T @ z / n
    sliced = t > 1
    if sliced:
        zs = np.concatenate([_standardise(x[s * m:(s + 1) * m]) for s in range(t)])
        Cs = [zs[s * m:(s + 1) * m].T @ zs[s * m:(s + 1) * m] / m for s in range(t)]
    js = rng.integers(d, size=n_iter)
    ss = rng.integers(t, size=n_iter)
    aa = rng.integers(m, size=n_iter)
    bb = rng.integers(m, size=n_iter)
    for j, s, a, b in zip(js, ss, aa, bb):
        if a == b:
            continue
        a, b = a + s * m, b + s * m
        dw = (z[b, j] - z[a, j]) * (z[a] - z[b]) / n
        dw[j] = 0.0
        change = float((2 * C[j] * dw + dw * dw).sum())
        if sliced:
            ds = (zs[b, j] - zs[a, j]) * (zs[a] - zs[b]) / m
            ds[j] = 0.0
            change += float((2 * Cs[s][j] * ds + ds * ds).sum())
        if change < 0:
            C[j] += dw
            C[:, j] = C[j]
            z[[a, b], j] = z[[b, a], j]
            x[[a, b], j] = x[[b, a], j]
            if sliced:
                Cs[s][j] += ds
                Cs[s][:, j] = Cs[s][j]
                zs[[a, b], j] = zs[[b, a], j]
    return x


def best_design(n_slices: int, per_slice: int, dim: int,
                seed: tuple[int, ...], n_candidates: int,
                decorrelate_iter: int = 0) -> Design:
    """Best of n_candidates sliced LHS designs, then (optionally) decorrelated.

    Fully determined by `seed`: candidate k comes from
    default_rng(SeedSequence(seed + (k,))), the decorrelation search from
    default_rng(SeedSequence(seed + (DECORRELATE_KEY,))).
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    best: Design | None = None
    for k in range(n_candidates):
        rng = np.random.default_rng(np.random.SeedSequence(list(seed) + [k]))
        pts = sliced_lhs(n_slices, per_slice, dim, rng)
        score, whole, slices = _score(pts, n_slices, per_slice)
        if best is None or score < best.score:
            best = Design(pts, n_slices, per_slice, k, score, whole, slices)
    assert best is not None
    before = max_abs_corr(best.points, n_slices, per_slice)
    pts = best.points
    if decorrelate_iter > 0:
        rng = np.random.default_rng(np.random.SeedSequence(list(seed) + [DECORRELATE_KEY]))
        pts = decorrelate(pts, n_slices, per_slice, rng, decorrelate_iter)
    score, whole, slices = _score(pts, n_slices, per_slice)
    return Design(pts, n_slices, per_slice, best.candidate, score, whole, slices,
                  before, max_abs_corr(pts, n_slices, per_slice))


def strata(points: np.ndarray, n_strata: int) -> np.ndarray:
    """Stratum index of every coordinate: floor(x * n_strata), shape like points."""
    return np.floor(points * n_strata).astype(int)


def is_latin_hypercube(points: np.ndarray) -> bool:
    """True when every column has exactly one point in each of len(points) strata."""
    n = len(points)
    s = strata(points, n)
    return all(np.array_equal(np.sort(s[:, j]), np.arange(n)) for j in range(points.shape[1]))
