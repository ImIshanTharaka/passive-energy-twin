"""
coverage_report.py - Climate coverage analysis (Sub-task 2.3.6).

Measures how well the training climates cover climate space, and where
the held-out climate sits relative to them. Produces the evidence behind
the climate-related claims in the methodology and results chapters.

WHAT THIS MODULE PRODUCES
-------------------------
  climate/climate_coverage_report.md   tables + findings, thesis-ready
  climate/coverage/pca_climates.png    the 11 climates on two PCA axes

It reads climate_metadata.csv through load_climate_registry() - never the
EPW files - so it always describes exactly the feature values the
surrogate will be trained on.

THE ANALYSES
------------
All on the seven continuous climate features, standardised with the
TRAINING climates' mean and standard deviation (population, ddof=0).
The holdout is projected with the same transform - it never influences
the scaling, just as it must never influence model training.

  Correlation      pairwise Pearson r between features (training set)
  Dimensionality   SVD of the standardised training matrix:
                     explained variance per component
                     effective rank = components needed for >= 99% variance
                     condition number = s_max / s_min
  PCA              the same SVD, read as principal components. Component
                   signs are fixed (largest-magnitude loading positive) so
                   the figure is identical on every run.
  Nearest          standardised Euclidean distance from each climate to its
  neighbours       closest training climate. Near-duplicates are flagged
                   below NEAR_DUPLICATE_FRACTION of the median distance.
  Envelope         for each holdout climate, feature by feature: inside the
                   training [min, max], or outside - and by how much, as a
                   percentage of the training range.
  Zone labels      which climates share an ASHRAE zone label (evidence for
                   decision D2: whether the label should be a model input).

WHY IT MATTERS
--------------
The surrogate sees only as many distinct climates as there are training
EPWs. Effective rank says how many independent climate dimensions those
climates actually span. The envelope table decides whether the holdout
test may be described as interpolation or (partial) extrapolation - a
claim an examiner will check.

USAGE
-----
    python climate/coverage_report.py
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:                                            # package import
    from .climate_features import (CONTINUOUS_FEATURES, DEFAULT_METADATA_CSV,
                                   ClimateRecord, holdout_keys,
                                   load_climate_registry, training_keys)
except ImportError:                             # script / test execution
    from climate_features import (CONTINUOUS_FEATURES, DEFAULT_METADATA_CSV,  # type: ignore
                                  ClimateRecord, holdout_keys,
                                  load_climate_registry, training_keys)

__all__ = [
    "CoverageAnalysis", "analyse", "standardise", "effective_rank",
    "nearest_neighbours", "envelope_table", "plot_pca", "write_report",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "climate" / "climate_coverage_report.md"
FIGURE_PATH = REPO_ROOT / "climate" / "coverage" / "pca_climates.png"

VARIANCE_THRESHOLD = 0.99        # effective rank = components to reach this
HIGH_CORRELATION = 0.90          # |r| at or above this is reported as redundant
NEAR_DUPLICATE_FRACTION = 0.5    # NN distance below this x median = near-duplicate

FEATURE_LABELS = {
    "hdd18": "HDD18", "cdd10": "CDD10", "annual_ghi": "GHI",
    "mean_dbt": "mean DBT", "mean_rh": "mean RH",
    "latitude": "latitude", "mean_wind": "mean wind",
}
FEATURE_UNITS = {
    "hdd18": "°C·day", "cdd10": "°C·day", "annual_ghi": "kWh/m²",
    "mean_dbt": "°C", "mean_rh": "%", "latitude": "°", "mean_wind": "m/s",
}

# Figure styling: reference palette slots 1-2 (validated: all-pairs CVD
# dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface). Text uses
# ink tokens, never series colours.
COLOR_TRAINING = "#2a78d6"
COLOR_HOLDOUT = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageAnalysis:
    features: tuple[str, ...]
    train_keys: tuple[str, ...]
    holdout_keys: tuple[str, ...]
    X_train: np.ndarray               # (n_train, n_features), natural units
    X_holdout: np.ndarray             # (n_holdout, n_features)
    mean: np.ndarray                  # training mean
    std: np.ndarray                   # training std (ddof=0)
    correlation: np.ndarray           # (n_features, n_features), training
    singular_values: np.ndarray
    explained: np.ndarray             # variance ratio per component
    effective_rank: int
    condition_number: float
    components: np.ndarray            # (n_components, n_features), sign-fixed
    scores_train: np.ndarray          # (n_train, n_components)
    scores_holdout: np.ndarray        # (n_holdout, n_components)
    nn_train: tuple[tuple[str, str, float], ...]
    nn_holdout: tuple[tuple[str, str, float], ...]
    envelope: tuple[tuple[str, str, float, float, float, str, float], ...]

    @property
    def nn_median(self) -> float:
        return float(np.median([d for _, _, d in self.nn_train]))


# ---------------------------------------------------------------------------
# Building blocks (pure functions - unit tested)
# ---------------------------------------------------------------------------

def standardise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(X - mean) / std. Raises if any feature has zero spread."""
    if np.any(std <= 0):
        raise ValueError("a feature has zero standard deviation across training climates")
    return (X - mean) / std


def effective_rank(explained: np.ndarray, threshold: float = VARIANCE_THRESHOLD) -> int:
    """Number of leading components whose cumulative variance reaches threshold."""
    cumulative = np.cumsum(explained)
    return int(np.searchsorted(cumulative, threshold - 1e-12) + 1)


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """Make each component's largest-magnitude loading positive (deterministic PCA)."""
    fixed = components.copy()
    for i, row in enumerate(fixed):
        if row[np.argmax(np.abs(row))] < 0:
            fixed[i] = -row
    return fixed


def nearest_neighbours(Z_ref: np.ndarray, ref_keys: list[str],
                       Z_query: np.ndarray, query_keys: list[str],
                       exclude_self: bool) -> list[tuple[str, str, float]]:
    """(query, nearest reference, standardised Euclidean distance) per query row."""
    out = []
    for i, q in enumerate(Z_query):
        d = np.linalg.norm(Z_ref - q, axis=1)
        if exclude_self:
            d[i] = np.inf
        j = int(np.argmin(d))
        out.append((query_keys[i], ref_keys[j], float(d[j])))
    return out


def envelope_table(X_train: np.ndarray, X_holdout: np.ndarray,
                   holdout_names: list[str], features: tuple[str, ...]
                   ) -> list[tuple[str, str, float, float, float, str, float]]:
    """Per holdout climate and feature: (key, feature, lo, hi, value, status, pct).

    status is "inside", "below" or "above"; pct is the distance outside the
    training range as a percentage of that range (0 when inside).
    """
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    rows = []
    for k, key in enumerate(holdout_names):
        for f, name in enumerate(features):
            v, span = X_holdout[k, f], hi[f] - lo[f]
            if v < lo[f]:
                status, pct = "below", 100.0 * (lo[f] - v) / span
            elif v > hi[f]:
                status, pct = "above", 100.0 * (v - hi[f]) / span
            else:
                status, pct = "inside", 0.0
            rows.append((key, name, float(lo[f]), float(hi[f]), float(v), status, float(pct)))
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(registry: dict[str, ClimateRecord],
            variance_threshold: float = VARIANCE_THRESHOLD) -> CoverageAnalysis:
    """Run every analysis on the registry's continuous features."""
    tr, ho = training_keys(registry), holdout_keys(registry)
    if len(tr) < 3:
        raise ValueError(f"need at least 3 training climates, have {len(tr)}")

    X_train = np.array([registry[k].features.continuous_vector() for k in tr])
    X_holdout = (np.array([registry[k].features.continuous_vector() for k in ho])
                 if ho else np.empty((0, len(CONTINUOUS_FEATURES))))

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    Z_train = standardise(X_train, mean, std)
    Z_holdout = standardise(X_holdout, mean, std) if ho else X_holdout

    correlation = np.corrcoef(X_train, rowvar=False)

    # Z_train is centred (standardised with its own mean), so its SVD is PCA.
    _, s, vt = np.linalg.svd(Z_train, full_matrices=False)
    explained = s ** 2 / np.sum(s ** 2)
    components = _fix_signs(vt)
    smallest = s[-1]
    condition = float(s[0] / smallest) if smallest > 1e-9 * s[0] else float("inf")

    return CoverageAnalysis(
        features=CONTINUOUS_FEATURES,
        train_keys=tuple(tr), holdout_keys=tuple(ho),
        X_train=X_train, X_holdout=X_holdout, mean=mean, std=std,
        correlation=correlation,
        singular_values=s, explained=explained,
        effective_rank=effective_rank(explained, variance_threshold),
        condition_number=condition,
        components=components,
        scores_train=Z_train @ components.T,
        scores_holdout=Z_holdout @ components.T if ho else np.empty((0, len(s))),
        nn_train=tuple(nearest_neighbours(Z_train, tr, Z_train, tr, exclude_self=True)),
        nn_holdout=tuple(nearest_neighbours(Z_train, tr, Z_holdout, ho, exclude_self=False)
                         if ho else ()),
        envelope=tuple(envelope_table(X_train, X_holdout, ho, CONTINUOUS_FEATURES)),
    )


def _axis_description(components: np.ndarray, i: int, n: int = 3) -> str:
    """'HDD18 −, CDD10 +, mean DBT +' - the n largest loadings of component i."""
    order = np.argsort(-np.abs(components[i]))[:n]
    return ", ".join(f"{FEATURE_LABELS[CONTINUOUS_FEATURES[j]]} "
                     f"{'+' if components[i, j] > 0 else '−'}" for j in order)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _place_labels(points: np.ndarray, span: np.ndarray) -> list[tuple[float, float, str, str]]:
    """Pick a label offset per point that stays clear of other points and labels.

    Deterministic greedy placement over eight candidate directions, scored in
    axis-normalised units. Returns (dx, dy, ha, va) offsets in points.
    """
    candidates = _LABEL_CANDIDATES
    norm = points / span
    placed: list[np.ndarray] = []
    choice = []
    for i, p in enumerate(norm):
        best, best_score = candidates[0], -np.inf
        for cand in candidates:
            anchor = p + np.array(cand[:2]) * 0.012
            others = np.delete(norm, i, axis=0)
            d_pts = np.min(np.linalg.norm(others - anchor, axis=1)) if len(others) else 1.0
            d_lab = min((np.linalg.norm(q - anchor) for q in placed), default=1.0)
            score = min(d_pts, d_lab * 1.3)
            if score > best_score:
                best, best_score = cand, score
        placed.append(p + np.array(best[:2]) * 0.03)
        choice.append(best)
    return choice


_LABEL_CANDIDATES = [(9, 0, "left", "center"), (-9, 0, "right", "center"),
                     (0, 9, "center", "bottom"), (0, -9, "center", "top"),
                     (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"),
                     (7, -7, "left", "top"), (-7, -7, "right", "top")]


def _resolve_label_collisions(fig, ax, labels, legend, points: np.ndarray,
                              marker_radius_pt: float = 7.0) -> None:
    """Measure rendered labels; move any that collide with anything.

    The greedy placement in _place_labels() cannot know text widths or the
    plot boundary. This pass renders once, then for each label that crosses
    the axes edge, or overlaps another label, the legend or any marker,
    tries the candidate offsets in order and keeps the first that fits.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer) if legend else None
    r = marker_radius_pt * fig.dpi / 72.0
    markers = [Bbox.from_extents(x - r, y - r, x + r, y + r)
               for x, y in ax.transData.transform(points)]

    def fits(label, others) -> bool:
        box = label.get_window_extent(renderer)
        inside = (box.x0 >= bounds.x0 and box.x1 <= bounds.x1
                  and box.y0 >= bounds.y0 and box.y1 <= bounds.y1)
        clear = all(not box.overlaps(o.get_window_extent(renderer)) for o in others)
        no_marker = all(not box.overlaps(m) for m in markers)
        return (inside and clear and no_marker
                and not (legend_box is not None and box.overlaps(legend_box)))

    for i, label in enumerate(labels):
        others = labels[:i] + labels[i + 1:]
        if fits(label, others):
            continue
        for dx, dy, ha, va in _LABEL_CANDIDATES:
            label.xyann = (dx, dy)
            label.set_ha(ha)
            label.set_va(va)
            if fits(label, others):
                break


def plot_pca(analysis: CoverageAnalysis, registry: dict[str, ClimateRecord],
             path: Path = FIGURE_PATH) -> Path:
    """Scatter of the climates on PC1-PC2, labelled, holdout distinguished."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = analysis
    pts = np.vstack([a.scores_train[:, :2], a.scores_holdout[:, :2]])
    names = [registry[k].city for k in a.train_keys] + \
            [f"{registry[k].city} (holdout)" for k in a.holdout_keys]
    zones = [registry[k].features.ashrae_zone for k in (*a.train_keys, *a.holdout_keys)]
    n_tr = len(a.train_keys)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.axhline(0, color=GRID, linewidth=0.9, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.9, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, width=0.6)

    ax.scatter(pts[:n_tr, 0], pts[:n_tr, 1], s=64, c=COLOR_TRAINING, marker="o",
               edgecolors=SURFACE, linewidths=2, zorder=3, label="Training climate")
    if len(pts) > n_tr:
        ax.scatter(pts[n_tr:, 0], pts[n_tr:, 1], s=90, c=COLOR_HOLDOUT, marker="D",
                   edgecolors=SURFACE, linewidths=2, zorder=4, label="Held-out climate")

    span = np.ptp(pts, axis=0)
    span[span == 0] = 1.0
    labels = []
    for (x, y), name, zone, (dx, dy, ha, va) in zip(pts, names, zones, _place_labels(pts, span)):
        labels.append(ax.annotate(f"{name} · {zone}", (x, y), xytext=(dx, dy),
                                  textcoords="offset points", ha=ha, va=va,
                                  fontsize=7.5, color=INK_PRIMARY, zorder=5))

    pad = 0.18 * span
    ax.set_xlim(pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    ax.set_ylim(pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])
    ax.set_xlabel(f"PC1 · {a.explained[0]:.0%} of variance   ({_axis_description(a.components, 0)})",
                  fontsize=8.5, color=INK_PRIMARY)
    ax.set_ylabel(f"PC2 · {a.explained[1]:.0%} of variance   ({_axis_description(a.components, 1)})",
                  fontsize=8.5, color=INK_PRIMARY)
    ax.set_title("Climate coverage: training climates and held-out climate",
                 fontsize=10.5, color=INK_PRIMARY, loc="left", pad=10)
    legend = ax.legend(loc="best", fontsize=8, frameon=True, framealpha=1.0,
                       facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)
    _resolve_label_collisions(fig, ax, labels, legend, pts)
    fig.text(0.01, 0.005,
             f"Seven continuous climate features, standardised on training climates. "
             f"PC1+PC2 = {a.explained[:2].sum():.0%} of variance. Labels: city · ASHRAE 169-2021 zone.",
             fontsize=6.5, color=INK_SECONDARY)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float, feature: str) -> str:
    return f"{v:,.0f}" if feature in ("hdd18", "cdd10", "annual_ghi") else f"{v:,.2f}"


def write_report(analysis: CoverageAnalysis, registry: dict[str, ClimateRecord],
                 csv_path: Path = DEFAULT_METADATA_CSV, path: Path = REPORT_PATH,
                 figure_path: Path = FIGURE_PATH) -> Path:
    a, F, L = analysis, analysis.features, FEATURE_LABELS
    csv_sha = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    any_rec = registry[a.train_keys[0]].diagnostics
    fig_rel = figure_path.relative_to(path.parent).as_posix()
    lines: list[str] = []
    w = lines.append

    w("# Climate Coverage Report — Sub-task 2.3.6\n")
    w("*Generated by `climate/coverage_report.py` from `climate/climate_metadata.csv` "
      f"(sha256 `{csv_sha}…`). Do not edit by hand — rerun the script.*\n")
    w(f"Training climates: **{len(a.train_keys)}** · held-out: **{len(a.holdout_keys)}** · "
      f"features: **{len(F)} continuous** · data: {any_rec.data_source} "
      f"{any_rec.period_of_record}\n")

    # --- 1. features
    w("## 1. Climate features\n")
    w("| Climate | Role | Zone | " + " | ".join(f"{L[f]} ({FEATURE_UNITS[f]})" for f in F) + " |")
    w("|---|---|---|" + "---|" * len(F))
    for k in (*a.train_keys, *a.holdout_keys):
        r = registry[k]
        vals = r.features.continuous_vector()
        w(f"| {r.city} | {r.role} | {r.features.ashrae_zone} | "
          + " | ".join(_fmt(v, f) for v, f in zip(vals, F)) + " |")
    w("")

    # --- 2. correlation
    w("## 2. Feature correlation (training climates)\n")
    w("| | " + " | ".join(L[f] for f in F) + " |")
    w("|---|" + "---|" * len(F))
    for i, f in enumerate(F):
        w(f"| **{L[f]}** | " + " | ".join(f"{a.correlation[i, j]:+.2f}" for j in range(len(F))) + " |")
    redundant = [(F[i], F[j], a.correlation[i, j]) for i in range(len(F)) for j in range(i + 1, len(F))
                 if abs(a.correlation[i, j]) >= HIGH_CORRELATION]
    w("")
    if redundant:
        w(f"Pairs with |r| ≥ {HIGH_CORRELATION:.2f} (largely redundant across these climates):\n")
        for f1, f2, r in sorted(redundant, key=lambda t: -abs(t[2])):
            w(f"- {L[f1]} ↔ {L[f2]}: r = {r:+.2f}")
    else:
        w(f"No pair reaches |r| ≥ {HIGH_CORRELATION:.2f}.")
    w("")

    # --- 3. dimensionality
    cum = np.cumsum(a.explained)
    w("## 3. Dimensionality\n")
    w("| Component | Singular value | Explained variance | Cumulative |")
    w("|---|---|---|---|")
    for i, (s, e, c) in enumerate(zip(a.singular_values, a.explained, cum)):
        w(f"| PC{i + 1} | {s:.3f} | {e:.1%} | {c:.1%} |")
    cond = "∞ (singular)" if np.isinf(a.condition_number) else f"{a.condition_number:,.1f}"
    w(f"\n**Effective rank (≥ {VARIANCE_THRESHOLD:.0%} variance): {a.effective_rank}** of {len(F)} "
      f"features. Condition number: {cond}.\n")
    w(f"The {len(F)} climate features carry about {a.effective_rank} independent dimensions of "
      f"information across these {len(a.train_keys)} training climates. Adding features cannot "
      f"raise this; only adding climates can.\n")

    # --- 4. PCA
    w("## 4. Principal components\n")
    w(f"![Climate coverage: PCA]({fig_rel})\n")
    w("Loadings (sign fixed so each component's largest loading is positive):\n")
    w("| Feature | " + " | ".join(f"PC{i + 1}" for i in range(min(3, len(F)))) + " |")
    w("|---|" + "---|" * min(3, len(F)))
    for j, f in enumerate(F):
        w(f"| {L[f]} | " + " | ".join(f"{a.components[i, j]:+.2f}" for i in range(min(3, len(F)))) + " |")
    w(f"\n- **PC1** ({a.explained[0]:.0%}): {_axis_description(a.components, 0)}")
    w(f"- **PC2** ({a.explained[1]:.0%}): {_axis_description(a.components, 1)}")
    w(f"- **PC3** ({a.explained[2]:.0%}): {_axis_description(a.components, 2)}\n")

    # --- 5. nearest neighbours
    med = a.nn_median
    w("## 5. Nearest neighbours (standardised distance)\n")
    w("| Climate | Nearest training climate | Distance | vs median |")
    w("|---|---|---|---|")
    for key, nb, d in sorted(a.nn_train, key=lambda t: t[2]):
        flag = " ⚠ near-duplicate" if d < NEAR_DUPLICATE_FRACTION * med else ""
        w(f"| {registry[key].city} | {registry[nb].city} | {d:.2f} | {d / med:.2f}×{flag} |")
    for key, nb, d in a.nn_holdout:
        w(f"| **{registry[key].city} (holdout)** | {registry[nb].city} | **{d:.2f}** | {d / med:.2f}× |")
    tr_d = [d for _, _, d in a.nn_train]
    w(f"\nTraining nearest-neighbour distances: min {min(tr_d):.2f}, median {med:.2f}, "
      f"max {max(tr_d):.2f}. Near-duplicate threshold: < {NEAR_DUPLICATE_FRACTION:.1f}× median.\n")

    # --- 6. holdout envelope
    w("## 6. Held-out climate vs training envelope\n")
    for hk in a.holdout_keys:
        rows = [r for r in a.envelope if r[0] == hk]
        w(f"### {registry[hk].city}\n")
        w("| Feature | Training range | Holdout value | Status |")
        w("|---|---|---|---|")
        for _, f, lo, hi, v, status, pct in rows:
            st = "inside" if status == "inside" else f"**{status} — {pct:.0f}% of range**"
            w(f"| {L[f]} | {_fmt(lo, f)} – {_fmt(hi, f)} | {_fmt(v, f)} | {st} |")
        outside = [(f, status, pct) for _, f, _, _, _, status, pct in rows if status != "inside"]
        nn = next(d for k, _, d in a.nn_holdout if k == hk)
        nn_name = registry[next(n for k, n, _ in a.nn_holdout if k == hk)].city
        within = min(tr_d) <= nn <= max(tr_d)
        w("")
        if outside:
            names = ", ".join(f"{L[f]} ({status}, {pct:.0f}%)" for f, status, pct in
                              sorted(outside, key=lambda t: -t[2]))
            w(f"**{registry[hk].city} is a partial extrapolation**: outside the training envelope on "
              f"{len(outside)} of {len(F)} features — {names}. Its nearest training climate "
              f"({nn_name}, {nn:.2f}) is {'within' if within else 'outside'} the spread of training "
              f"nearest-neighbour distances ({min(tr_d):.2f}–{max(tr_d):.2f}), so it sits "
              f"{'inside the training cloud by aggregate distance while' if within else 'away from the training cloud and'} "
              f"exceeding the envelope on individual axes. Describe the holdout test accordingly — "
              f"neither pure interpolation nor wholesale extrapolation.\n")
        else:
            w(f"**{registry[hk].city} lies inside the training envelope on every feature** — "
              f"the holdout test is an interpolation test.\n")

    # --- 7. zone labels (D2)
    w("## 7. ASHRAE zone labels (evidence for decision D2)\n")
    groups: dict[str, list[str]] = {}
    for k in (*a.train_keys, *a.holdout_keys):
        groups.setdefault(registry[k].features.ashrae_zone, []).append(registry[k].city)
    shared = {z: c for z, c in sorted(groups.items()) if len(c) > 1}
    w(f"{len(groups)} distinct labels across {len(a.train_keys) + len(a.holdout_keys)} climates.")
    for z, cities in shared.items():
        w(f"- **{z}** shared by {', '.join(cities)}")
    unseen = [registry[k].features.ashrae_zone for k in a.holdout_keys
              if registry[k].features.ashrae_zone not in
              {registry[t].features.ashrae_zone for t in a.train_keys}]
    w(f"- Holdout zone labels unseen in training: {', '.join(unseen) if unseen else 'none'}\n")
    w("A shared label tells the surrogate two climates are equivalent even when their continuous "
      "features differ substantially (§1, §6). The seven continuous features encode everything the "
      "label encodes, more precisely. **Recommendation (D2): keep `ashrae_zone` for stratification "
      "and reporting; exclude it from the surrogate input vector.**\n")

    # --- 8. methodology text
    top = a.explained[:2].sum()
    w("## 8. Methodology-ready statements\n")
    w(f"> Climate was represented by seven continuous features computed from each TMYx "
      f"{any_rec.period_of_record} weather file: HDD18, CDD10, annual global horizontal irradiation, "
      f"mean dry-bulb temperature, mean relative humidity, latitude and mean wind speed. Across the "
      f"{len(a.train_keys)} training climates these features span {a.effective_rank} independent "
      f"dimensions (≥ {VARIANCE_THRESHOLD:.0%} of variance), with the first two principal components "
      f"accounting for {top:.0%} (Figure X).\n")
    for hk in a.holdout_keys:
        outside = [L[r[1]] for r in a.envelope if r[0] == hk and r[5] != "inside"]
        if outside:
            w(f"> {registry[hk].city} was withheld from training as an out-of-distribution test. "
              f"It lies outside the training range on {', '.join(outside)}, and within it on the "
              f"remaining features; results for {registry[hk].city} therefore test partial "
              f"extrapolation rather than interpolation.\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main() -> int:
    registry = load_climate_registry(check_files=False)
    analysis = analyse(registry)
    fig = plot_pca(analysis, registry)
    report = write_report(analysis, registry)
    print(f"Effective rank: {analysis.effective_rank} of {len(analysis.features)} "
          f"(PC1+PC2 = {analysis.explained[:2].sum():.1%})")
    for key, nb, d in analysis.nn_holdout:
        out = [r[1] for r in analysis.envelope if r[0] == key and r[5] != "inside"]
        print(f"Holdout {key}: nearest {nb} ({d:.2f}); outside envelope on {out or 'nothing'}")
    print(f"Wrote {report.relative_to(REPO_ROOT)}")
    print(f"Wrote {fig.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
