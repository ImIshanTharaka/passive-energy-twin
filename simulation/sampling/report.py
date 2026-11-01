"""
report.py - Coverage report for the 2.4 samples.

    python sampling/sampler.py --report

Writes sampling/sampling_report.md and two figures in sampling/figures/:
    param_histograms.png   one histogram per sampled parameter (2,100 training designs)
    correlation.png        pairwise correlation of the sampled parameters

All metrics use normalised parameter values, (x - min) / (max - min), i.e.
the design as the surrogate will see it after min-max scaling.
Deterministic: same samples -> byte-identical markdown (figures depend on the
matplotlib version).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sampling import sampler as S
from sampling.lhs import centred_discrepancy
from sampling.transform import DesignSpace, load_design_space

FIG_DIR = S.SAMPLING_DIR / "figures"
REPORT = S.SAMPLING_DIR / "sampling_report.md"

# Chart tokens (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES1 = "#2a78d6"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"


def normalised(rows: list[dict], space: DesignSpace) -> np.ndarray:
    return np.array([[(r[p.name] - p.lo) / (p.hi - p.lo) for p in space.params] for r in rows])


def unique_designs(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["design_id"] not in seen:
            seen.add(r["design_id"])
            out.append(r)
    return out


def nn_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Distance from each row of a to its nearest row of b (or of a, excluding itself)."""
    b_ = a if b is None else b
    out = np.empty(len(a))
    for i in range(0, len(a), 256):
        d = np.sqrt(((a[i:i + 256, None, :] - b_[None, :, :]) ** 2).sum(-1))
        if b is None:
            idx = np.arange(i, min(i + 256, len(a)))
            d[np.arange(len(idx)), idx] = np.inf
        out[i:i + 256] = d.min(1)
    return out


def max_abs_corr(x: np.ndarray, names: list[str]) -> tuple[float, str, str]:
    c = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(c, 0)
    i, j = np.unravel_index(np.abs(c).argmax(), c.shape)
    return float(abs(c[i, j])), names[i], names[j]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7, length=0)


def fig_histograms(x: np.ndarray, space: DesignSpace, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, d = x.shape
    cols = 7
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.1 * rows), facecolor=SURFACE)
    bins = 10
    for j, ax in enumerate(axes.flat):
        if j >= d:
            ax.axis("off")
            continue
        p = space.params[j]
        _style(ax)
        if p.is_integer:
            levels = np.arange(int(p.lo), int(p.hi) + 1)
            vals = x[:, j] * (p.hi - p.lo) + p.lo
            counts = [int(np.sum(np.isclose(vals, v))) for v in levels]
            ax.bar(levels, counts, width=0.8, color=SERIES1, edgecolor=SURFACE, linewidth=2)
            ax.axhline(n / len(levels), color=INK2, lw=1, ls=(0, (3, 2)))
            ax.set_xticks(levels)
        else:
            counts, edges = np.histogram(x[:, j], bins=bins, range=(0, 1))
            centres = (edges[:-1] + edges[1:]) / 2 * (p.hi - p.lo) + p.lo
            width = (p.hi - p.lo) / bins
            ax.bar(centres, counts, width=width, color=SERIES1, edgecolor=SURFACE, linewidth=2)
            ax.axhline(n / bins, color=INK2, lw=1, ls=(0, (3, 2)))
            ax.set_xticks([p.lo, p.hi])
        ax.set_yticks([])
        ax.set_title(f"{p.name} [{p.unit}]" if p.unit else p.name, fontsize=8, color=INK, loc="left")
    fig.suptitle(f"Sampled parameters across the {n:,} unique training designs "
                 f"(dashed line = perfectly uniform count)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_correlation(x: np.ndarray, space: DesignSpace, path: Path) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = space.sampled_names
    c = np.corrcoef(x, rowvar=False)
    off = c.copy()
    np.fill_diagonal(off, 0)
    lim = max(0.1, float(np.ceil(np.abs(off).max() * 20) / 20))
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    shown = np.where(np.eye(len(names), dtype=bool), np.nan, c)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(shown, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7, color=INK2)
    ax.set_yticks(range(len(names)), names, fontsize=7, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    cb.set_label("Pearson r", color=INK2, fontsize=8)
    ax.set_title(f"Pairwise correlation between sampled parameters (training designs); "
                 f"max |r| = {np.abs(off).max():.3f}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return float(np.abs(off).max())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_dir: Path = S.SAMPLING_DIR) -> int:
    space = load_design_space()
    manifest = json.loads((out_dir / S.MANIFEST).read_text())
    main = S.load_samples(out_dir / S.MAIN_CSV, space)
    hold = S.load_samples(out_dir / S.HOLDOUT_CSV, space)
    pilot = S.load_samples(out_dir / S.PILOT_CSV, space)
    names = space.sampled_names

    train_u = unique_designs(main)
    x_train = normalised(train_u, space)
    sets = {
        "A anchors": [r for r in train_u if r["design_set"] == "anchor"],
        "B specific": [r for r in train_u if r["design_set"] == "specific"],
        "A+B training": train_u,
        "C holdout_id": unique_designs([r for r in hold if r["batch"] == "holdout_id"]),
        "P pilot LHS": unique_designs([r for r in pilot if r["design_set"] == "pilot_lhs"]),
    }

    FIG_DIR.mkdir(exist_ok=True)
    fig_histograms(x_train, space, FIG_DIR / "param_histograms.png")
    fig_correlation(x_train, space, FIG_DIR / "correlation.png")

    L: list[str] = []
    add = L.append
    add("# Sub-task 2.4 — Sampling report")
    add("")
    add("Generated by `python sampling/sampler.py --report` from the sample files. "
        "Do not edit by hand.")
    add("")
    add(f"Master seed `{manifest['master_seed']}` · design_space.yaml "
        f"`{manifest['inputs_sha256']['config/design_space.yaml'][:12]}` · "
        f"climate_config.yaml `{manifest['inputs_sha256']['config/climate_config.yaml'][:12]}`")
    add("")

    add("## 1. Runs")
    add("")
    add("| File | Batch | Runs | Unique designs | Climates |")
    add("|---|---|---|---|---|")
    for fname, rows in ((S.MAIN_CSV, main), (S.HOLDOUT_CSV, hold), (S.PILOT_CSV, pilot)):
        for batch in sorted({r["batch"] for r in rows}, key=["main", "holdout_id", "holdout_ood", "pilot"].index):
            br = [r for r in rows if r["batch"] == batch]
            add(f"| `{fname}` | {batch} | {len(br):,} | {len({r['design_id'] for r in br}):,} | "
                f"{len({r['climate_key'] for r in br})} |")
    total = len(main) + len(hold)
    add(f"| | **campaign (main + holdout)** | **{total:,}** | | |")
    add("")
    add(f"Each training climate sees {manifest['design_sets']['A']['n_designs']} anchors + "
        f"{manifest['design_sets']['B']['per_slice']} climate-specific designs; "
        f"{len(train_u):,} unique buildings across the training set.")
    add("")

    add("## 2. Space-filling quality")
    add("")
    add("Normalised parameter space (21 dimensions, each scaled to [0, 1]). "
        "Lower discrepancy = more uniform. Every set below is a Latin hypercube "
        "(B and C also per climate slice) — checked by the test suite.")
    add("")
    add("| Set | Designs | Centred L2 discrepancy | Max \\|r\\| between parameters | Nearest-neighbour distance (min / median) |")
    add("|---|---|---|---|---|")
    for label, rows in sets.items():
        x = normalised(rows, space)
        r, a, b = max_abs_corr(x, names)
        nn = nn_distances(x)
        add(f"| {label} | {len(rows):,} | {centred_discrepancy(x):.4f} | {r:.3f} ({a} × {b}) | "
            f"{nn.min():.3f} / {np.median(nn):.3f} |")
    add("")
    add("Chosen candidates (best of "
        f"{manifest['design_sets']['A']['n_candidates']} by discrepancy): " +
        ", ".join(f"{k} #{v['chosen_candidate']}" for k, v in manifest["design_sets"].items()
                  if "chosen_candidate" in v) + ".")
    add("")
    add("Chance correlation between parameters was then removed by within-slice value "
        "swaps that keep every Latin-hypercube property (`lhs.decorrelate`). Largest |r| "
        "over the whole set and every climate slice, before → after:")
    add("")
    add("| Set | Before | After |")
    add("|---|---|---|")
    for k, v in manifest["design_sets"].items():
        if "max_abs_corr_before_decorrelation" in v:
            add(f"| {k} ({v['label']}) | {v['max_abs_corr_before_decorrelation']:.3f} | "
                f"{v['max_abs_corr_after_decorrelation']:.3f} |")
    add("")

    add("## 3. Holdout placement")
    add("")
    x_hid = normalised(sets["C holdout_id"], space)
    d_hid = nn_distances(x_hid, x_train)
    d_tr = nn_distances(x_train)
    add("Distance from each in-distribution holdout design to its nearest training design, "
        "compared with the training set's own nearest-neighbour spacing:")
    add("")
    add("| | min | median | max |")
    add("|---|---|---|---|")
    add(f"| holdout_id → nearest training design | {d_hid.min():.3f} | {np.median(d_hid):.3f} | {d_hid.max():.3f} |")
    add(f"| training → nearest other training design | {d_tr.min():.3f} | {np.median(d_tr):.3f} | {d_tr.max():.3f} |")
    add("")
    add("No holdout design coincides with a training design (minimum distance > 0). "
        "Similar spacing means the holdout tests interpolation between training designs "
        "at the density the surrogate will meet in use. The out-of-distribution batch "
        "(London) reuses the anchor designs by design (Decision S4): only the climate is unseen.")
    add("")

    add("## 4. Number of floors (Decision S3)")
    add("")
    add("| Set | 1 | 2 | 3 | 4 | 5 |")
    add("|---|---|---|---|---|---|")
    for label, rows in sets.items():
        cnt = Counter(r["n_floors"] for r in rows)
        add(f"| {label} | " + " | ".join(str(cnt.get(k, 0)) for k in range(1, 6)) + " |")
    add("")

    add("## 5. Pilot edge cases")
    add("")
    add("Each runs in every pilot climate (" + ", ".join(manifest["climates"]["pilot"]) + ").")
    add("")
    add("| ID | Stresses |")
    add("|---|---|")
    for k, v in manifest["design_sets"]["E"]["cases"].items():
        add(f"| {k} | {v} |")
    add("")

    add("## 6. Pre-flight (build_idf on every unique design)")
    add("")
    pf_path = out_dir / S.PREFLIGHT_JSON
    if pf_path.exists():
        pf = json.loads(pf_path.read_text())
        add(f"**{pf['n_passed']:,} / {pf['n_designs']:,} designs built** "
            f"({pf['n_failed']} failed) against `{pf['idd_version'].lstrip('!')}` IDD; "
            f"mean build time {pf['build_seconds_mean']} s. Each IDF was also checked for "
            "5 zones and 4 windows per floor.")
        for f in pf["failures"][:20]:
            add(f"- `{f['design_id']}`: {f['error']}")
    else:
        add("Not run yet: `python sampling/sampler.py --preflight`.")
    add("")

    add("## 7. Figures")
    add("")
    add("![Parameter histograms](figures/param_histograms.png)")
    add("")
    add("![Parameter correlation](figures/correlation.png)")
    add("")

    add("## 8. Notes for later phases")
    add("")
    add("- **Model inputs:** the 21 sampled parameters (`manifest.parameters.model_inputs`) "
        "+ 7 climate features joined in 2.7. `glazing_vt` is a copy of `glazing_shgc` "
        "(Decision S2) and must not be a model input.")
    add("- **Phase 3 splits:** anchors appear in 10 climates. Use GroupKFold on `design_id`; "
        "a row-wise split would put the same building in train and test.")
    add("- **2.5 pilot:** also run one VT-only pair (identical IDFs, VT changed) to confirm "
        "VT has no effect on the targets (justification for S2).")
    add("")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(S.ROOT)} and 2 figures in {FIG_DIR.relative_to(S.ROOT)}/")
    return 0
