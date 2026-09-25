"""Quantify and localise zero-length branches and polytomies in the
reconstructed trees.

These are two different phenomena and must not be conflated (see
figures/README.md for the full writeup):

- Zero-length *leaf* edges are a SPEC 6 boundary optimum (t=0): the model
  found no evidence separating that cell from its parent. Intended
  behaviour, not a search defect, and unaffected by search/floor changes.
- Zero-length *internal* edges and polytomies are created by splices during
  SPR/NNI (they are zero at every pre-search stage) and are the smaller,
  still-open part -- whether a resolve pass finishes cleaning them up.

Two outputs:

1. A plain-text table (stdout) of zero-length-branch (split leaf/internal)
   and polytomy counts for ours/theirs/truth at every config, plus the
   pipeline-stage breakdown for "ours" (post-merge, post-SPR, post-NNI/
   final) where stage snapshots exist.
2. figures/<config>/degenerate_regions.png: the same radial layout as
   figures.py, with structural leaf-zero leaves and internal-associated
   (zero-length-internal-edge or polytomy descendant) leaves marked with
   different markers, plus a comparison of those leaves' mean per-cell
   standard deviation (ours/sds.csv) against the rest.

Usage:
    .venv/bin/python degeneracy.py [CONFIG ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection

import treeviz as tv
from figures import DPI, N_GROUPS, WORK_DIR, OUT_DIR, clade_colours

STAGE_FILES = ["2_merge.nwk", "5_spr.nwk", "6_nni.nwk"]


def _stage_dir(cdir: Path) -> Path | None:
    for name in ["ours_stages", "ours_stages_default"]:
        d = cdir / name
        if d.is_dir():
            return d
    return None


def print_degeneracy_table(config: str) -> None:
    cdir = WORK_DIR / config
    print(f"\n=== {config}: final-tree degeneracy (ours / theirs / truth) ===")
    for name in ["ours", "theirs", "truth"]:
        path = cdir / f"{name}.nwk"
        if not path.exists():
            continue
        t = tv.parse_newick(path.read_text())
        rep = tv.degeneracy_report(t)
        print(
            f"  {name:7s} leaves={rep['n_leaves']:6d} edges={rep['n_edges']:6d} "
            f"zero_len={rep['n_zero_length']:5d} ({rep['pct_zero_length']:5.2f}%) "
            f"[leaf={rep['n_zero_length_leaf']:5d} internal={rep['n_zero_length_internal']:4d}] "
            f"polytomies={rep['n_polytomies']:4d} "
            f"degenerate_leaves={rep['n_degenerate_leaves']:5d} ({rep['pct_degenerate_leaves']:5.2f}%) "
            f"depth_med[all/degen/rest]={rep['depth_all_median']:.0f}/"
            f"{rep['depth_degenerate_median']:.0f}/{rep['depth_nondegenerate_median']:.0f} "
            f"depth_max={rep['depth_all_max']}"
        )
        if rep["n_polytomies"]:
            hist = ", ".join(f"deg{k}:{v}" for k, v in sorted(rep["polytomy_degree_hist"].items()))
            spans = rep["polytomy_leaf_span"][:8]
            print(f"           polytomy degree histogram: {hist}; largest leaf spans: {spans}")

    stage_dir = _stage_dir(cdir)
    if stage_dir is not None:
        print(f"  --- ours, pipeline stages ({stage_dir.name}) ---")
        for stage in STAGE_FILES:
            p = stage_dir / stage
            if not p.exists():
                continue
            t = tv.parse_newick(p.read_text())
            rep = tv.degeneracy_report(t)
            print(
                f"  {stage:14s} zero_len={rep['n_zero_length']:5d} "
                f"polytomies={rep['n_polytomies']:4d}"
            )


def mean_sd_per_cell(sds_path: Path) -> np.ndarray:
    """Row-wise mean standard deviation per cell, computed lazily/streaming
    so the full matrix is never materialised."""
    lf = pl.scan_csv(sds_path, has_header=False)
    out = lf.select(pl.mean_horizontal(pl.all()).alias("mean_sd")).collect(engine="streaming")
    return out.to_numpy().ravel()


def draw_highlighted_panel(
    ax: plt.Axes,
    tree: tv.ParsedTree,
    label_to_group: dict[str, int],
    colours: np.ndarray,
    title: str,
) -> None:
    """Draw the radial layout with the two degeneracy mechanisms marked
    separately: structural leaf-zero (SPEC 6 boundary, own branch is zero)
    as grey rings, internal-associated (descends from a zero-length internal
    edge or a polytomy) as black triangles -- these are not the same thing
    and must not share a marker.
    """
    layout = tv.radial_layout(tree)
    n_leaves = int(tree.is_leaf.sum())

    edge_width = 0.5 if n_leaves <= 1000 else 0.2
    ax.add_collection(LineCollection(layout.edges, colors="0.75", linewidths=edge_width, alpha=0.6, zorder=1))

    leaf_ids = np.array([i for i in range(tree.n_nodes) if tree.is_leaf[i]])
    groups = np.array([label_to_group.get(tree.label[i], -1) for i in leaf_ids])
    known = groups >= 0
    marker_size = 10 if n_leaves <= 1000 else (3 if n_leaves <= 6000 else 1.5)
    ax.scatter(
        layout.x[leaf_ids[known]], layout.y[leaf_ids[known]],
        c=colours[groups[known]], s=marker_size, linewidths=0, alpha=0.5, zorder=2,
    )

    structural = tv.structural_leaf_zero_labels(tree)
    internal = tv.internal_degenerate_leaf_labels(tree)
    is_structural = np.array([tree.label[i] in structural for i in leaf_ids])
    is_internal = np.array([tree.label[i] in internal for i in leaf_ids])
    hi_size = max(marker_size * 4, 12)
    ax.scatter(
        layout.x[leaf_ids[is_structural]], layout.y[leaf_ids[is_structural]],
        facecolors="none", edgecolors="0.35", linewidths=0.6, s=hi_size, marker="o", zorder=3,
        label=f"leaf-zero, structural ({int(is_structural.sum())})",
    )
    ax.scatter(
        layout.x[leaf_ids[is_internal]], layout.y[leaf_ids[is_internal]],
        facecolors="none", edgecolors="black", linewidths=0.9, s=hi_size * 1.3, marker="^", zorder=4,
        label=f"internal-associated ({int(is_internal.sum())})",
    )

    ax.set_title(
        f"{title} (n={n_leaves}, leaf-zero={int(is_structural.sum())}, internal={int(is_internal.sum())})",
        fontsize=9,
    )
    if is_structural.any() or is_internal.any():
        ax.legend(loc="lower left", fontsize=6, framealpha=0.6, markerscale=0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_localisation_figure(config: str) -> None:
    cdir = WORK_DIR / config
    ours = tv.parse_newick((cdir / "ours.nwk").read_text())
    theirs = tv.parse_newick((cdir / "theirs.nwk").read_text())
    truth_path = cdir / "truth.nwk"
    truth = tv.parse_newick(truth_path.read_text()) if truth_path.exists() else None

    label_to_group = (
        tv.cut_into_groups(truth, target_groups=N_GROUPS)
        if truth is not None
        else {}
    )
    colours = clade_colours(N_GROUPS)

    ours_degen = tv.degenerate_leaf_labels(ours)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    draw_highlighted_panel(axes[0], ours, label_to_group, colours, "bonsai-rs")
    draw_highlighted_panel(axes[1], theirs, label_to_group, colours, "reference")

    sds_path = cdir / "ours" / "sds.csv"
    ax = axes[2]
    if sds_path.exists() and ours_degen:
        mean_sd = mean_sd_per_cell(sds_path)
        n_cells = mean_sd.shape[0]
        labels = [f"cell{k}" for k in range(n_cells)]
        is_degen = np.array([lbl in ours_degen for lbl in labels])
        ax.boxplot(
            [mean_sd[is_degen], mean_sd[~is_degen]],
            tick_labels=[f"degenerate\n(n={int(is_degen.sum())})", f"rest\n(n={int((~is_degen).sum())})"],
            showfliers=False,
        )
        ax.set_ylabel("mean per-cell s.d. (ours/sds.csv)")
        med_degen = float(np.median(mean_sd[is_degen]))
        med_rest = float(np.median(mean_sd[~is_degen]))
        ax.set_title(f"median {med_degen:.3f} vs {med_rest:.3f}", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no degenerate leaves\nin bonsai-rs tree", ha="center", va="center")

    fig.suptitle(
        f"{config}: structural leaf-zero (grey circles, SPEC 6 boundary) vs "
        "internal-associated (black triangles, zero-length internal edge or "
        "polytomy descendant), and input noise level",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = OUT_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "degenerate_regions.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    configs = sys.argv[1:] or sorted(p.name for p in WORK_DIR.iterdir() if p.is_dir() and p.name != "sim")
    for config in configs:
        print_degeneracy_table(config)
        make_localisation_figure(config)


if __name__ == "__main__":
    main()
