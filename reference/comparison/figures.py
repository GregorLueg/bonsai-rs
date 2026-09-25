"""Visual comparison of bonsai-rs, the reference implementation and ground
truth: tree layouts and distance-recovery scatters, for each configuration
under work_real/.

Aggregate scores (Robinson-Foulds, loglikelihood, distance_recovery) can
look fine while the tree itself is nonsense. This script draws the trees.

Usage:
    .venv/bin/python figures.py [--set NAME] [CONFIG ...]

With no arguments, runs every configuration found under work_real/.
`--set` picks which trees are drawn; see TREE_SETS. The default set is the
original bonsai-rs against reference pair; `e2e` draws the counts-to-tree
benchmark of docs/COMPARISON.md and writes `*_e2e.png`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection

import treeviz as tv

###########
# Globals #
###########

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work_real"
OUT_DIR = ROOT / "figures"

N_GROUPS = 10
N_PAIRS = 20_000
SEED = 0
DPI = 150
LABEL_LEAF_THRESHOLD = 5_000  # never label leaves above this; kept for clarity

# Trees drawn per set, as (panel title, Newick file under work_real/<config>/).
# A file that does not exist for a configuration is skipped for it.
TREE_SETS: dict[str, list[tuple[str, str]]] = {
    "default": [("bonsai-rs", "ours.nwk"), ("reference", "theirs.nwk")],
    "e2e": [
        ("bonsai-rs, GPU Sanity + approximate", "ours_e2e_gpu_approx.nwk"),
        ("bonsai-rs, CPU Sanity + exact", "ours_e2e_cpu_exact.nwk"),
        ("reference, 1 core", "theirs.nwk"),
        ("reference, 10 MPI ranks", "theirs_mpi10.nwk"),
    ],
}


def _log(msg: str) -> None:
    print(f"[figures] {msg}", flush=True)


def load_tree(path: Path) -> tv.ParsedTree:
    return tv.parse_newick(path.read_text())


def load_coords(path: Path) -> np.ndarray:
    """Load truth.csv (no header, one row per cell in cell0..cellN-1 order)."""
    return pl.read_csv(path, has_header=False).to_numpy().astype(np.float64)


def clade_colours(n_groups: int) -> np.ndarray:
    cmap = plt.get_cmap("tab10" if n_groups <= 10 else "tab20")
    return np.array([cmap(i % cmap.N) for i in range(n_groups)])


def clade_fragmentation(tree: tv.ParsedTree, label_to_group: dict[str, int]) -> tuple[int, int]:
    """Count contiguous same-colour runs in DFS leaf order.

    A clade that stays intact in a reconstruction appears as one run; a
    clade split across the tree by misplaced leaves appears as several,
    which the aggregate RF/loglikelihood scores do not surface directly.

    Returns:
        (total number of runs across all groups, number of distinct groups
        present in this tree) -- equal when every clade is fully contiguous.
    """
    leaves = tv.leaf_dfs_order(tree)
    seq = [label_to_group[tree.label[i]] for i in leaves if tree.label[i] in label_to_group]
    n_runs = sum(1 for k, g in enumerate(seq) if k == 0 or g != seq[k - 1])
    return n_runs, len(set(seq))


def draw_radial_panel(
    ax: plt.Axes,
    tree: tv.ParsedTree,
    label_to_group: dict[str, int],
    colours: np.ndarray,
    title: str,
) -> None:
    """Draw one radial tree layout into an existing axes."""
    layout = tv.radial_layout(tree)
    n_leaves = int(tree.is_leaf.sum())

    edge_width = 0.5 if n_leaves <= 1000 else 0.2
    lc = LineCollection(layout.edges, colors="0.6", linewidths=edge_width, alpha=0.6, zorder=1)
    ax.add_collection(lc)

    leaf_ids = np.array([i for i in range(tree.n_nodes) if tree.is_leaf[i]])
    groups = np.array([label_to_group.get(tree.label[i], -1) for i in leaf_ids])
    known = groups >= 0
    marker_size = 10 if n_leaves <= 1000 else (3 if n_leaves <= 6000 else 1.5)
    ax.scatter(
        layout.x[leaf_ids[known]],
        layout.y[leaf_ids[known]],
        c=colours[groups[known]],
        s=marker_size,
        linewidths=0,
        zorder=2,
    )
    if (~known).any():
        ax.scatter(
            layout.x[leaf_ids[~known]],
            layout.y[leaf_ids[~known]],
            c="0.3",
            s=marker_size,
            linewidths=0,
            zorder=2,
        )

    n_runs, n_groups = clade_fragmentation(tree, label_to_group)
    ax.set_title(f"{title} (n={n_leaves})\nclade fragments: {n_runs} (ideal {n_groups})", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_layout_figure(
    config: str,
    trees: list[tuple[str, tv.ParsedTree]],
    truth: tv.ParsedTree | None,
    label_to_group: dict[str, int],
    used_topology_cut: bool,
    out_path: Path,
) -> None:
    panels = trees + ([("truth", truth)] if truth is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5))
    colours = clade_colours(N_GROUPS)

    for ax, (title, tree) in zip(axes, panels):
        draw_radial_panel(ax, tree, label_to_group, colours, title)

    method = (
        f"leaves coloured by cutting the true tree topology into {N_GROUPS} clades"
        if used_topology_cut
        else f"leaves coloured by k-means ({N_GROUPS} clusters) on true coordinates "
        "(no true topology available)"
    )
    fig.suptitle(f"{config}: radial tree layout, equal-angle, radius = cumulative branch length\n{method}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def sample_pair_distances(
    tree: tv.ParsedTree,
    coords: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N_PAIRS distinct cell pairs and return (true squared euclidean,
    tree path distance)."""
    i = rng.integers(0, n_cells, N_PAIRS)
    j = rng.integers(0, n_cells, N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]

    node_i = np.array([tree.label_to_node[f"cell{k}"] for k in i])
    node_j = np.array([tree.label_to_node[f"cell{k}"] for k in j])

    depth, cumlen = tv.depth_and_cumlen(tree)
    up = tv.build_lca_tables(tree, depth)
    path_dist = tv.path_distances(up, depth, cumlen, node_i, node_j)

    sq_euclid = np.sum((coords[i] - coords[j]) ** 2, axis=1)
    return sq_euclid, path_dist


def make_distance_figure(
    config: str,
    trees: list[tuple[str, tv.ParsedTree]],
    coords: np.ndarray,
    truth_r: float | None,
    out_path: Path,
) -> None:
    n_cells = coords.shape[0]

    fig, axes = plt.subplots(1, len(trees), figsize=(5.5 * len(trees), 5))
    for ax, (name, tree) in zip(np.atleast_1d(axes), trees):
        # A fresh generator per panel so every tree is scored on the same pairs.
        rng = np.random.default_rng(SEED)
        sq_euclid, path_dist = sample_pair_distances(tree, coords, n_cells, rng)
        r = float(np.corrcoef(sq_euclid, path_dist)[0, 1])
        ax.scatter(sq_euclid, path_dist, s=3, alpha=0.15, linewidths=0, color="steelblue")
        ax.set_xlabel("true squared Euclidean distance")
        ax.set_ylabel("tree path distance (sum of branch lengths)")
        ax.set_title(f"{name}: r = {r:.4f}")

    suptitle = f"{config}: distance recovery, {N_PAIRS} sampled cell pairs"
    if truth_r is not None:
        suptitle += f" (generating tree: r = {truth_r:.4f})"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def run_config(config: str, tree_set: str) -> None:
    cdir = WORK_DIR / config
    t0 = time.time()
    _log(f"{config}: loading trees ({tree_set})")

    trees = [
        (title, load_tree(cdir / name))
        for title, name in TREE_SETS[tree_set]
        if (cdir / name).exists()
    ]
    truth_path = cdir / "truth.nwk"
    truth = load_tree(truth_path) if truth_path.exists() else None

    coords_path = cdir / "ours" / "truth.csv"
    coords = load_coords(coords_path)
    n_cells = coords.shape[0]
    labels = [f"cell{k}" for k in range(n_cells)]

    if truth is not None:
        label_to_group = tv.cut_into_groups(truth, target_groups=N_GROUPS)
        used_topology_cut = True
    else:
        label_to_group = tv.cluster_truth_coords(labels, coords, n_clusters=N_GROUPS)
        used_topology_cut = False

    out_dir = OUT_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"{config}: drawing tree layouts ({time.time() - t0:.1f}s elapsed)")
    suffix = "" if tree_set == "default" else f"_{tree_set}"
    make_layout_figure(
        config, trees, truth, label_to_group, used_topology_cut, out_dir / f"tree_layout{suffix}.png"
    )

    _log(f"{config}: computing distance recovery ({time.time() - t0:.1f}s elapsed)")
    truth_r = None
    if truth is not None:
        rng = np.random.default_rng(SEED)
        sq_euclid, path_dist = sample_pair_distances(truth, coords, n_cells, rng)
        truth_r = float(np.corrcoef(sq_euclid, path_dist)[0, 1])
    make_distance_figure(config, trees, coords, truth_r, out_dir / f"distance_recovery{suffix}.png")

    _log(f"{config}: done ({time.time() - t0:.1f}s total)")


def main() -> None:
    args = sys.argv[1:]
    tree_set = "default"
    if args[:1] == ["--set"]:
        tree_set, args = args[1], args[2:]
    configs = args or sorted(p.name for p in WORK_DIR.iterdir() if p.is_dir() and p.name != "sim")
    for config in configs:
        run_config(config, tree_set)


if __name__ == "__main__":
    main()
