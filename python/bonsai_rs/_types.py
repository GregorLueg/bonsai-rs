"""Frozen result records.

Plain dataclasses over numpy arrays. ``eq=False`` because elementwise ``==`` on
arrays does not return a bool, so a generated ``__eq__`` would raise.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from beartype import beartype


@dataclass(frozen=True, eq=False)
class Tree:
    """A rooted tree over the cells.

    Leaves are ``0..n_leaves`` in input order; internal nodes follow, ordered
    by height above the leaves. Every function in this package that takes a
    tree assumes that ordering, so build trees with this package (or
    `read_newick`) rather than by hand.

    Attributes:
        parent: Parent of each node, ``-1`` for the root. ``int64``.
        branch: Length of the branch above each node; the root's is ignored.
        n_leaves: Number of leaves.
    """

    parent: np.ndarray
    branch: np.ndarray
    n_leaves: int

    @property
    def n_nodes(self) -> int:
        """Leaves plus inferred ancestors."""
        return len(self.parent)


@dataclass(frozen=True, eq=False)
class Step:
    """What one step of the search bought.

    Attributes:
        step: Which step, ``"1 star"`` through ``"8 collapse"``.
        loglik: Tree loglikelihood after it, up to an additive constant.
        gain: Change from the previous step.
    """

    step: str
    loglik: float
    gain: float


@dataclass(frozen=True, eq=False)
class BonsaiResult:
    """A finished reconstruction.

    Attributes:
        tree: The tree. Leaf ``i`` is input cell ``i``.
        loglik: Final loglikelihood, meaningful only up to an additive constant.
        features: Input column of each retained feature, ascending. For
            `bonsai_from_counts` these index the genes of the count matrix.
        dropped: Genes the Sanity conversion dropped as ill-conditioned. Empty
            unless the run started from counts.
        node_means: Posterior mean of every node, ``(n_nodes, n_retained)``,
            in the input's units. Rows ``n_leaves:`` are the ancestors.
        node_sds: Posterior standard deviation, same layout.
        steps: The loglikelihood after each search step, in order.
    """

    tree: Tree
    loglik: float
    features: np.ndarray
    dropped: np.ndarray
    node_means: np.ndarray
    node_sds: np.ndarray
    steps: list[Step]

    # `ax` and the return value are matplotlib types; matplotlib is an
    # optional dependency, so they are `Any`.
    @beartype
    def plot(
        self,
        *,
        kind: Literal["daylight", "angle", "dendrogram"] = "angle",
        colours: Sequence[Any] | np.ndarray | None = None,
        ax: Any = None,
        figsize: tuple[float, float] = (6.0, 6.0),
    ) -> tuple[Any, Any]:
        """Draw the tree: leaves scattered over a radial layout, edges as lines.

        The look mirrors the comparison harness's tree-layout figure
        (``reference/comparison/figures.py``): grey edges, coloured leaf
        points, no axes chrome. Coordinates come from `layout`, not
        recomputed here.

        Args:
            kind: Layout to draw, as `layout`'s ``kind``. ``"angle"``
                (equal-angle radial) is the default, matching the comparison
                figures; ``"daylight"`` and ``"dendrogram"`` are the other
                choices `layout` accepts.
            colours: One value per leaf, in leaf order. Numeric arrays are
                mapped through a colormap; anything else is treated as
                categorical labels and given a discrete palette. ``None``
                draws every leaf the same colour.
            ax: Axes to draw into. ``None`` creates a new figure and axes.
            figsize: Figure size in inches. Ignored if ``ax`` is given.

        Returns:
            The ``Figure`` and ``Axes`` drawn into.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If ``colours`` is not one value per leaf.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
        except ImportError as e:
            raise ImportError(
                "BonsaiResult.plot needs matplotlib; install the `plot` extra, "
                "e.g. `pip install bonsai-rs[plot]`"
            ) from e

        from .tree import layout

        n_leaves = self.tree.n_leaves
        coords = layout(self.tree, kind=kind)

        non_root = np.flatnonzero(self.tree.parent >= 0)
        edges = np.stack([coords[self.tree.parent[non_root]], coords[non_root]], axis=1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        edge_width = 0.5 if n_leaves <= 1000 else 0.2
        ax.add_collection(
            LineCollection(
                edges, colors="0.6", linewidths=edge_width, alpha=0.6, zorder=1
            )
        )

        marker_size = 10 if n_leaves <= 1000 else (3 if n_leaves <= 6000 else 1.5)
        leaf_xy = coords[:n_leaves]
        scatter_kwargs = {"s": marker_size, "linewidths": 0, "zorder": 2}
        if colours is None:
            ax.scatter(leaf_xy[:, 0], leaf_xy[:, 1], **scatter_kwargs)
        else:
            values = np.asarray(colours)
            if values.ndim != 1 or len(values) != n_leaves:
                raise ValueError(
                    f"colours must be 1-D of length {n_leaves}, got shape "
                    f"{values.shape}"
                )
            if values.dtype.kind in "iuf":
                ax.scatter(
                    leaf_xy[:, 0],
                    leaf_xy[:, 1],
                    c=values,
                    cmap="viridis",
                    **scatter_kwargs,
                )
            else:
                categories, codes = np.unique(values, return_inverse=True)
                cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
                palette = np.array([cmap(i % cmap.N) for i in range(len(categories))])
                ax.scatter(
                    leaf_xy[:, 0], leaf_xy[:, 1], c=palette[codes], **scatter_kwargs
                )

        if kind != "dendrogram":
            ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax


@dataclass(frozen=True, eq=False)
class SanityResult:
    """Sanity posteriors.

    Attributes:
        log_fold_changes: Posterior log fold change ``d_c``, ``(n_cells,
            n_genes)``. Zero-centred per gene. This, not
            `log_transcription_quotients`, is what `from_sanity` wants.
        error_bars: Posterior SD on each log fold change, same layout.
        mean_log_quotient: Per-gene mean log transcription quotient ``m``.
        mean_log_quotient_error: Error bar on ``m``.
        variance: Per-gene variance of the log fold changes ``v``.
    """

    log_fold_changes: np.ndarray
    error_bars: np.ndarray
    mean_log_quotient: np.ndarray
    mean_log_quotient_error: np.ndarray
    variance: np.ndarray

    @property
    def log_transcription_quotients(self) -> np.ndarray:
        """Normalised expression ``m + d_c``, ``(n_cells, n_genes)``."""
        return self.log_fold_changes + self.mean_log_quotient


@dataclass(frozen=True, eq=False)
class Likelihood:
    """Likelihood means and SDs recovered from Sanity posteriors (S5).

    Attributes:
        means: ``(n_cells, n_kept)``, ready for `bonsai`.
        sds: Same layout.
        variances: Sanity's ``v`` for the kept genes; pass it to `bonsai`.
        features: Input column of each kept gene.
        dropped: Input columns dropped as ill-conditioned.
    """

    means: np.ndarray
    sds: np.ndarray
    variances: np.ndarray
    features: np.ndarray
    dropped: np.ndarray


@dataclass(frozen=True, eq=False)
class Clustering:
    """A cut of the tree into clusters.

    Attributes:
        leaf_cluster: Cluster of each leaf, numbered by decreasing size.
        centres: Representative node of each cluster.
        sizes: Leaves per cluster.
    """

    leaf_cluster: np.ndarray
    centres: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True, eq=False)
class SimulatedData:
    """Brownian motion on a known tree.

    Attributes:
        tree: The generating tree.
        truth: Noise-free leaf positions, ``(n_leaves, n_features)``.
        means: Observed means, truth plus noise.
        sds: Error bars on the means.
        variances: Per-feature variance the data was scaled by. ``truth``,
            ``means`` and ``sds`` are already divided by its square root.
    """

    tree: Tree
    truth: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True, eq=False)
class SimulatedCounts:
    """UMI counts drawn on a known tree.

    Attributes:
        tree: The generating tree.
        counts: ``(n_cells, n_genes)`` integer counts.
        cell_totals: The library size each cell was drawn with. Pass these,
            not the row sums: with a few hundred high-variance genes the row
            sums carry a per-cell compositional shift that costs the tree.
    """

    tree: Tree
    counts: np.ndarray
    cell_totals: np.ndarray
