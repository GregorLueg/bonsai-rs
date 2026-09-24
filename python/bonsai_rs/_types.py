"""Frozen result records.

Plain dataclasses over numpy arrays. ``eq=False`` because elementwise ``==`` on
arrays does not return a bool, so a generated ``__eq__`` would raise.
"""

from dataclasses import dataclass

import numpy as np


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
