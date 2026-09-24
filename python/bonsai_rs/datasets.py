"""Synthetic data on a known tree, for testing and for the docs."""

from typing import Literal

import numpy as np
from beartype import beartype

from . import _bonsai_rs as _core
from ._types import SimulatedCounts, SimulatedData, Tree


@beartype
def simulate(
    n_leaves: int = 64,
    n_features: int = 200,
    *,
    kind: Literal["binary", "random_branches", "unbalanced"] = "binary",
    branch_length: float = 1.0,
    noise_sd: float = 0.1,
    noise_spread: float = 2.0,
    seed: int = 0,
) -> SimulatedData:
    """Brownian motion on a known tree, with per-cell per-feature error bars.

    Args:
        n_leaves: Cells. A power of two for ``"binary"`` and
            ``"random_branches"``.
        n_features: Features.
        kind: ``"binary"`` (balanced, equal branches), ``"random_branches"``
            (balanced, log-uniform branches on ``[0.5, 2]``) or
            ``"unbalanced"`` (grown by splitting random leaves).
        branch_length: Branch length. Ignored by ``"random_branches"``.
        noise_sd: Error-bar scale relative to the spread of the data.
        noise_spread: Each error bar is ``noise_sd`` times a log-uniform draw
            on ``[1 / noise_spread, noise_spread]``. ``1.0`` is
            homoscedastic.
        seed: Seed.

    Returns:
        The tree and the data drawn on it.
    """
    d = _core.simulate(
        kind, n_leaves, n_features, branch_length, noise_sd, noise_spread, seed
    )
    return SimulatedData(
        tree=Tree(d["parent"], d["branch"], d["n_leaves"]),
        truth=d["truth"],
        means=d["means"],
        sds=d["sds"],
        variances=d["variances"],
    )


@beartype
def simulate_counts(
    n_leaves: int = 64,
    n_genes: int = 300,
    *,
    library_size: float = 3000.0,
    seed: int = 0,
) -> SimulatedCounts:
    """UMI counts drawn on a known tree.

    The noise-free leaf positions of `simulate` become log fold changes about
    a per-gene mean quotient, and counts are Poisson on top:
    ``count ~ Poisson(N_c * exp(log_q_g + x_gc))``. Library sizes ``N_c`` are
    log-normal about ``library_size``.

    Args:
        n_leaves: Cells, a power of two.
        n_genes: Genes.
        library_size: Median UMIs per cell.
        seed: Seed for both the tree and the counts.

    Returns:
        The tree, the counts and the library sizes they were drawn with.
    """
    sim = simulate(n_leaves, n_genes, seed=seed)
    rng = np.random.default_rng(seed)
    totals = rng.lognormal(np.log(library_size), 0.3, size=n_leaves)
    log_q = -np.log(n_genes) + 2.0 * (np.arange(n_genes) / n_genes - 0.5)
    lfc = sim.truth * np.sqrt(sim.variances)
    counts = rng.poisson(totals[:, None] * np.exp(log_q[None, :] + lfc))
    return SimulatedCounts(
        tree=sim.tree, counts=counts.astype(np.int64), cell_totals=totals
    )
