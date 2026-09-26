"""Tree representations of single-cell data under Brownian motion, in Rust.

A clean-room implementation of Bonsai (de Groot et al., Nat. Biotechnol. 2026).
Input is per-cell means *and* error bars; output is a tree, not coordinates.

    >>> import bonsai_rs as bs
    >>> sim = bs.datasets.simulate_counts(64, 300, seed=1)
    >>> res = bs.bonsai_from_counts(sim.counts, cell_totals=sim.cell_totals)
    >>> bs.robinson_foulds(res.tree, sim.tree)
    0

From raw UMIs, `bonsai_from_counts` runs Sanity first and is the path to use.
With your own means and error bars, call `bonsai` directly.
"""

from . import datasets
from ._bonsai_rs import BonsaiError, __core_version__, __version__, gpu_available
from ._types import (
    BonsaiResult,
    Clustering,
    Likelihood,
    SanityResult,
    SimulatedCounts,
    SimulatedData,
    Step,
    Tree,
)
from .core import bonsai, bonsai_from_counts, from_sanity, sanity
from .tree import (
    cluster,
    layout,
    read_newick,
    robinson_foulds,
    to_newick,
    tree_distances,
)

__all__ = [
    "BonsaiError",
    "BonsaiResult",
    "Clustering",
    "Likelihood",
    "SanityResult",
    "SimulatedCounts",
    "SimulatedData",
    "Step",
    "Tree",
    "__core_version__",
    "__version__",
    "bonsai",
    "bonsai_from_counts",
    "cluster",
    "datasets",
    "from_sanity",
    "gpu_available",
    "layout",
    "read_newick",
    "robinson_foulds",
    "sanity",
    "to_newick",
    "tree_distances",
]
