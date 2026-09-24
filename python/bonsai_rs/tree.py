"""Working with a finished tree: Newick, layouts, clustering and distances."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from beartype import beartype

from . import _bonsai_rs as _core
from ._types import Clustering, Tree


@beartype
def _args(tree: Tree) -> tuple[np.ndarray, np.ndarray, int]:
    """The three arrays a tree crosses the boundary as."""
    return (
        np.ascontiguousarray(tree.parent, dtype=np.int64),
        np.ascontiguousarray(tree.branch, dtype=np.float64),
        tree.n_leaves,
    )


@beartype
def to_newick(tree: Tree, labels: Sequence[str] | None = None) -> str:
    """Serialise a tree to Newick.

    Args:
        tree: The tree.
        labels: One label per leaf. ``None`` labels leaves by index.

    Returns:
        The Newick string, semicolon terminated.
    """
    if labels is None:
        labels = [str(i) for i in range(tree.n_leaves)]
    return _core.to_newick(*_args(tree), list(labels))


@beartype
def read_newick(text: str) -> tuple[Tree, list[str]]:
    """Parse a Newick string.

    Args:
        text: The Newick string.

    Returns:
        The tree and its leaf labels, in leaf order.
    """
    d = _core.read_newick(text)
    return Tree(d["parent"], d["branch"], d["n_leaves"]), d["labels"]


@beartype
def layout(
    tree: Tree,
    *,
    kind: Literal["daylight", "angle", "dendrogram"] = "daylight",
    hyperbolic: bool = False,
) -> np.ndarray:
    """Node coordinates for drawing.

    Args:
        tree: The tree.
        kind: ``"daylight"`` (equal-daylight radial, the default),
            ``"angle"`` (equal-angle radial, faster and cruder) or
            ``"dendrogram"``. Equal daylight falls back to equal angle above a
            node count where refinement stops paying.
        hyperbolic: Project onto the Poincare disk afterwards, which gives the
            crowded outer branches more room.

    Returns:
        ``(n_nodes, 2)`` coordinates. Draw an edge from every node ``i`` with
        ``tree.parent[i] >= 0`` to its parent.
    """
    x, y = _core.layout(*_args(tree), kind, hyperbolic)
    return np.column_stack([x, y])


@beartype
def cluster(tree: Tree, n_clusters: int) -> Clustering:
    """Cut the tree into clusters.

    Greedy branch cuts minimising the summed within-cluster leaf distance. The
    numbering depends on the tree alone.

    Args:
        tree: The tree.
        n_clusters: How many clusters, clamped to ``1..n_leaves``.

    Returns:
        The clustering.
    """
    leaf_cluster, centres, sizes = _core.cluster(*_args(tree), n_clusters)
    return Clustering(leaf_cluster, centres, sizes)


@beartype
def tree_distances(tree: Tree, pairs: np.ndarray | None = None) -> np.ndarray:
    """Path distance along the tree between leaves.

    Args:
        tree: The tree.
        pairs: ``(m, 2)`` leaf index pairs. ``None`` for every pair, returned
            as a square ``(n_leaves, n_leaves)`` matrix; that is quadratic in
            memory, so sample pairs on large trees.

    Returns:
        One distance per pair, or the full matrix.
    """
    if pairs is not None:
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must be (m, 2), got {pairs.shape}")
        return _core.tree_distances(
            *_args(tree),
            np.ascontiguousarray(pairs[:, 0]),
            np.ascontiguousarray(pairs[:, 1]),
        )
    n = tree.n_leaves
    i, j = np.triu_indices(n, k=1)
    d = _core.tree_distances(*_args(tree), i.astype(np.int64), j.astype(np.int64))
    out = np.zeros((n, n))
    out[i, j] = d
    out[j, i] = d
    return out


@beartype
def robinson_foulds(left: Tree, right: Tree) -> int:
    """Robinson-Foulds distance between two trees over the same leaves.

    Leaves are matched by index. Branch lengths are ignored.

    Args:
        left: First tree.
        right: Second tree.

    Returns:
        Splits in one tree but not the other.
    """
    return _core.robinson_foulds(_args(left), _args(right))
