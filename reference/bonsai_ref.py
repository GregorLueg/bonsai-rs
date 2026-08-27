"""Independent numpy reference for the Bonsai pruning recursion.

Written from docs/SPEC.md sections 4 and 5, not from any other implementation.
Two jobs: it is the numerical oracle the Rust kernels are checked against, and
it is the Python baseline the Rust timings are quoted relative to.

It is deliberately written the way a competent person would write it, fully
vectorised over the feature axis. The per-node Python loop is not a handicap we
invented; it is inherent to expressing a post-order sweep in numpy, and it is
the thing the Rust port is supposed to remove.

Run: uv run --with numpy reference/bonsai_ref.py --leaves 8192 --features 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def splitmix64(index: np.ndarray) -> np.ndarray:
    """Counter-based uniforms in [0, 1), matching the Rust fixture generator.

    Both sides must see bit-identical input or the loglikelihood comparison is
    meaningless. splitmix64 is stateless, so element `i` is a pure function of
    `i` and numpy can produce the whole stream without a Python loop.
    """
    z = (index.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    z = (z ^ (z >> np.uint64(31))).astype(np.uint64)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def make_fixture(n_leaves: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Leaf means and precisions in transformed units.

    Params
    ------
    n_leaves : number of cells
    n_features : number of genes

    Returns
    -------
    (means, precisions), each (n_leaves, n_features).
    """
    n = n_leaves * n_features
    means = (splitmix64(np.arange(n)) * 4.0 - 2.0).reshape(n_leaves, n_features)
    precisions = (0.25 + splitmix64(np.arange(n, 2 * n)) * 3.0).reshape(
        n_leaves, n_features
    )
    return means, precisions


def balanced_binary(n_leaves: int) -> np.ndarray:
    """Parent array for a balanced binary tree over a power-of-two leaf count.

    Node indices follow the same bottom-up allocation as the Rust arena, so
    every non-root node has a strictly larger parent index.

    Params
    ------
    n_leaves : leaf count, a power of two

    Returns
    -------
    Parent index per node, -1 for the root.
    """
    if n_leaves < 2 or n_leaves & (n_leaves - 1):
        raise ValueError(f"{n_leaves} is not a power of two of at least two")
    n_nodes = 2 * n_leaves - 1
    parent = np.full(n_nodes, -1, dtype=np.int64)
    level = list(range(n_leaves))
    nxt = n_leaves
    while len(level) > 1:
        up = []
        for i in range(0, len(level), 2):
            parent[level[i]] = nxt
            parent[level[i + 1]] = nxt
            up.append(nxt)
            nxt += 1
        level = up
    return parent


def prune(
    parent: np.ndarray,
    branch: np.ndarray,
    n_leaves: int,
    leaf_means: np.ndarray,
    leaf_precisions: np.ndarray,
) -> float:
    """Post-order pruning sweep returning the tree loglikelihood.

    SPEC.md section 4 for the effective leaf and section 5 for the
    loglikelihood, dropping the topology-independent constants of section 3.

    Params
    ------
    parent : parent index per node, -1 for the root
    branch : length of the branch above each node
    n_leaves : leaf count; leaves occupy indices 0..n_leaves
    leaf_means, leaf_precisions : (n_leaves, n_features) in transformed units

    Returns
    -------
    The tree loglikelihood, up to an additive constant.
    """
    n_nodes = len(parent)
    p = leaf_means.shape[1]

    m = np.zeros((n_nodes, p))
    w = np.zeros((n_nodes, p))
    m[:n_leaves] = leaf_means
    w[:n_leaves] = leaf_precisions

    kids: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, par in enumerate(parent):
        if par >= 0:
            kids[par].append(i)

    acc = 0.0
    for a in range(n_leaves, n_nodes):
        c = kids[a]
        wc = w[c]
        wd = wc / (1.0 + branch[c][:, None] * wc)
        wa = wd.sum(axis=0)
        ma = (wd * m[c]).sum(axis=0) / wa
        acc += np.log(wd).sum() - np.log(wa).sum()
        acc -= (wd * (ma - m[c]) ** 2).sum()
        m[a] = ma
        w[a] = wa

    return 0.5 * acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaves", type=int, default=8192)
    ap.add_argument("--features", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    t0 = time.perf_counter()
    means, precisions = make_fixture(args.leaves, args.features)
    gen_s = time.perf_counter() - t0

    parent = balanced_binary(args.leaves)
    branch = np.full(len(parent), 0.6)

    best = float("inf")
    loglik = float("nan")
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        loglik = prune(parent, branch, args.leaves, means, precisions)
        best = min(best, time.perf_counter() - t0)

    print(f"leaves        {args.leaves}")
    print(f"features      {args.features}")
    print(f"fixture_s     {gen_s:.3f}")
    print(f"loglik        {loglik:.12e}")
    print(f"prune_ms      {best * 1e3:.3f}")


if __name__ == "__main__":
    main()
