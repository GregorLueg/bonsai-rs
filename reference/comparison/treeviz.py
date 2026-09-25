"""Newick parsing, radial layout and tree-distance utilities for figures.py.

No dependency on the reference implementation or on bonsai-rs: this is a
from-scratch reader operating on Newick text files only.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"\(|\)|,|:|;|[^,():;]+")


@dataclass
class ParsedTree:
    """A parsed Newick tree as flat, numpy-friendly arrays.

    ``branch_length[v]`` is the length of the edge above node ``v`` (zero
    for the root). ``children[v]`` preserves the order children appeared
    in the Newick string.
    """

    parent: np.ndarray
    branch_length: np.ndarray
    is_leaf: np.ndarray
    label: list[str | None]
    children: list[list[int]]
    root: int
    label_to_node: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.label_to_node = {
            lbl: i for i, lbl in enumerate(self.label) if lbl is not None
        }

    @property
    def n_nodes(self) -> int:
        return len(self.parent)


def parse_newick(text: str) -> ParsedTree:
    """Parse a single Newick tree (one line, terminated by ';').

    Args:
        text: Newick string.

    Returns:
        The parsed tree as flat arrays.
    """
    text = text.strip()
    tokens = _TOKEN_RE.findall(text)
    if tokens and tokens[-1] == ";":
        tokens.pop()

    parents: list[int] = []
    branch_lengths: list[float] = []
    is_leaf: list[bool] = []
    labels: list[str | None] = []
    children: list[list[int]] = []

    def new_node(parent: int, leaf: bool, label: str | None) -> int:
        nid = len(parents)
        parents.append(parent)
        branch_lengths.append(0.0)
        is_leaf.append(leaf)
        labels.append(label)
        children.append([])
        if parent >= 0:
            children[parent].append(nid)
        return nid

    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def parse_clade(parent: int) -> int:
        nonlocal pos
        if peek() == "(":
            pos += 1
            nid = new_node(parent, False, None)
            parse_clade(nid)
            while peek() == ",":
                pos += 1
                parse_clade(nid)
            if peek() != ")":
                raise ValueError(f"malformed newick at token {pos}: {tokens[pos:pos+5]}")
            pos += 1
            if peek() not in (":", ",", ")", None):
                labels[nid] = peek()
                pos += 1
        else:
            name = peek()
            pos += 1
            nid = new_node(parent, True, name)
        if peek() == ":":
            pos += 1
            branch_lengths[nid] = float(peek())
            pos += 1
        return nid

    root = parse_clade(-1)

    return ParsedTree(
        parent=np.array(parents, dtype=np.int64),
        branch_length=np.array(branch_lengths, dtype=np.float64),
        is_leaf=np.array(is_leaf, dtype=bool),
        label=labels,
        children=children,
        root=root,
    )


def post_order(tree: ParsedTree) -> list[int]:
    """Iterative post-order traversal (children before parent)."""
    order: list[int] = []
    stack: list[tuple[int, bool]] = [(tree.root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        else:
            stack.append((node, True))
            for c in tree.children[node]:
                stack.append((c, False))
    return order


def leaf_dfs_order(tree: ParsedTree) -> list[int]:
    """Pre-order DFS leaf sequence, preserving original Newick child order."""
    order: list[int] = []
    stack: list[int] = [tree.root]
    while stack:
        u = stack.pop()
        if tree.is_leaf[u]:
            order.append(u)
        else:
            for c in reversed(tree.children[u]):
                stack.append(c)
    return order


def depth_and_cumlen(tree: ParsedTree) -> tuple[np.ndarray, np.ndarray]:
    """Topological depth (edge count) and cumulative branch length from the root."""
    n = tree.n_nodes
    depth = np.zeros(n, dtype=np.int64)
    cumlen = np.zeros(n, dtype=np.float64)
    q: deque[int] = deque([tree.root])
    while q:
        u = q.popleft()
        for c in tree.children[u]:
            depth[c] = depth[u] + 1
            cumlen[c] = cumlen[u] + tree.branch_length[c]
            q.append(c)
    return depth, cumlen


def build_lca_tables(tree: ParsedTree, depth: np.ndarray) -> np.ndarray:
    """Binary-lifting ancestor table, ``up[k][v]`` = 2^k-th ancestor of v."""
    n = tree.n_nodes
    log = max(1, int(np.ceil(np.log2(max(n, 2)))) + 1)
    up = np.empty((log, n), dtype=np.int64)
    up[0] = tree.parent
    up[0, tree.root] = tree.root
    for k in range(1, log):
        up[k] = up[k - 1][up[k - 1]]
    return up


def path_distances(
    up: np.ndarray,
    depth: np.ndarray,
    cumlen: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Sum-of-branch-lengths path distance between paired node arrays u and v.

    Vectorised binary-lifting LCA: bring the deeper node up to the shallower
    node's depth, then jump both up together until they meet.
    """
    deeper_is_u = depth[u] >= depth[v]
    deep = np.where(deeper_is_u, u, v).copy()
    shallow = np.where(deeper_is_u, v, u).copy()
    diff = np.abs(depth[u] - depth[v])
    log = up.shape[0]
    for k in range(log):
        mask = ((diff >> k) & 1).astype(bool)
        deep = np.where(mask, up[k][deep], deep)
    same = deep == shallow
    for k in reversed(range(log)):
        mask = (~same) & (up[k][deep] != up[k][shallow])
        deep = np.where(mask, up[k][deep], deep)
        shallow = np.where(mask, up[k][shallow], shallow)
        same = deep == shallow
    lca = np.where(same, deep, up[0][deep])
    return cumlen[u] + cumlen[v] - 2.0 * cumlen[lca]


def subtree_leaf_counts(tree: ParsedTree, order: list[int]) -> np.ndarray:
    """Leaf count of the subtree rooted at each node, via a post-order pass."""
    counts = np.zeros(tree.n_nodes, dtype=np.int64)
    for u in order:
        counts[u] = 1 if tree.is_leaf[u] else sum(counts[c] for c in tree.children[u])
    return counts


def cut_into_groups(tree: ParsedTree, target_groups: int = 10) -> dict[str, int]:
    """Cut a tree's topology into roughly ``target_groups`` clades.

    Greedily expands the current largest (by leaf count), non-leaf group into
    its children until the group count reaches the target. Returns a mapping
    from leaf label to group index (stable ordering: larger groups first).
    """
    order = post_order(tree)
    counts = subtree_leaf_counts(tree, order)
    groups = [tree.root]
    while len(groups) < target_groups:
        splittable = [(counts[g], i) for i, g in enumerate(groups) if not tree.is_leaf[g]]
        if not splittable:
            break
        _, idx = max(splittable)
        node = groups.pop(idx)
        groups.extend(tree.children[node])
    groups.sort(key=lambda g: -counts[g])

    label_to_group: dict[str, int] = {}
    for gi, root in enumerate(groups):
        stack = [root]
        while stack:
            u = stack.pop()
            if tree.is_leaf[u]:
                label_to_group[tree.label[u]] = gi
            else:
                stack.extend(tree.children[u])
    return label_to_group


def cluster_truth_coords(labels: list[str], coords: np.ndarray, n_clusters: int = 10) -> dict[str, int]:
    """Fallback clade assignment: k-means on true coordinates.

    Used only when a tree's generating topology is not available.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    assign = km.fit_predict(coords)
    order = np.argsort(-np.bincount(assign, minlength=n_clusters))
    remap = {old: new for new, old in enumerate(order)}
    return {lbl: remap[int(a)] for lbl, a in zip(labels, assign)}


def zero_length_mask(tree: ParsedTree, eps: float = 1e-12) -> np.ndarray:
    """Boolean mask over all nodes (root excluded) with branch_length <= eps."""
    mask = tree.branch_length <= eps
    mask = mask.copy()
    mask[tree.root] = False
    return mask


def polytomy_nodes(tree: ParsedTree) -> list[int]:
    """Internal nodes with more children than a strictly bifurcating tree allows.

    An unrooted tree's root conventionally has three children -- that is not
    a polytomy. Any other internal node with more than two children, or a
    root with more than three, is a genuine multifurcation.
    """
    out = []
    for u in range(tree.n_nodes):
        if tree.is_leaf[u]:
            continue
        limit = 3 if u == tree.root else 2
        if len(tree.children[u]) > limit:
            out.append(u)
    return out


def _flood_leaves(tree: ParsedTree, seeds: list[int]) -> set[str]:
    flagged: set[str] = set()
    seen: set[int] = set()
    stack = list(seeds)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if tree.is_leaf[u]:
            flagged.add(tree.label[u])
        else:
            stack.extend(tree.children[u])
    return flagged


def structural_leaf_zero_labels(tree: ParsedTree) -> set[str]:
    """Leaves whose own branch length is zero: SPEC 6 boundary optimum.

    The model found no evidence separating this cell from its parent, so
    the branch-length optimum sits at t=0. Intended behaviour, not a search
    defect -- no floor or resolve-pass change removes it.
    """
    mask = zero_length_mask(tree)
    return {tree.label[i] for i in range(tree.n_nodes) if tree.is_leaf[i] and mask[i]}


def internal_degenerate_leaf_labels(tree: ParsedTree) -> set[str]:
    """Leaves descending from a zero-length *internal* edge or a polytomy.

    This is the smaller, still-open part: whatever produces these is a
    candidate search/resolution defect, unlike structural_leaf_zero_labels.
    """
    mask = zero_length_mask(tree)
    seeds = [u for u in range(tree.n_nodes) if mask[u] and not tree.is_leaf[u]] + polytomy_nodes(tree)
    return _flood_leaves(tree, seeds)


def degenerate_leaf_labels(tree: ParsedTree) -> set[str]:
    """Leaf labels in the subtree below a zero-length branch or a polytomy node.

    Union of structural_leaf_zero_labels (own branch is zero) and
    internal_degenerate_leaf_labels (descends from an internal zero-length
    edge or polytomy). Kept as one combined count for the whole-tree
    localisation figure; use the two split functions when the mechanism
    matters, which per SPEC 6 it does -- they behave completely differently.
    """
    return structural_leaf_zero_labels(tree) | internal_degenerate_leaf_labels(tree)


def zero_length_leaf_internal_split(tree: ParsedTree) -> tuple[int, int]:
    """Zero-length branches split by which end they sit on.

    A zero-length *leaf* edge means that cell's branch-length optimum sits at
    the SPEC 6 boundary t=0 -- the model found no evidence separating it from
    its parent. That is intended behaviour, not a search defect, and no
    topology change removes it. A zero-length *internal* edge is what a
    resolve pass can (and per SPEC 9.2 step 3 should) collapse into a
    polytomy or remove; the two must not be conflated into one count.
    """
    mask = zero_length_mask(tree)
    n_leaf = int((mask & tree.is_leaf).sum())
    n_internal = int((mask & ~tree.is_leaf).sum())
    return n_leaf, n_internal


def degeneracy_report(tree: ParsedTree) -> dict:
    """Tabulate zero-length branches, polytomies, and subtree shape.

    Returns a plain dict of scalars/arrays suitable for printing as a table
    row; no plotting here.
    """
    n_leaves = int(tree.is_leaf.sum())
    n_edges = tree.n_nodes - 1
    zero_mask = zero_length_mask(tree)
    n_zero_leaf, n_zero_internal = zero_length_leaf_internal_split(tree)
    poly = polytomy_nodes(tree)
    degrees = [len(tree.children[u]) for u in poly]

    order = post_order(tree)
    counts = subtree_leaf_counts(tree, order)
    depth, _ = depth_and_cumlen(tree)

    degenerate = degenerate_leaf_labels(tree)
    leaf_ids = np.array([i for i in range(tree.n_nodes) if tree.is_leaf[i]])
    is_degenerate = np.array([tree.label[i] in degenerate for i in leaf_ids])
    leaf_depth = depth[leaf_ids]

    return {
        "n_leaves": n_leaves,
        "n_edges": n_edges,
        "n_zero_length": int(zero_mask.sum()),
        "pct_zero_length": 100.0 * zero_mask.sum() / n_edges,
        "n_zero_length_leaf": n_zero_leaf,
        "n_zero_length_internal": n_zero_internal,
        "n_polytomies": len(poly),
        "polytomy_degree_hist": dict(zip(*np.unique(degrees, return_counts=True))) if degrees else {},
        "polytomy_leaf_span": sorted((int(counts[u]) for u in poly), reverse=True),
        "n_degenerate_leaves": int(is_degenerate.sum()),
        "pct_degenerate_leaves": 100.0 * is_degenerate.sum() / n_leaves,
        "depth_all_median": float(np.median(leaf_depth)),
        "depth_all_max": int(leaf_depth.max()),
        "depth_degenerate_median": float(np.median(leaf_depth[is_degenerate])) if is_degenerate.any() else float("nan"),
        "depth_nondegenerate_median": float(np.median(leaf_depth[~is_degenerate])) if (~is_degenerate).any() else float("nan"),
        "subtree_leaf_count_p50": float(np.median(counts)),
        "subtree_leaf_count_p90": float(np.percentile(counts, 90)),
    }


def densest_degenerate_window(tree: ParsedTree, degenerate: set[str], window: int = 300) -> tuple[int, int]:
    """How concentrated the degenerate leaves are in the tree's leaf ordering.

    Returns (largest count of degenerate leaves in any `window`-leaf slice of
    DFS leaf order, total degenerate leaf count). A high ratio means the
    degenerate leaves form one localised region rather than being scattered.
    """
    leaves = leaf_dfs_order(tree)
    n = len(leaves)
    indicator = np.array([1 if tree.label[i] in degenerate else 0 for i in leaves])
    total = int(indicator.sum())
    if total == 0 or n <= window:
        return total, total
    csum = np.cumsum(indicator)
    windowed = csum[window - 1 :] - np.concatenate([[0], csum[:-window]])
    return int(windowed.max()), total


@dataclass
class RadialLayout:
    """Node positions and edge segments for a radial tree drawing."""

    x: np.ndarray
    y: np.ndarray
    edges: np.ndarray  # (n_edges, 2, 2): [[x_parent, y_parent], [x_child, y_child]]


def radial_layout(tree: ParsedTree) -> RadialLayout:
    """Equal-angle radial layout: leaves evenly spaced by angle, radius = cumulative
    branch length from the root, internal-node angle = mean of children's angles.
    """
    n = tree.n_nodes
    theta = np.zeros(n, dtype=np.float64)
    leaves = leaf_dfs_order(tree)
    n_leaves = len(leaves)
    for i, leaf in enumerate(leaves):
        theta[leaf] = 2.0 * np.pi * i / n_leaves

    for u in post_order(tree):
        if not tree.is_leaf[u]:
            theta[u] = float(np.mean([theta[c] for c in tree.children[u]]))

    _, cumlen = depth_and_cumlen(tree)
    x = cumlen * np.cos(theta)
    y = cumlen * np.sin(theta)

    non_root = np.array([v for v in range(n) if v != tree.root], dtype=np.int64)
    par = tree.parent[non_root]
    edges = np.stack(
        [np.stack([x[par], y[par]], axis=1), np.stack([x[non_root], y[non_root]], axis=1)],
        axis=1,
    )
    return RadialLayout(x=x, y=y, edges=edges)
