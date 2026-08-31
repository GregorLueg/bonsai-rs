//! Unsupervised clustering by iterative branch cutting, and the root selection
//! that follows from it (SPEC.md section 9 step 7, section 7.2).
//!
//! Cut a branch and the tree falls into two pieces; cut `k - 1` branches and it
//! falls into `k`. The clustering of the Methods picks those branches greedily,
//! each cut chosen to minimise the sum, over the pieces that result, of the
//! pairwise distances between the leaves within a piece. Distances are always
//! sums of branch lengths along the tree path (SPEC.md section 14), never
//! anything geometric.
//!
//! ### The `a * b` edge weighting
//!
//! Written out, the objective looks quadratic in the leaf count:
//!
//! ```text
//! S(T) = sum_{i < j} d(i, j)
//! ```
//!
//! It is not. `d(i, j)` is the sum of the lengths of the branches on the path
//! from `i` to `j`, so exchanging the order of summation counts each branch
//! once per leaf pair whose path crosses it. Deleting a branch splits the
//! leaves into `a` on one side and `b` on the other, and a path crosses that
//! branch exactly when it joins one side to the other, which happens for `a * b`
//! of the pairs. So
//!
//! ```text
//! S(T) = sum_{branches e} len(e) * a(e) * b(e)
//! ```
//!
//! which is one pass over the edges. This is the whole reason the procedure is
//! usable: it turns an `O(n^2)` sum into an `O(n)` one, and `a(e)` is just the
//! leaf count below `e`, settled by a single ascending scan of the arena.
//!
//! Note what the weighting is *not*: it is not the sum of branch lengths. A
//! branch in the middle of a tree carries far more leaf-to-leaf paths than one
//! at a tip, and `a * b` is exactly how much more.
//!
//! ### Which branch to cut
//!
//! Splitting a piece into `A` and `B` leaves
//! `S(A) + S(B) = S(A u B) - sum_{i in A, j in B} d(i, j)`, since the pairs that
//! stop being counted are exactly the crossing ones. So *minimising* what is
//! left is *maximising* the crossing sum, and the crossing sum has the same
//! rearrangement:
//!
//! ```text
//! cross(e) = b * D(v) + a * U(v)
//! ```
//!
//! where `v` is the lower end of `e`, `a` and `b` are the leaf counts below and
//! above it, `D(v)` is the summed distance from `v` down to the leaves below it
//! and `U(v)` the summed distance from `v` out to the leaves above it. `D` comes
//! from an ascending scan and `U` from the descending one that follows, so every
//! branch of a piece is scored in two linear passes.
//!
//! Only branches with a leaf on both sides are candidates, which is what keeps
//! every cluster non-empty and makes `k` clusters reachable for any
//! `k <= n_leaves`.
//!
//! ### The root
//!
//! [`root_edge`] is the branch this procedure would cut first and [`reroot`]
//! puts a new node on it. **This changes nothing about the model.** The
//! loglikelihood does not depend on where the root sits (SPEC.md section 2,
//! S14); the root is a bookkeeping choice that fixes what "downstream" means,
//! and moving it is a display decision and nothing more. A reader who assumes
//! otherwise will go looking for a likelihood change that is not there.
//!
//! ### Depth
//!
//! Every traversal here is a flat scan over the arena or an explicit stack.
//! Bonsai trees can be deep and laddery ([`Tree::ladder`] exists to exercise
//! that), and a recursive version would overflow the stack on a real dataset,
//! which is the same reason [`crate::tree::layout`] is written the way it is.

use crate::errors::BonsaiErrors;
use crate::tree::{NO_NODE, Tree};

///////////////
// Constants //
///////////////

/// Fraction of the cut branch that ends up below the new root, on the side of
/// the node the branch hangs from.
///
/// Ours, chosen 2026-08-31. The midpoint is the only split that does not depend
/// on which end of the branch you name first, and any split at all preserves
/// every leaf-to-leaf path length, since the two halves sum to the original. A
/// half is also exact in binary floating point, so the two pieces of a branch
/// add back to it bit for bit.
pub const ROOT_SPLIT: f64 = 0.5;

/////////////////
// The results //
/////////////////

/// A partition of the leaves into clusters, and what it cost.
///
/// Clusters are numbered by decreasing leaf count, ties broken by the smallest
/// leaf index they contain, so the numbering is a function of the tree alone.
#[derive(Clone, Debug)]
pub struct Clustering {
    /// Cluster of each leaf, indexed by leaf, each in `0..n_clusters`.
    pub leaf_cluster: Vec<u32>,
    /// Representative node of each cluster: the node of that piece of the tree
    /// minimising the summed distance to the piece's leaves, ties broken by the
    /// smallest node index. A singleton cluster is represented by its own leaf.
    pub centres: Vec<u32>,
    /// Leaf count of each cluster.
    pub sizes: Vec<usize>,
    /// Nodes whose upstream branch was cut, in the order they were cut. One
    /// shorter than the cluster count.
    pub cuts: Vec<u32>,
    /// Summed within-cluster pairwise leaf distance, the quantity the cutting
    /// minimises.
    pub objective: f64,
}

impl Clustering {
    /// Number of clusters.
    ///
    /// ### Returns
    ///
    /// The cluster count, always at least one.
    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.centres.len()
    }
}

//////////////////
// The objective //
//////////////////

/// Summed pairwise distance between every pair of leaves of a tree.
///
/// The `a * b` form derived in the module docs, so one pass over the arena
/// rather than the `O(n^2)` double loop over leaf pairs.
///
/// ### Params
///
/// * `tree` - Tree to measure
///
/// ### Returns
///
/// `sum_{i < j} d(i, j)` over the leaves, zero for a tree with one leaf.
pub fn summed_leaf_distance(tree: &Tree) -> f64 {
    let n_nodes = tree.n_nodes();
    let n_leaves = tree.n_leaves();
    let mut below = vec![0u32; n_nodes];
    for (i, slot) in below.iter_mut().enumerate() {
        *slot = u32::from(i < n_leaves);
    }
    // Ascending index order is a post-order (module docs of `crate::tree`), so
    // one forward scan settles every count and accumulates the sum on the way.
    let mut total = 0.0f64;
    for v in 0..n_nodes {
        let Some(p) = tree.parent(v as u32) else {
            continue;
        };
        let a = f64::from(below[v]);
        total += tree.branch(v as u32) * a * (n_leaves as f64 - a);
        below[p as usize] += below[v];
    }
    total
}

////////////////
// Clustering //
////////////////

/// Cluster the leaves by iteratively cutting branches.
///
/// The procedure of the module docs: `n_clusters - 1` cuts, each the branch
/// that minimises the summed within-piece pairwise leaf distance over the
/// pieces it produces. Every cut keeps at least one leaf on each side, so no
/// cluster is empty and the request is always met exactly.
///
/// Deterministic: ties on the objective are broken by the smaller node index,
/// and nothing here is parallel or randomised.
///
/// Cost is `O(n_nodes * n_clusters)` in the worst case, since a cut recomputes
/// only the two pieces it created and those can be as lopsided as the tree is.
/// On a balanced tree it is closer to `O(n_nodes * log n_clusters)`.
///
/// ### Params
///
/// * `tree` - Tree whose leaves to cluster
/// * `n_clusters` - Requested cluster count, clamped into `1..=n_leaves`
///
/// ### Returns
///
/// The clustering.
pub fn cluster(tree: &Tree, n_clusters: usize) -> Clustering {
    let n_leaves = tree.n_leaves();
    let wanted = n_clusters.clamp(1, n_leaves);
    let (pieces, cuts) = cut_greedily(tree, wanted);

    // Order clusters by decreasing size, ties by the smallest leaf they hold.
    // Both keys are functions of the tree alone, so the numbering is too.
    let mut keyed: Vec<(usize, u32, usize)> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.n_leaves, p.first_leaf, i))
        .collect();
    keyed.sort_unstable_by(|l, r| r.0.cmp(&l.0).then(l.1.cmp(&r.1)));

    let mut leaf_cluster = vec![0u32; n_leaves];
    let mut centres = Vec::with_capacity(keyed.len());
    let mut sizes = Vec::with_capacity(keyed.len());
    let mut objective = 0.0f64;
    for (slot, &(size, _, i)) in keyed.iter().enumerate() {
        let piece = &pieces[i];
        for &v in &piece.nodes {
            if (v as usize) < n_leaves {
                leaf_cluster[v as usize] = slot as u32;
            }
        }
        centres.push(piece.centre);
        sizes.push(size);
        objective += piece.objective;
    }

    Clustering {
        leaf_cluster,
        centres,
        sizes,
        cuts,
        objective,
    }
}

/// Nodes spread over the tree by the cutting procedure, one per cluster.
///
/// What SPEC.md section 7.2 asks for as the start points of the placement beam
/// search: the centres of the distance-based clustering, rather than an
/// arbitrary spread. Centres come back with the largest cluster first, which
/// matters where the caller gives the first start point a privileged position,
/// as [`crate::model::place::place`] does with its shared visited set.
///
/// Fewer than `n_centres` nodes come back when the tree has fewer leaves than
/// that, since a cluster per leaf is as fine as the partition goes.
///
/// ### Params
///
/// * `tree` - Tree to spread points over
/// * `n_centres` - Requested number of centres, clamped into `1..=n_leaves`
///
/// ### Returns
///
/// The centres, largest cluster first. Never empty, and never repeats a node.
pub fn cluster_centres(tree: &Tree, n_centres: usize) -> Vec<u32> {
    cluster(tree, n_centres).centres
}

////////////////////
// Root selection //
////////////////////

/// The branch the clustering would cut first, identified by its lower node.
///
/// SPEC.md section 9 step 7 puts the root here. The choice affects the drawing
/// and nothing else: the loglikelihood is independent of the root (SPEC.md
/// section 2, S14).
///
/// ### Params
///
/// * `tree` - Tree to place a root on
///
/// ### Returns
///
/// The node whose upstream branch would be cut first, or `MalformedTree` if the
/// tree is a single leaf and so has no branch to put a root on.
pub fn root_edge(tree: &Tree) -> Result<u32, BonsaiErrors> {
    let mut scratch = Scratch::new(tree.n_nodes());
    let mut whole = Piece::whole(tree);
    whole.analyse(tree, &mut scratch);
    whole
        .best
        .map(|(node, _)| node)
        .ok_or_else(|| BonsaiErrors::MalformedTree {
            reason: format!(
                "a tree with {} node(s) and {} leaf/leaves has no branch to root on",
                tree.n_nodes(),
                tree.n_leaves()
            ),
        })
}

/// Rebuild a tree with a fresh root sitting on one branch.
///
/// The branch above `edge` is replaced by two, meeting at a new node that
/// becomes the root; [`ROOT_SPLIT`] says where along it they meet. The two
/// halves sum to the original, so every leaf-to-leaf path length is unchanged
/// and so, by S14, is the loglikelihood.
///
/// **Leaf indices are preserved.** Only internal nodes are renumbered, so any
/// per-leaf data the caller holds, a [`crate::model::likelihood::NodeState`]
/// included, indexes the result unchanged.
///
/// ### The degree-two root
///
/// The result's root has exactly two children, which is what "on a branch"
/// means. That is the ordinary rooted representation of an unrooted tree and it
/// is what every tree in this crate already looks like, [`Tree::balanced_binary`]
/// included; the pruning recursion scores it identically to the same tree rooted
/// anywhere else. What a degree-two node *is* degenerate for is branch-length
/// optimisation, as the docs in [`crate::model::global`] set out: only the sum
/// of the two branches below it is identifiable, so the optimiser sees a flat
/// direction along them. That is a reason to reroot for display after the search
/// rather than during it, not a reason to avoid it.
///
/// If the old root had exactly two children it is suppressed, its two branches
/// merged into one, since rerooting elsewhere would otherwise leave it with a
/// single child and no arena holds that. So the result has the same node count
/// as the input for a binary-rooted tree, and one more for a polytomous one.
///
/// ### Params
///
/// * `tree` - Tree to reroot
/// * `edge` - Node whose upstream branch the root goes on; not the root itself
///
/// ### Returns
///
/// The rerooted tree, `NodeOutOfRange` if `edge` is not a node, or
/// `MalformedTree` if it is the root, which has no upstream branch.
pub fn reroot(tree: &Tree, edge: u32) -> Result<Tree, BonsaiErrors> {
    let n = tree.n_nodes();
    if edge as usize >= n {
        return Err(BonsaiErrors::NodeOutOfRange {
            index: edge as usize,
            n_nodes: n,
        });
    }
    let Some(above) = tree.parent(edge) else {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("node {edge} is the root and has no upstream branch to root on"),
        });
    };

    let n_leaves = tree.n_leaves();
    let old_root = tree.root();
    let new_root = n as u32;

    let mut parent: Vec<u32> = (0..n as u32)
        .map(|v| tree.parent(v).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    parent.push(NO_NODE);
    branch.push(0.0);

    // Everything off the root-ward path keeps its parent; the path from `above`
    // up to the old root reverses, each node inheriting the length of the edge
    // it used to point up along.
    let mut chain = vec![above];
    let mut walk = above;
    while let Some(p) = tree.parent(walk) {
        chain.push(p);
        walk = p;
    }

    let len = tree.branch(edge);
    let lower = ROOT_SPLIT * len;
    parent[edge as usize] = new_root;
    branch[edge as usize] = lower;

    let mut carried = len - lower;
    let mut prev = new_root;
    for &u in &chain {
        let next = branch[u as usize];
        parent[u as usize] = prev;
        branch[u as usize] = carried;
        carried = next;
        prev = u;
    }

    // The old root loses one child to the reversal; if it had only two it is now
    // a degree-two node hanging in the middle of the tree, which the arena
    // cannot hold and which contributes nothing anyway.
    let orphaned = match chain.len() {
        1 => edge,
        k => chain[k - 2],
    };
    let mut remaining = tree
        .children(old_root)
        .iter()
        .copied()
        .filter(|&c| c != orphaned);
    let dead = match (remaining.next(), remaining.next()) {
        (Some(only), None) => {
            branch[only as usize] += branch[old_root as usize];
            parent[only as usize] = parent[old_root as usize];
            Some(old_root)
        }
        _ => None,
    };

    relabel_from(parent, branch, n_leaves, new_root, dead)
}

/// Reroot a tree onto the branch the clustering would cut first.
///
/// [`root_edge`] then [`reroot`]. This is SPEC.md section 9 step 7 in full, and
/// it is a display step: the loglikelihood is unchanged (SPEC.md section 2,
/// S14).
///
/// ### Params
///
/// * `tree` - Tree to reroot
///
/// ### Returns
///
/// The rerooted tree, or `MalformedTree` if the tree is a single leaf.
pub fn reroot_for_display(tree: &Tree) -> Result<Tree, BonsaiErrors> {
    reroot(tree, root_edge(tree)?)
}

/// Renumber a rerooted parent array into the arena's ordering and build a tree.
///
/// [`Tree::from_parents`] relabels internal nodes into level order but *checks*
/// rather than fixes the requirement that every parent index exceed its
/// children's, so a rerooted array has to be brought into that shape first. A
/// breadth-first walk from the new root lists parents before children;
/// numbering the internal nodes in the reverse of that order therefore puts
/// every parent above its children. Leaves keep their indices, which is what
/// lets caller-held per-leaf data survive a reroot.
///
/// ### Params
///
/// * `parent` - Parent of each node in the rerooted tree, `NO_NODE` for the root
/// * `branch` - Length of the branch above each node, same indexing
/// * `n_leaves` - Leaf count; leaves occupy `0..n_leaves` in both indexings
/// * `new_root` - Index of the root in the incoming arrays
/// * `dead` - Node dropped by degree-two suppression, if any
///
/// ### Returns
///
/// The tree, or `MalformedTree` if the arrays do not describe one connected
/// tree over the surviving nodes.
fn relabel_from(
    parent: Vec<u32>,
    branch: Vec<f64>,
    n_leaves: usize,
    new_root: u32,
    dead: Option<u32>,
) -> Result<Tree, BonsaiErrors> {
    let total = parent.len();
    let live = total - usize::from(dead.is_some());

    let mut ptr = vec![0u32; total + 1];
    for v in 0..total {
        if Some(v as u32) == dead || parent[v] == NO_NODE {
            continue;
        }
        ptr[parent[v] as usize + 1] += 1;
    }
    for i in 0..total {
        ptr[i + 1] += ptr[i];
    }
    let mut cursor = ptr.clone();
    let mut kids = vec![0u32; ptr[total] as usize];
    for v in 0..total {
        if Some(v as u32) == dead || parent[v] == NO_NODE {
            continue;
        }
        let p = parent[v] as usize;
        kids[cursor[p] as usize] = v as u32;
        cursor[p] += 1;
    }

    let mut order = Vec::with_capacity(live);
    order.push(new_root);
    let mut head = 0usize;
    while head < order.len() {
        let u = order[head] as usize;
        head += 1;
        order.extend_from_slice(&kids[ptr[u] as usize..ptr[u + 1] as usize]);
    }
    if order.len() != live {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "rerooting reached {} of {live} nodes, so the parent array is not connected",
                order.len()
            ),
        });
    }

    let mut relabel = vec![NO_NODE; total];
    for (i, slot) in relabel.iter_mut().take(n_leaves).enumerate() {
        *slot = i as u32;
    }
    let mut next = n_leaves as u32;
    for &u in order.iter().rev() {
        if (u as usize) >= n_leaves {
            relabel[u as usize] = next;
            next += 1;
        }
    }

    let mut new_parent = vec![NO_NODE; live];
    let mut new_branch = vec![0.0f64; live];
    for &u in &order {
        let i = relabel[u as usize] as usize;
        new_parent[i] = match parent[u as usize] {
            NO_NODE => NO_NODE,
            p => relabel[p as usize],
        };
        new_branch[i] = branch[u as usize];
    }
    Tree::from_parents(new_parent, new_branch, n_leaves)
}

///////////////////////
// The greedy cutter //
///////////////////////

/// Buffers the per-piece sweeps reuse, sized once for the whole tree.
struct Scratch {
    /// Leaves of the piece below each node.
    count: Vec<u32>,
    /// Summed distance from a node down to the piece's leaves below it.
    down: Vec<f64>,
    /// Summed distance from a node to every leaf of its piece.
    total: Vec<f64>,
    /// Membership flag, set only while a piece is being split.
    mark: Vec<bool>,
}

impl Scratch {
    /// Allocate for a tree of `n_nodes` nodes.
    ///
    /// ### Params
    ///
    /// * `n_nodes` - Node count of the tree
    ///
    /// ### Returns
    ///
    /// Zeroed buffers.
    fn new(n_nodes: usize) -> Self {
        Self {
            count: vec![0; n_nodes],
            down: vec![0.0; n_nodes],
            total: vec![0.0; n_nodes],
            mark: vec![false; n_nodes],
        }
    }
}

/// A connected piece of the tree left by the cuts made so far.
struct Piece {
    /// Its nodes, ascending. The last is its top: every parent sits above its
    /// children in the arena, so the ancestor-most node of a connected piece is
    /// also its highest-numbered one.
    nodes: Vec<u32>,
    /// Original leaves it holds, which is what a cluster is.
    n_leaves: usize,
    /// Smallest leaf index it holds, or `u32::MAX` if it holds none. Only a tie
    /// break for the cluster numbering.
    first_leaf: u32,
    /// Best cut inside it: the node whose upstream branch to cut, and the
    /// reduction in the objective it buys. `None` when nothing inside it can be
    /// cut with a leaf left on both sides.
    best: Option<(u32, f64)>,
    /// Node minimising the summed distance to the piece's leaves.
    centre: u32,
    /// Summed pairwise distance between the leaves it holds.
    objective: f64,
}

impl Piece {
    /// The piece covering an entire tree, before any cut.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to cover
    ///
    /// ### Returns
    ///
    /// The piece, unanalysed.
    fn whole(tree: &Tree) -> Self {
        Self::new((0..tree.n_nodes() as u32).collect())
    }

    /// A piece over a node list, with its measurements left unset.
    ///
    /// ### Params
    ///
    /// * `nodes` - Its nodes, ascending
    ///
    /// ### Returns
    ///
    /// The piece, which [`Piece::analyse`] must fill in before use.
    fn new(nodes: Vec<u32>) -> Self {
        let centre = nodes.first().copied().unwrap_or(0);
        Self {
            nodes,
            n_leaves: 0,
            first_leaf: u32::MAX,
            best: None,
            centre,
            objective: 0.0,
        }
    }

    /// Measure the piece: its leaf count, its objective, its centre and its
    /// best cut.
    ///
    /// Two linear passes over the piece's nodes. Ascending order is a post-order
    /// so the first settles the leaf counts `a` and the downward distance sums
    /// `D`; descending order visits parents first so the second turns `D` into
    /// the summed distance to *every* leaf of the piece, from which the outward
    /// sum `U` follows by subtraction. `cross = b * D + a * U` then scores every
    /// branch, as derived in the module docs.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree the piece belongs to
    /// * `s` - Scratch buffers, whose contents are not preserved
    fn analyse(&mut self, tree: &Tree, s: &mut Scratch) {
        let Some(&top) = self.nodes.last() else {
            return;
        };
        let n_leaves = tree.n_leaves() as u32;

        for &v in &self.nodes {
            s.count[v as usize] = u32::from(v < n_leaves);
            s.down[v as usize] = 0.0;
        }
        for &v in &self.nodes {
            if v == top {
                continue;
            }
            let Some(p) = tree.parent(v) else {
                continue;
            };
            let a = f64::from(s.count[v as usize]);
            s.count[p as usize] += s.count[v as usize];
            s.down[p as usize] += s.down[v as usize] + a * tree.branch(v);
        }

        let m = f64::from(s.count[top as usize]);
        for &v in self.nodes.iter().rev() {
            if v == top {
                s.total[v as usize] = s.down[v as usize];
                continue;
            }
            let Some(p) = tree.parent(v) else {
                continue;
            };
            // Moving the vantage point from `p` to `v` walks towards the `a`
            // leaves below `v` and away from the other `m - a`.
            let a = f64::from(s.count[v as usize]);
            s.total[v as usize] = s.total[p as usize] + tree.branch(v) * (m - 2.0 * a);
        }

        self.n_leaves = s.count[top as usize] as usize;
        self.first_leaf = self
            .nodes
            .first()
            .copied()
            .filter(|&v| v < n_leaves)
            .unwrap_or(u32::MAX);
        self.objective = 0.0;
        self.best = None;
        let mut centre = top;
        let mut centre_total = f64::INFINITY;
        for &v in &self.nodes {
            // Strict comparisons with an ascending scan send every tie to the
            // smallest node index, which is what makes the whole module
            // deterministic.
            if s.total[v as usize] < centre_total {
                centre_total = s.total[v as usize];
                centre = v;
            }
            if v == top {
                continue;
            }
            let a = f64::from(s.count[v as usize]);
            let b = m - a;
            if a == 0.0 || b == 0.0 {
                continue;
            }
            self.objective += tree.branch(v) * a * b;
            let up = s.total[v as usize] - s.down[v as usize];
            let cross = b * s.down[v as usize] + a * up;
            if self.best.is_none_or(|(_, seen)| cross > seen) {
                self.best = Some((v, cross));
            }
        }
        self.centre = centre;
    }
}

/// Cut the tree greedily into `n_clusters` pieces.
///
/// Each round takes the best cut over all pieces, splits the piece it belongs
/// to, and remeasures only the two halves. A piece with a single leaf offers no
/// cut, so the loop stops early if the tree runs out of separable leaves, which
/// the clamp on `n_clusters` already rules out.
///
/// ### Params
///
/// * `tree` - Tree to cut
/// * `n_clusters` - Number of pieces wanted, at least one
///
/// ### Returns
///
/// The pieces, measured, and the cut nodes in the order they were cut.
fn cut_greedily(tree: &Tree, n_clusters: usize) -> (Vec<Piece>, Vec<u32>) {
    let mut scratch = Scratch::new(tree.n_nodes());
    let mut pieces = vec![Piece::whole(tree)];
    pieces[0].analyse(tree, &mut scratch);

    let mut cut = vec![false; tree.n_nodes()];
    let mut cuts = Vec::with_capacity(n_clusters.saturating_sub(1));
    while pieces.len() < n_clusters {
        let mut chosen: Option<(usize, u32, f64)> = None;
        for (i, piece) in pieces.iter().enumerate() {
            let Some((node, cross)) = piece.best else {
                continue;
            };
            let better = match chosen {
                None => true,
                Some((_, seen, best)) => cross > best || (cross == best && node < seen),
            };
            if better {
                chosen = Some((i, node, cross));
            }
        }
        let Some((i, node, _)) = chosen else {
            break;
        };

        cut[node as usize] = true;
        cuts.push(node);
        let mut below = split_off(tree, &mut pieces[i], node, &cut, &mut scratch);
        pieces[i].analyse(tree, &mut scratch);
        below.analyse(tree, &mut scratch);
        pieces.push(below);
    }
    (pieces, cuts)
}

/// Split the subtree below `node` out of a piece.
///
/// The subtree is collected with an explicit stack that stops at branches
/// already cut, then the piece's node list is partitioned in one pass, which
/// keeps both halves in ascending order and so keeps the "top is last"
/// invariant.
///
/// ### Params
///
/// * `tree` - Tree the piece belongs to
/// * `piece` - Piece to split, left holding everything above the cut
/// * `node` - Node whose upstream branch has just been cut
/// * `cut` - Which branches are cut, indexed by their lower node
/// * `s` - Scratch buffers; `mark` is left as it was found
///
/// ### Returns
///
/// The piece below the cut, unanalysed.
fn split_off(tree: &Tree, piece: &mut Piece, node: u32, cut: &[bool], s: &mut Scratch) -> Piece {
    let mut stack = vec![node];
    while let Some(u) = stack.pop() {
        s.mark[u as usize] = true;
        for &c in tree.children(u) {
            if !cut[c as usize] {
                stack.push(c);
            }
        }
    }

    let mut below = Vec::new();
    let mut above = Vec::with_capacity(piece.nodes.len());
    for &v in &piece.nodes {
        if s.mark[v as usize] {
            below.push(v);
        } else {
            above.push(v);
        }
    }
    for &v in &below {
        s.mark[v as usize] = false;
    }
    piece.nodes = above;
    Piece::new(below)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::likelihood::NodeState;
    use crate::tree::simulate::{SimulationParams, simulate_binary};
    use crate::utils::rng::splitmix64_at;
    use approx::assert_relative_eq;
    use std::collections::BTreeSet;

    /// Every leaf-to-leaf distance, the honest `O(n^2)` way.
    ///
    /// A breadth-first walk over the undirected tree from each leaf in turn,
    /// sharing no code and no identity with the thing it checks.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to measure
    ///
    /// ### Returns
    ///
    /// Row-major `[leaf][leaf]` distances.
    fn brute_distances(tree: &Tree) -> Vec<f64> {
        let n = tree.n_nodes();
        let n_leaves = tree.n_leaves();
        let mut out = vec![0.0f64; n_leaves * n_leaves];
        for start in 0..n_leaves {
            let mut dist = vec![f64::NAN; n];
            dist[start] = 0.0;
            let mut queue = vec![start as u32];
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head];
                head += 1;
                let d = dist[u as usize];
                let mut visit = |v: u32, len: f64| {
                    if dist[v as usize].is_nan() {
                        dist[v as usize] = d + len;
                        queue.push(v);
                    }
                };
                for &c in tree.children(u) {
                    visit(c, tree.branch(c));
                }
                if let Some(p) = tree.parent(u) {
                    visit(p, tree.branch(u));
                }
            }
            for leaf in 0..n_leaves {
                out[start * n_leaves + leaf] = dist[leaf];
            }
        }
        out
    }

    /// Summed pairwise leaf distance from the brute-force matrix.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to measure
    ///
    /// ### Returns
    ///
    /// `sum_{i < j} d(i, j)`.
    fn brute_total(tree: &Tree) -> f64 {
        let n_leaves = tree.n_leaves();
        let d = brute_distances(tree);
        let mut total = 0.0;
        for i in 0..n_leaves {
            for j in i + 1..n_leaves {
                total += d[i * n_leaves + j];
            }
        }
        total
    }

    /// Give a tree spread-out branch lengths from the crate's fixed stream.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to modify in place
    /// * `seed` - Offset into the stream
    fn jitter(tree: &mut Tree, seed: u64) {
        let n = tree.n_nodes() as u64;
        for (i, b) in tree.branches_mut().iter_mut().enumerate() {
            *b = 0.05 + 4.0 * splitmix64_at(seed * n + i as u64);
        }
    }

    /// A tree with a polytomy: two cherries and three loose leaves under one
    /// node, itself under the root with a fourth loose leaf.
    ///
    /// ### Returns
    ///
    /// The tree, on nine leaves.
    fn polytomous() -> Tree {
        // Leaves 0..9. Node 9 = (0,1), node 10 = (2,3), node 11 = {9,10,4,5,6},
        // node 12 = {11,7,8}.
        let parent = vec![9, 9, 10, 10, 11, 11, 11, 12, 12, 11, 11, 12, NO_NODE];
        let branch: Vec<f64> = (0..13).map(|i| 0.3 + 0.17 * i as f64).collect();
        Tree::from_parents(parent, branch, 9).expect("fixture is a valid tree")
    }

    /// Every shape the tests sweep over.
    ///
    /// ### Returns
    ///
    /// Named trees, all with non-uniform branch lengths.
    fn shapes() -> Vec<(String, Tree)> {
        let mut out = Vec::new();
        for n in [8usize, 16, 32] {
            let mut t = Tree::balanced_binary(n, 1.0).expect("power of two");
            jitter(&mut t, 1);
            out.push((format!("balanced-{n}"), t));
            let mut t = Tree::ladder(n, 1.0).expect("at least two leaves");
            jitter(&mut t, 2);
            out.push((format!("ladder-{n}"), t));
            let parent: Vec<u32> = (0..n).map(|_| n as u32).chain([NO_NODE]).collect();
            let mut t = Tree::from_parents(parent, vec![1.0; n + 1], n).expect("star");
            jitter(&mut t, 3);
            out.push((format!("star-{n}"), t));
        }
        let mut t = polytomous();
        jitter(&mut t, 4);
        out.push(("polytomy-9".to_string(), t));
        out
    }

    #[test]
    fn test_edge_weighting_matches_brute_force() {
        // The load-bearing identity: `sum_{i<j} d(i,j)` equals the one-pass sum
        // of `len * a * b` over branches. Everything else in the module is
        // built on it.
        for (name, tree) in shapes() {
            let fast = summed_leaf_distance(&tree);
            let slow = brute_total(&tree);
            assert_relative_eq!(fast, slow, max_relative = 1e-12);
            assert!(fast > 0.0, "{name} summed to nothing");
        }
    }

    #[test]
    fn test_piece_objective_matches_brute_force_per_cluster() {
        // The same identity, but restricted to a piece, which is the form the
        // greedy cutter actually uses.
        for (name, tree) in shapes() {
            let n_leaves = tree.n_leaves();
            let d = brute_distances(&tree);
            for k in 1..=n_leaves.min(5) {
                let c = cluster(&tree, k);
                let mut slow = 0.0;
                for i in 0..n_leaves {
                    for j in i + 1..n_leaves {
                        if c.leaf_cluster[i] == c.leaf_cluster[j] {
                            slow += d[i * n_leaves + j];
                        }
                    }
                }
                assert_relative_eq!(c.objective, slow, max_relative = 1e-12, epsilon = 1e-12);
                assert!(c.objective <= summed_leaf_distance(&tree) + 1e-9, "{name}");
            }
        }
    }

    #[test]
    fn test_cutting_reduces_the_objective_monotonically() {
        for (name, tree) in shapes() {
            let mut previous = f64::INFINITY;
            for k in 1..=tree.n_leaves() {
                let c = cluster(&tree, k);
                assert_eq!(c.n_clusters(), k, "{name} at k = {k}");
                assert!(
                    c.objective <= previous + 1e-9,
                    "{name}: objective rose from {previous} to {} at k = {k}",
                    c.objective
                );
                previous = c.objective;
            }
            // Every leaf on its own leaves no pair inside any cluster.
            assert_relative_eq!(previous, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cuts_partition_the_leaves_with_none_empty() {
        for (name, tree) in shapes() {
            let n_leaves = tree.n_leaves();
            for k in 1..=n_leaves {
                let c = cluster(&tree, k);
                assert_eq!(c.cuts.len(), k - 1, "{name}: k - 1 cuts");
                assert_eq!(c.sizes.len(), k);
                assert_eq!(c.sizes.iter().sum::<usize>(), n_leaves, "{name}");
                assert!(c.sizes.iter().all(|&s| s > 0), "{name}: an empty cluster");
                let mut counted = vec![0usize; k];
                for &g in &c.leaf_cluster {
                    counted[g as usize] += 1;
                }
                assert_eq!(counted, c.sizes, "{name}: labels disagree with sizes");
                // Sizes come back sorted, largest first.
                assert!(c.sizes.windows(2).all(|w| w[0] >= w[1]), "{name}");
            }
        }
    }

    #[test]
    fn test_clusters_recover_planted_clades() {
        // Four clades of eight, separated by long branches, with short jittered
        // branches inside them. `tree::simulate`'s generators all give constant
        // or log-uniform branch lengths with no planted separation, so the
        // planting is done here; the jitter still comes from the crate's fixed
        // stream, so the fixture is reproducible.
        let mut tree = Tree::balanced_binary(32, 1.0).expect("power of two");
        for (i, b) in tree.branches_mut().iter_mut().enumerate() {
            *b = 0.05 + 0.1 * splitmix64_at(7 * 63 + i as u64);
        }
        let (lo, hi) = tree.level(2);
        assert_eq!(
            hi - lo,
            4,
            "level 2 of a 32-leaf balanced tree is the clades"
        );
        for v in lo..hi {
            tree.branches_mut()[v] = 20.0;
        }

        let c = cluster(&tree, 4);
        let planted: BTreeSet<BTreeSet<u32>> = (0..4)
            .map(|k| (8 * k..8 * k + 8).collect::<BTreeSet<u32>>())
            .collect();
        let mut found: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); 4];
        for (leaf, &g) in c.leaf_cluster.iter().enumerate() {
            found[g as usize].insert(leaf as u32);
        }
        let found: BTreeSet<BTreeSet<u32>> = found.into_iter().collect();

        println!(
            "planted clades: sizes {:?}, centres {:?}, cuts {:?}",
            c.sizes, c.centres, c.cuts
        );
        println!(
            "objective: {:.3} at k = 1, {:.3} at k = 4, {:.3} at k = 8",
            summed_leaf_distance(&tree),
            c.objective,
            cluster(&tree, 8).objective
        );
        assert_eq!(found, planted, "recovered partition is not the planted one");
        assert_eq!(c.sizes, vec![8, 8, 8, 8]);
        // The cuts are not necessarily the four long branches themselves: the
        // branch above the pair {clade 0, clade 1} carries sixteen leaves at
        // forty apart on either side, which crosses more than any single clade
        // branch does, so the greedy split is the top branch first and one
        // clade branch inside each half. What must hold is that every cut
        // separates whole clades, never splitting one.
        for &node in &c.cuts {
            let below = leaves_below(&tree, node);
            assert_eq!(below.len() % 8, 0, "cut {node} split a clade");
            assert!(
                below.iter().all(|&l| below.contains(&(l - l % 8))),
                "cut {node} split a clade: {below:?}"
            );
        }
    }

    /// Leaves of the subtree below a node.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to walk
    /// * `node` - Root of the subtree
    ///
    /// ### Returns
    ///
    /// The leaves, ascending.
    fn leaves_below(tree: &Tree, node: u32) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        let mut stack = vec![node];
        while let Some(u) = stack.pop() {
            if (u as usize) < tree.n_leaves() {
                out.insert(u);
            }
            stack.extend_from_slice(tree.children(u));
        }
        out
    }

    #[test]
    fn test_reroot_preserves_every_leaf_to_leaf_path() {
        for (name, tree) in shapes() {
            let n_leaves = tree.n_leaves();
            let before = brute_distances(&tree);
            for edge in 0..tree.n_nodes() as u32 - 1 {
                let rerooted = reroot(&tree, edge).expect("every non-root node is an edge");
                assert_eq!(rerooted.n_leaves(), n_leaves);
                let after = brute_distances(&rerooted);
                for i in 0..n_leaves * n_leaves {
                    assert_relative_eq!(before[i], after[i], max_relative = 1e-12);
                }
            }
            // And the objective, which is a function of those distances.
            let rerooted = reroot_for_display(&tree).expect("a tree with branches");
            assert_relative_eq!(
                summed_leaf_distance(&tree),
                summed_leaf_distance(&rerooted),
                max_relative = 1e-12
            );
            assert_eq!(rerooted.children(rerooted.root()).len(), 2, "{name}");
        }
    }

    #[test]
    fn test_reroot_preserves_the_loglikelihood() {
        // The sharpest available statement of "same tree" (SPEC.md section 2,
        // S14). Leaf indices survive a reroot, so one set of leaf rows serves
        // both trees.
        let params = SimulationParams {
            n_leaves: 32,
            n_features: 120,
            seed: 11,
            ..SimulationParams::default()
        };
        let data = simulate_binary::<f64>(Some(params)).expect("simulation");
        let precisions = data.precisions();
        let p = data.n_features;

        let mut state =
            NodeState::new(data.tree.n_nodes(), p, &data.means, &precisions).expect("state");
        let base = state.prune(&data.tree);

        for edge in 0..data.tree.n_nodes() as u32 - 1 {
            let rerooted = reroot(&data.tree, edge).expect("non-root node");
            let mut other =
                NodeState::new(rerooted.n_nodes(), p, &data.means, &precisions).expect("state");
            let moved = other.prune(&rerooted);
            assert_relative_eq!(base, moved, max_relative = 1e-12);
        }

        let display = reroot_for_display(&data.tree).expect("a tree with branches");
        let mut other =
            NodeState::new(display.n_nodes(), p, &data.means, &precisions).expect("state");
        assert_relative_eq!(base, other.prune(&display), max_relative = 1e-12);
    }

    #[test]
    fn test_deterministic() {
        for (name, tree) in shapes() {
            let first = cluster(&tree, 5);
            for _ in 0..3 {
                let again = cluster(&tree, 5);
                assert_eq!(first.leaf_cluster, again.leaf_cluster, "{name}");
                assert_eq!(first.cuts, again.cuts, "{name}");
                assert_eq!(first.centres, again.centres, "{name}");
            }
            let edge = root_edge(&tree).expect("a tree with branches");
            for _ in 0..3 {
                assert_eq!(
                    edge,
                    root_edge(&tree).expect("a tree with branches"),
                    "{name}"
                );
            }
            assert_eq!(
                Some(&edge),
                first.cuts.first(),
                "{name}: root is the first cut"
            );
        }
    }

    #[test]
    fn test_deep_ladder_does_not_overflow_the_stack() {
        // 100k leaves, 200k nodes, one node per level. Recursion dies here.
        let tree = Tree::ladder(100_000, 1.0).expect("at least two leaves");
        let total = summed_leaf_distance(&tree);
        assert!(total.is_finite() && total > 0.0);
        let c = cluster(&tree, 8);
        assert_eq!(c.n_clusters(), 8);
        assert_eq!(c.sizes.iter().sum::<usize>(), 100_000);
        assert!(c.objective < total);
        let rerooted = reroot_for_display(&tree).expect("a tree with branches");
        assert_eq!(rerooted.n_leaves(), 100_000);
        assert_relative_eq!(summed_leaf_distance(&rerooted), total, max_relative = 1e-12);
    }

    #[test]
    fn test_two_leaf_tree() {
        let tree = Tree::from_parents(vec![2, 2, NO_NODE], vec![0.75, 1.25, 0.0], 2)
            .expect("valid two-leaf tree");
        assert_relative_eq!(summed_leaf_distance(&tree), 2.0, max_relative = 1e-12);

        let one = cluster(&tree, 1);
        assert_eq!(one.sizes, vec![2]);
        assert_relative_eq!(one.objective, 2.0, max_relative = 1e-12);

        let two = cluster(&tree, 2);
        assert_eq!(two.sizes, vec![1, 1]);
        assert_relative_eq!(two.objective, 0.0, epsilon = 1e-15);
        assert_eq!(two.leaf_cluster.len(), 2);
        assert_ne!(two.leaf_cluster[0], two.leaf_cluster[1]);
        // A singleton is represented by its own leaf.
        let mut centres = two.centres.clone();
        centres.sort_unstable();
        assert_eq!(centres, vec![0, 1]);

        // Both branches carry the same pair, so the tie goes to the lower index.
        assert_eq!(root_edge(&tree).expect("a tree with branches"), 0);
        let rerooted = reroot_for_display(&tree).expect("a tree with branches");
        assert_eq!(rerooted.n_nodes(), 3);
        assert_relative_eq!(summed_leaf_distance(&rerooted), 2.0, max_relative = 1e-12);
    }

    #[test]
    fn test_star_cuts_the_longest_branches_first() {
        // Around a star every leaf sits at the same place topologically, so the
        // cut order is purely the branch lengths, longest first.
        let parent = vec![6, 6, 6, 6, 6, 6, NO_NODE];
        let branch = vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 0.0];
        let tree = Tree::from_parents(parent, branch, 6).expect("star");
        let c = cluster(&tree, 4);
        assert_eq!(c.cuts, vec![5, 1, 3]);
        assert_eq!(c.sizes, vec![3, 1, 1, 1]);
        assert_eq!(root_edge(&tree).expect("a tree with branches"), 5);
    }

    #[test]
    fn test_polytomy() {
        let tree = polytomous();
        assert_relative_eq!(
            summed_leaf_distance(&tree),
            brute_total(&tree),
            max_relative = 1e-12
        );
        let c = cluster(&tree, 3);
        assert_eq!(c.sizes.iter().sum::<usize>(), 9);
        assert_eq!(c.cuts.len(), 2);
        let rerooted = reroot_for_display(&tree).expect("a tree with branches");
        // The old root had three children, so nothing was suppressed and the
        // new root is an extra node.
        assert_eq!(rerooted.n_nodes(), tree.n_nodes() + 1);
        assert_relative_eq!(
            summed_leaf_distance(&rerooted),
            summed_leaf_distance(&tree),
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_more_clusters_than_leaves_is_clamped() {
        let mut tree = Tree::balanced_binary(8, 1.0).expect("power of two");
        jitter(&mut tree, 5);
        for asked in [8usize, 9, 100, usize::MAX] {
            let c = cluster(&tree, asked);
            assert_eq!(c.n_clusters(), 8);
            assert_eq!(c.sizes, vec![1; 8]);
            assert_relative_eq!(c.objective, 0.0, epsilon = 1e-12);
        }
        assert_eq!(cluster_centres(&tree, 1000).len(), 8);
    }

    #[test]
    fn test_one_cluster_and_zero_clusters() {
        let mut tree = Tree::balanced_binary(8, 1.0).expect("power of two");
        jitter(&mut tree, 6);
        for asked in [0usize, 1] {
            let c = cluster(&tree, asked);
            assert_eq!(c.n_clusters(), 1);
            assert!(c.cuts.is_empty());
            assert_eq!(c.sizes, vec![8]);
            assert_eq!(c.leaf_cluster, vec![0; 8]);
            assert_relative_eq!(
                c.objective,
                summed_leaf_distance(&tree),
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_centres_minimise_the_distance_to_their_cluster() {
        for (name, tree) in shapes() {
            let n_leaves = tree.n_leaves();
            let d_all = brute_distances_from_all(&tree);
            let c = cluster(&tree, 3.min(n_leaves));
            assert_eq!(c.centres.len(), c.n_clusters());
            let mut seen = BTreeSet::new();
            for (g, &centre) in c.centres.iter().enumerate() {
                assert!(seen.insert(centre), "{name}: centre {centre} repeated");
                let cost = |node: u32| -> f64 {
                    (0..n_leaves)
                        .filter(|&leaf| c.leaf_cluster[leaf] as usize == g)
                        .map(|leaf| d_all[node as usize * n_leaves + leaf])
                        .sum()
                };
                let best = cost(centre);
                for node in 0..tree.n_nodes() as u32 {
                    // Only nodes of the same piece are candidates, and a node of
                    // another piece can genuinely be closer, so the check is
                    // against the piece the centre came from: every node whose
                    // cost is lower must belong elsewhere.
                    if cost(node) < best - 1e-9 {
                        assert!(
                            !same_piece(&tree, &c, node, g),
                            "{name}: node {node} beats centre {centre} in its own cluster"
                        );
                    }
                }
            }
        }
    }

    /// Distance from every node to every leaf, row-major `[node][leaf]`.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to measure
    ///
    /// ### Returns
    ///
    /// The distances.
    fn brute_distances_from_all(tree: &Tree) -> Vec<f64> {
        let n = tree.n_nodes();
        let n_leaves = tree.n_leaves();
        let mut out = vec![0.0f64; n * n_leaves];
        for start in 0..n {
            let mut dist = vec![f64::NAN; n];
            dist[start] = 0.0;
            let mut queue = vec![start as u32];
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head];
                head += 1;
                let d = dist[u as usize];
                let mut visit = |v: u32, len: f64| {
                    if dist[v as usize].is_nan() {
                        dist[v as usize] = d + len;
                        queue.push(v);
                    }
                };
                for &c in tree.children(u) {
                    visit(c, tree.branch(c));
                }
                if let Some(p) = tree.parent(u) {
                    visit(p, tree.branch(u));
                }
            }
            out[start * n_leaves..(start + 1) * n_leaves].copy_from_slice(&dist[..n_leaves]);
        }
        out
    }

    /// Whether a node lies in the piece that cluster `g` occupies.
    ///
    /// Recovered from the clustering rather than from the cutter: a node is in
    /// the piece when the nearest leaf reachable without leaving it belongs to
    /// `g`, which for a node of that piece is any of its leaves.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree the clustering came from
    /// * `c` - The clustering
    /// * `node` - Node to test
    /// * `g` - Cluster index
    ///
    /// ### Returns
    ///
    /// Whether the node sits inside that cluster's piece.
    fn same_piece(tree: &Tree, c: &Clustering, node: u32, g: usize) -> bool {
        // Walk out from the node and stop at any leaf; the first leaves met are
        // the ones sharing its piece, since a piece is cut at branches only.
        let n = tree.n_nodes();
        let mut seen = vec![false; n];
        seen[node as usize] = true;
        let mut stack = vec![node];
        while let Some(u) = stack.pop() {
            if (u as usize) < tree.n_leaves() {
                if c.leaf_cluster[u as usize] as usize == g {
                    return true;
                }
                continue;
            }
            for &v in tree.children(u).iter().chain(tree.parent(u).iter()) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        false
    }

    #[test]
    fn test_reroot_rejects_the_root_and_out_of_range_nodes() {
        let tree = Tree::balanced_binary(4, 1.0).expect("power of two");
        assert!(matches!(
            reroot(&tree, tree.root()),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            reroot(&tree, tree.n_nodes() as u32),
            Err(BonsaiErrors::NodeOutOfRange { .. })
        ));
    }

    #[test]
    fn test_single_leaf_tree_has_no_root_branch() {
        let tree = Tree::from_parents(vec![NO_NODE], vec![0.0], 1).expect("one leaf is a tree");
        assert_relative_eq!(summed_leaf_distance(&tree), 0.0, epsilon = 1e-15);
        let c = cluster(&tree, 4);
        assert_eq!(c.n_clusters(), 1);
        assert_eq!(c.centres, vec![0]);
        assert!(matches!(
            root_edge(&tree),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_rerooting_twice_is_stable() {
        // Rerooting a tree already rooted on its first cut should find the same
        // branch again, so the display root is a fixed point.
        let mut tree = Tree::balanced_binary(16, 1.0).expect("power of two");
        jitter(&mut tree, 8);
        let once = reroot_for_display(&tree).expect("a tree with branches");
        let twice = reroot_for_display(&once).expect("a tree with branches");
        assert_eq!(once.n_nodes(), twice.n_nodes());
        let a = brute_distances(&once);
        let b = brute_distances(&twice);
        for i in 0..a.len() {
            assert_relative_eq!(a[i], b[i], max_relative = 1e-12);
        }
    }
}
