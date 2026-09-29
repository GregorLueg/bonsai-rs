//! Placing a node on an existing tree: which node should `q` hang off, used to
//! regraft the pruned subtree of an SPR move.
//!
//! Attaching `q` below `a` adds the old tree's loglikelihood, `q`'s own, and one
//! edge term. The first two do not depend on `a` (the likelihood is root
//! independent), so the edge term alone ranks attachment points and
//! `model::branch` already solves it.

use crate::errors::BonsaiErrors;
use crate::model::branch::optimise_edge_loglik;
use crate::model::merge::EffLeaf;
use crate::tree::Tree;
use crate::utils::kernels::prep_edge;
use crate::utils::traits::BonsaiFloat;

////////////
// Consts //
////////////

/// Beam tolerance, in nats of loglikelihood.
///
/// A neighbour is recursed into when its score is within this of the best so
/// far. Zero is greedy, infinity exhaustive. Ours: the smallest value in a grid
/// of 2 to 6 that returns the exhaustive node on every swept fixture; smaller
/// values fail badly on ladders. It is an absolute difference, not scaled by
/// feature count, as neighbouring gaps stay `O(1)`.
const DEFAULT_TOLERANCE: f64 = 4.0;

/// Number of start points the beam search fans out from (SPEC.md section 7.2).
///
/// Ours, not the paper's `log(n)`. A balanced tree needs one; a ladder fails at
/// fewer than eight (measured over the fixtures behind [`DEFAULT_TOLERANCE`]).
const DEFAULT_STARTS: usize = 8;

///////////
// Types //
///////////

/// Tuning knobs for the beam search of SPEC.md section 7.2.
///
/// Neither changes the model; both trade attachment scores evaluated (one root
/// find over `p` features each) against the risk of a local optimum.
#[derive(Clone, Copy, Debug)]
pub struct PlacementParams {
    /// How far below the best score seen so far a neighbour may fall and still
    /// be recursed into, in nats. `0.0` is greedy, `f64::INFINITY` exhaustive.
    pub tolerance: f64,
    /// Number of start points, clamped to at least one and at most the node
    /// count.
    pub n_starts: usize,
}

impl Default for PlacementParams {
    /// The shipped defaults: a tolerance of four nats and eight start points.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            n_starts: DEFAULT_STARTS,
        }
    }
}

/// The result of attaching a node below one particular node of the tree.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    /// Optimal length of the new edge. Zero means the two effective leaves are
    /// already closer than their combined uncertainty, which is the case
    /// SPEC.md section 7.3's polytomy resolution then has to deal with.
    pub branch: f64,
    /// The edge's loglikelihood at that length, which is the attachment score.
    /// Comparable across attachment points, not an absolute quantity.
    pub loglik: f64,
}

/// Where a node should be attached, and what that costs.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Node of the existing tree to attach below.
    pub node: u32,
    /// Optimal length of the new edge.
    pub branch: f64,
    /// Attachment score there, the best found by the search.
    pub loglik: f64,
    /// How many distinct nodes were scored; the search's cost in `O(p)` root
    /// finds.
    pub scored: usize,
}

/// What the beam search needs from the tree it walks.
///
/// A [`Tree`] is one; the other is the view an SPR cut would leave behind (see
/// [`crate::search::masked`]). Both must present the same start points and
/// neighbour order.
pub(crate) trait Walk {
    /// Size of the node id space, for the visited set.
    fn id_space(&self) -> usize;
    /// The spread start points, root first; [`start_points`] for a tree.
    fn spread_starts(&self, n_starts: usize) -> Vec<u32>;
    /// A node's children in arena order, then its parent.
    fn neighbours(&self, node: u32, out: &mut Vec<u32>);
}

impl Walk for Tree {
    fn id_space(&self) -> usize {
        self.n_nodes()
    }

    fn spread_starts(&self, n_starts: usize) -> Vec<u32> {
        start_points(self, n_starts)
    }

    fn neighbours(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend(neighbours(self, node));
    }
}

/////////////////////
// Private helpers //
/////////////////////

/// Neighbours of a node in the unrooted sense: its children and its parent.
///
/// The likelihood is unrooted, so the search must be able to walk upwards.
///
/// ### Params
///
/// * `tree` - Tree being searched
/// * `node` - Node whose neighbours are wanted
///
/// ### Returns
///
/// An iterator over the neighbours, children first in arena order and the
/// parent last. The fixed order makes the search deterministic.
fn neighbours(tree: &Tree, node: u32) -> impl Iterator<Item = u32> + '_ {
    tree.children(node).iter().copied().chain(tree.parent(node))
}

///////////////
// Functions //
///////////////

/// Score attaching `q` below a node summarised by the effective leaf `a`.
///
/// ```text
/// dL(t) = -1/2 * sum_g [ log(t + 1/W[g,a] + 1/W[g,q])
///                        + (M[g,a] - M[g,q])^2 / (t + 1/W[g,a] + 1/W[g,q]) ]
/// ```
///
/// which is the edge expression with `s[g] = 1/W[g,a] + 1/W[g,q]` and
/// `d[g] = (M[g,a] - M[g,q])^2`, as produced by `prep_edge` and maximised by
/// `optimise_edge_loglik`.
///
/// ### Params
///
/// * `a` - Effective leaf summarising the whole existing tree seen from the
///   candidate node
/// * `q` - Effective leaf summarising the node being attached
/// * `s` - Scratch for the summed inverse precisions, length `p`
/// * `d` - Scratch for the squared separations, length `p`
///
/// ### Returns
///
/// The optimal branch length and the score there, or `RootFindDiverged` if the
/// bracketed solve fails.
pub fn attachment_score<T: BonsaiFloat>(
    a: EffLeaf<'_, T>,
    q: EffLeaf<'_, T>,
    s: &mut [f64],
    d: &mut [f64],
) -> Result<Attachment, BonsaiErrors> {
    debug_assert_eq!(a.m.len(), a.w.len());
    debug_assert_eq!(q.m.len(), q.w.len());
    debug_assert_eq!(a.m.len(), q.m.len());
    debug_assert_eq!(a.m.len(), s.len());
    debug_assert_eq!(a.m.len(), d.len());

    let upper = prep_edge(a.m, a.w, q.m, q.w, s, d);
    let (branch, loglik) = optimise_edge_loglik(s, d, upper)?;
    Ok(Attachment { branch, loglik })
}

/// Start points for the beam search.
///
/// **Placeholder.** SPEC.md section 7.2 takes the centres of a distance-based
/// clustering this crate does not have; this spreads the starts evenly over the
/// node index, which samples the leaves and then every internal level (arena
/// order).
///
/// The root comes first: starts share one visited set (see [`place`]), so the
/// first is the only one guaranteed to explore unimpeded, which makes multi-start
/// no worse than a single search from the root.
///
/// ### Params
///
/// * `tree` - Tree being searched
/// * `n_starts` - Requested number of start points, clamped into `1..=n_nodes`
///
/// ### Returns
///
/// The start points, root first. Never empty.
pub fn start_points(tree: &Tree, n_starts: usize) -> Vec<u32> {
    let n = tree.n_nodes();
    let k = n_starts.clamp(1, n);
    let mut out = Vec::with_capacity(k);
    out.push(tree.root());
    // May collide with the root as `k` approaches `n`; the visited set absorbs it.
    for i in 1..k {
        out.push((i * n / k) as u32);
    }
    out
}

/// Find the best node of a tree to attach `q` below.
///
/// From each start point, score the node, score each unvisited neighbour, and
/// recurse into those whose score is within `tolerance` of the best score seen
/// anywhere so far. The best node over all start points wins.
///
/// The comparison is against the best seen before the neighbour is folded in,
/// so a tolerance of zero is hill-climbing and `f64::INFINITY` is exhaustive.
///
/// Each node is scored at most once (one visited set across all starts). The
/// search is sequential and depth-first with a fixed neighbour order, so the
/// answer does not depend on the thread count.
///
/// `eff` maps a node index to the effective leaf of the entire existing tree
/// collapsed onto that node (SPEC.md section 4 with that node as root). It must
/// be defined for every index in `0..tree.n_nodes()` and return leaves with
/// `q`'s feature count. Do not pass [`crate::model::global::UpState`] rows
/// directly: they sit at the node's parent and are not diffused along the branch
/// above. Use `model::global::collapse_onto_every_node`.
///
/// ### Params
///
/// * `tree` - The existing tree, which is not modified
/// * `q` - Effective leaf summarising the node being attached
/// * `eff` - Effective leaf of the whole tree seen from each node; see above
/// * `params` - Search parameters, or `None` for [`PlacementParams::default`]
///
/// ### Returns
///
/// The best attachment found, or `RootFindDiverged` if a branch-length solve
/// fails. The caller still owes the polytomy resolution of SPEC.md section 7.3.
pub fn place<'a, T: BonsaiFloat, F>(
    tree: &Tree,
    q: EffLeaf<'_, T>,
    eff: F,
    params: Option<PlacementParams>,
) -> Result<Placement, BonsaiErrors>
where
    F: Fn(u32) -> EffLeaf<'a, T>,
{
    place_walk(tree, q, eff, params)
}

/// [`place`] over anything the search can walk.
///
/// ### Params
///
/// * `walk` - The tree, or a view of one
/// * `q` - Effective leaf summarising the node being attached
/// * `eff` - Effective leaf of the whole tree seen from each node
/// * `params` - Search parameters, or `None` for [`PlacementParams::default`]
///
/// ### Returns
///
/// As [`place`], with the node in the walk's own ids.
pub(crate) fn place_walk<'a, T: BonsaiFloat, F, W: Walk>(
    walk: &W,
    q: EffLeaf<'_, T>,
    eff: F,
    params: Option<PlacementParams>,
) -> Result<Placement, BonsaiErrors>
where
    F: Fn(u32) -> EffLeaf<'a, T>,
{
    let params = params.unwrap_or_default();
    let p = q.m.len();
    let mut s = vec![0.0f64; p];
    let mut d = vec![0.0f64; p];

    let mut visited = vec![false; walk.id_space()];
    let mut stack: Vec<u32> = Vec::new();
    let mut around: Vec<u32> = Vec::new();
    let spread = walk.spread_starts(params.n_starts);
    // The root is the first start and always scored, so this is overwritten.
    let mut best = Placement {
        node: spread[0],
        branch: 0.0,
        loglik: f64::NEG_INFINITY,
        scored: 0,
    };

    for start in spread {
        if visited[start as usize] {
            continue;
        }
        visited[start as usize] = true;
        let at = attachment_score(eff(start), q, &mut s, &mut d)?;
        best.scored += 1;
        if at.loglik > best.loglik {
            best.node = start;
            best.branch = at.branch;
            best.loglik = at.loglik;
        }
        stack.push(start);

        while let Some(a) = stack.pop() {
            walk.neighbours(a, &mut around);
            for &nb in &around {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let at = attachment_score(eff(nb), q, &mut s, &mut d)?;
                best.scored += 1;

                // Threshold against the incumbent before the neighbour joins it.
                let admit = at.loglik > best.loglik - params.tolerance;
                if at.loglik > best.loglik {
                    best.node = nb;
                    best.branch = at.branch;
                    best.loglik = at.loglik;
                }
                if admit {
                    stack.push(nb);
                }
            }
        }
    }

    Ok(best)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::NO_NODE;
    use crate::utils::rng::SplitMix64;
    use approx::assert_relative_eq;

    /// A tree with data simulated on it, plus the two-sided effective leaves.
    ///
    /// Everything is `f64` and flat, `[node][feature]`.
    struct Fixture {
        tree: Tree,
        p: usize,
        /// Effective mean of the whole tree collapsed onto each node.
        eff_m: Vec<f64>,
        /// Effective precision of the whole tree collapsed onto each node.
        eff_w: Vec<f64>,
        /// Leaf means as measured, `[leaf][feature]`.
        leaf_m: Vec<f64>,
        /// Leaf precisions as measured, `[leaf][feature]`.
        leaf_w: Vec<f64>,
    }

    impl Fixture {
        /// Simulate leaf data on a tree and collapse it onto every node.
        ///
        /// Positions random-walk down the tree in the units of SPEC.md section
        /// 3.1, then leaves pick up measurement noise of standard deviation
        /// `sigma`.
        ///
        /// ### Params
        ///
        /// * `tree` - Tree to simulate on
        /// * `p` - Number of features
        /// * `sigma` - Measurement standard deviation on every leaf and feature
        /// * `seed` - Generator seed
        ///
        /// ### Returns
        ///
        /// The fixture.
        fn new(tree: Tree, p: usize, sigma: f64, seed: u64) -> Self {
            let n = tree.n_nodes();
            let mut rng = SplitMix64::new(seed);
            let mut pos = vec![0.0f64; n * p];
            // Parents have larger indices than their children, so descending
            // index order visits every parent before its children.
            for node in (0..n - 1).rev() {
                let par = match tree.parent(node as u32) {
                    Some(par) => par as usize,
                    None => continue,
                };
                let sd = tree.branch(node as u32).sqrt();
                for g in 0..p {
                    pos[node * p + g] = pos[par * p + g] + sd * rng.normal();
                }
            }
            let n_leaves = tree.n_leaves();
            let mut leaf_m = vec![0.0f64; n_leaves * p];
            let leaf_w = vec![1.0 / (sigma * sigma); n_leaves * p];
            for i in 0..n_leaves {
                for g in 0..p {
                    leaf_m[i * p + g] = pos[i * p + g] + sigma * rng.normal();
                }
            }

            let mut eff_m = vec![0.0f64; n * p];
            let mut eff_w = vec![0.0f64; n * p];
            for a in 0..n {
                let (w, m) = collapse(&tree, &leaf_m, &leaf_w, p, a as u32, NO_NODE);
                eff_w[a * p..(a + 1) * p].copy_from_slice(&w);
                eff_m[a * p..(a + 1) * p].copy_from_slice(&m);
            }

            Self {
                tree,
                p,
                eff_m,
                eff_w,
                leaf_m,
                leaf_w,
            }
        }

        /// The effective leaf of the whole tree seen from a node.
        ///
        /// ### Params
        ///
        /// * `node` - Node to collapse onto
        ///
        /// ### Returns
        ///
        /// The effective leaf.
        fn eff(&self, node: u32) -> EffLeaf<'_, f64> {
            let lo = node as usize * self.p;
            EffLeaf {
                m: &self.eff_m[lo..lo + self.p],
                w: &self.eff_w[lo..lo + self.p],
            }
        }

        /// One leaf's measured data as an effective leaf, for use as the node
        /// being attached.
        ///
        /// ### Params
        ///
        /// * `leaf` - Leaf index
        ///
        /// ### Returns
        ///
        /// The effective leaf.
        fn leaf(&self, leaf: u32) -> EffLeaf<'_, f64> {
            let lo = leaf as usize * self.p;
            EffLeaf {
                m: &self.leaf_m[lo..lo + self.p],
                w: &self.leaf_w[lo..lo + self.p],
            }
        }

        /// Score every node of the tree by brute force.
        ///
        /// The independent reference the beam search is pinned against.
        ///
        /// ### Params
        ///
        /// * `q` - Effective leaf being attached
        ///
        /// ### Returns
        ///
        /// The score of every node, indexed by node.
        fn scan(&self, q: EffLeaf<'_, f64>) -> Vec<f64> {
            let mut s = vec![0.0; self.p];
            let mut d = vec![0.0; self.p];
            (0..self.tree.n_nodes() as u32)
                .map(|a| {
                    attachment_score(self.eff(a), q, &mut s, &mut d)
                        .expect("root find diverged in the brute-force scan")
                        .loglik
                })
                .collect()
        }

        /// The brute-force best node and its score, ties going to the lowest
        /// index, matching what the beam search's strict comparison does.
        ///
        /// ### Params
        ///
        /// * `q` - Effective leaf being attached
        ///
        /// ### Returns
        ///
        /// The best node and its score.
        fn best(&self, q: EffLeaf<'_, f64>) -> (u32, f64) {
            let scores = self.scan(q);
            let mut node = 0u32;
            for (i, &sc) in scores.iter().enumerate() {
                if sc > scores[node as usize] {
                    node = i as u32;
                }
            }
            (node, scores[node as usize])
        }
    }

    /// Collapse the component of the tree containing `x`, with the edge to
    /// `from` cut, onto `x`.
    ///
    /// Slow recursive reference, sharing no code with
    /// [`crate::model::global::collapse_onto_every_node`].
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `leaf_m` - Leaf means, `[leaf][feature]`
    /// * `leaf_w` - Leaf precisions, same layout
    /// * `p` - Number of features
    /// * `x` - Node to collapse onto
    /// * `from` - Neighbour whose edge is cut, `NO_NODE` to collapse the whole
    ///   tree
    ///
    /// ### Returns
    ///
    /// The effective precisions and means at `x`.
    fn collapse(
        tree: &Tree,
        leaf_m: &[f64],
        leaf_w: &[f64],
        p: usize,
        x: u32,
        from: u32,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut w = vec![0.0f64; p];
        let mut m = vec![0.0f64; p];
        if (x as usize) < tree.n_leaves() {
            let lo = x as usize * p;
            w.copy_from_slice(&leaf_w[lo..lo + p]);
            m.copy_from_slice(&leaf_m[lo..lo + p]);
        }
        for nb in neighbours(tree, x) {
            if nb == from {
                continue;
            }
            // The branch between two neighbours is stored on whichever of them
            // is the child.
            let t = if tree.parent(x) == Some(nb) {
                tree.branch(x)
            } else {
                tree.branch(nb)
            };
            let (wn, mn) = collapse(tree, leaf_m, leaf_w, p, nb, x);
            for g in 0..p {
                let wd = wn[g] / (1.0 + t * wn[g]);
                w[g] += wd;
                m[g] += (mn[g] - m[g]) * (wd / w[g]);
            }
        }
        (w, m)
    }

    /// A balanced binary tree of eight leaves with one very tight cherry.
    ///
    /// Leaves 0 and 1 are near duplicates far from everything else.
    ///
    /// ### Returns
    ///
    /// The tree.
    fn tight_cherry() -> Tree {
        let mut tree = Tree::balanced_binary(8, 1.0).expect("balanced tree");
        tree.branches_mut()[0] = 0.01;
        tree.branches_mut()[1] = 0.01;
        tree
    }

    #[test]
    fn test_exhaustive_tolerance_agrees_with_the_brute_force_scan() {
        for tree in [
            Tree::balanced_binary(16, 1.0).expect("balanced tree"),
            Tree::ladder(12, 0.8).expect("ladder"),
            tight_cherry(),
        ] {
            let fix = Fixture::new(tree, 24, 0.3, 20260827);
            for leaf in [0u32, 3, 7] {
                let q = fix.leaf(leaf);
                let (node, loglik) = fix.best(q);
                let got = place(
                    &fix.tree,
                    q,
                    |a| fix.eff(a),
                    Some(PlacementParams {
                        tolerance: f64::INFINITY,
                        n_starts: 3,
                    }),
                )
                .expect("placement");
                assert_eq!(got.node, node);
                assert_relative_eq!(got.loglik, loglik, max_relative = 1e-12);
                // Exhaustive means exactly that: every node scored, once.
                assert_eq!(got.scored, fix.tree.n_nodes());
            }
        }
    }

    #[test]
    fn test_returned_node_maximises_the_attachment_score() {
        let fix = Fixture::new(
            Tree::balanced_binary(16, 1.0).expect("balanced tree"),
            32,
            0.25,
            77,
        );
        let q = fix.leaf(5);
        let scores = fix.scan(q);
        let got = place(&fix.tree, q, |a| fix.eff(a), None).expect("placement");
        for (i, &sc) in scores.iter().enumerate() {
            assert!(
                got.loglik >= sc,
                "node {i} scores {sc} against the returned {}",
                got.loglik
            );
        }
        assert_relative_eq!(got.loglik, scores[got.node as usize], max_relative = 1e-12);
    }

    #[test]
    fn test_greedy_multi_start_is_not_worse_than_a_single_root_start() {
        let fix = Fixture::new(Tree::ladder(24, 0.5).expect("ladder"), 48, 0.3, 31337);
        let mut improved = 0usize;
        for leaf in 0..fix.tree.n_leaves() as u32 {
            let q = fix.leaf(leaf);
            let one = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams {
                    tolerance: 0.0,
                    n_starts: 1,
                }),
            )
            .expect("placement");
            let many = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams {
                    tolerance: 0.0,
                    n_starts: 8,
                }),
            )
            .expect("placement");
            assert!(
                many.loglik >= one.loglik,
                "leaf {leaf}: eight starts scored {} against one start's {}",
                many.loglik,
                one.loglik
            );
            if many.loglik > one.loglik {
                improved += 1;
            }
        }
        assert!(improved > 0, "no leaf where the extra starts mattered");
    }

    #[test]
    fn test_a_detached_leaf_is_placed_back_next_to_its_sibling() {
        // Detach leaf 0, suppress the degree-two node it leaves, and ask where
        // it belongs: its old sibling.
        let full = Fixture::new(tight_cherry(), 64, 0.05, 4242);

        // Old indices: leaves 0..7, cherries 8..11, then 12, 13, root 14.
        // Dropping leaf 0 leaves node 8 with one child, so leaf 1 joins node 12
        // directly across the summed branch. Relabel the survivors: old leaf
        // `i` becomes `i - 1`, old internal 9,10,11,12,13,14 become 7..12.
        let parent = vec![10, 7, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, NO_NODE];
        let old_of = [1u32, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14];
        let mut branch = vec![0.0f64; parent.len()];
        for (new, &old) in old_of.iter().enumerate() {
            branch[new] = full.tree.branch(old);
        }
        branch[0] += full.tree.branch(8);
        let reduced = Tree::from_parents(parent, branch, 7).expect("reduced tree");

        // Carry the measured leaf data across, minus the detached leaf.
        let p = full.p;
        let leaf_m = full.leaf_m[p..].to_vec();
        let leaf_w = full.leaf_w[p..].to_vec();
        let n = reduced.n_nodes();
        let mut eff_m = vec![0.0f64; n * p];
        let mut eff_w = vec![0.0f64; n * p];
        for a in 0..n {
            let (w, m) = collapse(&reduced, &leaf_m, &leaf_w, p, a as u32, NO_NODE);
            eff_w[a * p..(a + 1) * p].copy_from_slice(&w);
            eff_m[a * p..(a + 1) * p].copy_from_slice(&m);
        }
        let fix = Fixture {
            tree: reduced,
            p,
            eff_m,
            eff_w,
            leaf_m,
            leaf_w,
        };

        let q = full.leaf(0);
        let got = place(&fix.tree, q, |a| fix.eff(a), None).expect("placement");
        // New leaf 0 is old leaf 1, the sibling it was detached from.
        assert_eq!(got.node, 0);
        assert_eq!(got.node, fix.best(q).0);
    }

    #[test]
    fn test_wider_tolerance_scores_more_nodes_and_never_scores_worse() {
        let fix = Fixture::new(Tree::ladder(20, 0.7).expect("ladder"), 48, 0.3, 99);
        let q = fix.leaf(11);
        let mut previous: Option<Placement> = None;
        for tolerance in [0.0f64, 0.5, 2.0, 8.0, 64.0, f64::INFINITY] {
            let got = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams {
                    tolerance,
                    n_starts: 4,
                }),
            )
            .expect("placement");
            if let Some(prev) = previous {
                assert!(
                    got.scored >= prev.scored,
                    "tolerance {tolerance} scored {} against {}",
                    got.scored,
                    prev.scored
                );
                assert!(
                    got.loglik >= prev.loglik,
                    "tolerance {tolerance} found {} against {}",
                    got.loglik,
                    prev.loglik
                );
            }
            previous = Some(got);
        }
        // The widest setting is the exhaustive one, so it must be the optimum.
        let got = previous.expect("at least one tolerance");
        assert_eq!(got.node, fix.best(q).0);
        assert_eq!(got.scored, fix.tree.n_nodes());
    }

    #[test]
    fn test_greedy_scores_fewer_nodes_than_exhaustive() {
        let fix = Fixture::new(
            Tree::balanced_binary(32, 1.0).expect("balanced tree"),
            32,
            0.3,
            5,
        );
        let q = fix.leaf(9);
        let greedy = place(
            &fix.tree,
            q,
            |a| fix.eff(a),
            Some(PlacementParams {
                tolerance: 0.0,
                n_starts: 1,
            }),
        )
        .expect("placement");
        assert!(greedy.scored < fix.tree.n_nodes());
    }

    #[test]
    fn test_a_duplicate_of_a_leaf_attaches_to_that_leaf_with_no_branch() {
        let fix = Fixture::new(
            Tree::balanced_binary(8, 1.0).expect("balanced tree"),
            32,
            0.4,
            2718,
        );
        // An exact copy of leaf 3, so the edge wants no length.
        let q = fix.leaf(3);
        let got = place(&fix.tree, q, |a| fix.eff(a), None).expect("placement");
        assert_eq!(got.node, 3);
        assert_eq!(got.branch, 0.0);
    }

    #[test]
    fn test_two_leaf_tree_scores_all_three_nodes() {
        let tree =
            Tree::from_parents(vec![2, 2, NO_NODE], vec![1.0, 1.0, 0.0], 2).expect("two-leaf tree");
        let fix = Fixture::new(tree, 16, 0.3, 8);
        let q = fix.leaf(0);
        let exhaustive = place(
            &fix.tree,
            q,
            |a| fix.eff(a),
            Some(PlacementParams {
                tolerance: f64::INFINITY,
                n_starts: 2,
            }),
        )
        .expect("placement");
        assert_eq!(exhaustive.scored, 3);
        assert_eq!(exhaustive.node, fix.best(q).0);
        let default = place(&fix.tree, q, |a| fix.eff(a), None).expect("placement");
        assert_eq!(default.node, exhaustive.node);
    }

    #[test]
    fn test_star_tree_is_searched_through_its_root() {
        // Leaves are reachable only through the root: fails if the walk never
        // goes upwards.
        let parent = vec![6, 6, 6, 6, 6, 6, NO_NODE];
        let tree = Tree::from_parents(parent, vec![1.0; 7], 6).expect("star");
        let fix = Fixture::new(tree, 32, 0.3, 606);
        for leaf in 0..6u32 {
            let q = fix.leaf(leaf);
            let got = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams {
                    tolerance: f64::INFINITY,
                    n_starts: 1,
                }),
            )
            .expect("placement");
            assert_eq!(got.scored, 7);
            assert_eq!(got.node, fix.best(q).0);
        }
    }

    #[test]
    fn test_attachment_score_matches_the_spec_expression() {
        let fix = Fixture::new(
            Tree::balanced_binary(8, 1.0).expect("balanced tree"),
            16,
            0.3,
            13,
        );
        let (a, q) = (fix.eff(9), fix.leaf(2));
        let mut s = vec![0.0; fix.p];
        let mut d = vec![0.0; fix.p];
        let at = attachment_score(a, q, &mut s, &mut d).expect("score");

        // S27 written out literally, with no reuse of prep_edge.
        let spec = |t: f64| -> f64 {
            let mut acc = 0.0;
            for g in 0..fix.p {
                let u = t + 1.0 / a.w[g] + 1.0 / q.w[g];
                let diff = a.m[g] - q.m[g];
                acc += u.ln() + diff * diff / u;
            }
            -0.5 * acc
        };
        assert_relative_eq!(at.loglik, spec(at.branch), max_relative = 1e-12);
        for delta in [-0.05f64, 0.05, 0.5] {
            let other = (at.branch + delta).max(0.0);
            assert!(at.loglik >= spec(other), "beaten at t = {other}");
        }
    }

    #[test]
    fn test_placement_is_the_same_whatever_the_start_count() {
        // Exhaustive tolerance: start points must not change the answer.
        let fix = Fixture::new(
            Tree::balanced_binary(16, 1.0).expect("balanced tree"),
            32,
            0.3,
            1000,
        );
        let q = fix.leaf(6);
        let reference = place(
            &fix.tree,
            q,
            |a| fix.eff(a),
            Some(PlacementParams {
                tolerance: f64::INFINITY,
                n_starts: 1,
            }),
        )
        .expect("placement");
        for n_starts in [2usize, 5, 31, 1000] {
            let got = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams {
                    tolerance: f64::INFINITY,
                    n_starts,
                }),
            )
            .expect("placement");
            assert_eq!(got.node, reference.node);
            assert_eq!(got.scored, reference.scored);
        }
    }

    #[test]
    fn test_start_points_are_in_range_and_lead_with_the_root() {
        let tree = Tree::balanced_binary(16, 1.0).expect("balanced tree");
        for n_starts in [0usize, 1, 4, 31, 100] {
            let starts = start_points(&tree, n_starts);
            assert!(!starts.is_empty());
            assert!(starts.len() <= tree.n_nodes());
            assert_eq!(starts[0], tree.root());
            assert!(starts.iter().all(|&s| (s as usize) < tree.n_nodes()));
        }
    }
}
