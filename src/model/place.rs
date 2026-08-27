//! Placing a node on an existing tree.
//!
//! Implements SPEC.md section 7. Used twice by the search: regrafting the
//! pruned subtree of an SPR move (section 9.3), and adding a cell to an
//! existing backbone (section 15). Both reduce to the same question, "which
//! node of this tree should `q` hang off", so both go through [`place`].
//!
//! ### Why the score is an ordinary edge
//!
//! Attaching `q` below `a` gives a tree whose loglikelihood is the old tree's,
//! plus `q`'s own, plus the contribution of the single edge joining them
//! (SPEC.md section 7.1, S27). The first two terms do not depend on `a`: the
//! collapse of the existing tree onto `a` accumulates the same total whatever
//! `a` is, because the likelihood does not depend on the choice of root (S14).
//! So the edge term alone ranks attachment points, and the edge term is exactly
//! what `model::branch` already solves. No new kernel appears here.
//!
//! ### What the caller still owes
//!
//! Attaching to a node makes a polytomy there, since the node already had
//! three neighbours in the unrooted sense. SPEC.md section 7.3 says not to
//! special-case attachment to the middle of an edge but to follow every
//! attachment with the polytomy resolution of section 9.2, which covers edge
//! attachment as a special case. That resolution lives in `search`, and this
//! module does not perform it: [`place`] reports where to attach and with what
//! branch length, it does not modify the tree.

use crate::errors::BonsaiErrors;
use crate::model::branch::optimise_edge_loglik;
use crate::model::merge::EffLeaf;
use crate::tree::Tree;
use crate::utils::kernels::prep_edge;
use crate::utils::traits::BonsaiFloat;

////////////////
// Parameters //
////////////////

/// Beam tolerance, in nats of loglikelihood.
///
/// A neighbour is recursed into when its attachment score is within this much
/// of the best score seen so far, so it buys breadth against cost. Zero is
/// greedy hill-climbing and infinity is an exhaustive scan; both are supported
/// and both are used by the tests.
///
/// Ours, not theirs, and set by measurement on 2026-08-27. Simulated data on
/// balanced trees of 64 and 256 leaves and a ladder of 128 leaves, 64 features,
/// three seeds, sixteen queries each; a query is a second noisy measurement of
/// a cell already in the tree, and it counts as recovered when a single-start
/// search returns the same node as the exhaustive scan. Over the grid
/// `{2, 2.5, 3, 3.5, 4, 6}` this is the smallest value that recovers 48 out of
/// 48. The ones below it fail only on the ladder, and they fail badly rather
/// than marginally: 3.5 loses one query by 10 nats, 2.0 loses five by 55.
///
/// Cost is shape-dependent and worth knowing. The balanced fixtures score 13
/// of 127 and 18 of 511 nodes; the ladder scores 137 of 255, and does not score
/// more than that at any wider tolerance, so on the shape that needs the beam
/// the beam is already covering everything it will ever cover.
///
/// It is an absolute loglikelihood difference and does not scale with the
/// feature count, deliberately: what is being compared is the gap between two
/// neighbouring attachment points, and that gap stays `O(1)` as features are
/// added because neighbouring nodes agree on most features.
const DEFAULT_TOLERANCE: f64 = 4.0;

/// Number of start points the beam search fans out from.
///
/// The reference uses `log(n)` centres from a distance-based clustering; our
/// count is ours to choose (SPEC.md section 7.2). Measured on 2026-08-27 on the
/// fixtures described on [`DEFAULT_TOLERANCE`], and the honest summary is that
/// at the shipped tolerance the extra starts change no answer and cost scored
/// nodes: 13 against 23 on the 127-node fixture, 18 against 29 on the 511-node
/// one, 137 against 144 on the ladder.
///
/// Eight is kept anyway, as insurance against a caller tightening the
/// tolerance, and because that is the point at which the insurance actually
/// pays: at a tolerance of 2 the ladder recovers 16 queries out of 16 from
/// eight starts, 13 or 14 from four, and 11 to 13 from one. It is also roughly
/// `log2(n)` over the few hundred nodes a backbone round works with, which is
/// the same order as the reference's `log(n)`. A caller placing millions of
/// cells against a fixed backbone should drop it to one and keep the tolerance.
const DEFAULT_STARTS: usize = 8;

/// Tuning knobs for the beam search of SPEC.md section 7.2.
///
/// Neither changes the model. Both trade the number of attachment scores
/// evaluated, each of which is a root find over `p` features, against the risk
/// of stopping at a local optimum.
#[derive(Clone, Copy, Debug)]
pub struct PlacementParams {
    /// How far below the best score seen so far a neighbour may fall and still
    /// be recursed into, in nats. `0.0` is greedy, `f64::INFINITY` exhaustive.
    pub tolerance: f64,
    /// Number of start points, clamped to at least one and at most the node
    /// count.
    pub n_starts: usize,
}

impl PlacementParams {
    /// Build parameters explicitly.
    ///
    /// ### Params
    ///
    /// * `tolerance` - Beam tolerance in nats; `0.0` for greedy hill-climbing,
    ///   `f64::INFINITY` for an exhaustive scan
    /// * `n_starts` - Number of start points; clamped into `1..=n_nodes` by
    ///   [`start_points`]
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub fn new(tolerance: f64, n_starts: usize) -> Self {
        Self {
            tolerance,
            n_starts,
        }
    }
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

//////////////////////
// Attachment score //
//////////////////////

/// The result of attaching a node below one particular node of the tree.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    /// Optimal length of the new edge. Zero means the two effective leaves are
    /// already closer than their combined uncertainty, which is the case
    /// SPEC.md section 7.3's polytomy resolution then has to deal with.
    pub branch: f64,
    /// The edge's loglikelihood at that length, which is the attachment score.
    /// Comparable across attachment points but not an absolute quantity; see
    /// the module docs.
    pub loglik: f64,
}

/// Score attaching `q` below a node summarised by the effective leaf `a`.
///
/// SPEC.md section 7.1 (S27):
///
/// ```text
/// dL(t) = -1/2 * sum_g [ log(t + 1/W[g,a] + 1/W[g,q])
///                        + (M[g,a] - M[g,q])^2 / (t + 1/W[g,a] + 1/W[g,q]) ]
/// ```
///
/// which is the edge expression of SPEC.md section 6 with
/// `s[g] = 1/W[g,a] + 1/W[g,q]` and `d[g] = (M[g,a] - M[g,q])^2`. Those are
/// what `prep_edge` produces, along with the bracket, on the one pass it makes
/// over the two arrays, and the maximisation over `t` is what
/// `optimise_edge_loglik` does. So this function is a two-line composition
/// rather than a kernel, which is the point: there is only one branch-length
/// solver in the crate.
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

/////////////////
// Beam search //
/////////////////

/// Where a node should be attached, and what that costs.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Node of the existing tree to attach below.
    pub node: u32,
    /// Optimal length of the new edge.
    pub branch: f64,
    /// Attachment score there, the best found by the search.
    pub loglik: f64,
    /// How many distinct nodes were scored. Diagnostic only: it is the search's
    /// cost in units of one `O(p)` root find, and the handle on whether the
    /// tolerance is doing anything.
    pub scored: usize,
}

/// Start points for the beam search.
///
/// **Placeholder.** SPEC.md section 7.2 takes the centres of the distance-based
/// clustering of the Methods, which depends on machinery this crate does not
/// have yet; until it does, this spreads the starts evenly over the node index.
/// That is a real spread rather than an arbitrary one, because the arena
/// invariant makes the index ordering meaningful: leaves come first, then
/// internal nodes by height, so an even sweep over indices samples the leaves
/// broadly and then samples every level of the internal skeleton.
///
/// The root always comes first. That is load-bearing rather than cosmetic:
/// start points share one visited set (see [`place`]), so the first start is
/// the only one guaranteed to explore unimpeded, and putting the root there
/// makes a multi-start search provably no worse than a single search from the
/// root.
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
    // `i * n / k` for `i` in `1..k` lands inside `0..n`. It can collide with the
    // root once `k` approaches `n`; the shared visited set absorbs that.
    for i in 1..k {
        out.push((i * n / k) as u32);
    }
    out
}

/// Neighbours of a node in the unrooted sense: its children and its parent.
///
/// The arena is rooted for bookkeeping but the likelihood is not (SPEC.md
/// section 2, S14), so the search has to be able to walk upwards. Forgetting
/// the parent here would confine the search to the subtree below its start
/// point, which is the single easiest way to get this module quietly wrong.
///
/// ### Params
///
/// * `tree` - Tree being searched
/// * `node` - Node whose neighbours are wanted
///
/// ### Returns
///
/// An iterator over the neighbours, children first in arena order and the
/// parent last. Allocation-free, and the fixed order is what makes the search
/// deterministic.
fn neighbours(tree: &Tree, node: u32) -> impl Iterator<Item = u32> + '_ {
    tree.children(node).iter().copied().chain(tree.parent(node))
}

/// Find the best node of a tree to attach `q` below.
///
/// The beam search of SPEC.md section 7.2. From each start point, score the
/// node, score each unvisited neighbour, and recurse into those whose score is
/// within `tolerance` of the best score seen anywhere so far. The best node
/// over all start points wins.
///
/// The comparison is against the best seen *before* the neighbour itself is
/// folded in, so a tolerance of zero means "recurse only into a neighbour that
/// strictly improves on everything seen so far", which is hill-climbing.
/// A tolerance of `f64::INFINITY` admits every neighbour and therefore visits
/// the whole tree, which is an exhaustive scan.
///
/// Nodes are scored at most once across the whole call: one visited set is
/// shared by every start point, so the total cost is bounded by `n_nodes` root
/// finds however many starts are requested. The search is sequential and
/// depth-first with a fixed neighbour order, so the answer does not depend on
/// the thread count.
///
/// ### The effective-leaf seam
///
/// `eff` maps a node index to the effective leaf summarising **the entire
/// existing tree collapsed onto that node**: the pruning recursion of SPEC.md
/// section 4 run with that node as the root, which for a leaf includes the
/// leaf's own observation. That is a two-sided quantity, one upward and one
/// downward sweep over the arena, and it belongs to `model::global` rather than
/// here, so it is passed in. It must be defined for every index in
/// `0..tree.n_nodes()` and every leaf it returns must have `q`'s feature count.
///
/// A plain closure is used rather than a trait because the provider is free to
/// store the sweep in whatever layout suits it, needs no wrapper type, and the
/// borrow it hands back is checked at the call site. The one thing the closure
/// signature does impose is that the effective leaves are materialised
/// somewhere the closure can borrow from, rather than computed into a local
/// buffer on each call.
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
    let params = params.unwrap_or_default();
    let p = q.m.len();
    let mut s = vec![0.0f64; p];
    let mut d = vec![0.0f64; p];

    let mut visited = vec![false; tree.n_nodes()];
    let mut stack: Vec<u32> = Vec::new();
    // The root is always the first start point and is therefore always scored,
    // so this placeholder is always overwritten before it is returned.
    let mut best = Placement {
        node: tree.root(),
        branch: 0.0,
        loglik: f64::NEG_INFINITY,
        scored: 0,
    };

    for start in start_points(tree, params.n_starts) {
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
            for nb in neighbours(tree, a) {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let at = attachment_score(eff(nb), q, &mut s, &mut d)?;
                best.scored += 1;

                // Threshold against the incumbent before the neighbour joins
                // it, otherwise a zero tolerance could never admit anything.
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
    use approx::assert_relative_eq;

    /// One draw from a deterministic uniform stream.
    ///
    /// A splitmix64 step. The fixtures need reproducible pseudo-randomness and
    /// nothing else, so this avoids pinning the tests to a particular version
    /// of an external generator.
    ///
    /// ### Params
    ///
    /// * `state` - Generator state, advanced in place
    ///
    /// ### Returns
    ///
    /// A uniform draw on `[0, 1)`.
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One standard normal draw, Box-Muller.
    ///
    /// ### Params
    ///
    /// * `state` - Generator state, advanced in place
    ///
    /// ### Returns
    ///
    /// A draw from `N(0, 1)`.
    fn gauss(state: &mut u64) -> f64 {
        let u = uniform(state).max(1e-12);
        let v = uniform(state);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// A tree with data simulated on it, plus the two-sided effective leaves.
    ///
    /// Stands in for what `model::global` will provide. Everything is `f64` and
    /// flat, `[node][feature]`.
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
        /// Positions random-walk down the tree with the tree's own branch
        /// lengths, in the transformed units of SPEC.md section 3.1 where the
        /// diffusion has unit variance per feature, then leaves pick up
        /// measurement noise of standard deviation `sigma`.
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
            let mut state = seed;
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
                    pos[node * p + g] = pos[par * p + g] + sd * gauss(&mut state);
                }
            }
            let n_leaves = tree.n_leaves();
            let mut leaf_m = vec![0.0f64; n_leaves * p];
            let leaf_w = vec![1.0 / (sigma * sigma); n_leaves * p];
            for i in 0..n_leaves {
                for g in 0..p {
                    leaf_m[i * p + g] = pos[i * p + g] + sigma * gauss(&mut state);
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
        /// The independent reference the beam search is pinned against: no
        /// traversal, no tolerance, no visited set, just every node in index
        /// order.
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
    /// The two-sided sweep written the slow, obvious, recursive way: `O(n)` per
    /// node rather than `O(1)` amortised. This is the reference the effective
    /// leaves in the fixture come from, so the placement tests do not depend on
    /// `model::global` existing yet.
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
    /// Leaves 0 and 1 sit on branches short enough that they are near
    /// duplicates of each other and far from everything else, which makes the
    /// answer to "where does leaf 0 belong" unambiguous.
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
                    Some(PlacementParams::new(f64::INFINITY, 3)),
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
        // Greedy from the root alone stops at the first node that beats
        // everything before it, so on a deep tree it can be trapped a long way
        // from the optimum. The extra starts are what get it out.
        let fix = Fixture::new(Tree::ladder(24, 0.5).expect("ladder"), 48, 0.3, 31337);
        let mut improved = 0usize;
        for leaf in 0..fix.tree.n_leaves() as u32 {
            let q = fix.leaf(leaf);
            let one = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams::new(0.0, 1)),
            )
            .expect("placement");
            let many = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams::new(0.0, 8)),
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
        // Balanced tree of eight, leaves 0 and 1 a tight cherry. Detach leaf 0
        // and suppress the degree-two node it leaves behind, then ask where it
        // belongs. The answer has to be its old sibling.
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
                Some(PlacementParams::new(tolerance, 4)),
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
            Some(PlacementParams::new(0.0, 1)),
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
        // The node being attached is an exact copy of leaf 3's measurement, so
        // its squared separation from leaf 3 is identically zero and the edge
        // wants no length at all.
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
            Some(PlacementParams::new(f64::INFINITY, 2)),
        )
        .expect("placement");
        assert_eq!(exhaustive.scored, 3);
        assert_eq!(exhaustive.node, fix.best(q).0);
        // A tree this small has no room for the beam to matter, so the default
        // parameters must land in the same place.
        let default = place(&fix.tree, q, |a| fix.eff(a), None).expect("placement");
        assert_eq!(default.node, exhaustive.node);
    }

    #[test]
    fn test_star_tree_is_searched_through_its_root() {
        // Every leaf of a star is reachable only through the root, so this is
        // the case that fails if the search does not walk upwards.
        let parent = vec![6, 6, 6, 6, 6, 6, NO_NODE];
        let tree = Tree::from_parents(parent, vec![1.0; 7], 6).expect("star");
        let fix = Fixture::new(tree, 32, 0.3, 606);
        for leaf in 0..6u32 {
            let q = fix.leaf(leaf);
            let got = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams::new(f64::INFINITY, 1)),
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
        // With a tolerance wide enough to cover the tree, the start points are
        // bookkeeping and must not change the answer.
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
            Some(PlacementParams::new(f64::INFINITY, 1)),
        )
        .expect("placement");
        for n_starts in [2usize, 5, 31, 1000] {
            let got = place(
                &fix.tree,
                q,
                |a| fix.eff(a),
                Some(PlacementParams::new(f64::INFINITY, n_starts)),
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
