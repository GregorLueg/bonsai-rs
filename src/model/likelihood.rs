//! The pruning recursion and the tree loglikelihood.
//!
//! Every subtree collapses into an effective leaf carrying a mean and a
//! precision per feature, and the tree loglikelihood accumulates as that
//! collapse proceeds. This is the continuous-trait form of Felsenstein's
//! pruning algorithm.

use crate::errors::BonsaiErrors;
use crate::tree::Tree;
use crate::utils::kernels::prune_general;
use crate::utils::simd::prune_binary;
use crate::utils::traits::BonsaiFloat;

///////////////
// NodeState //
///////////////

/// Effective means and precisions for every node in a tree.
///
/// Stored as two flat row-major blocks of `n_nodes * p`, so a node's row is one
/// contiguous span and a merge writes exactly one row. Leaf rows hold the input
/// data in the transformed units of SPEC.md section 3.1; internal rows are
/// filled by [`NodeState::prune`].
#[derive(Clone, Debug)]
pub struct NodeState<T> {
    /// Effective means, `[node][feature]`, row-major.
    m: Vec<T>,
    /// Effective precisions, `[node][feature]`, row-major.
    w: Vec<T>,
    /// Number of features.
    p: usize,
    /// Number of nodes, leaves included.
    n_nodes: usize,
    /// Leaf rows the caller actually supplied.
    ///
    /// Kept so `prune` can tell a state built for this tree from one built for
    /// a smaller leaf set: the constructor takes a node count, so it cannot
    /// check this itself.
    n_leaf_rows: usize,
    /// Per-node loglikelihood contributions, the terms `prune` sums. Zero at
    /// the leaves and at any internal node not yet pruned.
    ///
    /// Kept so a tree that differs from this one at a few nodes can be scored
    /// by summing these where it agrees and recomputing where it does not,
    /// which is what search step 5 does with its candidates.
    contrib: Vec<f64>,
    /// Scratch for the polytomy path, grown on demand.
    scratch: Vec<f64>,
}

impl<T: BonsaiFloat> NodeState<T> {
    /// Allocate state for a tree and fill the leaf rows from the input data.
    ///
    /// ### Params
    ///
    /// * `n_nodes` - Total node count of the tree this state will serve
    /// * `p` - Number of features
    /// * `leaf_means` - Transformed means, row-major `[leaf][feature]`
    /// * `leaf_precisions` - Transformed precisions, same layout
    ///
    /// ### Returns
    ///
    /// The state, with internal rows zeroed, or `ShapeMismatch` if the two leaf
    /// blocks disagree.
    pub fn new(
        n_nodes: usize,
        p: usize,
        leaf_means: &[T],
        leaf_precisions: &[T],
    ) -> Result<Self, BonsaiErrors> {
        if leaf_means.len() != leaf_precisions.len() {
            return Err(BonsaiErrors::ShapeMismatch {
                mean_cells: leaf_means.len() / p.max(1),
                mean_features: p,
                sd_cells: leaf_precisions.len() / p.max(1),
                sd_features: p,
            });
        }
        // A leaf block that is not a whole number of rows, or that does not fit
        // the arena, is a caller error rather than a broken invariant. Before
        // this check the over-long case panicked inside `copy_from_slice` and
        // the short case under-filled silently, leaving zero-precision rows
        // that turn the loglikelihood into `NaN` at the first logarithm.
        if p == 0 || !leaf_means.len().is_multiple_of(p) || leaf_means.len() > n_nodes * p {
            return Err(BonsaiErrors::ShapeMismatch {
                mean_cells: leaf_means.len() / p.max(1),
                mean_features: p,
                sd_cells: n_nodes,
                sd_features: p,
            });
        }
        let n_leaf_rows = leaf_means.len() / p;
        let mut m = vec![T::zero(); n_nodes * p];
        let mut w = vec![T::zero(); n_nodes * p];
        m[..leaf_means.len()].copy_from_slice(leaf_means);
        w[..leaf_precisions.len()].copy_from_slice(leaf_precisions);
        Ok(Self {
            m,
            w,
            p,
            n_nodes,
            n_leaf_rows,
            contrib: vec![0.0; n_nodes],
            scratch: Vec::new(),
        })
    }

    /// Assemble a state from rows that are already settled.
    ///
    /// No validation beyond the lengths, because the one caller is
    /// [`crate::search::spr`], which forms the rows through the same kernels
    /// [`NodeState::prune`] dispatches to and is tested bit for bit against it.
    ///
    /// ### Params
    ///
    /// * `p` - Number of features
    /// * `n_leaf_rows` - Number of leaf rows, which `prune` checks against the
    ///   tree it is handed
    /// * `m` - Effective means, `[node][feature]`, row-major, every row settled
    /// * `w` - Effective precisions, same layout
    /// * `contrib` - Per-node loglikelihood contributions, one per node
    ///
    /// ### Returns
    ///
    /// The state.
    ///
    /// ### Panics
    ///
    /// If the three lengths do not describe the same node count.
    pub(crate) fn from_rows(
        p: usize,
        n_leaf_rows: usize,
        m: Vec<T>,
        w: Vec<T>,
        contrib: Vec<f64>,
    ) -> Self {
        assert_eq!(m.len(), w.len());
        assert_eq!(m.len(), contrib.len() * p);
        Self {
            n_nodes: contrib.len(),
            m,
            w,
            p,
            n_leaf_rows,
            contrib,
            scratch: Vec::new(),
        }
    }

    /// One node's loglikelihood contribution from the last `prune`.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose term is wanted
    ///
    /// ### Returns
    ///
    /// The term, zero for a leaf.
    #[inline]
    pub fn contribution(&self, node: u32) -> f64 {
        self.contrib[node as usize]
    }

    /// Number of features.
    ///
    /// ### Returns
    ///
    /// The feature count.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.p
    }

    /// Effective means of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's row of means.
    #[inline]
    pub fn means(&self, node: u32) -> &[T] {
        let lo = node as usize * self.p;
        &self.m[lo..lo + self.p]
    }

    /// Effective precisions of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's row of precisions.
    #[inline]
    pub fn precisions(&self, node: u32) -> &[T] {
        let lo = node as usize * self.p;
        &self.w[lo..lo + self.p]
    }

    /// Run the pruning recursion over the whole tree and return its
    /// loglikelihood.
    ///
    /// Sequential, and measurement says it should stay that way: the whole
    /// prune is 2.4 per cent of a pipeline run, so parallelising it cannot
    /// matter. Walks levels from the leaves up and, within a level, ascending
    /// node index, which the arena invariant makes a valid post-order.
    ///
    /// Per-node contributions are summed within a level and only then added to
    /// the running total, which keeps the association fixed to the tree so that
    /// this routine's own answer does not depend on how the levels happen to be
    /// walked. That is what makes this routine's answer a property of the tree
    /// rather than of the traversal.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose topology and branch lengths to use
    ///
    /// ### Returns
    ///
    /// The tree loglikelihood, up to the dropped additive constants.
    ///
    /// ### Panics
    ///
    /// If `tree` has a different node count from the one this state was built
    /// for, or a different leaf count from the number of leaf rows supplied to
    /// [`NodeState::new`]. Both are a mismatched pair of arguments rather than
    /// bad data.
    pub fn prune(&mut self, tree: &Tree) -> f64 {
        assert_eq!(
            tree.n_nodes(),
            self.n_nodes,
            "this state was built for a tree of {} nodes",
            self.n_nodes
        );
        // Same reasoning as above, for the other half of the shape. An
        // under-filled leaf block leaves zero-precision rows, which the first
        // logarithm turns into `-inf`: a finite-looking `Ok` carrying a
        // meaningless number, since `Leaves` has public fields and nothing ties
        // its length to the tree.
        assert_eq!(
            tree.n_leaves(),
            self.n_leaf_rows,
            "this state was filled with {} leaf rows",
            self.n_leaf_rows
        );
        let mut total = 0.0f64;
        for level in 0..tree.n_levels() {
            let (start, end) = tree.level(level);
            let mut level_total = 0.0f64;
            for node in start..end {
                let here = prune_node_into(
                    tree,
                    node as u32,
                    self.p,
                    &mut self.m,
                    &mut self.w,
                    &mut self.scratch,
                );
                self.contrib[node] = here;
                level_total += here;
            }
            total += level_total;
        }
        total
    }
}

///////////////////////
// Per-node dispatch //
///////////////////////

/// Prune one internal node into a slab, reading its children's settled rows.
///
/// A "row" here is `[node][feature]` over all `p` features. This is written
/// against a slab of equal-length rows indexed by node rather than against
/// `NodeState` itself, which is what let a second layout share it; that layout
/// is gone, but the shape is still the right one to write the dispatch between
/// the binary and polytomy kernels against. The traversal order and the order
/// the per-node contributions are summed in stay each module's own business,
/// which is what makes the two independent enough to cross-check.
///
/// ### Params
///
/// * `tree` - Tree whose topology and branch lengths to use
/// * `node` - Internal node to settle
/// * `len` - Row stride, that is, features per node in this slab
/// * `m` - Effective means slab, `node`'s row written in place
/// * `w` - Effective precisions slab, same
/// * `scratch` - Polytomy scratch, grown on demand and reused across calls
///
/// ### Returns
///
/// The node's loglikelihood contribution.
pub(crate) fn prune_node_into<T: BonsaiFloat>(
    tree: &Tree,
    node: u32,
    len: usize,
    m: &mut [T],
    w: &mut [T],
    scratch: &mut Vec<f64>,
) -> f64 {
    let split = node as usize * len;
    let (m_lo, m_hi) = m.split_at_mut(split);
    let (w_lo, w_hi) = w.split_at_mut(split);
    let m_out = &mut m_hi[..len];
    let w_out = &mut w_hi[..len];

    let kids = tree.children(node);
    match kids.len() {
        2 => {
            let (k, l) = (kids[0] as usize, kids[1] as usize);
            prune_binary(
                &m_lo[k * len..k * len + len],
                &w_lo[k * len..k * len + len],
                tree.branch(kids[0]),
                &m_lo[l * len..l * len + len],
                &w_lo[l * len..l * len + len],
                tree.branch(kids[1]),
                m_out,
                w_out,
            )
        }
        n_child => {
            if scratch.len() < len * n_child {
                scratch.resize(len * n_child, 0.0);
            }
            let children: Vec<(&[T], &[T], f64)> = kids
                .iter()
                .map(|&c| {
                    let lo = c as usize * len;
                    (&m_lo[lo..lo + len], &w_lo[lo..lo + len], tree.branch(c))
                })
                .collect();
            prune_general(&children, m_out, w_out, &mut scratch[..len * n_child])
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::{NO_NODE, Tree};
    use crate::utils::rng::SplitMix64;
    use approx::assert_relative_eq;

    /// Deterministic pseudo-random leaf data, so tests do not need an rng
    /// dependency and always describe the same scenario.
    ///
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// Row-major means and precisions, `[leaf][feature]`.
    pub(crate) fn leaf_data(n_leaves: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(0x2545_F491_4F6C_DD1D);
        let mut m = Vec::with_capacity(n_leaves * p);
        let mut w = Vec::with_capacity(n_leaves * p);
        for _ in 0..n_leaves * p {
            m.push(rng.uniform() * 4.0 - 2.0);
            w.push(0.25 + rng.uniform() * 3.0);
        }
        (m, w)
    }

    #[test]
    fn test_rejects_leaf_blocks_that_do_not_fit_the_arena() {
        // Regression. An over-long leaf block
        // panicked inside `copy_from_slice`, in library code, on the crate's
        // most-used type. A block that was not a whole number of rows was
        // accepted and under-filled, leaving zero-precision rows that make the
        // loglikelihood NaN at the first logarithm.
        let p = 8usize;

        let too_long = vec![1.0f64; 32];
        assert!(
            matches!(
                NodeState::new(3, p, &too_long, &too_long),
                Err(BonsaiErrors::ShapeMismatch { .. })
            ),
            "a leaf block larger than the arena was accepted"
        );

        let ragged = vec![1.0f64; 13];
        assert!(
            matches!(
                NodeState::new(9, p, &ragged, &ragged),
                Err(BonsaiErrors::ShapeMismatch { .. })
            ),
            "a leaf block that is not a whole number of rows was accepted"
        );

        // Exactly filling the arena is legal, as is leaving room for the
        // internal rows the sweep will write.
        assert!(NodeState::new(3, p, &[1.0f64; 24], &[1.0f64; 24]).is_ok());
        assert!(NodeState::new(9, p, &vec![1.0f64; 40], &vec![1.0f64; 40]).is_ok());
    }

    #[test]
    fn test_the_loglikelihood_is_the_same_at_every_rooting() {
        // S14, listed in SPEC.md section 13.2 as a required invariant. The
        // root is a bookkeeping choice: the model is defined on the unrooted
        // tree, so
        // moving the root along any edge, which splits that edge in two and
        // reverses the path back to the old root, has to leave the
        // loglikelihood exactly where it was.
        use crate::tree::cluster::reroot;

        let p = 12usize;
        for (name, tree) in [
            ("balanced", Tree::balanced_binary(8, 0.7).expect("balanced")),
            ("ladder", Tree::ladder(7, 0.4).expect("ladder")),
            (
                "zero branches",
                Tree::balanced_binary(8, 0.0).expect("zero"),
            ),
            (
                "internal polytomy",
                Tree::from_parents(
                    vec![6, 6, 6, 7, 7, 7, 7, NO_NODE],
                    vec![0.35, 0.8, 1.4, 0.2, 0.6, 1.1, 0.45, 0.0],
                    6,
                )
                .expect("polytomy"),
            ),
        ] {
            let (m, w) = leaf_data(tree.n_leaves(), p);
            let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).expect("state");
            let base = state.prune(&tree);
            assert!(base.is_finite());

            // Every edge of the tree, which is every node but the root.
            let mut worst = 0.0f64;
            for node in 0..tree.n_nodes() as u32 {
                if node == tree.root() {
                    continue;
                }
                let moved = reroot(&tree, node).expect("reroot");
                // Leaf indices survive a reroot, so the same leaf block still
                // describes the same cells.
                let mut state = NodeState::new(moved.n_nodes(), p, &m, &w).expect("state");
                let here = state.prune(&moved);
                let error = (here - base).abs() / base.abs().max(1.0);
                worst = worst.max(error);
            }
            // The worst measured here is `O(1e-16)` relative on every shape,
            // including the polytomy path. The bound sits two orders above
            // that, so it pins the invariant rather than the summation order.
            assert!(
                worst < 1e-14,
                "{name}: rerooting moved the loglikelihood by {worst:e} relative"
            );
        }
    }

    #[test]
    #[should_panic(expected = "this state was built for a tree of")]
    fn test_pruning_a_tree_the_state_was_not_built_for_is_caught() {
        // This was a `debug_assert`, so a release build indexed happily
        // inside the larger state's rows and
        // returned a well-formed loglikelihood computed from the wrong ones.
        let p = 4usize;
        let big = Tree::balanced_binary(8, 0.5).expect("big");
        let small = Tree::balanced_binary(4, 0.5).expect("small");
        let (m, w) = leaf_data(8, p);
        let mut state = NodeState::new(big.n_nodes(), p, &m, &w).expect("state");
        state.prune(&small);
    }

    #[test]
    fn test_the_loglikelihood_is_l_and_not_twice_it() {
        // SPEC.md section 12: the paper states thresholds in `2L` and this
        // crate works in `L`, so anything transcribed has to be halved. Not a
        // claim about the constants, none of which are theirs: a claim about
        // the units the whole crate is denominated in, and the only place to
        // pin it is against the formula as written.
        //
        // S20 transcribed literally, `1/2` and all, in the direct form with the
        // ancestor's mean formed rather than the pairwise identity, over a tree
        // with a polytomy in it so both prune kernels are covered.
        let p = 6usize;
        let n_leaves = 6usize;
        let (m, w) = leaf_data(n_leaves, p);
        // Leaves 0 and 1 under node 6, which is the binary kernel, and
        // everything else straight onto the root, which is the general one.
        let parent = vec![6, 6, 7, 7, 7, 7, 7, NO_NODE];
        let branch = vec![0.35, 0.8, 1.4, 0.2, 0.6, 1.1, 0.45, 0.0];
        let tree = Tree::from_parents(parent, branch.clone(), n_leaves).expect("tree");

        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).expect("state");
        let got = state.prune(&tree);

        // Post-order over the arena, which the invariant makes ascending index.
        let mut wd = vec![0.0f64; tree.n_nodes() * p];
        let mut mm = vec![0.0f64; tree.n_nodes() * p];
        let mut sum = 0.0f64;
        for node in 0..tree.n_nodes() {
            let kids = tree.children(node as u32);
            if kids.is_empty() {
                for g in 0..p {
                    let wi = w[node * p + g];
                    mm[node * p + g] = m[node * p + g];
                    wd[node * p + g] = wi / (1.0 + branch[node] * wi);
                }
                continue;
            }
            for g in 0..p {
                let total: f64 = kids.iter().map(|&k| wd[k as usize * p + g]).sum();
                let mean: f64 = kids
                    .iter()
                    .map(|&k| wd[k as usize * p + g] * mm[k as usize * p + g])
                    .sum::<f64>()
                    / total;
                sum += kids
                    .iter()
                    .map(|&k| wd[k as usize * p + g].ln())
                    .sum::<f64>()
                    - total.ln()
                    - kids
                        .iter()
                        .map(|&k| {
                            let d = mean - mm[k as usize * p + g];
                            wd[k as usize * p + g] * d * d
                        })
                        .sum::<f64>();
                mm[node * p + g] = mean;
                wd[node * p + g] = total / (1.0 + branch[node] * total);
            }
        }
        let l = 0.5 * sum;

        assert_relative_eq!(got, l, max_relative = 1e-12);
        // Not vacuous: `L` and `2L` are far apart on this fixture, so a factor
        // of two anywhere in the recursion would fail the assertion above.
        assert!(
            (got - 2.0 * l).abs() > 1.0,
            "L and 2L are indistinguishable here, so this fixture pins nothing"
        );
    }

    #[test]
    fn test_star_tree_loglik_matches_closed_form() {
        // For a star tree the loglikelihood is the single root term of S20,
        // which we can write out directly.
        let (n_leaves, p) = (5usize, 7usize);
        let (m, w) = leaf_data(n_leaves, p);
        let t = [0.4, 1.1, 0.2, 2.3, 0.9];

        let mut parent = vec![n_leaves as u32; n_leaves];
        parent.push(NO_NODE);
        let mut branch = t.to_vec();
        branch.push(0.0);
        let tree = Tree::from_parents(parent, branch, n_leaves).unwrap();

        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let got = state.prune(&tree);

        let mut expect = 0.0f64;
        for g in 0..p {
            let wd: Vec<f64> = (0..n_leaves)
                .map(|i| {
                    let wi = w[i * p + g];
                    wi / (1.0 + t[i] * wi)
                })
                .collect();
            let wa: f64 = wd.iter().sum();
            let ma: f64 = (0..n_leaves).map(|i| wd[i] * m[i * p + g]).sum::<f64>() / wa;
            expect += wd.iter().map(|x| x.ln()).sum::<f64>() - wa.ln();
            expect -= (0..n_leaves)
                .map(|i| wd[i] * (ma - m[i * p + g]).powi(2))
                .sum::<f64>();
        }
        assert_relative_eq!(got, 0.5 * expect, epsilon = 1e-12);
    }

    #[test]
    fn test_effective_leaf_summary_reproduces_the_full_tree() {
        // SPEC.md section 4: replacing a subtree by its effective leaf must not
        // change the loglikelihood of what remains. Score a four-leaf tree, then
        // score a three-leaf tree in which one cherry has been replaced by its
        // summary, and check the difference is exactly the cherry's own term.
        let p = 6usize;
        let (m, w) = leaf_data(4, p);
        let branch_len = 0.7;

        // ((0,1)a, 2, 3) rooted at r.
        let tree =
            Tree::from_parents(vec![4, 4, 5, 5, 5, NO_NODE], vec![branch_len; 6], 4).unwrap();
        let mut full = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let l_full = full.prune(&tree);

        // The cherry on its own, to get its contribution and its summary.
        let cherry = Tree::from_parents(vec![2, 2, NO_NODE], vec![branch_len; 3], 2).unwrap();
        let mut cherry_state = NodeState::new(3, p, &m[..2 * p], &w[..2 * p]).unwrap();
        let l_cherry = cherry_state.prune(&cherry);

        // Now a three-leaf star whose first leaf is the cherry's summary.
        let mut m_sum = cherry_state.means(2).to_vec();
        let mut w_sum = cherry_state.precisions(2).to_vec();
        m_sum.extend_from_slice(&m[2 * p..4 * p]);
        w_sum.extend_from_slice(&w[2 * p..4 * p]);
        let collapsed = Tree::from_parents(vec![3, 3, 3, NO_NODE], vec![branch_len; 4], 3).unwrap();
        let mut collapsed_state = NodeState::new(4, p, &m_sum, &w_sum).unwrap();
        let l_collapsed = collapsed_state.prune(&collapsed);

        assert_relative_eq!(l_full, l_cherry + l_collapsed, epsilon = 1e-11);
    }

    #[test]
    fn test_relabelling_into_level_order_preserves_the_loglikelihood() {
        // The same topology described with internal nodes numbered in a
        // different (still legal) order must score identically, since
        // `from_parents` relabels into level order either way.
        //
        // Topology: ((0,1)a, (2,3)b)r, first with a=4,b=5,r=6, then with the
        // cherries allocated in the other order.
        let p = 8usize;
        let (m, w) = leaf_data(4, p);
        let branch = vec![0.3, 1.7, 0.9, 0.5, 1.1, 0.2, 0.0];

        let straight =
            Tree::from_parents(vec![4, 4, 5, 5, 6, 6, NO_NODE], branch.clone(), 4).unwrap();
        let mut s1 = NodeState::new(straight.n_nodes(), p, &m, &w).unwrap();
        let l1 = s1.prune(&straight);

        // Swap which internal index holds which cherry, and swap the matching
        // branch lengths so the tree is genuinely the same.
        let swapped_branch = vec![0.3, 1.7, 0.9, 0.5, 0.2, 1.1, 0.0];
        let swapped =
            Tree::from_parents(vec![5, 5, 4, 4, 6, 6, NO_NODE], swapped_branch, 4).unwrap();
        let mut s2 = NodeState::new(swapped.n_nodes(), p, &m, &w).unwrap();
        let l2 = s2.prune(&swapped);

        assert_relative_eq!(l1, l2, epsilon = 1e-12);
    }

    #[test]
    fn test_levels_partition_the_internal_nodes() {
        let tree = Tree::balanced_binary(64, 1.0).unwrap();
        assert_eq!(tree.n_levels(), 6);
        let mut covered = 0usize;
        for level in 0..tree.n_levels() {
            let (start, end) = tree.level(level);
            // Every child of a node in this level must already be settled.
            for node in start..end {
                for &c in tree.children(node as u32) {
                    assert!((c as usize) < start);
                }
            }
            covered += end - start;
        }
        assert_eq!(covered, tree.n_nodes() - tree.n_leaves());
    }

    #[test]
    fn test_f32_and_f64_storage_agree() {
        let (n_leaves, p) = (64usize, 128usize);
        let (m, w) = leaf_data(n_leaves, p);
        let tree = Tree::balanced_binary(n_leaves, 0.6).unwrap();

        let mut s64 = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let l64 = s64.prune(&tree);

        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let mut s32 = NodeState::new(tree.n_nodes(), p, &m32, &w32).unwrap();
        let l32 = s32.prune(&tree);

        assert_relative_eq!(l32, l64, max_relative = 1e-4);
    }
}
