//! The pruning recursion and the tree loglikelihood.
//!
//! Implements SPEC.md sections 4 and 5: every subtree collapses into an
//! effective leaf carrying a mean and a precision per feature, and the tree
//! loglikelihood accumulates as that collapse proceeds. This is the
//! continuous-trait form of Felsenstein's pruning algorithm.
//!
//! Values are up to the additive constants dropped in SPEC.md section 3, so
//! only differences between loglikelihoods are meaningful.

use crate::errors::BonsaiErrors;
use crate::tree::Tree;
use crate::utils::kernels::prune_general;
use crate::utils::simd::prune_binary;
use crate::utils::traits::BonsaiFloat;

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
        let mut m = vec![T::zero(); n_nodes * p];
        let mut w = vec![T::zero(); n_nodes * p];
        m[..leaf_means.len()].copy_from_slice(leaf_means);
        w[..leaf_precisions.len()].copy_from_slice(leaf_precisions);
        Ok(Self {
            m,
            w,
            p,
            n_nodes,
            scratch: Vec::new(),
        })
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
    /// Sequential, and deliberately so: this is the straightforward reference
    /// that [`crate::model::blocked::BlockedState`] is checked against, and the
    /// layout that makes a whole node row contiguous. Walks levels from the
    /// leaves up and, within a level, ascending node index, which the arena
    /// invariant makes a valid post-order.
    ///
    /// Per-node contributions are summed within a level and only then added to
    /// the running total. Floating-point addition is not associative, so fixing
    /// the association to the tree rather than to the traversal is what makes
    /// the two implementations comparable at all.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose topology and branch lengths to use
    ///
    /// ### Returns
    ///
    /// The tree loglikelihood, up to the dropped additive constants.
    pub fn prune(&mut self, tree: &Tree) -> f64 {
        debug_assert_eq!(tree.n_nodes(), self.n_nodes);
        let mut total = 0.0f64;
        let mut parts: Vec<f64> = Vec::new();

        for level in 0..tree.n_levels() {
            let (start, end) = tree.level(level);
            parts.clear();
            parts.extend((start..end).map(|node| self.prune_node(tree, node as u32)));
            total += parts.iter().sum::<f64>();
        }
        total
    }

    /// Prune one internal node, reading its children's settled rows.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose topology and branch lengths to use
    /// * `node` - Internal node to settle
    ///
    /// ### Returns
    ///
    /// The node's loglikelihood contribution.
    fn prune_node(&mut self, tree: &Tree, node: u32) -> f64 {
        let p = self.p;
        let split = node as usize * p;
        let (m_lo, m_hi) = self.m.split_at_mut(split);
        let (w_lo, w_hi) = self.w.split_at_mut(split);
        let m_out = &mut m_hi[..p];
        let w_out = &mut w_hi[..p];

        let kids = tree.children(node);
        match kids.len() {
            2 => {
                let (k, l) = (kids[0] as usize, kids[1] as usize);
                prune_binary(
                    &m_lo[k * p..k * p + p],
                    &w_lo[k * p..k * p + p],
                    tree.branch(kids[0]),
                    &m_lo[l * p..l * p + p],
                    &w_lo[l * p..l * p + p],
                    tree.branch(kids[1]),
                    m_out,
                    w_out,
                )
            }
            n_child => {
                if self.scratch.len() < p * n_child {
                    self.scratch.resize(p * n_child, 0.0);
                }
                let children: Vec<(&[T], &[T], f64)> = kids
                    .iter()
                    .map(|&c| {
                        let lo = c as usize * p;
                        (&m_lo[lo..lo + p], &w_lo[lo..lo + p], tree.branch(c))
                    })
                    .collect();
                prune_general(&children, m_out, w_out, &mut self.scratch[..p * n_child])
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{NO_NODE, Tree};
    use approx::assert_relative_eq;

    /// Deterministic pseudo-random leaf data, so tests do not need an rng
    /// dependency and always describe the same scenario.
    fn leaf_data(n_leaves: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut m = Vec::with_capacity(n_leaves * p);
        let mut w = Vec::with_capacity(n_leaves * p);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n_leaves * p {
            m.push(next() * 4.0 - 2.0);
            w.push(0.25 + next() * 3.0);
        }
        (m, w)
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
        let tree = Tree::from_parents(
            vec![4, 4, 5, 5, 5, NO_NODE],
            vec![branch_len; 6],
            4,
        )
        .unwrap();
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
        let collapsed =
            Tree::from_parents(vec![3, 3, 3, NO_NODE], vec![branch_len; 4], 3).unwrap();
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

        let straight = Tree::from_parents(vec![4, 4, 5, 5, 6, 6, NO_NODE], branch.clone(), 4)
            .unwrap();
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
