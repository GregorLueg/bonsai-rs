//! Feature-blocked node state: the same pruning recursion, laid out so that the
//! parallel axis is the feature axis rather than the tree level.
//!
//! The model factorises over features (SPEC.md section 2, S13), so a feature is
//! independent of every other feature through the whole recursion. Splitting the
//! features into blocks and running an entire sweep per block therefore needs no
//! synchronisation at all, and, unlike level-parallelism, does not care what
//! shape the tree is. A ladder parallelises exactly as well as a balanced tree.
//!
//! The cost is a layout change. State is stored `[block][node][feature]`, so one
//! block's whole sweep streams through one contiguous region and different
//! threads never touch the same cache line. In the `[node][feature]` layout of
//! [`crate::model::likelihood::NodeState`] a feature block is strided across the
//! entire allocation, which is the wrong access pattern for this axis.

use rayon::prelude::*;

use crate::errors::BonsaiErrors;
use crate::tree::Tree;
use crate::model::likelihood::prune_node_into;
use crate::utils::traits::BonsaiFloat;

/// Default features per block.
///
/// Sets the working set of one thread's sweep: `2 * n_nodes * BLOCK` elements.
/// Wants to be large enough that a node's slice is several cache lines and the
/// per-call overhead of the kernels is amortised, and small enough that the
/// block stays cheap to stream. 128 puts a node's `f64` slice at 1 KiB and, at
/// 16k nodes, a block's two arrays at 32 MiB. Chosen 2026-08-27; see the
/// `block_size` sweep in `benches/prune_sweep.rs` before changing it.
pub const DEFAULT_BLOCK: usize = 128;

/// Effective means and precisions, laid out `[block][node][feature]`.
#[derive(Clone, Debug)]
pub struct BlockedState<T> {
    /// Effective means.
    m: Vec<T>,
    /// Effective precisions.
    w: Vec<T>,
    /// Number of features.
    p: usize,
    /// Number of nodes, leaves included.
    n_nodes: usize,
    /// Features per block; the last block may be shorter.
    block: usize,
    /// Offset into `m` and `w` at which each block starts, length
    /// `n_blocks + 1`. Precomputed because the last block is ragged.
    block_start: Vec<usize>,
}

impl<T: BonsaiFloat> BlockedState<T> {
    /// Allocate state for a tree and fill the leaf rows from row-major input.
    ///
    /// The input is the natural `[cell][feature]` matrix; this constructor does
    /// the transpose into blocked order once, up front.
    ///
    /// ### Params
    ///
    /// * `n_nodes` - Total node count of the tree this state will serve
    /// * `p` - Number of features
    /// * `leaf_means` - Transformed means, row-major `[leaf][feature]`
    /// * `leaf_precisions` - Transformed precisions, same layout
    /// * `block` - Features per block; `DEFAULT_BLOCK` is the usual choice
    ///
    /// ### Returns
    ///
    /// The state, or `ShapeMismatch` if the two leaf blocks disagree.
    pub fn new(
        n_nodes: usize,
        p: usize,
        leaf_means: &[T],
        leaf_precisions: &[T],
        block: usize,
    ) -> Result<Self, BonsaiErrors> {
        if leaf_means.len() != leaf_precisions.len() {
            return Err(BonsaiErrors::ShapeMismatch {
                mean_cells: leaf_means.len() / p.max(1),
                mean_features: p,
                sd_cells: leaf_precisions.len() / p.max(1),
                sd_features: p,
            });
        }
        let block = block.max(1);
        let n_blocks = p.div_ceil(block);

        let mut block_start = Vec::with_capacity(n_blocks + 1);
        let mut acc = 0usize;
        for b in 0..n_blocks {
            block_start.push(acc);
            acc += n_nodes * (block.min(p - b * block));
        }
        block_start.push(acc);

        let mut m = vec![T::zero(); acc];
        let mut w = vec![T::zero(); acc];
        let n_leaves = leaf_means.len() / p.max(1);
        for b in 0..n_blocks {
            let lo = b * block;
            let len = block.min(p - lo);
            for leaf in 0..n_leaves {
                let dst = block_start[b] + leaf * len;
                let src = leaf * p + lo;
                m[dst..dst + len].copy_from_slice(&leaf_means[src..src + len]);
                w[dst..dst + len].copy_from_slice(&leaf_precisions[src..src + len]);
            }
        }

        Ok(Self {
            m,
            w,
            p,
            n_nodes,
            block,
            block_start,
        })
    }

    /// Number of feature blocks.
    ///
    /// ### Returns
    ///
    /// The block count.
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.block_start.len() - 1
    }

    /// Effective means of one node, in the natural `[feature]` order.
    ///
    /// Gathers across blocks, so this is for inspection and output, not for the
    /// hot path.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's `p` effective means.
    pub fn means(&self, node: u32) -> Vec<T> {
        self.gather(&self.m, node)
    }

    /// Effective precisions of one node, in the natural `[feature]` order.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's `p` effective precisions.
    pub fn precisions(&self, node: u32) -> Vec<T> {
        self.gather(&self.w, node)
    }

    /// Collect one node's row out of the blocked layout.
    ///
    /// ### Params
    ///
    /// * `src` - Either the means or the precisions array
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The row in feature order.
    fn gather(&self, src: &[T], node: u32) -> Vec<T> {
        let mut out = Vec::with_capacity(self.p);
        for b in 0..self.n_blocks() {
            let len = self.block.min(self.p - b * self.block);
            let lo = self.block_start[b] + node as usize * len;
            out.extend_from_slice(&src[lo..lo + len]);
        }
        out
    }

    /// Run the pruning recursion, fanning out over feature blocks.
    ///
    /// Each block runs the full post-order sweep over its own slice of the
    /// features, independently of every other block. Block subtotals are summed
    /// in block order, so the answer does not depend on the thread count.
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
        let (p, block) = (self.p, self.block);
        let starts = &self.block_start;
        let n_blocks = starts.len() - 1;

        // Hand each block its own disjoint mutable window of both arrays.
        let mut m_rest: &mut [T] = &mut self.m;
        let mut w_rest: &mut [T] = &mut self.w;
        let mut windows = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let len = starts[b + 1] - starts[b];
            let (m_here, m_tail) = m_rest.split_at_mut(len);
            let (w_here, w_tail) = w_rest.split_at_mut(len);
            m_rest = m_tail;
            w_rest = w_tail;
            windows.push((b, m_here, w_here));
        }

        let mut parts = vec![0.0f64; n_blocks];
        windows
            .into_par_iter()
            .map(|(b, m, w)| {
                let len = block.min(p - b * block);
                let mut scratch: Vec<f64> = Vec::new();
                let mut acc = 0.0f64;

                for node in tree.internal_postorder() {
                    acc += prune_node_into(tree, node, len, m, w, &mut scratch);
                }
                acc
            })
            .collect_into_vec(&mut parts);

        parts.iter().sum()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::likelihood::{NodeState, tests::leaf_data};
    use crate::tree::Tree;
    use approx::assert_relative_eq;

    #[test]
    fn test_blocked_matches_row_major_on_a_balanced_tree() {
        let (n_leaves, p) = (256usize, 300usize);
        let (m, w) = leaf_data(n_leaves, p);
        let tree = Tree::balanced_binary(n_leaves, 0.55).unwrap();

        let mut flat = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let l_flat = flat.prune(&tree);

        // 128 does not divide 300, so the ragged last block is exercised.
        let mut blocked = BlockedState::new(tree.n_nodes(), p, &m, &w, 128).unwrap();
        let l_blocked = blocked.prune(&tree);

        assert_relative_eq!(l_blocked, l_flat, max_relative = 1e-12);
        for node in 0..tree.n_nodes() as u32 {
            for g in 0..p {
                assert_relative_eq!(
                    blocked.means(node)[g],
                    flat.means(node)[g],
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_blocked_matches_row_major_on_a_ladder() {
        let (n_leaves, p) = (200usize, 64usize);
        let (m, w) = leaf_data(n_leaves, p);
        let tree = Tree::ladder(n_leaves, 0.8).unwrap();

        let mut flat = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let mut blocked = BlockedState::new(tree.n_nodes(), p, &m, &w, 16).unwrap();

        assert_relative_eq!(blocked.prune(&tree), flat.prune(&tree), max_relative = 1e-12);
    }

    #[test]
    fn test_block_size_does_not_change_the_answer() {
        let (n_leaves, p) = (128usize, 257usize);
        let (m, w) = leaf_data(n_leaves, p);
        let tree = Tree::balanced_binary(n_leaves, 0.4).unwrap();

        let mut reference = BlockedState::new(tree.n_nodes(), p, &m, &w, 1).unwrap();
        let want = reference.prune(&tree);

        for block in [7usize, 32, 64, 256, 1024] {
            let mut got = BlockedState::new(tree.n_nodes(), p, &m, &w, block).unwrap();
            assert_relative_eq!(got.prune(&tree), want, max_relative = 1e-12);
        }
    }
}
