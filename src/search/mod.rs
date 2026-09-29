//! The tree search of SPEC.md section 9.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::tree::Tree;
use crate::utils::rng::SplitMix64;
use crate::utils::traits::BonsaiFloat;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

pub mod bounds;
pub mod candidates;
pub(crate) mod live;
pub(crate) mod masked;
pub mod nni;
pub mod polytomy;
pub mod spr;
pub mod star;

/// The leaf data a search step scores its trees against.
///
/// Both blocks are row-major `[leaf][feature]` in the transformed units of
/// SPEC.md section 3.1, the layout
/// [`crate::model::likelihood::NodeState::new`] expects.
#[derive(Clone, Copy, Debug)]
pub struct Leaves<'a, T> {
    /// Transformed means, `[leaf][feature]`.
    pub means: &'a [T],
    /// Transformed precisions, same layout.
    pub precisions: &'a [T],
    /// Number of features.
    pub n_features: usize,
}

///////////////////
// Settled trees //
///////////////////

/// Loglikelihood of a tree, from the leaf data alone.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// The tree loglikelihood, or the error the state allocation failed with.
pub(crate) fn tree_loglik<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
) -> Result<f64, BonsaiErrors> {
    Ok(settled_down(tree, leaves)?.1)
}

/// Settle a tree's down rows alone.
///
/// Skips the up sweep.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// The settled rows and the tree loglikelihood, or the error the state
/// allocation failed with.
pub(crate) fn settled_down<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
) -> Result<(NodeState<T>, f64), BonsaiErrors> {
    let mut down = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )?;
    let loglik = down.prune(tree);
    Ok((down, loglik))
}

/// Settle a tree's down and up rows.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// The settled rows and the tree loglikelihood, or the error the state
/// allocation failed with.
pub(crate) fn settle<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
) -> Result<(NodeState<T>, UpState<T>, f64), BonsaiErrors> {
    let (down, loglik) = settled_down(tree, leaves)?;
    let mut up = UpState::new(tree.n_nodes(), leaves.n_features);
    up.sweep(tree, &down);
    Ok((down, up, loglik))
}

/// Leaves standing below every node of a tree.
///
/// ### Params
///
/// * `tree` - The tree
///
/// ### Returns
///
/// One count per node, indexed by node id.
pub(crate) fn leaves_below(tree: &Tree) -> Vec<usize> {
    let mut below = vec![0usize; tree.n_nodes()];
    for leaf in 0..tree.n_leaves() {
        below[leaf] = 1;
    }
    for node in tree.internal_postorder() {
        below[node as usize] = tree.children(node).iter().map(|&c| below[c as usize]).sum();
    }
    below
}

////////////////////////
// Split fingerprints //
////////////////////////

/// Order-independent fingerprint of a tree's unrooted splits.
///
/// Leaves get fixed pseudo-random words, an internal node's word is the sum of
/// its subtree's, and each split contributes a hash of the smaller of its word
/// and complement. Blind to root position, sibling order and node numbering.
/// NNI and SPR use it to reject proposals that change no split (SPEC.md
/// section 9.4 deviation note); without it the greedy phase loops on
/// branch-length descent.
///
/// The second child of a degree-two root is skipped: both children describe
/// one split and would otherwise be counted twice.
///
/// ### Params
///
/// * `tree` - Tree to fingerprint
///
/// ### Returns
///
/// The fingerprint. Two trees with the same unrooted splits share it; two
/// with different splits collide with probability about `2^-64`.
#[cfg(test)]
pub(crate) fn split_fingerprint(tree: &Tree) -> u64 {
    split_fingerprint_with(tree, &leaf_words(tree))
}

/// [`split_fingerprint`] over words the caller already has.
///
/// ### Params
///
/// * `tree` - Tree to fingerprint
/// * `word` - Its [`leaf_words`]
///
/// ### Returns
///
/// The fingerprint.
#[cfg(any(test, debug_assertions))]
pub(crate) fn split_fingerprint_with(tree: &Tree, word: &[u64]) -> u64 {
    split_fingerprint_counted(tree, word, &leaves_below(tree))
}

/// One split's term in [`split_fingerprint_with`].
///
/// The leaf word canonicalised against its complement, then mixed. The
/// fingerprint is the wrapping sum of these over the distinct non-trivial
/// splits.
///
/// ### Params
///
/// * `here` - Leaf word of one side of the split
/// * `total` - Leaf word of every leaf
///
/// ### Returns
///
/// The split's term.
#[inline]
pub(crate) fn split_hash(here: u64, total: u64) -> u64 {
    SplitMix64::new(here.min(total.wrapping_sub(here))).next_u64()
}

/// [`split_fingerprint_with`] over leaf counts the caller has as well.
///
/// ### Params
///
/// * `tree` - Tree to fingerprint
/// * `word` - Its [`leaf_words`]
/// * `below` - Its [`leaves_below`]
///
/// ### Returns
///
/// The fingerprint.
pub(crate) fn split_fingerprint_counted(tree: &Tree, word: &[u64], below: &[usize]) -> u64 {
    let root = tree.root();
    let total = word[root as usize];
    let root_kids = tree.children(root);
    let duplicate = if root_kids.len() == 2 {
        root_kids[1]
    } else {
        crate::tree::NO_NODE
    };

    let n_leaves = tree.n_leaves();

    tree.internal_postorder()
        .filter(|&node| tree.parent(node).is_some() && node != duplicate)
        .filter(|&node| {
            // Trivial splits (under two leaves a side) are shared by every tree;
            // excluding them keeps the fingerprint stable when rooting on a leaf branch.
            let here = below[node as usize];
            here >= 2 && n_leaves - here >= 2
        })
        .fold(0u64, |acc, node| {
            acc.wrapping_add(split_hash(word[node as usize], total))
        })
}

/// Per-node word summarising which leaves sit below it.
///
/// A leaf's word is fixed pseudo-random, an internal node's is the sum of its
/// subtree's, so it names a leaf set and survives the renumbering of
/// `Tree::from_parents`. Distinct sets collide at roughly `2^-64`.
///
/// ### Params
///
/// * `tree` - Tree to summarise
///
/// ### Returns
///
/// One word per node, indexed by node id.
pub(crate) fn leaf_words(tree: &Tree) -> Vec<u64> {
    let mut word = vec![0u64; tree.n_nodes()];
    for leaf in 0..tree.n_leaves() {
        word[leaf] = SplitMix64::new(leaf as u64).next_u64();
    }
    for node in tree.internal_postorder() {
        word[node as usize] = tree
            .children(node)
            .iter()
            .fold(0u64, |acc, &child| acc.wrapping_add(word[child as usize]));
    }
    word
}

/// Words of every node within `radius` edges of a clade a move created.
///
/// A created clade is a node of the new tree with no counterpart in the old
/// one. The walk is unrooted, so it also reaches the moved subtree and its new
/// siblings.
///
/// ### Params
///
/// * `tree` - The tree the move produced
/// * `word` - [`leaf_words`] of `tree`
/// * `is_new` - Whether an internal node of `tree` is a clade the move created
/// * `radius` - How many edges out to mark
/// * `out` - Set the words are added to
pub(crate) fn mark_near_new_clades(
    tree: &Tree,
    word: &[u64],
    is_new: impl Fn(usize) -> bool,
    radius: usize,
    out: &mut FxHashSet<u64>,
) {
    let n = tree.n_nodes();
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    for v in tree.n_leaves()..n {
        if is_new(v) {
            dist[v] = 0;
            queue.push_back(v as u32);
        }
    }
    while let Some(v) = queue.pop_front() {
        out.insert(word[v as usize]);
        let d = dist[v as usize];
        if d == radius {
            continue;
        }
        for nb in tree.children(v).iter().copied().chain(tree.parent(v)) {
            if dist[nb as usize] == usize::MAX {
                dist[nb as usize] = d + 1;
                queue.push_back(nb);
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
    use crate::tree::cluster::reroot;

    /// Regression: a degree-two root's children describe one split, which must
    /// be counted once.
    #[test]
    fn test_the_fingerprint_is_blind_to_where_the_tree_is_rooted() {
        let tree = Tree::balanced_binary(16, 0.7).expect("balanced fixture");
        let want = split_fingerprint(&tree);

        // Every internal edge is a legal place to put the root, and none of
        // them may change the fingerprint.
        let mut checked = 0usize;
        for edge in 0..tree.n_nodes() as u32 {
            if tree.parent(edge).is_none() {
                continue;
            }
            let rerooted = reroot(&tree, edge).expect("reroot");
            assert_eq!(
                split_fingerprint(&rerooted),
                want,
                "rooting on the branch above node {edge} changed the fingerprint"
            );
            checked += 1;
        }
        assert!(checked > 20, "only {checked} edges exercised");
    }

    #[test]
    fn test_the_fingerprint_separates_genuinely_different_topologies() {
        // The other half: a filter that never fires is as useless as one that
        // always does.
        let balanced = Tree::balanced_binary(16, 0.7).expect("balanced fixture");
        let ladder = Tree::ladder(16, 0.7).expect("ladder fixture");
        assert_ne!(split_fingerprint(&balanced), split_fingerprint(&ladder));
    }

    #[test]
    fn test_leaf_words_name_a_leaf_set_not_a_node() {
        // The property SPR relies on: the word identifies the set of leaves
        // below a node, so it survives the renumbering a rebuild performs.
        let tree = Tree::balanced_binary(8, 1.0).expect("balanced fixture");
        let word = leaf_words(&tree);
        for node in tree.internal_postorder() {
            let summed: u64 = tree
                .children(node)
                .iter()
                .fold(0u64, |acc, &c| acc.wrapping_add(word[c as usize]));
            assert_eq!(word[node as usize], summed);
        }
        // Distinct leaves get distinct words, so no two leaves alias.
        let mut leaves: Vec<u64> = (0..tree.n_leaves()).map(|l| word[l]).collect();
        leaves.sort_unstable();
        let before = leaves.len();
        leaves.dedup();
        assert_eq!(leaves.len(), before);
    }
}
