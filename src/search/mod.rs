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
pub mod nni;
pub mod polytomy;
pub mod spr;
pub mod star;

/// The leaf data a search step scores its trees against.
///
/// Both blocks are row-major `[leaf][feature]` in the transformed units of
/// SPEC.md section 3.1, which is the layout
/// [`crate::model::likelihood::NodeState::new`] expects. Search steps rebuild
/// the tree, so they rebuild the node state with it and need the leaf rows
/// rather than a settled state.
#[derive(Clone, Copy, Debug)]
pub struct Leaves<'a, T> {
    /// Transformed means, `[leaf][feature]`.
    pub means: &'a [T],
    /// Transformed precisions, same layout.
    pub precisions: &'a [T],
    /// Number of features.
    pub n_features: usize,
}

//////////////////////
// Settled trees    //
//////////////////////

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
/// The up sweep is the more expensive half of settling a tree, so a caller that
/// only reads down rows takes this.
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
/// Every leaf gets a fixed pseudo-random word; an internal node's word is the
/// sum of its subtree's. A split is then canonicalised by taking the smaller of
/// the word and its complement, so the fingerprint does not depend on which
/// node is the root, on sibling order, or on internal node numbering. The
/// fingerprint is the sum of a hash of each split, which is a multiset hash:
/// the same set of splits gives the same word whatever order they are met in.
/// Used by both NNI and SPR to reject a proposal that changes no split. That
/// filter decides whether the search terminates: without it, collapsing and
/// re-resolving a star reports a gain from reoptimising branch lengths while
/// leaving the topology alone, and the greedy phase does branch-length descent
/// forever. See the deviation note on SPEC.md section 9.4.
///
/// **Skipping the second child of a degree-two root is not tidying.** Its two
/// children describe the same split, so the sum would carry it twice and two
/// representations of one unrooted tree would compare unequal. Skipping it is
/// what makes the fingerprint blind to rerooting, which is what the filter
/// needs it to be.
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
pub(crate) fn split_fingerprint_with(tree: &Tree, word: &[u64]) -> u64 {
    let root = tree.root();
    let total = word[root as usize];
    let root_kids = tree.children(root);
    let duplicate = if root_kids.len() == 2 {
        root_kids[1]
    } else {
        crate::tree::NO_NODE
    };

    // Leaf counts, to drop the splits that carry no information.
    let below = leaves_below(tree);
    let n_leaves = tree.n_leaves();

    tree.internal_postorder()
        .filter(|&node| tree.parent(node).is_some() && node != duplicate)
        .filter(|&node| {
            // A split with fewer than two leaves on a side is trivial: every
            // tree over the same leaves has it, so it distinguishes nothing.
            // Excluding it is what makes the fingerprint survive rooting on a
            // leaf's own branch, which puts the old root one step above a tip
            // and makes it describe exactly such a split.
            let here = below[node as usize];
            here >= 2 && n_leaves - here >= 2
        })
        .fold(0u64, |acc, node| {
            let here = word[node as usize];
            let split = here.min(total.wrapping_sub(here));
            acc.wrapping_add(SplitMix64::new(split).next_u64())
        })
}

/// Per-node word summarising which leaves sit below it.
///
/// Each leaf gets a fixed pseudo-random word and an internal node gets the sum
/// of its subtree's, so the value identifies a *set of leaves* rather than a
/// node index. That is what makes it survive the renumbering `Tree::from_parents`
/// performs: SPR uses it to name a subtree across a rebuild, and
/// [`split_fingerprint`] to name a split.
///
/// Summation means two different leaf sets can collide, at roughly `2^-64` per
/// comparison. Fine for both callers, neither of which is deciding correctness
/// on the result alone.
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
/// one, which is exactly the path the move rewired. The walk is unrooted, so it
/// reaches the moved subtree and its new siblings as well as the ancestors.
/// SPR uses it to choose what the next sweep proposes, NNI to choose which
/// cached gains to throw away.
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

    /// Regression. NNI and SPR independently grew the same split
    /// fingerprint, but only SPR's deduplicated. A degree-two root's two
    /// children describe one split, so the raw key carries it twice and the
    /// same unrooted tree compares unequal to itself under a different root.
    /// NNI's move filter would then accept a proposal that changed nothing but
    /// where the root sat, which is the branch-length descent the filter exists
    /// to stop.
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
