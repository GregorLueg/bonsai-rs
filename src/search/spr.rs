//! Subtree pruning and regrafting, search step 5 (SPEC.md section 9.3).
//!
//! One move detaches a node and everything below it, then hangs it back
//! elsewhere. The subtree is summarised as an effective leaf by the pruning
//! recursion of SPEC.md section 4, its new home is found by the beam search of
//! [`place`] (SPEC.md section 7.2), and the polytomy the attachment creates is
//! resolved by [`crate::search::polytomy::splice_star`] (SPEC.md sections 7.3
//! and 9.2).
//!
//! ### Degree-two nodes
//!
//! Removing a child from a node of degree three leaves a node that carries no
//! information: the pruning recursion passes its child's effective leaf through
//! with the two branch lengths added. The arena refuses to hold one
//! ([`Tree::from_parents`] wants two children per internal node), so the two
//! remaining neighbours are joined by the summed branch. A root left with two
//! children is suppressed the same way, by making one child the root and giving
//! the other the summed branch. The exception is a remaining tree of two leaves,
//! where the degree-two root is the tree's single edge.
//!
//! ### What may not be pruned
//!
//! The root, and a child of a root that has only two children. Both are rejected
//! before any work is done.
//!
//! ### Attachment points
//!
//! The beam search runs on the remaining tree, from which the pruned subtree is
//! absent, so it can never be regrafted onto itself.
//!
//! ### Accepting a move
//!
//! On the candidate tree's own per-node terms, never on
//! [`crate::search::polytomy::Splice::gain`]: that gain is exact only against the
//! tree its star was built from, and a regraft moves a subtree across the tree.
//! The terms come from the masked views of [`crate::search::masked`], which
//! recompute only the paths a move changed and score it as the current total
//! less the changed terms plus the new ones ([`fixed`]); where the views
//! decline, [`LazyRows::score`] does the same on a built candidate. An accepted
//! candidate is applied in one relabel of the arena
//! ([`crate::search::masked::apply_move`]) and its rows written into the current
//! tree's [`RowStore`]. [`LazyRows`] forms only the rows a proposal reads.
//!
//! ### Topology only
//!
//! A regraft onto the split the subtree came off only reoptimises nearby
//! branches, which steps 4 and 7 already do globally. So a proposal whose split
//! fingerprint matches the current tree's is discarded, the same deviation
//! SPEC.md section 9.4 records for the interchanges.

use crate::errors::BonsaiErrors;
use crate::model::global::{LOGLIK_SCALE_FLOOR, up_part};
use crate::model::likelihood::NodeState;
use crate::model::merge::EffLeaf;
use crate::model::place::{PlacementParams, place, place_walk};
use crate::search::live::LiveTree;
use crate::search::masked::{
    Attached, MoveData, Pruned as PrunedView, PrunedRows, Regraft, ViewCache, apply_move, attach,
    score_move,
};
#[cfg(debug_assertions)]
use crate::search::polytomy::splice_result;
use crate::search::polytomy::{CentreStar, splice_edits};
#[cfg(any(test, debug_assertions))]
use crate::search::split_fingerprint_with;
use crate::search::star::{StarParams, StarResult, resolve_star};
#[cfg(test)]
use crate::search::tree_loglik;
use crate::search::{
    Leaves, leaf_words, leaves_below, mark_near_new_clades, settled_down,
    split_fingerprint_counted, split_hash,
};
use crate::tree::{NO_NODE, Tree};
use crate::utils::kernels::prune_general;
use crate::utils::rng::SplitMix64;
use crate::utils::simd::prune_binary;
use crate::utils::traits::{BonsaiFloat, narrow, wide};
use crate::utils::verbosity::Verbosity;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

////////////
// Consts //
////////////

/// Default for [`SprParams::max_rounds`].
///
/// A runaway guard: every accepted move exceeds [`acceptance_floor`], so the
/// sweeps terminate on their own. Simulated fixtures of 16 to 64 leaves need two
/// to four rounds (measured).
const DEFAULT_MAX_ROUNDS: usize = 100;

/// Default for [`SprParams::min_relative_gain`], as a fraction of `|L|`.
///
/// The acceptance gate compares a whole-tree loglikelihood of magnitude
/// `O(n p)`, so the floor scales with it, as in
/// [`crate::model::global::optimise_branch_lengths`]:
/// `min_relative_gain * max(|L|, LOGLIK_SCALE_FLOOR)`. An absolute floor cycled
/// on rounding noise at ten thousand cells, `|L|` around `1e7` (measured).
const DEFAULT_MIN_RELATIVE_GAIN: f64 = 1e-12;

/// Fewest candidates a sweep proposes in parallel before deciding any of them.
///
/// The chunk halves on every acceptance down to this, since the first accepted
/// move invalidates the rest of its chunk. Ours: wall time is flat below eight
/// and the tree is unchanged at every floor (measured).
const PROPOSAL_CHUNK_MIN: usize = 8;

/// Most candidates a sweep proposes in parallel before deciding any of them.
///
/// Doubles from [`PROPOSAL_CHUNK_MIN`] on every chunk that accepts nothing.
/// Bounds memory: a candidate the views declined is a whole arena.
const PROPOSAL_CHUNK_MAX: usize = 256;

/// Default for [`SprApprox::recheck`].
const DEFAULT_RECHECK: bool = true;

/// Default for [`SprApprox::revisit_radius`].
const DEFAULT_REVISIT_RADIUS: usize = 5;

/// Scale of the fixed-point loglikelihood totals SPR accepts on, `2^64`.
///
/// Truncating every term onto a grid of `2^-64` nats and summing integers makes
/// a total independent of arena order; an `i128` holds up to `9e18` nats.
const FIXED_SCALE: f64 = 18_446_744_073_709_551_616.0;

////////////////
// PruneOrder //
////////////////

/// Which subtree the sweep considers next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneOrder {
    /// Descending length of the branch above the subtree, ties going to the
    /// lower node index.
    ///
    /// The paper's default (SPEC.md section 9.3).
    LongestBranch,
    /// A uniform shuffle of the eligible subtrees, from [`SprParams::seed`].
    Random,
}

///////////////
// SprApprox //
///////////////

/// Knobs of the approximate search; each changes the answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SprApprox {
    /// After the first sweep, propose only subtrees within this many edges of
    /// a clade the previous sweep's accepted moves created. `0` switches this
    /// approximation off and proposes every subtree every sweep.
    ///
    /// Don't-look bits (Bentley, *ORSA Journal on Computing*, 1992). Not exact,
    /// because every effective leaf depends on the whole tree.
    pub revisit_radius: usize,
    /// After a move is accepted, keep the rest of its chunk rather than
    /// proposing it again: a proposal that moved something is re-applied to the
    /// current tree at the same attachment point and re-scored exactly there.
    /// The sweep stays monotone; the attachment point was found on the tree
    /// before the acceptance.
    pub recheck: bool,
}

impl Default for SprApprox {
    /// Every approximation at its measured default.
    ///
    /// ### Returns
    ///
    /// The default knobs.
    fn default() -> Self {
        Self {
            revisit_radius: DEFAULT_REVISIT_RADIUS,
            recheck: DEFAULT_RECHECK,
        }
    }
}

/// The specified search, or this crate's approximation of it.
///
/// Both never lower the loglikelihood; they differ in which moves they look at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprSearch {
    /// Every subtree every sweep, as SPEC.md section 9.3 specifies.
    Exact,
    /// The approximations in [`SprApprox`]. The default: within a few nats of
    /// [`SprSearch::Exact`] and up to twice as fast over steps 5 to 8 (measured).
    Approximate(SprApprox),
}

impl Default for SprSearch {
    /// [`SprSearch::Approximate`] at its measured defaults.
    ///
    /// ### Returns
    ///
    /// The default search.
    fn default() -> Self {
        Self::Approximate(SprApprox::default())
    }
}

impl SprSearch {
    /// The revisit radius in force, `0` when every subtree is proposed.
    ///
    /// ### Returns
    ///
    /// The radius.
    fn revisit_radius(&self) -> usize {
        match self {
            Self::Exact => 0,
            Self::Approximate(a) => a.revisit_radius,
        }
    }

    /// Whether a sweep keeps the rest of a chunk after an acceptance.
    ///
    /// ### Returns
    ///
    /// [`SprApprox::recheck`], false for the exact search.
    fn recheck(&self) -> bool {
        match self {
            Self::Exact => false,
            Self::Approximate(a) => a.recheck,
        }
    }
}

///////////////
// SprParams //
///////////////

/// Tuning knobs for search step 5.
#[derive(Clone, Copy, Debug)]
pub struct SprParams {
    /// Order the sweep visits candidate subtrees in.
    pub order: PruneOrder,
    /// Seed for [`PruneOrder::Random`], unused otherwise.
    pub seed: u64,
    /// Cap on sweeps over the tree.
    pub max_rounds: usize,
    /// Beam-search knobs handed to [`place`].
    pub placement: PlacementParams,
    /// Star primitive knobs handed to the polytomy resolution. Its `min_gain`
    /// is also an absolute floor on an accepted move; see [`acceptance_floor`].
    pub star: StarParams,
    /// Smallest improvement in the whole-tree loglikelihood that will be
    /// accepted as a move, as a fraction of `max(|L|, 1)`.
    ///
    /// Strictly greater than, and the maximum with `star.min_gain`.
    pub min_relative_gain: f64,
    /// Whether to run the specified search or this crate's approximation of it.
    pub search: SprSearch,
}

impl Default for SprParams {
    /// The paper's ordering, `DEFAULT_MAX_ROUNDS`, and the shipped
    /// placement and star defaults.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            order: PruneOrder::LongestBranch,
            seed: 0,
            max_rounds: DEFAULT_MAX_ROUNDS,
            placement: PlacementParams::default(),
            star: StarParams::default(),
            min_relative_gain: DEFAULT_MIN_RELATIVE_GAIN,
            search: SprSearch::default(),
        }
    }
}

///////////////
// Threshold //
///////////////

/// Smallest gain, in nats, that this round will accept as a move: the larger of
/// `star.min_gain` and `min_relative_gain * max(|L|, LOGLIK_SCALE_FLOOR)`.
///
/// ### Params
///
/// * `params` - The round's knobs
/// * `best` - Loglikelihood of the tree as it currently stands
///
/// ### Returns
///
/// The floor a proposal's gain has to exceed, strictly.
fn acceptance_floor(params: &SprParams, best: f64) -> f64 {
    params
        .star
        .min_gain
        .max(params.min_relative_gain * best.abs().max(LOGLIK_SCALE_FLOOR))
}

/// One loglikelihood term on the [`FIXED_SCALE`] grid.
///
/// ### Params
///
/// * `x` - The term, in nats
///
/// ### Returns
///
/// The term in units of `2^-64` nats, truncated towards zero.
#[inline]
pub(crate) fn fixed(x: f64) -> i128 {
    (x * FIXED_SCALE) as i128
}

/// A fixed-point total back in nats.
///
/// ### Params
///
/// * `x` - The total, in units of `2^-64` nats
///
/// ### Returns
///
/// The total, rounded to `f64`.
#[inline]
pub(crate) fn unfixed(x: i128) -> f64 {
    x as f64 / FIXED_SCALE
}

////////////
// Output //
////////////

/// What one accepted move did.
#[derive(Clone, Copy, Debug)]
pub struct SprGain {
    /// The subtree that was pruned, as a node of the tree the move started
    /// from. Renumbering makes it meaningless in the tree that came out.
    pub pruned: u32,
    /// Loglikelihood gain of the whole move, in nats, covering the regraft and
    /// the polytomy resolution that followed it.
    pub gain: f64,
}

/// What a run of the moves did.
#[derive(Clone, Debug)]
pub struct SprResult {
    /// The tree the sweeps finished on.
    pub tree: Tree,
    /// Its loglikelihood, from [`NodeState::prune`] on the tree itself.
    pub loglik: f64,
    /// One entry per accepted move, in the order they were performed.
    pub gains: Vec<SprGain>,
    /// Number of sweeps over the tree, the last of which found nothing.
    pub rounds: usize,
}

//////////////////////
// Arena rebuilding //
//////////////////////

/// Renumber a parent array into the arena invariant and build the tree.
///
/// Anything the walk from `root` does not reach is dropped. Kept leaves are
/// compacted into `0..k` in original order, and kept internal nodes are numbered
/// by height then original index, the order [`Tree::from_parents`] relabels
/// into, so [`Tree::from_level_ordered`] builds the arena directly and the
/// returned map refers to the tree that comes back.
///
/// ### Params
///
/// * `parent` - Parent index per node, [`NO_NODE`] where there is none
/// * `branch` - Branch above each node, same indexing
/// * `root` - Node to walk from
/// * `n_leaves` - Leaf-index boundary of the *input* space: indices below it
///   are leaves, and the ones the walk reaches become the new tree's leaves
///
/// ### Returns
///
/// The tree and, per input index, its index in that tree or [`NO_NODE`] if it
/// was dropped. `MalformedTree` if the arena rejected the result.
pub(crate) fn assemble(
    parent: &[u32],
    branch: &[f64],
    root: u32,
    n_leaves: usize,
) -> Result<(Tree, Vec<u32>), BonsaiErrors> {
    let n = parent.len();

    let mut ptr = vec![0u32; n + 1];
    for &par in parent {
        if par != NO_NODE {
            ptr[par as usize + 1] += 1;
        }
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let mut cursor = ptr.clone();
    let mut kids = vec![0u32; ptr[n] as usize];
    for (i, &par) in parent.iter().enumerate() {
        if par != NO_NODE {
            kids[cursor[par as usize] as usize] = i as u32;
            cursor[par as usize] += 1;
        }
    }

    // A cycle cannot be reached from the root, so the walk terminates.
    let mut reached = vec![false; n];
    let mut height = vec![0u32; n];
    let mut stack = vec![(root, false)];
    reached[root as usize] = true;
    while let Some((node, expanded)) = stack.pop() {
        let (lo, hi) = (ptr[node as usize] as usize, ptr[node as usize + 1] as usize);
        if expanded {
            height[node as usize] = kids[lo..hi]
                .iter()
                .map(|&c| height[c as usize] + 1)
                .max()
                .unwrap_or(0);
            continue;
        }
        stack.push((node, true));
        for &child in &kids[lo..hi] {
            reached[child as usize] = true;
            stack.push((child, false));
        }
    }

    let mut new_id = vec![NO_NODE; n];
    let mut next = 0u32;
    for (i, &alive) in reached.iter().enumerate().take(n_leaves) {
        if alive {
            new_id[i] = next;
            next += 1;
        }
    }
    let n_kept_leaves = next as usize;
    // Counting sort by height; index order is kept within each height.
    let mut count = vec![0u32; n + 1];
    for i in n_leaves..n {
        if reached[i] {
            count[height[i] as usize + 1] += 1;
        }
    }
    for h in 0..n {
        count[h + 1] += count[h];
    }
    for i in n_leaves..n {
        if reached[i] {
            let h = height[i] as usize;
            new_id[i] = next + count[h];
            count[h] += 1;
        }
    }
    next += count[n];

    let n_new = next as usize;
    let mut new_parent = vec![NO_NODE; n_new];
    let mut new_branch = vec![0.0f64; n_new];
    for old in 0..n {
        let here = new_id[old];
        if here == NO_NODE {
            continue;
        }
        new_parent[here as usize] = match parent[old] {
            NO_NODE => NO_NODE,
            par => new_id[par as usize],
        };
        new_branch[here as usize] = branch[old];
    }
    let tree = Tree::from_level_ordered(new_parent, new_branch, n_kept_leaves)?;
    Ok((tree, new_id))
}

/////////////
// Pruning //
/////////////

/// A detached subtree and the tree left behind.
///
/// The remaining tree is a real arena with its own leaf numbering, because
/// [`place`] needs a [`Tree`]. The original-space arrays are kept alongside it
/// for assembling the regraft. Leaf data is not copied; [`LazyRows`] reads the
/// original tree's rows.
struct Pruned {
    /// The remaining tree, leaves renumbered into `0..k`.
    tree: Tree,
    /// Original node index of each node of `tree`.
    to_old: Vec<u32>,
    /// Detached parent array in the original index space: [`NO_NODE`] at the
    /// pruned node and at any node the suppression removed.
    parent: Vec<u32>,
    /// Branch lengths in the same space, carrying the suppression's sums.
    branch: Vec<f64>,
    /// Root of the remaining tree in the original index space, which the
    /// suppression of a degree-two root moves.
    root: u32,
}

/// Whether a node may be pruned.
///
/// ### Params
///
/// * `tree` - The tree
/// * `x` - Node that would be detached, together with everything below it
///
/// ### Returns
///
/// False for the root, which the arena needs, and for a child of a root that
/// has only two children, which would leave the root with one child and no
/// suppression that could fix it. True otherwise, leaves included.
fn can_prune(tree: &Tree, x: u32) -> bool {
    match tree.parent(x) {
        None => false,
        Some(par) => tree.parent(par).is_some() || tree.children(par).len() > 2,
    }
}

/// Detach `x` and everything below it, suppressing the degree-two node that
/// leaves behind.
///
/// ### Params
///
/// * `tree` - The tree; not modified
/// * `x` - Node to detach
///
/// ### Returns
///
/// The remaining tree and the maps back, `None` if `x` may not be pruned, or
/// the error the arena failed with.
fn prune_subtree(tree: &Tree, x: u32) -> Result<Option<Pruned>, BonsaiErrors> {
    let Some((parent, branch, root)) = cut(tree, x) else {
        return Ok(None);
    };
    let (remaining, to_new) = assemble(&parent, &branch, root, tree.n_leaves())?;
    let mut to_old = vec![NO_NODE; remaining.n_nodes()];
    for (old, &new) in to_new.iter().enumerate() {
        if new != NO_NODE {
            to_old[new as usize] = old as u32;
        }
    }
    Ok(Some(Pruned {
        tree: remaining,
        to_old,
        parent,
        branch,
        root,
    }))
}

/// [`prune_subtree`]'s arrays without the arena: the parent and branch arrays
/// of the remaining tree in the original index space, and its root. All
/// [`regraft`] reads.
///
/// ### Params
///
/// * `tree` - The tree; not modified
/// * `x` - Node to detach
///
/// ### Returns
///
/// The arrays and the root, `None` if `x` may not be pruned.
fn cut(tree: &Tree, x: u32) -> Option<(Vec<u32>, Vec<f64>, u32)> {
    let par = tree.parent(x).filter(|_| can_prune(tree, x))?;
    let n = tree.n_nodes();
    let mut parent: Vec<u32> = (0..n)
        .map(|i| tree.parent(i as u32).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    let mut root = tree.root();

    parent[x as usize] = NO_NODE;
    let kept: Vec<u32> = tree
        .children(par)
        .iter()
        .copied()
        .filter(|&c| c != x)
        .collect();
    match (tree.parent(par), kept.as_slice()) {
        // Degree-two internal node: join its parent and surviving child.
        (Some(above), &[s]) => {
            parent[s as usize] = above;
            branch[s as usize] += branch[par as usize];
            parent[par as usize] = NO_NODE;
        }
        // Degree-two root: one child becomes the root; a leaf cannot, and when
        // both are leaves the root is the tree's only edge and stays.
        (None, &[a, b]) => {
            if let Some(r) = [a, b].into_iter().find(|&c| c as usize >= tree.n_leaves()) {
                let other = if r == a { b } else { a };
                parent[other as usize] = r;
                branch[other as usize] = branch[a as usize] + branch[b as usize];
                parent[r as usize] = NO_NODE;
                parent[par as usize] = NO_NODE;
                root = r;
            }
        }
        _ => {}
    }
    Some((parent, branch, root))
}

//////////////////
// Settled rows //
//////////////////

/// The current tree's settled down rows, stored by slot rather than by node.
///
/// A node points at a slot, so [`RowStore::accept`] repoints the nodes whose
/// subtree a move left alone and writes only the changed rows, into freed slots,
/// instead of copying `O(n p)` per acceptance.
#[derive(Clone)]
pub(crate) struct RowStore<T> {
    /// Number of features.
    p: usize,
    /// Per node of the current tree, its slot.
    slot: Vec<u32>,
    /// Down means, `[slot][feature]`, row-major.
    m: Vec<T>,
    /// Down precisions, same layout.
    w: Vec<T>,
    /// Loglikelihood contribution of each slot's node, zero for a leaf.
    contrib: Vec<f64>,
}

impl<T: BonsaiFloat> RowStore<T> {
    /// Copy a settled state into slots, one per node in node order.
    ///
    /// ### Params
    ///
    /// * `state` - Down rows settled against the tree
    /// * `n_nodes` - Node count of that tree
    ///
    /// ### Returns
    ///
    /// The store.
    pub(crate) fn from_state(state: &NodeState<T>, n_nodes: usize) -> Self {
        let p = state.n_features();
        let mut m = Vec::with_capacity(n_nodes * p);
        let mut w = Vec::with_capacity(n_nodes * p);
        for v in 0..n_nodes as u32 {
            m.extend_from_slice(state.means(v));
            w.extend_from_slice(state.precisions(v));
        }
        Self {
            p,
            slot: (0..n_nodes as u32).collect(),
            m,
            w,
            contrib: (0..n_nodes as u32).map(|v| state.contribution(v)).collect(),
        }
    }

    /// Number of features.
    ///
    /// ### Returns
    ///
    /// The feature count.
    #[inline]
    pub(crate) fn n_features(&self) -> usize {
        self.p
    }

    /// Down means of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the current tree
    ///
    /// ### Returns
    ///
    /// Its row.
    #[inline]
    pub(crate) fn means(&self, node: u32) -> &[T] {
        let lo = self.slot[node as usize] as usize * self.p;
        &self.m[lo..lo + self.p]
    }

    /// Down precisions of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the current tree
    ///
    /// ### Returns
    ///
    /// Its row.
    #[inline]
    pub(crate) fn precisions(&self, node: u32) -> &[T] {
        let lo = self.slot[node as usize] as usize * self.p;
        &self.w[lo..lo + self.p]
    }

    /// Loglikelihood contribution of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the current tree
    ///
    /// ### Returns
    ///
    /// The term, zero for a leaf.
    #[inline]
    pub(crate) fn contribution(&self, node: u32) -> f64 {
        self.contrib[self.slot[node as usize] as usize]
    }

    /// The tree loglikelihood as a [`fixed`] total.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree the store describes
    ///
    /// ### Returns
    ///
    /// The sum of every internal node's term on the fixed-point grid.
    pub(crate) fn score(&self, tree: &Tree) -> i128 {
        (tree.n_leaves()..tree.n_nodes())
            .map(|v| fixed(self.contribution(v as u32)))
            .sum()
    }

    /// Write one node's row, in its own slot or a new one.
    ///
    /// ### Params
    ///
    /// * `node` - Node id, possibly past every id the store has seen
    /// * `m` - Its down means
    /// * `w` - Its down precisions
    /// * `contrib` - Its loglikelihood term
    pub(crate) fn write_row(&mut self, node: u32, m: &[T], w: &[T], contrib: f64) {
        let p = self.p;
        if node as usize >= self.slot.len() {
            self.slot.resize(node as usize + 1, NO_NODE);
        }
        let s = match self.slot[node as usize] {
            NO_NODE => {
                let s = self.contrib.len();
                self.m.resize((s + 1) * p, T::zero());
                self.w.resize((s + 1) * p, T::zero());
                self.contrib.push(0.0);
                self.slot[node as usize] = s as u32;
                s
            }
            s => s as usize,
        };
        self.m[s * p..(s + 1) * p].copy_from_slice(m);
        self.w[s * p..(s + 1) * p].copy_from_slice(w);
        self.contrib[s] = contrib;
    }

    /// Key the store by another numbering of the same nodes.
    ///
    /// ### Params
    ///
    /// * `id_of` - Per new key, the node it names under the current keys
    ///
    /// ### Returns
    ///
    /// The slot map before, for [`RowStore::restore`].
    pub(crate) fn remap(&mut self, id_of: &[u32]) -> Vec<u32> {
        let old = std::mem::take(&mut self.slot);
        self.slot = id_of.iter().map(|&v| old[v as usize]).collect();
        old
    }

    /// Undo a [`RowStore::remap`].
    ///
    /// ### Params
    ///
    /// * `slot` - What the remap returned
    pub(crate) fn restore(&mut self, slot: Vec<u32>) {
        self.slot = slot;
    }

    /// Make the store describe the tree an accepted move produced.
    ///
    /// Row for row what [`NodeState::prune`] would produce on `tree`: every row
    /// is a current row or was formed by [`LazyRows`] through the same kernels.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree the move produced
    /// * `to_old` - Per node of `tree`, its node in `was`, or [`NO_NODE`]
    /// * `was` - The tree the store currently describes
    ///
    /// ### Returns
    ///
    /// `Ok` once the store describes `tree`, or the error [`LazyRows::new`]
    /// failed with.
    pub(crate) fn accept(
        &mut self,
        tree: &Tree,
        to_old: &[u32],
        was: &Tree,
    ) -> Result<(), BonsaiErrors> {
        let fresh = LazyRows::new(tree, to_old, was, self)?.into_fresh();
        self.accept_fresh(fresh);
        Ok(())
    }

    /// [`RowStore::accept`] with the rows already formed, by the
    /// [`LazyRows`] that scored the move.
    ///
    /// ### Params
    ///
    /// * `fresh` - [`LazyRows::into_fresh`] of the rows of the tree the move
    ///   produced, formed against this store
    pub(crate) fn accept_fresh(&mut self, fresh: Fresh<T>) {
        let Fresh {
            inherited,
            slot: fresh_slot,
            fresh_m,
            fresh_w,
            fresh_contrib,
        } = fresh;
        let p = self.p;

        let mut used = vec![false; self.contrib.len()];
        let mut slot: Vec<u32> = inherited
            .iter()
            .map(|&old| {
                if old == NO_NODE {
                    return NO_NODE;
                }
                let s = self.slot[old as usize];
                used[s as usize] = true;
                s
            })
            .collect();
        let mut free = (0..used.len() as u32).filter(|&s| !used[s as usize]);
        for (v, &f) in fresh_slot.iter().enumerate() {
            if f == NO_NODE {
                continue;
            }
            let s = match free.next() {
                Some(s) => s as usize,
                None => {
                    self.m.resize(self.m.len() + p, T::zero());
                    self.w.resize(self.w.len() + p, T::zero());
                    self.contrib.push(0.0);
                    self.contrib.len() - 1
                }
            };
            let f = f as usize;
            self.m[s * p..(s + 1) * p].copy_from_slice(&fresh_m[f * p..(f + 1) * p]);
            self.w[s * p..(s + 1) * p].copy_from_slice(&fresh_w[f * p..(f + 1) * p]);
            self.contrib[s] = fresh_contrib[f];
            slot[v] = s as u32;
        }
        self.slot = slot;
    }
}

/// The rows of the tree a move produced, as [`RowStore::accept_fresh`] takes
/// them: per node, either the current node whose row it keeps or a row formed
/// for it.
pub(crate) struct Fresh<T> {
    /// Per node, the current node whose row it keeps, or [`NO_NODE`].
    inherited: Vec<u32>,
    /// Per node, its row in the formed rows, or [`NO_NODE`].
    slot: Vec<u32>,
    /// Formed means, `[slot][feature]`, row-major.
    fresh_m: Vec<T>,
    /// Formed precisions, same layout.
    fresh_w: Vec<T>,
    /// Loglikelihood term of each formed row, `[slot]`.
    fresh_contrib: Vec<f64>,
}

/// One node's settled row, means and precisions.
///
/// Boxed so a borrow stays valid while later rows are formed.
pub(crate) type Row<T> = (Box<[T]>, Box<[T]>);

/// A tree's settled rows, computed only where they are read.
///
/// The beam search scores a few dozen nodes and the resolution reads a handful
/// of rows, so rows are computed on demand and remembered.
pub(crate) struct LazyRows<'a, T> {
    /// The tree the rows describe.
    tree: &'a Tree,
    /// Number of features.
    p: usize,
    /// Down rows of the tree the move started from.
    base: &'a RowStore<T>,
    /// Per node, the node of that tree whose down row it still has, or
    /// [`NO_NODE`] where the move changed it.
    inherited: Vec<u32>,
    /// Per node, its row in `fresh_m` and `fresh_w`, or [`NO_NODE`].
    slot: Vec<u32>,
    /// Recomputed down means, `[slot][feature]`, row-major.
    fresh_m: Vec<T>,
    /// Recomputed down precisions, same layout.
    fresh_w: Vec<T>,
    /// Loglikelihood contribution of each recomputed row, `[slot]`.
    fresh_contrib: Vec<f64>,
    /// Up rows, filled a chain at a time; cells allocated on first use.
    up: OnceLock<Vec<OnceLock<Row<T>>>>,
    /// Effective leaves, filled as the beam search asks for them; allocated
    /// like `up`.
    eff: OnceLock<Vec<OnceLock<Row<T>>>>,
}

impl<'a, T: BonsaiFloat> LazyRows<'a, T> {
    /// Mark what the move changed and recompute exactly that.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree the rows are to describe
    /// * `to_old` - Per node of `tree`, its index in `was`, or [`NO_NODE`] for
    ///   a node the move created
    /// * `was` - The tree the move started from
    /// * `base` - Down rows settled against `was`
    ///
    /// ### Returns
    ///
    /// The rows, ready to be read, or `MalformedTree` if `to_old` sends a leaf
    /// anywhere but to a leaf.
    pub(crate) fn new(
        tree: &'a Tree,
        to_old: &[u32],
        was: &Tree,
        base: &'a RowStore<T>,
    ) -> Result<Self, BonsaiErrors> {
        let p = base.n_features();
        let n = tree.n_nodes();

        // A node keeps its row when its children, order and branches are
        // unchanged and each child kept its row; one ascending pass suffices.
        let mut inherited = vec![NO_NODE; n];
        for v in 0..n {
            let old = to_old[v];
            if old == NO_NODE || old as usize >= was.n_nodes() {
                continue;
            }
            if v < tree.n_leaves() {
                // A leaf's row is its observation, always inherited.
                if old as usize >= was.n_leaves() {
                    return Err(BonsaiErrors::MalformedTree {
                        reason: format!("leaf {v} maps to node {old}, which is not a leaf"),
                    });
                }
                inherited[v] = old;
                continue;
            }
            let kids = tree.children(v as u32);
            let before = was.children(old);
            let same = kids.len() == before.len()
                && kids
                    .iter()
                    .zip(before)
                    .all(|(&c, &b)| inherited[c as usize] == b && tree.branch(c) == was.branch(b));
            if same {
                inherited[v] = old;
            }
        }

        let dirty: Vec<u32> = (tree.n_leaves()..n)
            .filter(|&v| inherited[v] == NO_NODE)
            .map(|v| v as u32)
            .collect();
        let mut slot = vec![NO_NODE; n];
        for (i, &v) in dirty.iter().enumerate() {
            slot[v as usize] = i as u32;
        }

        let mut rows = Self {
            tree,
            p,
            base,
            inherited,
            slot,
            fresh_m: vec![T::zero(); dirty.len() * p],
            fresh_w: vec![T::zero(); dirty.len() * p],
            fresh_contrib: vec![0.0; dirty.len()],
            up: OnceLock::new(),
            eff: OnceLock::new(),
        };

        // Ascending index is a post-order: dirty children are done first.
        let mut scratch: Vec<f64> = Vec::new();
        for &v in &dirty {
            let here = rows.slot[v as usize] as usize * p;
            let kids = rows.tree.children(v);
            let children: Vec<(&[T], &[T], f64)> = kids
                .iter()
                .map(|&c| {
                    let (m, w) = read_down(&rows, c);
                    (m, w, rows.tree.branch(c))
                })
                .collect();
            // Children may borrow the slab the answer goes into.
            let mut m_out = vec![T::zero(); p];
            let mut w_out = vec![T::zero(); p];
            let contrib = if children.len() == 2 {
                prune_binary(
                    children[0].0,
                    children[0].1,
                    children[0].2,
                    children[1].0,
                    children[1].1,
                    children[1].2,
                    &mut m_out,
                    &mut w_out,
                )
            } else {
                if scratch.len() < p * children.len() {
                    scratch.resize(p * children.len(), 0.0);
                }
                prune_general(
                    &children,
                    &mut m_out,
                    &mut w_out,
                    &mut scratch[..p * children.len()],
                )
            };
            drop(children);
            rows.fresh_m[here..here + p].copy_from_slice(&m_out);
            rows.fresh_w[here..here + p].copy_from_slice(&w_out);
            rows.fresh_contrib[here / p] = contrib;
        }
        Ok(rows)
    }

    /// The tree loglikelihood as a [`fixed`] total, from the rows.
    ///
    /// The terms [`NodeState::prune`] sums, summed in fixed point so the total
    /// is independent of arena order; [`crate::search::masked::score_move`]
    /// relies on that. Agrees with `prune`'s `f64` sum to rounding, not the bit.
    ///
    /// ### Returns
    ///
    /// The fixed-point total, up to the dropped additive constants.
    pub(crate) fn score(&self) -> i128 {
        let mut total = 0i128;
        for v in self.tree.n_leaves()..self.tree.n_nodes() {
            let old = self.inherited[v];
            total += fixed(if old != NO_NODE {
                self.base.contribution(old)
            } else {
                self.fresh_contrib[self.slot[v] as usize]
            });
        }
        total
    }

    /// The down row of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree
    ///
    /// ### Returns
    ///
    /// Its effective means and precisions.
    pub(crate) fn down_row(&self, node: u32) -> (&[T], &[T]) {
        read_down(self, node)
    }

    /// The up row of one node, filling the chain above it if it is not there.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree
    ///
    /// ### Returns
    ///
    /// Everything outside the node's subtree, positioned at its parent and not
    /// diffused along the branch above it, which is the up sweep's convention.
    pub(crate) fn up_row(&self, node: u32) -> &Row<T> {
        let up = self.up_cells();
        let mut chain: Vec<u32> = Vec::new();
        let mut here = node;
        while up[here as usize].get().is_none() {
            chain.push(here);
            match self.tree.parent(here) {
                None => break,
                Some(a) => here = a,
            }
        }
        for &v in chain.iter().rev() {
            let row = self.compute_up(v);
            let _ = up[v as usize].set(row);
        }
        // Filled above; the initialiser never runs (no unwrap in this module).
        up[node as usize].get_or_init(|| self.compute_up(node))
    }

    /// Keep the rows this formed and drop the borrows.
    ///
    /// ### Returns
    ///
    /// The formed rows, for [`RowStore::accept_fresh`].
    pub(crate) fn into_fresh(self) -> Fresh<T> {
        Fresh {
            inherited: self.inherited,
            slot: self.slot,
            fresh_m: self.fresh_m,
            fresh_w: self.fresh_w,
            fresh_contrib: self.fresh_contrib,
        }
    }

    /// The up-row cells, allocated on first use.
    ///
    /// ### Returns
    ///
    /// One cell per node.
    fn up_cells(&self) -> &[OnceLock<Row<T>>] {
        self.up
            .get_or_init(|| (0..self.tree.n_nodes()).map(|_| OnceLock::new()).collect())
    }

    /// The effective-leaf cells, allocated on first use.
    ///
    /// ### Returns
    ///
    /// One cell per node.
    fn eff_cells(&self) -> &[OnceLock<Row<T>>] {
        self.eff
            .get_or_init(|| (0..self.tree.n_nodes()).map(|_| OnceLock::new()).collect())
    }

    /// One node's up row, from its parent's.
    ///
    /// [`crate::model::global::UpState::sweep`]'s per-node step for one child,
    /// with the same expressions in the same order, so the bits match.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree whose parent's up row is settled
    ///
    /// ### Returns
    ///
    /// The node's up row.
    fn compute_up(&self, node: u32) -> Row<T> {
        let Some(a) = self.tree.parent(node) else {
            return root_up(self.p);
        };
        let above = self.up_row(a);
        let kids = self.tree.children(a);
        let side = if kids.len() == 2 {
            let other = if kids[0] == node { kids[1] } else { kids[0] };
            let (m_o, w_o) = self.down_row(other);
            UpSide::Sibling(self.tree.branch(other), m_o, w_o)
        } else {
            let (m_a, w_a) = self.down_row(a);
            let (m_c, w_c) = self.down_row(node);
            UpSide::Parent(m_a, w_a, self.tree.branch(node), m_c, w_c)
        };
        up_step(
            self.tree.parent(a).is_none(),
            self.tree.branch(a),
            (&above.0, &above.1),
            side,
        )
    }

    /// The whole tree collapsed onto one node.
    ///
    /// The up row sits at the node's parent and is diffused down the branch
    /// above the node before it is combined with the down row. The root's row is
    /// its down row; that is guarded on the root, not on a zero up precision,
    /// because the root's branch entry may hold anything, a NaN included.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree
    ///
    /// ### Returns
    ///
    /// The effective leaf the beam search scores an attachment against.
    fn eff_leaf(&self, node: u32) -> EffLeaf<'_, T> {
        let rows = self.eff_cells()[node as usize].get_or_init(|| {
            let above = self.up_row(node);
            eff_step(
                self.tree.parent(node).is_none(),
                self.tree.branch(node),
                self.down_row(node),
                (&above.0, &above.1),
            )
        });
        EffLeaf {
            m: &rows.0,
            w: &rows.1,
        }
    }
}

/// The star around a node, read off rows rather than off a settled tree.
///
/// The same star [`centre_star`] builds, in the same member order, which the
/// primitive's pair scan and tie-breaking read.
///
/// ### Params
///
/// * `tree` - The tree
/// * `rows` - Its rows
/// * `centre` - Internal node to build the star around
///
/// ### Returns
///
/// The star, or `NodeOutOfRange` for an index outside the arena, or
/// `MalformedTree` if the centre is a leaf.
pub(crate) fn lazy_centre_star<T: BonsaiFloat>(
    tree: &Tree,
    rows: &LazyRows<'_, T>,
    centre: u32,
) -> Result<CentreStar<T>, BonsaiErrors> {
    if centre as usize >= tree.n_nodes() {
        return Err(BonsaiErrors::NodeOutOfRange {
            index: centre as usize,
            n_nodes: tree.n_nodes(),
        });
    }
    let kids = tree.children(centre);
    if kids.is_empty() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("node {centre} is a leaf and has no star around it"),
        });
    }

    let p = rows.p;
    let parent = tree.parent(centre);
    let n = kids.len() + usize::from(parent.is_some());
    let mut star = CentreStar {
        centre,
        member_nodes: Vec::with_capacity(n),
        has_upstream: parent.is_some(),
        deleted: Vec::new(),
        means: Vec::with_capacity(n * p),
        precisions: Vec::with_capacity(n * p),
        branch: Vec::with_capacity(n),
        n_features: p,
    };
    for &child in kids {
        let (m, w) = rows.down_row(child);
        star.member_nodes.push(child);
        star.means.extend_from_slice(m);
        star.precisions.extend_from_slice(w);
        star.branch.push(tree.branch(child));
    }
    if let Some(par) = parent {
        let above = rows.up_row(centre);
        star.member_nodes.push(par);
        star.means.extend_from_slice(&above.0);
        star.precisions.extend_from_slice(&above.1);
        star.branch.push(tree.branch(centre));
    }
    Ok(star)
}

/// The up row of a root: nothing outside it.
///
/// ### Params
///
/// * `p` - Number of features
///
/// ### Returns
///
/// Zero means and precisions.
pub(crate) fn root_up<T: BonsaiFloat>(p: usize) -> Row<T> {
    (
        vec![T::zero(); p].into_boxed_slice(),
        vec![T::zero(); p].into_boxed_slice(),
    )
}

/// What [`up_step`] reads besides the parent's own up row.
pub(crate) enum UpSide<'r, T> {
    /// The parent is binary: the other child's branch and down row.
    Sibling(f64, &'r [T], &'r [T]),
    /// Anything else: the parent's down row, then the node's branch and down
    /// row, which is peeled back off the parent's total.
    Parent(&'r [T], &'r [T], f64, &'r [T], &'r [T]),
}

/// One node's up row from its parent's.
///
/// [`crate::model::global::UpState::sweep`]'s per-node step, shared by
/// [`LazyRows`] and [`crate::search::masked`] so all give the same bits.
///
/// ### Params
///
/// * `parent_is_root` - Whether the node's parent is the root
/// * `t_a` - Branch above the parent
/// * `up_a` - The parent's up row
/// * `side` - The rest of what the step reads; see [`UpSide`]
///
/// ### Returns
///
/// The node's up row.
pub(crate) fn up_step<T: BonsaiFloat>(
    parent_is_root: bool,
    t_a: f64,
    up_a: (&[T], &[T]),
    side: UpSide<'_, T>,
) -> Row<T> {
    let (up_m, up_w) = up_a;
    let p = up_m.len();
    let is_root = parent_is_root;
    let mut m_out = vec![T::zero(); p];
    let mut w_out = vec![T::zero(); p];
    match side {
        UpSide::Sibling(t_o, m_o, w_o) => {
            for g in 0..p {
                let (w_up, m_up) = up_part(is_root, t_a, up_w[g], up_m[g]);
                let wo = wide(w_o[g]);
                let wdo = wo / (1.0 + t_o * wo);
                let mo = wide(m_o[g]);
                let tot = wdo + w_up;
                w_out[g] = narrow(tot);
                m_out[g] = narrow(mo + (m_up - mo) * (w_up / tot));
            }
        }
        UpSide::Parent(m_a, w_a, t_c, m_c, w_c) => {
            for g in 0..p {
                let (w_up, m_up) = up_part(is_root, t_a, up_w[g], up_m[g]);
                let ma = wide(m_a[g]);
                let tot = wide(w_a[g]) + w_up;
                let m_tot = ma + (m_up - ma) * (w_up / tot);
                let wc = wide(w_c[g]);
                let wdc = wc / (1.0 + t_c * wc);
                let rest = tot - wdc;
                let mc = wide(m_c[g]);
                w_out[g] = narrow(rest);
                m_out[g] = narrow(m_tot + (m_tot - mc) * (wdc / rest));
            }
        }
    }
    (m_out.into_boxed_slice(), w_out.into_boxed_slice())
}

/// The whole tree collapsed onto one node, from its down and up rows.
///
/// Shared by [`LazyRows::eff_leaf`] and the masked views; see there.
///
/// ### Params
///
/// * `is_root` - Whether the node is the root
/// * `t` - Branch above the node
/// * `down` - Its down row
/// * `up` - Its up row
///
/// ### Returns
///
/// The effective leaf's means and precisions.
pub(crate) fn eff_step<T: BonsaiFloat>(
    is_root: bool,
    t: f64,
    down: (&[T], &[T]),
    up: (&[T], &[T]),
) -> Row<T> {
    let (m_down, w_down) = down;
    let (m_up, w_up) = up;
    let p = m_down.len();
    let mut m = vec![T::zero(); p];
    let mut w = vec![T::zero(); p];
    for g in 0..p {
        let w_above = if is_root {
            0.0
        } else {
            let wu = wide(w_up[g]);
            wu / (1.0 + t * wu)
        };
        let below = wide(w_down[g]);
        let total = below + w_above;
        let md = wide(m_down[g]);
        m[g] = narrow(md + (wide(m_up[g]) - md) * (w_above / total));
        w[g] = narrow(total);
    }
    (m.into_boxed_slice(), w.into_boxed_slice())
}

/// One node's down row, from wherever it lives.
///
/// A free function so the constructor can call it while filling the slab.
///
/// ### Params
///
/// * `rows` - The rows
/// * `node` - Node of the pruned tree
///
/// ### Returns
///
/// Its effective means and precisions.
fn read_down<'r, T: BonsaiFloat>(rows: &'r LazyRows<'_, T>, node: u32) -> (&'r [T], &'r [T]) {
    let old = rows.inherited[node as usize];
    if old != NO_NODE {
        return (rows.base.means(old), rows.base.precisions(old));
    }
    let lo = rows.slot[node as usize] as usize * rows.p;
    (
        &rows.fresh_m[lo..lo + rows.p],
        &rows.fresh_w[lo..lo + rows.p],
    )
}

//////////////
// One move //
//////////////

/// A candidate move, proposed and scored but not yet decided.
struct Proposal<T> {
    /// The tree the move produces.
    tree: Tree,
    /// Per node of `tree`, the node of the tree the move started from whose
    /// leaf set it shares, or [`NO_NODE`]; leaves map to themselves.
    to_old: Vec<u32>,
    /// The subtree that was pruned, as a node of the tree the move started
    /// from.
    pruned: u32,
    /// Loglikelihood of `tree` as a [`fixed`] total, from
    /// [`LazyRows::score`]; what acceptance compares.
    score: i128,
    /// Where the subtree was attached, as a node of the tree the move started
    /// from.
    target: u32,
    /// Length of the new branch.
    branch: f64,
    /// The rows `tree` has that the tree the move started from does not.
    fresh: Fresh<T>,
}

/// A scored move the sweep has yet to decide on.
///
/// Where the views scored it, nothing was built; the tree is assembled only on
/// acceptance.
struct Candidate<T> {
    /// The subtree to prune, as a node of the tree it was scored on.
    pruned: u32,
    /// Where to attach it, same ids; read by the debug cross-check only.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    target: u32,
    /// Length of the new branch.
    branch: f64,
    /// [`fixed`] total of the tree the move produces.
    score: i128,
    /// Leaf word of the pruned subtree; with the next two, what
    /// [`SprApprox::recheck`] re-applies to a tree that has moved since.
    pruned_word: u64,
    /// Leaf word of the attachment point.
    target_word: u64,
    /// The built move, in the arena's ids, when the built path scored it.
    built: Option<Proposal<T>>,
    /// The views the move was scored on, kept for the candidate the sweep
    /// will accept if it accepts any; [`apply_move`] consumes them.
    data: Option<Box<MoveData<T>>>,
}

impl<T> Candidate<T> {
    /// A candidate from a move the built path scored.
    ///
    /// ### Params
    ///
    /// * `built` - The proposal
    /// * `word` - Leaf words of the tree it was proposed on
    ///
    /// ### Returns
    ///
    /// The candidate, carrying the proposal.
    fn from_built(built: Proposal<T>, word: &[u64]) -> Self {
        Self {
            pruned: built.pruned,
            target: built.target,
            branch: built.branch,
            score: built.score,
            pruned_word: word[built.pruned as usize],
            target_word: word[built.target as usize],
            built: Some(built),
            data: None,
        }
    }
}

/// Hang a detached subtree back onto the remaining tree below `target`.
///
/// Attaching below an internal node adds a child. Below a leaf, a new node
/// takes the leaf's place and the leaf hangs off it on a zero-length branch.
///
/// ### Params
///
/// * `pruned` - What [`prune_subtree`] left
/// * `x` - The detached node, in the original index space
/// * `target` - Node to attach below, in the original index space
/// * `branch` - Length of the new branch
/// * `n_leaves` - Leaf count of the original tree
///
/// ### Returns
///
/// The reassembled tree, the node the attachment centred on, and the original
/// index of each of its nodes ([`NO_NODE`] for the node an attachment below a
/// leaf creates), or the error the arena failed with.
fn regraft(
    pruned: &Pruned,
    x: u32,
    target: u32,
    branch: f64,
    n_leaves: usize,
) -> Result<(Tree, u32, Vec<u32>), BonsaiErrors> {
    regraft_cut(
        (&pruned.parent, &pruned.branch, pruned.root),
        x,
        target,
        branch,
        n_leaves,
    )
}

/// [`regraft`] onto [`cut`]'s arrays.
///
/// ### Params
///
/// * `cut` - Parent array, branch array and root of the remaining tree
/// * `x` - The detached node
/// * `target` - Node to attach below
/// * `branch` - Length of the new branch
/// * `n_leaves` - Leaf count of the original tree
///
/// ### Returns
///
/// As [`regraft`].
fn regraft_cut(
    cut: (&[u32], &[f64], u32),
    x: u32,
    target: u32,
    branch: f64,
    n_leaves: usize,
) -> Result<(Tree, u32, Vec<u32>), BonsaiErrors> {
    let (parent, lengths, root) = cut;
    let mut par = parent.to_vec();
    let mut len = lengths.to_vec();
    // The root is never a leaf, so this arm never moves it.
    let centre = if (target as usize) < n_leaves {
        let fresh = par.len() as u32;
        par.push(par[target as usize]);
        len.push(len[target as usize]);
        par[target as usize] = fresh;
        len[target as usize] = 0.0;
        par[x as usize] = fresh;
        len[x as usize] = branch;
        fresh
    } else {
        par[x as usize] = target;
        len[x as usize] = branch;
        target
    };
    let (tree, map) = assemble(&par, &len, root, n_leaves)?;
    let mut to_old = vec![NO_NODE; tree.n_nodes()];
    for (old, &new) in map.iter().enumerate() {
        if new != NO_NODE {
            to_old[new as usize] = old as u32;
        }
    }
    let centre = map[centre as usize];
    Ok((tree, centre, to_old))
}

/// Index a tree's [`leaf_words`] by word.
///
/// ### Params
///
/// * `word` - One word per node
///
/// ### Returns
///
/// Word to node.
fn word_index(word: &[u64]) -> FxHashMap<u64, u32> {
    word.iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u32))
        .collect()
}

/// Propose the move that prunes `x` and regrafts it wherever the beam search
/// likes best, and score it.
///
/// A proposal whose splits match the current tree's is dropped (see the module
/// docs). The rest are scored by [`LazyRows::score`], with candidate nodes
/// matched to the current tree's by [`leaf_words`]; [`LazyRows`] checks children
/// and branches before trusting a match.
///
/// The attachment is followed by the polytomy resolution of SPEC.md section 7.3
/// at the attachment point only, the one node whose degree the move raised; its
/// rows come from [`lazy_centre_star`].
///
/// ### Params
///
/// * `tree` - The tree; not modified
/// * `down` - Down rows, settled against `tree`
/// * `x` - Node to prune
/// * `params` - Knobs
/// * `here` - Split fingerprint of `tree`
/// * `by_word` - [`word_index`] of `tree`
/// * `placed` - The attachment node, in `tree`'s ids, and branch, when
///   they are already known; the pruned tree is then never built
///
/// ### Returns
///
/// The scored candidate, `None` if `x` may not be pruned or the move changes
/// no split, or the error the placement, the primitive or the arena failed
/// with.
fn propose<T: BonsaiFloat>(
    tree: &Tree,
    down: &RowStore<T>,
    x: u32,
    params: &SprParams,
    here: u64,
    by_word: &FxHashMap<u64, u32>,
    placed: Option<(u32, f64)>,
) -> Result<Option<Proposal<T>>, BonsaiErrors> {
    let (arrays, target, placed_branch) = match placed {
        Some((target, branch)) => {
            let Some(arrays) = cut(tree, x) else {
                return Ok(None);
            };
            (arrays, target, branch)
        }
        None => {
            let Some(pruned) = prune_subtree(tree, x)? else {
                return Ok(None);
            };
            let rows = LazyRows::new(&pruned.tree, &pruned.to_old, tree, down)?;
            let q = EffLeaf {
                m: down.means(x),
                w: down.precisions(x),
            };
            let best = place(
                &pruned.tree,
                q,
                |node: u32| rows.eff_leaf(node),
                Some(params.placement),
            )?;
            let target = pruned.to_old[best.node as usize];
            drop(rows);
            (
                (pruned.parent, pruned.branch, pruned.root),
                target,
                best.branch,
            )
        }
    };
    let (attached, centre, to_old) = regraft_cut(
        (&arrays.0, &arrays.1, arrays.2),
        x,
        target,
        placed_branch,
        tree.n_leaves(),
    )?;

    if centre == NO_NODE || attached.n_leaves() != tree.n_leaves() {
        return Ok(None);
    }

    let members = attached.children(centre).len() + usize::from(attached.parent(centre).is_some());
    let attached_word = leaf_words(&attached);
    let attached_below = leaves_below(&attached);
    let attached_print = split_fingerprint_counted(&attached, &attached_word, &attached_below);
    let (candidate, word) = if members <= crate::search::polytomy::RESOLVED_STAR_MEMBERS {
        if attached_print == here {
            return Ok(None);
        }
        (attached, attached_word)
    } else {
        let attached_rows = LazyRows::new(&attached, &to_old, tree, down)?;
        let star = lazy_centre_star(&attached, &attached_rows, centre)?;
        let result = resolve_star(star.view(), Some(params.star))?;
        let n_leaves = attached.n_leaves();
        let total = attached_word[attached.root() as usize];
        let last = star.member_nodes.len() - 1;
        let (member_word, member_count): (Vec<u64>, Vec<usize>) = star
            .member_nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| {
                if star.has_upstream && i == last {
                    (
                        total.wrapping_sub(attached_word[centre as usize]),
                        n_leaves - attached_below[centre as usize],
                    )
                } else {
                    (attached_word[node as usize], attached_below[node as usize])
                }
            })
            .unzip();
        let print = attached_print.wrapping_add(resolution_splits(
            &member_word,
            &member_count,
            n_leaves,
            total,
            &result,
        ));
        #[cfg(debug_assertions)]
        {
            let full = splice_result(&attached, &star, &result)?;
            debug_assert_eq!(
                split_fingerprint_with(&full, &leaf_words(&full)),
                print,
                "the resolution's splits did not predict the spliced tree's fingerprint"
            );
        }
        if print == here {
            return Ok(None);
        }
        let candidate = splice_assembled(&attached, &star, &result)?;
        let word = leaf_words(&candidate);
        (candidate, word)
    };

    let n_leaves = tree.n_leaves();
    let to_old: Vec<u32> = (0..candidate.n_nodes())
        .map(|v| {
            if v < n_leaves {
                v as u32
            } else {
                by_word
                    .get(&word[v])
                    .copied()
                    .filter(|&old| old as usize >= n_leaves)
                    .unwrap_or(NO_NODE)
            }
        })
        .collect();
    let rows = LazyRows::new(&candidate, &to_old, tree, down)?;
    let score = rows.score();
    let fresh = rows.into_fresh();
    Ok(Some(Proposal {
        fresh,
        tree: candidate,
        to_old,
        pruned: x,
        score,
        target,
        branch: placed_branch,
    }))
}

/// Splice a resolved star into a tree and number the result by [`assemble`].
///
/// Unlike [`crate::search::polytomy::splice_result`], [`assemble`] keeps each
/// level's existing order with the made ancestors last, so a move reorders only
/// nodes whose height changed and the arena can be updated locally.
///
/// ### Params
///
/// * `tree` - The tree the star was built on
/// * `star` - The star
/// * `result` - Its resolution
///
/// ### Returns
///
/// The spliced tree, or the error the arena failed with.
fn splice_assembled<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    result: &StarResult<T>,
) -> Result<Tree, BonsaiErrors> {
    let n = tree.n_nodes();
    let n_made = result.parent.len() - star.member_nodes.len();
    let mut parent: Vec<u32> = (0..n)
        .map(|v| tree.parent(v as u32).unwrap_or(NO_NODE))
        .chain(std::iter::repeat_n(NO_NODE, n_made))
        .collect();
    let mut branch = tree.branches().to_vec();
    branch.resize(n + n_made, 0.0);
    debug_assert!(star.deleted.is_empty(), "an SPR star deletes nothing");
    for (v, up, t) in splice_edits(star, result, n as u32) {
        parent[v as usize] = up;
        branch[v as usize] = t;
    }
    Ok(assemble(&parent, &branch, tree.root(), tree.n_leaves())?.0)
}

/// Fingerprint terms of the splits a star resolution adds.
///
/// Resolving the star at `centre` keeps every edge of `attached` and adds one
/// per created ancestor, so the spliced tree's fingerprint is the attached
/// tree's plus this, bit for bit, without splicing. The upstream member stands
/// for everything outside the centre's subtree.
///
/// ### Params
///
/// * `member_word` - Leaf word of each star member, upstream last
/// * `member_count` - Leaf count of each, the same
/// * `n_leaves` - Leaves in the tree
/// * `total` - Leaf word of every leaf
/// * `result` - What the primitive built from the star
///
/// ### Returns
///
/// The wrapping sum of [`split_hash`] over the added non-trivial splits.
fn resolution_splits<T: BonsaiFloat>(
    member_word: &[u64],
    member_count: &[usize],
    n_leaves: usize,
    total: u64,
    result: &StarResult<T>,
) -> u64 {
    let n_members = result.n_members;
    let n_local = n_members + result.merges.len();
    let mut group = vec![0u64; n_local];
    let mut count = vec![0usize; n_local];
    group[..n_members].copy_from_slice(member_word);
    count[..n_members].copy_from_slice(member_count);
    let mut print = 0u64;
    for (j, merge) in result.merges.iter().enumerate() {
        let a = n_members + j;
        debug_assert_eq!(merge.ancestor as usize, a);
        let (l, r) = (merge.left as usize, merge.right as usize);
        group[a] = group[l].wrapping_add(group[r]);
        count[a] = count[l] + count[r];
        if count[a] >= 2 && n_leaves - count[a] >= 2 {
            print = print.wrapping_add(split_hash(group[a], total));
        }
    }
    print
}

/// What the views read off the current tree besides its rows.
struct Prints<'a> {
    /// Leaf words of the current tree.
    word: &'a [u64],
    /// Leaf counts of the current tree.
    below: &'a [usize],
    /// Its split fingerprint.
    here: u64,
    /// Its [`fixed`] loglikelihood total.
    score: i128,
}

/// Score pruning `x` and regrafting it without building either tree.
///
/// The pruned and regrafted trees are views of the current one
/// ([`crate::search::masked`]); moves that change a split are scored by
/// [`score_move`] and applied only if the sweep accepts them.
///
/// ### Params
///
/// * `tree` - The current tree
/// * `down` - Its rows
/// * `x` - Node to prune
/// * `params` - Knobs
/// * `prints` - The current tree's words, counts, fingerprint and total
/// * `placed` - The attachment node and branch when they are already known,
///   which [`SprApprox::recheck`] carries over from an earlier tree; `None`
///   runs the beam search
/// * `keep` - Given the move's total, whether to keep its views
/// * `cache` - Up-row cells for the pruned view, returned emptied
/// * `mirror` - The tree as an arena, to check the views against the built
///   path; debug builds only
///
/// ### Returns
///
/// See [`Fast`]; or the error the placement or the primitive failed with.
#[allow(clippy::too_many_arguments)]
fn view_move<T: BonsaiFloat>(
    tree: &LiveTree,
    down: &RowStore<T>,
    x: u32,
    params: &SprParams,
    prints: &Prints<'_>,
    placed: Option<(u32, f64)>,
    keep: &dyn Fn(i128) -> bool,
    cache: &mut ViewCache<T>,
    mirror: Option<&Mirror<T>>,
) -> Result<Fast<T>, BonsaiErrors> {
    let Some(view) = PrunedView::new(tree, x) else {
        return Ok(Fast::Declined);
    };
    let q = EffLeaf {
        m: down.means(x),
        w: down.precisions(x),
    };
    let fast = (|| -> Result<Fast<T>, BonsaiErrors> {
        let rows = PrunedRows::new(&view, down, cache);
        let (target, branch) = match placed {
            Some(placed) => placed,
            None => {
                let best = place_walk(&view, q, |v| rows.eff_leaf(v), Some(params.placement))?;
                (best.node, best.branch)
            }
        };
        let attached = match attach(
            &rows,
            target,
            branch,
            prints.word,
            prints.below,
            prints.here,
            mirror.is_none(),
        ) {
            Regraft::Attached(attached) => *attached,
            Regraft::Unchanged => return Ok(Fast::NoMove),
            // A carried-over target the cut suppresses has nowhere to go, as
            // on the built path.
            Regraft::Declined => {
                return Ok(if placed.is_some() && tree.parent(x) == Some(target) {
                    Fast::NoMove
                } else {
                    Fast::Declined
                });
            }
        };
        if let (Some(mirror), None) = (mirror, placed) {
            check_against_built(mirror, x, params, (target, branch), &attached)?;
        }
        let small =
            attached.star.member_nodes.len() <= crate::search::polytomy::RESOLVED_STAR_MEMBERS;
        let result = if small {
            None
        } else {
            Some(resolve_star(attached.star.view(), Some(params.star))?)
        };
        let print = match &result {
            None => attached.print,
            Some(result) => attached.print.wrapping_add(resolution_splits(
                &attached.member_word,
                &attached.member_count,
                tree.n_leaves(),
                prints.word[tree.root() as usize],
                result,
            )),
        };
        if print == prints.here {
            return Ok(Fast::NoMove);
        }
        let (score, data) = score_move(rows, attached, result.as_ref(), prints.score, print);
        Ok(Fast::Move(
            target,
            branch,
            score,
            keep(score).then(|| Box::new(data)),
        ))
    })();
    cache.reset();
    fast
}

/// What the views decided about one proposal.
enum Fast<T> {
    /// The proposal changes no split, or does not apply to this tree.
    NoMove,
    /// It does: the attachment node in the current tree's ids, its branch,
    /// the [`fixed`] total of the tree the move produces, the same bits the
    /// built path finds, and the views it was scored on if they were kept.
    Move(u32, f64, i128, Option<Box<MoveData<T>>>),
    /// The views declined; the built path decides.
    Declined,
}

/// The live tree numbered as an arena, with its rows and word index keyed
/// the same way: what the built path runs on.
///
/// Debug builds keep one per state of the sweep and check the views against
/// it; release builds never make one.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
struct Mirror<T> {
    /// The arena tree.
    tree: Tree,
    /// Arena index of each live id, [`NO_NODE`] at a free id.
    arena_of: Vec<u32>,
    /// Rows keyed by arena index.
    down: RowStore<T>,
    /// Word index keyed by arena index.
    by_word: FxHashMap<u64, u32>,
    /// Split fingerprint.
    here: u64,
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
impl<T: BonsaiFloat> Mirror<T> {
    /// Number the live tree as an arena and key a copy of its rows to match.
    ///
    /// ### Params
    ///
    /// * `tree` - The live tree
    /// * `down` - Its rows
    /// * `word` - Its leaf words
    /// * `below` - Its leaf counts
    /// * `here` - Its split fingerprint
    ///
    /// ### Returns
    ///
    /// The mirror; panics if the live tree's words, counts or fingerprint are
    /// not the arena's, which is what it is for.
    fn new(
        tree: &LiveTree,
        down: &RowStore<T>,
        word: &[u64],
        below: &[usize],
        here: u64,
    ) -> Result<Self, BonsaiErrors> {
        let (arena, id_of, arena_of) = tree.to_tree()?;
        let mut rows = down.clone();
        rows.remap(&id_of);
        let word_a: Vec<u64> = id_of.iter().map(|&v| word[v as usize]).collect();
        let below_a: Vec<usize> = id_of.iter().map(|&v| below[v as usize]).collect();
        assert_eq!(word_a, leaf_words(&arena), "live words");
        assert_eq!(below_a, leaves_below(&arena), "live counts");
        assert_eq!(
            here,
            split_fingerprint_counted(&arena, &word_a, &below_a),
            "live fingerprint"
        );
        Ok(Self {
            by_word: word_index(&word_a),
            tree: arena,
            arena_of,
            down: rows,
            here,
        })
    }
}

/// Debug builds: the views' placement, star and fingerprint against the built
/// path's, bit for bit.
///
/// ### Params
///
/// * `mirror` - The current tree as an arena
/// * `x` - The pruned node, a live id
/// * `params` - Knobs
/// * `best` - The attachment node, a live id, and branch the view found
/// * `attached` - The star and fingerprint the view built
///
/// ### Returns
///
/// `Ok` if they agree; panics otherwise, which is the point.
fn check_against_built<T: BonsaiFloat>(
    mirror: &Mirror<T>,
    x: u32,
    params: &SprParams,
    best: (u32, f64),
    attached: &Attached<'_, T>,
) -> Result<(), BonsaiErrors> {
    let (tree, down) = (&mirror.tree, &mirror.down);
    let x = mirror.arena_of[x as usize];
    let pruned = prune_subtree(tree, x)?.expect("the view accepted the cut");
    let rows = LazyRows::new(&pruned.tree, &pruned.to_old, tree, down)?;
    let q = EffLeaf {
        m: down.means(x),
        w: down.precisions(x),
    };
    let built = place(
        &pruned.tree,
        q,
        |node: u32| rows.eff_leaf(node),
        Some(params.placement),
    )?;
    assert_eq!(
        pruned.to_old[built.node as usize], mirror.arena_of[best.0 as usize],
        "placement node"
    );
    assert_eq!(built.branch.to_bits(), best.1.to_bits(), "placement branch");

    let target = pruned.to_old[built.node as usize];
    let (tree_a, centre, to_old) = regraft(&pruned, x, target, built.branch, tree.n_leaves())?;
    let word = leaf_words(&tree_a);
    let below = leaves_below(&tree_a);
    assert_eq!(
        split_fingerprint_counted(&tree_a, &word, &below),
        attached.print,
        "regrafted fingerprint"
    );
    let rows_a = LazyRows::new(&tree_a, &to_old, tree, down)?;
    let star = lazy_centre_star(&tree_a, &rows_a, centre)?;
    assert_eq!(star.branch, attached.star.branch, "star branches");
    assert_eq!(
        star.has_upstream, attached.star.has_upstream,
        "star upstream"
    );
    assert!(
        star.means
            .iter()
            .zip(&attached.star.means)
            .all(|(a, b)| a.to_f64().map(f64::to_bits) == b.to_f64().map(f64::to_bits)),
        "star means"
    );
    assert!(
        star.precisions
            .iter()
            .zip(&attached.star.precisions)
            .all(|(a, b)| a.to_f64().map(f64::to_bits) == b.to_f64().map(f64::to_bits)),
        "star precisions"
    );
    Ok(())
}

/// Debug builds: the built path's version of a move, on the mirror.
///
/// ### Params
///
/// * `mirror` - The current tree as an arena
/// * `x` - The pruned node, a live id
/// * `params` - Knobs
/// * `placed` - The attachment node, a live id, and branch, if known
///
/// ### Returns
///
/// What [`propose`] makes of it.
fn built_on_mirror<T: BonsaiFloat>(
    mirror: &Mirror<T>,
    x: u32,
    params: &SprParams,
    placed: Option<(u32, f64)>,
) -> Result<Option<Proposal<T>>, BonsaiErrors> {
    propose(
        &mirror.tree,
        &mirror.down,
        mirror.arena_of[x as usize],
        params,
        mirror.here,
        &mirror.by_word,
        placed.map(|(t, b)| (mirror.arena_of[t as usize], b)),
    )
}

/// Turn what the views decided into a candidate.
///
/// ### Params
///
/// * `x` - The pruned node
/// * `params` - Knobs
/// * `prints` - What the views read off the current tree
/// * `placed` - The carried-over attachment, if any
/// * `fast` - What the views decided
/// * `mirror` - The tree as an arena, to check against; debug builds only
///
/// ### Returns
///
/// The candidate, `Ok(None)` if the move changes nothing or does not apply,
/// or `Err(())` if the views declined and the built path has to decide.
fn candidate<T: BonsaiFloat>(
    x: u32,
    params: &SprParams,
    prints: &Prints<'_>,
    placed: Option<(u32, f64)>,
    fast: Fast<T>,
    mirror: Option<&Mirror<T>>,
) -> Result<Result<Option<Candidate<T>>, ()>, BonsaiErrors> {
    match fast {
        Fast::NoMove => {
            if let Some(mirror) = mirror {
                assert!(
                    built_on_mirror(mirror, x, params, placed)?.is_none(),
                    "the views called a move no move"
                );
            }
            Ok(Ok(None))
        }
        Fast::Move(target, branch, score, data) => {
            if let Some(mirror) = mirror {
                assert_eq!(
                    built_on_mirror(mirror, x, params, Some((target, branch)))?.map(|b| b.score),
                    Some(score),
                    "the views scored a move differently from the built path"
                );
            }
            Ok(Ok(Some(Candidate {
                pruned: x,
                target,
                branch,
                score,
                pruned_word: prints.word[x as usize],
                target_word: prints.word[target as usize],
                built: None,
                data,
            })))
        }
        Fast::Declined => Ok(Err(())),
    }
}

/// One move to score: a subtree by its leaf word, and where to put it.
struct Want {
    /// Leaf word of the subtree to prune.
    pruned_word: u64,
    /// Leaf word of the attachment point and the branch, carried over from
    /// an earlier tree by [`SprApprox::recheck`]; `None` runs the beam search.
    placed: Option<(u64, f64)>,
}

/// Score a chunk of moves against the tree as it stands.
///
/// In parallel on the views; the few the views decline (a cut that leaves the
/// root of degree two, a binary root) are then scored on the built path through
/// an arena numbering, before anything is decided.
///
/// ### Params
///
/// * `tree` - The current tree
/// * `down` - Its rows; keyed by arena index while the built path runs, and
///   restored
/// * `params` - Knobs
/// * `prints` - What the views read off the current tree
/// * `by_word` - Word index of `tree`
/// * `caches` - Spare view caches, taken and returned
/// * `wants` - The moves, `None` for a slot with nothing to score
/// * `mirror` - The tree as an arena, to check against; debug builds only
///
/// ### Returns
///
/// One candidate per slot, `None` where the move changes nothing, does not
/// apply, or a leaf word no longer names a node; or the error the placement,
/// the primitive or the arena failed with. A candidate the built path scored
/// is in the arena's ids.
#[allow(clippy::too_many_arguments)]
fn score_chunk<T: BonsaiFloat>(
    tree: &LiveTree,
    down: &mut RowStore<T>,
    params: &SprParams,
    prints: &Prints<'_>,
    by_word: &FxHashMap<u64, u32>,
    caches: &Mutex<Vec<ViewCache<T>>>,
    wants: &[Option<Want>],
    mirror: Option<&Mirror<T>>,
) -> Result<Vec<Option<Candidate<T>>>, BonsaiErrors> {
    // Only the first candidate over the floor is accepted, so only it keeps
    // its views.
    let floor = acceptance_floor(params, unfixed(prints.score));
    let first = AtomicUsize::new(usize::MAX);
    let space = tree.id_space();
    let rows: &RowStore<T> = down;
    type Scored<T> = Result<Option<Candidate<T>>, (u32, Option<(u32, f64)>)>;
    let scored: Vec<Scored<T>> = wants
        .par_iter()
        .enumerate()
        .map(|(i, want)| -> Result<Scored<T>, BonsaiErrors> {
            let Some(want) = want else {
                return Ok(Ok(None));
            };
            let Some(&x) = by_word.get(&want.pruned_word) else {
                return Ok(Ok(None));
            };
            let placed = match want.placed {
                None => None,
                Some((target_word, branch)) => match by_word.get(&target_word) {
                    Some(&target) => Some((target, branch)),
                    None => return Ok(Ok(None)),
                },
            };
            // The star primitive runs rayon work, so a worker may pick up
            // another proposal: each proposal takes its own cache.
            let taken = caches.lock().expect("cache pool").pop();
            let mut cache = match taken {
                Some(cache) if cache.len() >= space => cache,
                _ => ViewCache::new(space),
            };
            let keep = |score: i128| {
                unfixed(score - prints.score) > floor && first.fetch_min(i, Ordering::Relaxed) >= i
            };
            let fast = view_move(
                tree, rows, x, params, prints, placed, &keep, &mut cache, mirror,
            );
            caches.lock().expect("cache pool").push(cache);
            Ok(candidate(x, params, prints, placed, fast?, mirror)?.map_err(|()| (x, placed)))
        })
        .collect::<Result<_, _>>()?;

    let mut arena: Option<Arena> = None;
    let mut out = Vec::with_capacity(scored.len());
    for s in scored {
        let (x, placed) = match s {
            Ok(c) => {
                out.push(c);
                continue;
            }
            Err(declined) => declined,
        };
        if arena.is_none() {
            let (t, id_of, arena_of) = tree.to_tree()?;
            let word: Vec<u64> = id_of.iter().map(|&v| prints.word[v as usize]).collect();
            arena = Some(Arena {
                by_word: word_index(&word),
                slot: down.remap(&id_of),
                tree: t,
                arena_of,
                word,
            });
        }
        let a = arena.as_ref().expect("made above");
        let built = propose(
            &a.tree,
            down,
            a.arena_of[x as usize],
            params,
            prints.here,
            &a.by_word,
            placed.map(|(g, b)| (a.arena_of[g as usize], b)),
        )?;
        out.push(built.map(|b| Candidate::from_built(b, &a.word)));
    }
    if let Some(a) = arena {
        down.restore(a.slot);
    }
    Ok(out)
}

/// The live tree numbered as an arena, for the built path's few proposals.
struct Arena {
    /// The arena tree.
    tree: Tree,
    /// The store's slot map before it was keyed by arena index.
    slot: Vec<u32>,
    /// Leaf words by arena index.
    word: Vec<u64>,
    /// Word index by arena index.
    by_word: FxHashMap<u64, u32>,
    /// Arena index of each live id.
    arena_of: Vec<u32>,
}

/// Words of every node within `radius` edges of a clade a move created.
///
/// [`mark_near_new_clades`] over the live tree, from the created clades
/// alone, so it touches only the neighbourhood it marks.
///
/// ### Params
///
/// * `tree` - The tree the move produced
/// * `word` - Its leaf words
/// * `seeds` - The clades the move created
/// * `radius` - How many edges out to mark
/// * `out` - Set the words are added to
fn mark_near_live(
    tree: &LiveTree,
    word: &[u64],
    seeds: &[u32],
    radius: usize,
    out: &mut FxHashSet<u64>,
) {
    let mut dist: FxHashMap<u32, usize> = FxHashMap::default();
    let mut queue = std::collections::VecDeque::new();
    for &v in seeds {
        if dist.insert(v, 0).is_none() {
            queue.push_back(v);
        }
    }
    while let Some(v) = queue.pop_front() {
        out.insert(word[v as usize]);
        let d = dist[&v];
        if d == radius {
            continue;
        }
        for nb in tree.children(v).iter().copied().chain(tree.parent(v)) {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nb) {
                e.insert(d + 1);
                queue.push_back(nb);
            }
        }
    }
}

////////////
// Rounds //
////////////

/// The subtrees a sweep will try, in the order it will try them.
///
/// ### Params
///
/// * `tree` - The tree the sweep starts from
/// * `params` - Knobs, whose `order` this selects on
/// * `rng` - Stream for [`PruneOrder::Random`], advanced only by that arm
///
/// ### Returns
///
/// One [`crate::search::leaf_words`] entry per eligible subtree, in sweep order.
fn candidate_order(tree: &Tree, params: &SprParams, rng: &mut SplitMix64) -> Vec<u64> {
    let word = crate::search::leaf_words(tree);
    let mut nodes: Vec<u32> = (0..tree.n_nodes() as u32)
        .filter(|&x| can_prune(tree, x))
        .collect();
    match params.order {
        PruneOrder::LongestBranch => nodes
            .sort_unstable_by(|&a, &b| tree.branch(b).total_cmp(&tree.branch(a)).then(a.cmp(&b))),
        PruneOrder::Random => {
            // Fisher-Yates: the draw count depends on the candidate count only.
            for i in (1..nodes.len()).rev() {
                let j = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
                nodes.swap(i, j);
            }
        }
    }
    nodes.iter().map(|&x| word[x as usize]).collect()
}

/// One sweep, optionally restricted to the subtrees a previous sweep touched.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs
/// * `look` - [`leaf_words`] of the subtrees to propose, or `None` for all of
///   them
/// * `start` - Down rows settled against `tree` and its loglikelihood, as
///   [`settled_down`] returns them
///
/// ### Returns
///
/// The round's result, the words [`mark_near_new_clades`] collected from its
/// accepted moves, empty when the search revisits everything, and the settled
/// rows of the tree it finished on, `None` if it moved nothing; or the error
/// the placement, the primitive or the arena failed with.
#[allow(clippy::type_complexity)]
fn sweep<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: SprParams,
    look: Option<&FxHashSet<u64>>,
    start: (NodeState<T>, f64),
) -> Result<(SprResult, FxHashSet<u64>, Option<NodeState<T>>), BonsaiErrors> {
    let mut rng = SplitMix64::new(params.seed);
    let mut gains = Vec::new();
    let radius = params.search.revisit_radius();
    let recheck = params.search.recheck();

    // These describe the current tree; they change only on acceptance.
    let (settled, loglik_in) = start;
    let mut down = RowStore::from_state(&settled, tree.n_nodes());
    drop(settled);
    let mut best = down.score(tree);
    let mut word = leaf_words(tree);
    let mut below = leaves_below(tree);
    let mut here = split_fingerprint_counted(tree, &word, &below);
    let mut by_word = word_index(&word);
    let mut live = LiveTree::from_tree(tree);
    let caches: Mutex<Vec<ViewCache<T>>> = Mutex::new(Vec::new());
    let mut mirror = if cfg!(debug_assertions) {
        Some(Mirror::new(&live, &down, &word, &below, here)?)
    } else {
        None
    };

    let mut order = candidate_order(tree, &params, &mut rng);
    if let Some(look) = look {
        order.retain(|w| look.contains(w));
    }
    let mut revisit = FxHashSet::default();
    let mut next = 0usize;
    let mut chunk = PROPOSAL_CHUNK_MIN;
    while next < order.len() {
        let end = (next + chunk).min(order.len());
        let wants: Vec<Option<Want>> = order[next..end]
            .iter()
            .map(|&w| {
                Some(Want {
                    pruned_word: w,
                    placed: None,
                })
            })
            .collect();
        let prints = Prints {
            word: &word,
            below: &below,
            here,
            score: best,
        };
        let mut batch = score_chunk(
            &live,
            &mut down,
            &params,
            &prints,
            &by_word,
            &caches,
            &wants,
            mirror.as_ref(),
        )?;
        // Decided in order; after an acceptance with the recheck, the rest is
        // rescored against the new tree and deciding resumes there.
        let mut accepted = None;
        let mut offset = 0usize;
        loop {
            let mut hit = None;
            for (k, cand) in batch.iter_mut().enumerate() {
                let Some(mut cand) = cand.take() else {
                    continue;
                };
                if unfixed(cand.score - best) <= acceptance_floor(&params, unfixed(best)) {
                    continue;
                }
                gains.push(SprGain {
                    pruned: cand.pruned,
                    gain: unfixed(cand.score - best),
                });
                best = cand.score;
                match cand.built.take() {
                    Some(proposal) => {
                        // Scored on the tree as the batch found it, unchanged since.
                        let (_, id_of, _) = live.to_tree()?;
                        down.remap(&id_of);
                        down.accept_fresh(proposal.fresh);
                        live = LiveTree::from_tree(&proposal.tree);
                        word = leaf_words(&proposal.tree);
                        below = leaves_below(&proposal.tree);
                        here = split_fingerprint_counted(&proposal.tree, &word, &below);
                        by_word = word_index(&word);
                        if radius > 0 {
                            mark_near_new_clades(
                                &proposal.tree,
                                &word,
                                |v| proposal.to_old[v] == NO_NODE,
                                radius,
                                &mut revisit,
                            );
                        }
                    }
                    None => {
                        let data = cand
                            .data
                            .take()
                            .expect("the first candidate over the floor kept its views");
                        let expected = match &mirror {
                            Some(m) => built_on_mirror(
                                m,
                                cand.pruned,
                                &params,
                                Some((cand.target, cand.branch)),
                            )?,
                            None => None,
                        };
                        let applied =
                            apply_move(&mut live, &mut down, *data, &mut word, &mut below);
                        let n_leaves = live.n_leaves();
                        let created: Vec<u32> = applied
                            .changed
                            .iter()
                            .copied()
                            .filter(|&v| {
                                by_word
                                    .get(&word[v as usize])
                                    .is_none_or(|&old| (old as usize) < n_leaves)
                            })
                            .collect();
                        for &(w, v) in &applied.stale {
                            if by_word.get(&w) == Some(&v) {
                                by_word.remove(&w);
                            }
                        }
                        for &v in &applied.changed {
                            by_word.insert(word[v as usize], v);
                        }
                        here = applied.print;
                        if let (Some(m), Some(built)) = (&mirror, expected) {
                            check_applied(&live, &down, m, &built, &created)?;
                        }
                        if radius > 0 {
                            mark_near_live(&live, &word, &created, radius, &mut revisit);
                        }
                    }
                }
                if mirror.is_some() {
                    mirror = Some(Mirror::new(&live, &down, &word, &below, here)?);
                }
                hit = Some(k);
                break;
            }
            let Some(k) = hit else {
                break;
            };
            accepted.get_or_insert(offset + k);
            if !recheck {
                break;
            }
            // Same subtree, attachment and branch; dropped if either end no
            // longer names a node.
            let rest: Vec<Option<Want>> = batch[k + 1..]
                .iter()
                .map(|c| {
                    c.as_ref().map(|c| Want {
                        pruned_word: c.pruned_word,
                        placed: Some((c.target_word, c.branch)),
                    })
                })
                .collect();
            if rest.iter().all(Option::is_none) {
                break;
            }
            offset += k + 1;
            let prints = Prints {
                word: &word,
                below: &below,
                here,
                score: best,
            };
            batch = score_chunk(
                &live,
                &mut down,
                &params,
                &prints,
                &by_word,
                &caches,
                &rest,
                mirror.as_ref(),
            )?;
        }

        // Without the recheck the rest of an accepting chunk is proposed afresh.
        match accepted {
            Some(_) if recheck => {
                next = end;
                chunk = (chunk / 2).max(PROPOSAL_CHUNK_MIN);
            }
            Some(k) => {
                next += k + 1;
                chunk = (chunk / 2).max(PROPOSAL_CHUNK_MIN);
            }
            None => {
                next = end;
                chunk = (chunk * 2).min(PROPOSAL_CHUNK_MAX);
            }
        }
    }

    let tree = live.to_tree()?.0;
    // The fresh prune is also the next sweep's start.
    let (loglik, settled) = if gains.is_empty() {
        (loglik_in, None)
    } else {
        let (settled, loglik) = settled_down(&tree, leaves)?;
        (loglik, Some(settled))
    };
    Ok((
        SprResult {
            tree,
            loglik,
            gains,
            rounds: 1,
        },
        revisit,
        settled,
    ))
}

/// Debug builds: a move applied to the live tree against the built path's.
///
/// ### Params
///
/// * `tree` - The live tree after the move
/// * `down` - Its rows after the move
/// * `mirror` - The arena before the move
/// * `built` - The built path's move, on the mirror
/// * `created` - The clades the live path marked as created
///
/// ### Returns
///
/// `Ok` if they agree; panics otherwise.
fn check_applied<T: BonsaiFloat>(
    tree: &LiveTree,
    down: &RowStore<T>,
    mirror: &Mirror<T>,
    built: &Proposal<T>,
    created: &[u32],
) -> Result<(), BonsaiErrors> {
    let (a, id_of, _) = tree.to_tree()?;
    let b = &built.tree;
    assert_eq!(a.n_nodes(), b.n_nodes(), "node count");
    for v in 0..a.n_nodes() as u32 {
        assert_eq!(a.parent(v), b.parent(v), "parent of {v}");
        assert_eq!(
            a.branch(v).to_bits(),
            b.branch(v).to_bits(),
            "branch of {v}"
        );
    }
    let bits = |xs: &[T]| -> Vec<u64> {
        xs.iter()
            .map(|x| x.to_f64().map_or(u64::MAX, f64::to_bits))
            .collect()
    };
    let p = down.n_features();
    let fresh = &built.fresh;
    for v in 0..a.n_nodes() {
        let (m, w, c) = match fresh.inherited[v] {
            NO_NODE => {
                let k = fresh.slot[v] as usize;
                (
                    bits(&fresh.fresh_m[k * p..(k + 1) * p]),
                    bits(&fresh.fresh_w[k * p..(k + 1) * p]),
                    fresh.fresh_contrib[k].to_bits(),
                )
            }
            old => (
                bits(mirror.down.means(old)),
                bits(mirror.down.precisions(old)),
                mirror.down.contribution(old).to_bits(),
            ),
        };
        let id = id_of[v];
        assert_eq!(bits(down.means(id)), m, "means of {v}");
        assert_eq!(bits(down.precisions(id)), w, "precisions of {v}");
        assert_eq!(down.contribution(id).to_bits(), c, "term of {v}");
    }
    let mut want: Vec<u32> = (b.n_leaves()..b.n_nodes())
        .filter(|&v| built.to_old[v] == NO_NODE)
        .map(|v| id_of[v])
        .collect();
    let mut got = created.to_vec();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, want, "created clades");
    Ok(())
}

/// One sweep over the candidate subtrees, performing every improving move.
///
/// Each candidate is looked up by its [`leaf_words`] entry in the tree as it
/// stands; one whose word no longer names a node was swallowed by an earlier
/// move and is skipped. The tree is a [`LiveTree`] for the length of the sweep.
///
/// A candidate is scored as the current [`fixed`] total less the terms the move
/// changes plus the ones it forms, the per-node terms a fresh
/// [`NodeState::prune`] sums, so the sweep is monotone. The result's loglikelihood
/// is a fresh prune of the tree the sweep finished on.
///
/// ### Chunks
///
/// Scoring is parallel and decisions are sequential. Candidates are scored a
/// chunk at a time against one tree and decided in order; at an acceptance the
/// rest of the chunk is scored again against the new tree. Without
/// [`SprApprox::recheck`] they are proposed afresh; with it they keep the
/// attachment found on the older tree. Every candidate is decided against the
/// tree it was scored on, so the result is independent of the thread count. The
/// chunk halves on an acceptance and doubles on a clean chunk, between
/// [`PROPOSAL_CHUNK_MIN`] and [`PROPOSAL_CHUNK_MAX`].
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults
///
/// ### Returns
///
/// The improved tree, whose loglikelihood is never below the input's, or the
/// error the placement, the primitive or the arena failed with.
pub fn spr_round<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<SprParams>,
) -> Result<SprResult, BonsaiErrors> {
    let start = settled_down(tree, leaves)?;
    Ok(sweep(tree, leaves, params.unwrap_or_default(), None, start)?.0)
}

/// Search step 5: sweep until a sweep finds nothing.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults
/// * `verbosity` - [`Verbosity::Detailed`] prints one line per sweep
///
/// ### Returns
///
/// The tree the sweeps finished on, whose loglikelihood is never below the
/// input's, or the error the placement, the primitive or the arena failed with.
pub fn spr<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<SprParams>,
    verbosity: Verbosity,
) -> Result<SprResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let (mut settled, loglik) = settled_down(tree, leaves)?;
    let mut out = SprResult {
        tree: tree.clone(),
        loglik,
        gains: Vec::new(),
        rounds: 0,
    };

    let mut look: Option<FxHashSet<u64>> = None;
    while out.rounds < params.max_rounds {
        let started = Instant::now();
        // The random order must differ between rounds.
        let (round, revisit, next) = sweep(
            &out.tree,
            leaves,
            SprParams {
                seed: params.seed.wrapping_add(out.rounds as u64),
                ..params
            },
            look.as_ref(),
            (settled, out.loglik),
        )?;
        if params.search.revisit_radius() > 0 {
            look = Some(revisit);
        }
        out.rounds += 1;
        if verbosity.detailed_verbosity() {
            println!(
                "    sweep {}: {} moves, loglik {:.6e} ({:.2?})",
                out.rounds,
                round.gains.len(),
                round.loglik,
                started.elapsed()
            );
        }
        let Some(next) = next else {
            break;
        };
        settled = next;
        out.gains.extend_from_slice(&round.gains);
        out.loglik = round.loglik;
        out.tree = round.tree;
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::global::UpState;
    use crate::model::global::optimise_branch_lengths;
    use crate::model::place::attachment_score;
    use crate::search::polytomy::{CentreStar, centre_star};
    use crate::tree::simulate::{
        SimulatedData, SimulationParams, robinson_foulds, simulate_binary, splits,
    };
    use approx::assert_relative_eq;

    /// The whole tree collapsed onto every node, over the whole arena.
    ///
    /// The independent reference [`LazyRows`] is checked against. The up row
    /// is diffused down the branch above the node; the root's row is its down
    /// row.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `down` - Down rows, settled by [`NodeState::prune`] against this tree
    /// * `up` - Up rows, settled by the up sweep against the same
    ///
    /// ### Returns
    ///
    /// The means and precisions, `[node][feature]` row-major.
    fn collapse<T: BonsaiFloat>(
        tree: &Tree,
        down: &NodeState<T>,
        up: &UpState<T>,
    ) -> (Vec<T>, Vec<T>) {
        let p = down.n_features();
        let n = tree.n_nodes();
        let mut means = vec![T::zero(); n * p];
        let mut precisions = vec![T::zero(); n * p];

        for a in 0..n {
            let node = a as u32;
            let is_root = tree.parent(node).is_none();
            let t = tree.branch(node);
            let (m_down, w_down) = (down.means(node), down.precisions(node));
            let (m_up, w_up) = (up.means(node), up.precisions(node));
            let base = a * p;
            for g in 0..p {
                // Guarded on the root: its branch entry may hold a NaN.
                let w_above = if is_root {
                    0.0
                } else {
                    let w = wide(w_up[g]);
                    w / (1.0 + t * w)
                };
                let below = wide(w_down[g]);
                let total = below + w_above;
                let m = wide(m_down[g]);
                // Convex combination: cannot cancel.
                means[base + g] = narrow(m + (wide(m_up[g]) - m) * (w_above / total));
                precisions[base + g] = narrow(total);
            }
        }
        (means, precisions)
    }

    /// Settle a tree's down and up rows, over the whole arena.
    ///
    /// The independent reference [`LazyRows`] is checked against.
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
    fn settle<T: BonsaiFloat>(
        tree: &Tree,
        leaves: Leaves<'_, T>,
    ) -> Result<(NodeState<T>, UpState<T>, f64), BonsaiErrors> {
        let mut down = NodeState::new(
            tree.n_nodes(),
            leaves.n_features,
            leaves.means,
            leaves.precisions,
        )?;
        let loglik = down.prune(tree);
        let mut up = UpState::new(tree.n_nodes(), leaves.n_features);
        up.sweep(tree, &down);
        Ok((down, up, loglik))
    }

    /// A small simulated dataset.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, a power of two
    /// * `n_features` - Number of features
    /// * `seed` - Simulation seed
    ///
    /// ### Returns
    ///
    /// The dataset with its precisions already formed.
    fn dataset(n_leaves: usize, n_features: usize, seed: u64) -> (SimulatedData<f64>, Vec<f64>) {
        let data = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves,
            n_features,
            seed,
            ..SimulationParams::default()
        }))
        .expect("simulate");
        let precisions = data.precisions();
        (data, precisions)
    }

    /// A starting tree with its branch lengths optimised, as search step 4
    /// leaves them.
    ///
    /// ### Params
    ///
    /// * `tree` - Starting topology
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The tree with its branch lengths optimised.
    fn optimised(tree: &Tree, leaves: Leaves<'_, f64>) -> Tree {
        let mut tree = tree.clone();
        let mut state = NodeState::new(
            tree.n_nodes(),
            leaves.n_features,
            leaves.means,
            leaves.precisions,
        )
        .expect("state");
        optimise_branch_lengths(&mut tree, &mut state, None).expect("branch lengths");
        tree
    }

    /// The tree search steps 1 to 4 leave for step 5.
    ///
    /// The merge scan numbers the arena its own way and leaves polytomies that
    /// step 3 does not always clear, which the proposals have to cope with.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The tree, with its branch lengths optimised.
    fn searched(n_leaves: usize, leaves: Leaves<'_, f64>) -> Tree {
        use crate::search::bounds::EllipsoidBounds;
        use crate::search::candidates::KnnCandidates;
        use crate::search::polytomy::resolve_polytomies;
        use crate::search::star::{Star, star_tree_with};

        let p = leaves.n_features;
        let mut parent = vec![n_leaves as u32; n_leaves];
        parent.push(NO_NODE);
        let mut branch = vec![1.0f64; n_leaves];
        branch.push(0.0);
        let mut star = Tree::from_parents(parent, branch, n_leaves).expect("star");
        let mut state =
            NodeState::new(star.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
        optimise_branch_lengths(&mut star, &mut state, None).expect("step 1");
        let mut candidates = EllipsoidBounds::new(KnnCandidates::new(None));
        let (tree, _) = star_tree_with(
            Star {
                means: leaves.means,
                precisions: leaves.precisions,
                branch: &star.branches()[..n_leaves],
                n_features: p,
            },
            None,
            &mut candidates,
            Verbosity::Quiet,
        )
        .expect("step 2");
        let tree = resolve_polytomies(&tree, leaves, None)
            .expect("step 3")
            .tree;
        optimised(&tree, leaves)
    }

    /// Every proposal step 5 would make from `tree`, with everything needed to
    /// check the star it built.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree to propose from
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// Per prunable subtree, the regrafted tree, the attachment point, the star
    /// read off rows and the star a settled attached tree gives.
    #[allow(clippy::type_complexity)]
    fn proposals(
        tree: &Tree,
        leaves: Leaves<'_, f64>,
    ) -> Vec<(Tree, u32, CentreStar<f64>, CentreStar<f64>)> {
        let down = settled_down(tree, leaves).expect("down").0;
        let down = RowStore::from_state(&down, tree.n_nodes());
        let params = SprParams::default();
        let mut out = Vec::new();
        for x in 0..tree.n_nodes() as u32 {
            let Some(pruned) = prune_subtree(tree, x).expect("prune") else {
                continue;
            };
            let rows = LazyRows::new(&pruned.tree, &pruned.to_old, tree, &down).expect("rows");
            let q = EffLeaf {
                m: down.means(x),
                w: down.precisions(x),
            };
            let best = place(
                &pruned.tree,
                q,
                |node: u32| rows.eff_leaf(node),
                Some(params.placement),
            )
            .expect("place");
            let target = pruned.to_old[best.node as usize];
            let (attached, centre, to_old) =
                regraft(&pruned, x, target, best.branch, tree.n_leaves()).expect("regraft");
            let members =
                attached.children(centre).len() + usize::from(attached.parent(centre).is_some());
            if members <= crate::search::polytomy::RESOLVED_STAR_MEMBERS {
                continue;
            }
            let attached_rows = LazyRows::new(&attached, &to_old, tree, &down).expect("rows");
            let cheap = lazy_centre_star(&attached, &attached_rows, centre).expect("lazy star");
            let (a_down, a_up, _) = settle(&attached, leaves).expect("settle attached");
            let settled = centre_star(&attached, &a_down, &a_up, centre).expect("centre star");
            out.push((attached, centre, cheap, settled));
        }
        out
    }

    /// The leaf data of the remaining tree, in its own leaf numbering.
    ///
    /// Used by the tests that settle the remaining tree as a reference.
    ///
    /// ### Params
    ///
    /// * `pruned` - What [`prune_subtree`] left
    /// * `leaves` - The original leaf data
    ///
    /// ### Returns
    ///
    /// The means and precisions, `[leaf][feature]` row-major.
    fn pruned_leaves(pruned: &Pruned, leaves: Leaves<'_, f64>) -> (Vec<f64>, Vec<f64>) {
        let p = leaves.n_features;
        let n = pruned.tree.n_leaves();
        let mut means = Vec::with_capacity(n * p);
        let mut precisions = Vec::with_capacity(n * p);
        for &old in pruned.to_old.iter().take(n) {
            let lo = old as usize * p;
            means.extend_from_slice(&leaves.means[lo..lo + p]);
            precisions.extend_from_slice(&leaves.precisions[lo..lo + p]);
        }
        (means, precisions)
    }

    /// Every node of the subtree rooted at `x`, `x` itself included.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `x` - Root of the subtree
    ///
    /// ### Returns
    ///
    /// The nodes, in no particular order.
    fn subtree_nodes(tree: &Tree, x: u32) -> Vec<u32> {
        let mut out = vec![x];
        let mut i = 0;
        while i < out.len() {
            let node = out[i];
            out.extend_from_slice(tree.children(node));
            i += 1;
        }
        out
    }

    /// A six-leaf tree with a four-way root, so that pruning one of its
    /// children leaves no degree-two node behind.
    ///
    /// ### Returns
    ///
    /// The tree.
    fn wide_root() -> Tree {
        let parent = vec![6, 6, 7, 7, 8, 8, 8, 8, NO_NODE];
        let branch: Vec<f64> = (1..=9).map(|i| 0.1 * i as f64).collect();
        Tree::from_parents(parent, branch, 6).expect("fixture")
    }

    #[test]
    fn test_a_round_never_lowers_the_loglikelihood() {
        // Each round is scored by an independent `NodeState::prune`, from a deliberately wrong start.
        for seed in [1u64, 2, 3] {
            let (p, n) = (128usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let mut tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let mut before = tree_loglik(&tree, leaves).expect("loglik");

            for round in 0..6 {
                let out = spr_round(&tree, leaves, None).expect("round");
                let after = tree_loglik(&out.tree, leaves).expect("loglik");
                assert!(
                    after >= before,
                    "seed {seed} round {round}: {before} fell to {after}"
                );
                // Relative: both sides are sums of magnitude `O(n p)`.
                assert_relative_eq!(after, out.loglik, max_relative = 1e-12);
                let claimed: f64 = out.gains.iter().map(|g| g.gain).sum();
                let drift = (after - before - claimed).abs();
                assert!(
                    drift <= 1e-12 * after.abs(),
                    "seed {seed} round {round}: drift {drift:e} on |L| {:e}",
                    after.abs()
                );
                before = after;
                tree = out.tree;
            }
        }
    }

    #[test]
    fn test_the_whole_run_never_lowers_the_loglikelihood() {
        for seed in [4u64, 5] {
            let (p, n) = (128usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = tree_loglik(&start, leaves).expect("loglik");
            let out = spr(&start, leaves, None, Verbosity::Quiet).expect("spr");
            let after = tree_loglik(&out.tree, leaves).expect("loglik");
            assert!(after >= before, "seed {seed}: {before} fell to {after}");
            assert_relative_eq!(after, out.loglik, max_relative = 1e-12);
            assert!(out.rounds >= 1);
        }
    }

    #[test]
    fn test_prune_and_regraft_where_it_came_from_is_the_original_tree() {
        // Detach a subtree whose parent survives the cut and put it straight back.
        let (p, n) = (64usize, 6usize);
        let (data, w) = dataset(8, p, 11);
        let means: Vec<f64> = data.means[..n * p].to_vec();
        let precisions: Vec<f64> = w[..n * p].to_vec();
        let leaves = Leaves {
            means: &means,
            precisions: &precisions,
            n_features: p,
        };
        let tree = wide_root();
        let before = tree_loglik(&tree, leaves).expect("loglik");

        let x = 6u32;
        let pruned = prune_subtree(&tree, x)
            .expect("prune")
            .expect("node 6 is prunable");
        assert_eq!(pruned.tree.n_leaves(), 4);
        let (back, _, _) = regraft(&pruned, x, tree.root(), tree.branch(x), n).expect("regraft");

        assert_eq!(back.n_nodes(), tree.n_nodes());
        assert_eq!(splits(&back), splits(&tree));
        assert_eq!(back.branches(), tree.branches());
        assert_relative_eq!(
            tree_loglik(&back, leaves).expect("loglik"),
            before,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_the_lazy_rows_match_a_full_settle() {
        // Every `LazyRows` row must equal a full settle plus collapse, bit for bit, with and
        // without polytomies, for both storage types.
        for (n_leaves, seed, pipeline) in [
            (16usize, 3u64, false),
            (32, 5, false),
            (64, 7, true),
            (64, 11, true),
            (128, 13, true),
        ] {
            let p = 24usize;
            let (data, w) = dataset(n_leaves, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let tree = if pipeline {
                searched(n_leaves, leaves)
            } else {
                optimised(&data.tree, leaves)
            };
            let down = settled_down(&tree, leaves).expect("down").0;
            let down = RowStore::from_state(&down, tree.n_nodes());

            let mut checked = 0usize;
            let mut dirty_seen = 0usize;
            for x in 0..tree.n_nodes() as u32 {
                let Some(pruned) = prune_subtree(&tree, x).expect("prune") else {
                    continue;
                };
                let (rest_m, rest_w) = pruned_leaves(&pruned, leaves);
                let rest = Leaves {
                    means: &rest_m,
                    precisions: &rest_w,
                    n_features: p,
                };
                let (ref_down, ref_up, _) = settle(&pruned.tree, rest).expect("settle");
                let (eff_m, eff_w) = collapse(&pruned.tree, &ref_down, &ref_up);
                let rows = LazyRows::new(&pruned.tree, &pruned.to_old, &tree, &down).expect("rows");

                for v in 0..pruned.tree.n_nodes() as u32 {
                    if rows.inherited[v as usize] == NO_NODE {
                        dirty_seen += 1;
                    }
                    let (m, w) = rows.down_row(v);
                    assert_eq!(m, ref_down.means(v), "down mean, pruned {x}, node {v}");
                    assert_eq!(
                        w,
                        ref_down.precisions(v),
                        "down precision, pruned {x}, node {v}"
                    );
                    let up = rows.up_row(v);
                    assert_eq!(&up.0[..], ref_up.means(v), "up mean, pruned {x}, node {v}");
                    assert_eq!(
                        &up.1[..],
                        ref_up.precisions(v),
                        "up precision, pruned {x}, node {v}"
                    );
                    let lo = v as usize * p;
                    let e = rows.eff_leaf(v);
                    assert_eq!(e.m, &eff_m[lo..lo + p], "effective mean, node {v}");
                    assert_eq!(e.w, &eff_w[lo..lo + p], "effective precision, node {v}");
                }

                // Again on the tree the regraft builds.
                let target = pruned.to_old[pruned.tree.root() as usize];
                let (attached, _, to_old) =
                    regraft(&pruned, x, target, 0.25, tree.n_leaves()).expect("regraft");
                let a_rows = LazyRows::new(&attached, &to_old, &tree, &down).expect("rows");
                let (a_down, a_up, _) = settle(&attached, leaves).expect("settle attached");
                for v in 0..attached.n_nodes() as u32 {
                    let (m, w) = a_rows.down_row(v);
                    assert_eq!(m, a_down.means(v), "attached down mean, node {v}");
                    assert_eq!(w, a_down.precisions(v), "attached down precision, node {v}");
                    let up = a_rows.up_row(v);
                    assert_eq!(&up.0[..], a_up.means(v), "attached up mean, node {v}");
                    assert_eq!(
                        &up.1[..],
                        a_up.precisions(v),
                        "attached up precision, node {v}"
                    );
                }
                checked += 1;
            }
            assert!(checked > n_leaves, "only {checked} subtrees were prunable");
            // The fixtures must reach the recomputed rows.
            assert!(dirty_seen > 0, "no row was ever recomputed");
        }
    }

    #[test]
    fn test_the_lazy_rows_only_form_what_is_asked_for() {
        // A proposal on a balanced tree of 512 nodes must touch a small fraction of the rows.
        let p = 24usize;
        let (data, w) = dataset(256, p, 17);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&data.tree, leaves);
        let down = settled_down(&tree, leaves).expect("down").0;
        let down = RowStore::from_state(&down, tree.n_nodes());

        let mut total = 0usize;
        let mut formed = 0usize;
        for x in 0..tree.n_nodes() as u32 {
            let Some(pruned) = prune_subtree(&tree, x).expect("prune") else {
                continue;
            };
            let rows = LazyRows::new(&pruned.tree, &pruned.to_old, &tree, &down).expect("rows");
            let q = EffLeaf {
                m: down.means(x),
                w: down.precisions(x),
            };
            place(
                &pruned.tree,
                q,
                |node: u32| rows.eff_leaf(node),
                Some(PlacementParams::default()),
            )
            .expect("place");
            total += rows.eff_cells().len();
            formed += rows
                .eff_cells()
                .iter()
                .filter(|c| c.get().is_some())
                .count();
        }
        assert!(total > 0);
        // A measurement: a regression here slows the run while every answer stays right.
        assert!(
            formed * 10 < total,
            "the beam formed {formed} of {total} effective leaves"
        );
    }

    #[test]
    fn test_the_polytomy_threshold_matches_the_primitive() {
        // `propose` decides from the arena whether an attachment left a polytomy; this
        // checks it agrees with the star of a settled attached tree.
        for members in 1..=6usize {
            let star = CentreStar::<f64> {
                centre: 0,
                member_nodes: (0..members as u32).collect(),
                has_upstream: false,
                deleted: Vec::new(),
                means: Vec::new(),
                precisions: Vec::new(),
                branch: Vec::new(),
                n_features: 0,
            };
            assert_eq!(
                star.is_polytomy(),
                members > crate::search::polytomy::RESOLVED_STAR_MEMBERS,
                "{members} members"
            );
        }
    }

    #[test]
    fn test_the_star_read_off_rows_matches_a_settled_one() {
        // The star handed to the resolution must equal that of a freshly settled attached
        // tree bit for bit: the primitive picks its pair by strict comparison.
        let mut checked = 0usize;
        for (n_leaves, seed, pipeline) in [
            (16usize, 3u64, false),
            (32, 5, false),
            (64, 7, false),
            (16, 3, true),
            (32, 5, true),
            (64, 7, true),
            (64, 11, true),
        ] {
            let p = 32usize;
            let (data, w) = dataset(n_leaves, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let tree = if pipeline {
                searched(n_leaves, leaves)
            } else {
                optimised(&data.tree, leaves)
            };
            for (_, centre, cheap, settled) in proposals(&tree, leaves) {
                assert_eq!(cheap.centre, settled.centre);
                assert_eq!(cheap.member_nodes, settled.member_nodes);
                assert_eq!(cheap.has_upstream, settled.has_upstream);
                assert_eq!(cheap.branch, settled.branch);
                assert_eq!(cheap.n_features, settled.n_features);
                assert_eq!(cheap.means, settled.means, "centre {centre}");
                assert_eq!(cheap.precisions, settled.precisions, "centre {centre}");
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} proposals were checked");
    }

    #[test]
    fn test_degree_two_suppression_preserves_the_loglikelihood() {
        // Pruning leaf 0 leaves node 5 with one child; the suppression joins leaf 1 to
        // node 6 with `b1 + b5`.
        let p = 64usize;
        let (data, w) = dataset(8, p, 12);
        let means: Vec<f64> = data.means[..5 * p].to_vec();
        let precisions: Vec<f64> = w[..5 * p].to_vec();
        let b: Vec<f64> = (1..=8).map(|i| 0.15 * i as f64).collect();
        let tree =
            Tree::from_parents(vec![5, 5, 6, 7, 7, 6, 7, NO_NODE], b.clone(), 5).expect("fixture");

        let pruned = prune_subtree(&tree, 0).expect("prune").expect("prunable");
        assert_eq!(pruned.tree.n_leaves(), 4);
        assert_eq!(pruned.tree.n_nodes(), 6);

        // Leaves 1..=4 become 0..=3; node 6 becomes 4 and node 7 the root. Leaf 0 carries `b1 + b5`.
        let rest = Leaves {
            means: &means[p..],
            precisions: &precisions[p..],
            n_features: p,
        };
        let reference = Tree::from_parents(
            vec![4, 4, 5, 5, 5, NO_NODE],
            vec![b[1] + b[5], b[2], b[3], b[4], b[6], 0.0],
            4,
        )
        .expect("reference");
        assert_eq!(splits(&pruned.tree), splits(&reference));
        // The root's slot is not a branch and is not compared.
        let last = reference.n_nodes() - 1;
        assert_eq!(pruned.tree.branches()[..last], reference.branches()[..last]);
        assert_relative_eq!(
            tree_loglik(&pruned.tree, rest).expect("loglik"),
            tree_loglik(&reference, rest).expect("loglik"),
            epsilon = 1e-12
        );

        // Dropping the suppressed node's branch changes the answer, so this is not topology alone.
        let dropped = Tree::from_parents(
            vec![4, 4, 5, 5, 5, NO_NODE],
            vec![b[1], b[2], b[3], b[4], b[6], 0.0],
            4,
        )
        .expect("reference");
        let got = tree_loglik(&pruned.tree, rest).expect("loglik");
        let wrong = tree_loglik(&dropped, rest).expect("loglik");
        assert!((got - wrong).abs() > 1e-6, "{got} against {wrong}");
    }

    #[test]
    fn test_a_degree_two_root_is_suppressed_by_rerooting() {
        // Pruning a child of a three-way root leaves a degree-two root, joined by making
        // the internal child the root.
        let p = 64usize;
        let (data, w) = dataset(8, p, 13);
        let means: Vec<f64> = data.means[..5 * p].to_vec();
        let precisions: Vec<f64> = w[..5 * p].to_vec();
        let b: Vec<f64> = (1..=8).map(|i| 0.15 * i as f64).collect();
        // Root 7 holds leaf 4, node 5 and node 6; node 5 holds leaves 0 and 1;
        // node 6 holds leaves 2 and 3.
        let tree =
            Tree::from_parents(vec![5, 5, 6, 6, 7, 7, 7, NO_NODE], b.clone(), 5).expect("fixture");
        let pruned = prune_subtree(&tree, 4).expect("prune").expect("prunable");

        // Four leaves, two internal nodes: the old root is gone.
        assert_eq!(pruned.tree.n_leaves(), 4);
        assert_eq!(pruned.tree.n_nodes(), 6);
        for node in pruned.tree.internal_postorder() {
            assert!(pruned.tree.children(node).len() >= 2);
        }
        let reference = Tree::from_parents(
            vec![4, 4, 5, 5, 5, NO_NODE],
            vec![b[0], b[1], b[2], b[3], b[5] + b[6], 0.0],
            4,
        )
        .expect("reference");
        let rest = Leaves {
            means: &means[..4 * p],
            precisions: &precisions[..4 * p],
            n_features: p,
        };
        assert_eq!(splits(&pruned.tree), splits(&reference));
        assert_relative_eq!(
            tree_loglik(&pruned.tree, rest).expect("loglik"),
            tree_loglik(&reference, rest).expect("loglik"),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_the_pruned_subtree_is_never_a_target() {
        // No node of `pruned.tree` may map back into the subtree.
        let (p, n) = (32usize, 16usize);
        let (data, w) = dataset(n, p, 7);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);

        let mut checked = 0usize;
        for x in 0..tree.n_nodes() as u32 {
            let Some(pruned) = prune_subtree(&tree, x).expect("prune") else {
                continue;
            };
            let inside = subtree_nodes(&tree, x);
            for &old in &pruned.to_old {
                assert!(
                    !inside.contains(&old),
                    "pruning {x} left node {old} of its own subtree in the remaining tree"
                );
            }
            // The remaining tree holds every other node bar at most the suppressed one.
            let lost = tree.n_nodes() - inside.len() - pruned.tree.n_nodes();
            assert!(lost <= 1, "pruning {x} lost {lost} nodes");
            checked += 1;
        }
        assert!(checked > 0, "the fixture had no eligible subtree");
    }

    #[test]
    fn test_the_node_map_survives_the_arena() {
        // `Tree::from_parents` must relabel nothing, or the returned map is wrong.
        let (p, n) = (32usize, 16usize);
        let (data, w) = dataset(n, p, 8);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&Tree::balanced_binary(n, 1.0).expect("balanced"), leaves);

        for x in 0..tree.n_nodes() as u32 {
            let Some(pruned) = prune_subtree(&tree, x).expect("prune") else {
                continue;
            };
            for node in 0..pruned.tree.n_nodes() as u32 {
                let old = pruned.to_old[node as usize];
                assert_relative_eq!(pruned.tree.branch(node), pruned.branch[old as usize]);
                match pruned.tree.parent(node) {
                    None => assert_eq!(old, pruned.root),
                    Some(par) => {
                        assert_eq!(pruned.to_old[par as usize], pruned.parent[old as usize])
                    }
                }
            }
        }
    }

    #[test]
    fn test_the_attachment_score_ranks_the_real_trees() {
        // SPEC.md section 7.1: the regrafted loglikelihood is the remaining tree's plus
        // the subtree's plus the attachment score, so the difference is the same constant
        // at every target, leaf targets included.
        let (p, n) = (32usize, 8usize);
        let (data, w) = dataset(n, p, 51);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&data.tree, leaves);
        let (down, _, _) = settle(&tree, leaves).expect("settle");

        let mut checked = 0usize;
        for x in 0..tree.n_nodes() as u32 {
            let Some(pruned) = prune_subtree(&tree, x).expect("prune") else {
                continue;
            };
            let (rest_m, rest_w) = pruned_leaves(&pruned, leaves);
            let rest = Leaves {
                means: &rest_m,
                precisions: &rest_w,
                n_features: p,
            };
            let (rest_down, rest_up, _) = settle(&pruned.tree, rest).expect("settle");
            let (eff_m, eff_w) = collapse(&pruned.tree, &rest_down, &rest_up);
            let q = EffLeaf {
                m: down.means(x),
                w: down.precisions(x),
            };
            let (mut s, mut d) = (vec![0.0; p], vec![0.0; p]);

            let mut offset: Option<f64> = None;
            for node in 0..pruned.tree.n_nodes() as u32 {
                let lo = node as usize * p;
                let scored = attachment_score(
                    EffLeaf {
                        m: &eff_m[lo..lo + p],
                        w: &eff_w[lo..lo + p],
                    },
                    q,
                    &mut s,
                    &mut d,
                )
                .expect("score");
                let (built, _, _) = regraft(
                    &pruned,
                    x,
                    pruned.to_old[node as usize],
                    scored.branch,
                    tree.n_leaves(),
                )
                .expect("regraft");
                let here = tree_loglik(&built, leaves).expect("loglik") - scored.loglik;
                match offset {
                    None => offset = Some(here),
                    Some(first) => assert_relative_eq!(here, first, epsilon = 1e-8),
                }
            }
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_recovery_from_a_wrong_topology() {
        // A ladder is a bad start for balanced data. Branch lengths first, as the step order has it.
        let (p, n) = (256usize, 32usize);
        for seed in [1u64, 2, 3, 4] {
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = robinson_foulds(&start, &data.tree).expect("rf");
            let out = spr(&start, leaves, None, Verbosity::Quiet).expect("spr");
            let after = robinson_foulds(&out.tree, &data.tree).expect("rf");
            // Asserted exactly so a regression that merely improves the tree is caught.
            assert_eq!(
                after, 0,
                "seed {seed}: Robinson-Foulds {before} fell only to {after}"
            );
            assert!(out.rounds <= 5, "seed {seed} took {} rounds", out.rounds);
        }
    }

    #[test]
    fn test_the_two_orders_reach_the_same_topology() {
        // Both orders recover the generating tree. Move count is not asserted: it is noise at
        // 32 leaves (seed 1: 19 moves on macOS, 20 on Linux, 2026-09-29).
        let (p, n) = (256usize, 32usize);
        for seed in [1u64, 2] {
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let ordered = spr(&start, leaves, None, Verbosity::Quiet).expect("spr");
            let random = spr(
                &start,
                leaves,
                Some(SprParams {
                    order: PruneOrder::Random,
                    seed: 11,
                    ..SprParams::default()
                }),
                Verbosity::Quiet,
            )
            .expect("spr");
            assert_eq!(robinson_foulds(&ordered.tree, &data.tree).expect("rf"), 0);
            assert_eq!(robinson_foulds(&random.tree, &data.tree).expect("rf"), 0);
        }
    }

    #[test]
    fn test_a_tree_too_small_for_a_legal_move() {
        // Two leaves under one root: nothing may be pruned.
        let p = 16usize;
        let (data, w) = dataset(8, p, 21);
        let means: Vec<f64> = data.means[..2 * p].to_vec();
        let precisions: Vec<f64> = w[..2 * p].to_vec();
        let leaves = Leaves {
            means: &means,
            precisions: &precisions,
            n_features: p,
        };
        let tree = Tree::from_parents(vec![2, 2, NO_NODE], vec![0.4, 0.6, 0.0], 2).expect("pair");
        for x in 0..3u32 {
            assert!(prune_subtree(&tree, x).expect("prune").is_none());
        }
        let out = spr(&tree, leaves, None, Verbosity::Quiet).expect("spr");
        assert_eq!(out.gains.len(), 0);
        assert_eq!(out.rounds, 1);
        assert_eq!(splits(&out.tree), splits(&tree));
    }

    #[test]
    fn test_a_star_tree() {
        // Every regraft lands on a polytomy root and drags the section 9.2 resolution in;
        // the result must stay monotone and a tree.
        let (p, n) = (64usize, 8usize);
        let (data, w) = dataset(n, p, 22);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let star = Tree::from_parents(parent, vec![1.0; n + 1], n).expect("star");
        let before = tree_loglik(&star, leaves).expect("loglik");

        let out = spr(&star, leaves, None, Verbosity::Quiet).expect("spr");
        assert!(out.loglik >= before);
        assert_eq!(out.tree.n_leaves(), n);
        for node in out.tree.internal_postorder() {
            assert!(out.tree.children(node).len() >= 2);
        }
    }

    #[test]
    fn test_pruning_a_single_leaf() {
        let (p, n) = (64usize, 16usize);
        let (data, w) = dataset(n, p, 23);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let (down, _, _) = settle(&tree, leaves).expect("settle");
        let down = RowStore::from_state(&down, tree.n_nodes());

        let mut checked = 0usize;
        for leaf in 0..tree.n_leaves() as u32 {
            if !can_prune(&tree, leaf) {
                continue;
            }
            let pruned = prune_subtree(&tree, leaf)
                .expect("prune")
                .expect("prunable");
            assert_eq!(pruned.tree.n_leaves(), tree.n_leaves() - 1);
            let word = leaf_words(&tree);
            let here = split_fingerprint_with(&tree, &word);
            let Some(candidate) = propose(
                &tree,
                &down,
                leaf,
                &SprParams::default(),
                here,
                &word_index(&word),
                None,
            )
            .expect("propose") else {
                continue;
            };
            assert_eq!(candidate.tree.n_leaves(), tree.n_leaves());
            // A proposal may come back worse but not malformed.
            let _ = tree_loglik(&candidate.tree, leaves).expect("loglik");
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_the_incremental_loglik_matches_a_fresh_prune() {
        // The incremental figure must match a fresh prune to rounding, and the state an
        // acceptance builds must match it to the bit.
        let (p, n) = (96usize, 32usize);
        let (data, w) = dataset(n, p, 5);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let (down, _, _) = settle(&tree, leaves).expect("settle");
        let down = RowStore::from_state(&down, tree.n_nodes());
        let word = leaf_words(&tree);
        let here = split_fingerprint_with(&tree, &word);
        let by_word = word_index(&word);

        let mut checked = 0usize;
        for x in 0..tree.n_nodes() as u32 {
            let Some(proposal) =
                propose(&tree, &down, x, &SprParams::default(), here, &by_word, None)
                    .expect("propose")
            else {
                continue;
            };
            let (fresh, loglik) = settled_down(&proposal.tree, leaves).expect("settle");
            assert_relative_eq!(unfixed(proposal.score), loglik, max_relative = 1e-12);
            let mut built = down.clone();
            built
                .accept(&proposal.tree, &proposal.to_old, &tree)
                .expect("accept");
            assert_store_is_fresh(&built, &fresh, &proposal.tree);
            checked += 1;
        }
        assert!(checked > n, "only {checked} candidates exercised");
    }

    /// Every row and term of a store against a fresh prune of the same tree, to
    /// the bit.
    ///
    /// ### Params
    ///
    /// * `store` - The store under test
    /// * `fresh` - [`NodeState::prune`] of `tree`
    /// * `tree` - The tree both describe
    fn assert_store_is_fresh(store: &RowStore<f64>, fresh: &NodeState<f64>, tree: &Tree) {
        for v in 0..tree.n_nodes() as u32 {
            assert_eq!(store.means(v), fresh.means(v), "means at node {v}");
            assert_eq!(
                store.precisions(v),
                fresh.precisions(v),
                "precisions at node {v}"
            );
            assert_eq!(
                store.contribution(v).to_bits(),
                fresh.contribution(v).to_bits(),
                "term at node {v}"
            );
        }
    }

    #[test]
    fn test_a_revisit_radius_wider_than_the_tree_changes_nothing() {
        // With the radius past the diameter every move marks every node, so the next sweep
        // matches an unrestricted one.
        let (p, n) = (96usize, 32usize);
        let (data, w) = dataset(n, p, 7);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let with = |search: SprSearch| {
            spr(
                &start,
                leaves,
                Some(SprParams {
                    search,
                    ..SprParams::default()
                }),
                Verbosity::Quiet,
            )
            .expect("spr")
        };
        let all = with(SprSearch::Exact);
        assert!(all.rounds >= 2, "only {} rounds", all.rounds);
        // A radius of zero is the exact search.
        for radius in [0, 4 * n] {
            let got = with(SprSearch::Approximate(SprApprox {
                revisit_radius: radius,
                recheck: false,
            }));
            assert_eq!(got.rounds, all.rounds, "radius {radius}");
            assert_eq!(
                got.loglik.to_bits(),
                all.loglik.to_bits(),
                "radius {radius}"
            );
            assert_eq!(
                crate::search::split_fingerprint(&got.tree),
                crate::search::split_fingerprint(&all.tree),
                "radius {radius}"
            );
        }
    }

    #[test]
    fn test_a_revisit_radius_never_lowers_the_loglikelihood() {
        let (p, n) = (96usize, 32usize);
        let (data, w) = dataset(n, p, 7);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let before = tree_loglik(&start, leaves).expect("loglik");
        for radius in 1..=3 {
            let out = spr(
                &start,
                leaves,
                Some(SprParams {
                    search: SprSearch::Approximate(SprApprox {
                        revisit_radius: radius,
                        recheck: false,
                    }),
                    ..SprParams::default()
                }),
                Verbosity::Quiet,
            )
            .expect("spr");
            assert!(out.loglik > before, "radius {radius}");
            assert_relative_eq!(
                out.loglik,
                tree_loglik(&out.tree, leaves).expect("loglik"),
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_a_chain_of_accepted_moves_leaves_the_store_a_fresh_prune() {
        // One acceptance never reuses a freed slot; a chain does, so each acceptance in a
        // ladder's first round is checked against a fresh prune.
        let (p, n) = (64usize, 32usize);
        let (data, w) = dataset(n, p, 11);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let mut tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let (down, mut best) = settled_down(&tree, leaves).expect("settle");
        let mut store = RowStore::from_state(&down, tree.n_nodes());
        let params = SprParams::default();

        let mut accepted = 0usize;
        let mut x = 0u32;
        while (x as usize) < tree.n_nodes() {
            let word = leaf_words(&tree);
            let here = split_fingerprint_with(&tree, &word);
            let by_word = word_index(&word);
            let proposal =
                propose(&tree, &store, x, &params, here, &by_word, None).expect("propose");
            x += 1;
            let Some(proposal) = proposal else {
                continue;
            };
            if unfixed(proposal.score) <= best + acceptance_floor(&params, best) {
                continue;
            }
            store
                .accept(&proposal.tree, &proposal.to_old, &tree)
                .expect("accept");
            let (fresh, loglik) = settled_down(&proposal.tree, leaves).expect("settle");
            assert_store_is_fresh(&store, &fresh, &proposal.tree);
            best = loglik;
            tree = proposal.tree;
            accepted += 1;
            x = 0;
        }
        assert!(accepted >= 8, "only {accepted} moves accepted");
    }

    #[test]
    fn test_the_sweep_is_deterministic_whatever_the_thread_count() {
        // The sweep must be independent of chunk scheduling; a ladder has tens of accepted
        // moves inside chunks.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let reference = spr_round(&start, leaves, None).expect("round");
        // Some sixty candidates and a chunk of thirty-two: not all moves fall on boundaries.
        assert!(
            reference.gains.len() >= 8,
            "the fixture has to accept inside chunks, got {} moves",
            reference.gains.len()
        );

        for threads in [1usize, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| spr_round(&start, leaves, None).expect("round"));
            assert_eq!(
                splits(&got.tree),
                splits(&reference.tree),
                "topology moved at {threads} threads"
            );
            assert_eq!(got.tree.branches(), reference.tree.branches());
            assert_eq!(got.loglik.to_bits(), reference.loglik.to_bits());
            assert_eq!(got.gains.len(), reference.gains.len());
            for (a, b) in got.gains.iter().zip(&reference.gains) {
                assert_eq!(a.pruned, b.pruned);
                assert_eq!(a.gain.to_bits(), b.gain.to_bits());
            }
        }
    }

    #[test]
    fn test_a_sweep_that_improves_nothing_changes_nothing() {
        // The generating tree at the step 4 optimum: the sweep must return it unchanged.
        for seed in [31u64, 32] {
            let (p, n) = (256usize, 16usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&data.tree, leaves);
            let out = spr_round(&start, leaves, None).expect("round");
            assert_eq!(out.gains.len(), 0, "seed {seed} moved a tree it should not");
            assert_eq!(splits(&out.tree), splits(&start));
        }
    }

    #[test]
    fn test_the_random_order_is_deterministic_across_thread_counts() {
        let (p, n) = (128usize, 16usize);
        let (data, w) = dataset(n, p, 41);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let params = SprParams {
            order: PruneOrder::Random,
            seed: 9,
            ..SprParams::default()
        };

        let runs: Vec<SprResult> = [1usize, 2, 8]
            .into_iter()
            .map(|threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool");
                pool.install(|| spr(&start, leaves, Some(params), Verbosity::Quiet).expect("spr"))
            })
            .collect();

        for run in &runs[1..] {
            assert_eq!(run.tree.branches(), runs[0].tree.branches());
            assert_eq!(splits(&run.tree), splits(&runs[0].tree));
            assert_eq!(run.loglik.to_bits(), runs[0].loglik.to_bits());
            assert_eq!(run.gains.len(), runs[0].gains.len());
        }
    }

    /// Magnitude of the whole-tree loglikelihood on the realistic dataset an
    /// absolute floor failed on, ten thousand cells by a few thousand genes; the
    /// extrapolation target.
    const REALISTIC_LOGLIK: f64 = 1.1e7;

    /// Worst gain accepted on that dataset that turned out to be rounding.
    const REALISTIC_NOISE: f64 = 5.96e-8;

    #[test]
    fn test_the_acceptance_floor_outgrows_the_loglikelihood_rounding_floor() {
        // Both sides are sums over every internal node, so the resolvable difference grows
        // with `|L|`, `O(n p)`. With the floor at `StarParams::min_gain` alone the search
        // cycled on neutral moves at 10,000 cells and ended on the round cap.
        //
        // Fixtures cannot reach that magnitude, so noise is measured over a range of sizes,
        // its growth law checked, and the floor tested against the extrapolation.
        let params = SprParams::default();
        let mut rungs: Vec<(f64, f64)> = Vec::new();
        for (n, p) in [(16usize, 64usize), (32, 128), (64, 256), (128, 512)] {
            let (data, w) = dataset(n, p, 5);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let (down, best) = settled_down(&tree, leaves).expect("settle");
            let down = RowStore::from_state(&down, tree.n_nodes());
            let word = leaf_words(&tree);
            let here = split_fingerprint_with(&tree, &word);
            let by_word = word_index(&word);

            // Noise: the disagreement between a candidate's accepted figure and a fresh prune of
            // the same tree.
            let mut noise = 0.0f64;
            let mut seen = 0usize;
            for x in 0..tree.n_nodes() as u32 {
                let Some(proposal) =
                    propose(&tree, &down, x, &params, here, &by_word, None).expect("propose")
                else {
                    continue;
                };
                let (_, fresh) = settled_down(&proposal.tree, leaves).expect("settle");
                noise = noise.max((unfixed(proposal.score) - fresh).abs());
                seen += 1;
            }
            assert!(seen > n / 2, "only {seen} candidates at {n} leaves");
            assert!(
                acceptance_floor(&params, best) > 100.0 * noise,
                "at {n} by {p}, |L| = {:.3e}: floor {:.3e} against noise {noise:.3e}",
                best.abs(),
                acceptance_floor(&params, best)
            );
            rungs.push((best.abs(), noise / best.abs()));
        }

        // Relative noise grew as sqrt(`|L|`): 1.2e-16 at |L| = 4.7e2 to 6.2e-16 at 1.1e4.
        // Pin the law, then extrapolate. A rung can be exactly zero (below one ulp), so the
        // base is floored at unit roundoff.
        let (l0, rel0) = rungs[0];
        let rel0 = rel0.max(f64::EPSILON / 2.0);
        let (l1, rel1) = *rungs.last().expect("rungs");
        assert!(
            rel1 / rel0 < 4.0 * (l1 / l0).sqrt(),
            "relative noise grew faster than the square root: {rel0:.3e} to {rel1:.3e}"
        );
        // Ten, not a hundred: the projection over-predicts by about 4x (2.2e-7 against 6.0e-8
        // measured on the real 10k dataset). The hundredfold check is against the measurement.
        let projected = rel1 * (REALISTIC_LOGLIK / l1).sqrt() * REALISTIC_LOGLIK;
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) > 10.0 * projected,
            "extrapolated to |L| = {REALISTIC_LOGLIK:.3e}: floor {:.3e} against noise \
             {projected:.3e}",
            acceptance_floor(&params, -REALISTIC_LOGLIK)
        );

        // And against what was observed.
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) > 100.0 * REALISTIC_NOISE,
            "floor {:.3e} against the measured 10k noise {REALISTIC_NOISE:.3e}",
            acceptance_floor(&params, -REALISTIC_LOGLIK)
        );

        // The smallest gain accepted on the subsample ladder was 2e-5 nats at |L| = 1.7e6;
        // scale it down as the floor scales and check the margin.
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) < 2e-5 * (REALISTIC_LOGLIK / 1.7e6),
            "the floor would reject a real move"
        );
    }

    #[test]
    fn test_the_sweeps_reach_a_fixed_point_rather_than_cycling() {
        // The failure is a cycle: one neutral move per round until the cap. `spr` must stop
        // on its own and a second run from its output must be the identity, at several sizes.
        for (n, p) in [(16usize, 64usize), (32, 128), (64, 256)] {
            let (data, w) = dataset(n, p, 7);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = searched(n, leaves);
            let params = SprParams::default();
            let first = spr(&start, leaves, Some(params), Verbosity::Quiet).expect("spr");
            assert!(
                first.rounds < params.max_rounds,
                "{n} by {p} ran to the cap at {} rounds",
                first.rounds
            );
            let again = spr(&first.tree, leaves, Some(params), Verbosity::Quiet).expect("spr");
            assert_eq!(
                again.gains.len(),
                0,
                "{n} by {p} kept moving after it stopped"
            );
            assert_eq!(again.tree.branches(), first.tree.branches());
            assert_eq!(again.loglik.to_bits(), first.loglik.to_bits());
        }
    }
}
