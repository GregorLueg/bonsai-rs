//! Subtree pruning and regrafting, search step 5 (SPEC.md section 9.3).
//!
//! One move detaches a node and everything below it, then hangs it back
//! somewhere else. The three halves of that are already written elsewhere and
//! this module is mostly the plumbing between them: the subtree is summarised
//! as an effective leaf by the pruning recursion of SPEC.md section 4, its new
//! home is found by the beam search of [`place`] (SPEC.md section 7.2), and the
//! polytomy the attachment creates is resolved by [`splice_star`] (SPEC.md
//! sections 7.3 and 9.2).
//!
//! ### Detaching leaves a degree-two node behind
//!
//! Removing a child from a node of degree three leaves a node with two
//! neighbours, which carries no information: the pruning recursion contributes
//! exactly zero at such a node and passes its child's effective leaf straight
//! through with the two branch lengths added, so the tree with it and the tree
//! without it have the same loglikelihood to the last bit. The arena refuses to
//! hold one at all, since [`Tree::from_parents`] wants at least two children on
//! every internal node, which is the practical reason the suppression is not
//! optional: the two remaining neighbours are joined by the sum of their branch
//! lengths.
//!
//! The root needs the same treatment and gets it. A root left with two children
//! is a degree-two vertex in the unrooted sense. It still scores correctly,
//! because those two branches enter the likelihood only through their sum, but
//! it is a node whose children can never be pruned again. So it is suppressed
//! the same way, by making one of the two children the root and giving the
//! other the summed branch. The one case with nowhere to go is a remaining tree
//! of two leaves, where the degree-two root *is* the tree's single edge.
//!
//! ### What may not be pruned
//!
//! The root, because the arena needs one. And a child of a root that already
//! has only two children, because that leaves the root with one child and no
//! suppression can fix it. Both are rejected before any work is done rather
//! than discovered inside [`Tree::from_parents`].
//!
//! ### The pruned subtree is not a candidate attachment point
//!
//! Regrafting a subtree onto itself makes a cycle or a disconnected component.
//! Nothing here has to test for that, because the beam search runs on a tree
//! the subtree is not in: pruning builds the remaining tree explicitly, with
//! its leaves renumbered into a contiguous block, and the subtree's nodes are
//! simply absent from it. `test_the_pruned_subtree_is_never_a_target` asserts
//! that rather than assuming it.
//!
//! ### Accepting a move
//!
//! On the candidate tree's own per-node terms, never on
//! [`crate::search::polytomy::Splice::gain`]: that gain is exact only against
//! the tree its star was built from, and by the time a move has been proposed
//! the tree has been cut, rebuilt and possibly re-rooted. A regraft moves a
//! subtree across the tree, which no single star summarises.
//! [`crate::search::nni`] can accept on its merge gains plus a collapse delta
//! because an interchange edits one internal edge and the star it resolves
//! summarises the rest of the tree exactly.
//!
//! The terms come from [`LazyRows::loglik`], which reads the current tree's
//! terms wherever the candidate's subtrees are the current tree's and
//! recomputes the rest; an accepted candidate's rows are assembled from the
//! same rows by [`LazyRows::into_state`] rather than swept. A fresh
//! [`NodeState::prune`] per candidate is correct and is what this replaced, but
//! it pays an `O(n p)` sweep for every candidate that passes the split filter,
//! and at the noise levels where the step earns its keep that is a third of the
//! step. `test_the_incremental_loglik_matches_a_fresh_prune` pins both against
//! the sweep, the state to the bit.
//!
//! Proposing was what cost before that: settling the tree the cut leaves
//! behind and the tree the regraft builds, and collapsing the first onto every
//! node, five `O(n p)` sweeps per candidate for rows of which a few dozen are
//! ever read. [`LazyRows`] forms the ones that are read and no others, which
//! is what took step 5 off `n^1.9`.
//!
//! ### This is a topology search and only a topology search
//!
//! A regraft that puts the subtree back on the split it came off is not a move.
//! It reoptimises the branch the subtree hangs on, and the three the resolution
//! makes where the cut left a degree-two node behind, and it gains a little
//! nearly every time. Accepting those turns the sweep into a very expensive
//! branch-length descent, which is what steps 4 and 7 already do globally. Left
//! unfiltered, most accepted moves change no split at all and carry a few per
//! cent of the nats the real moves carry. So a proposal whose split fingerprint
//! matches the current tree's is discarded, which is the deviation SPEC.md
//! section 9.4 records for the interchanges, arrived at independently and for
//! the same reason.

use crate::errors::BonsaiErrors;
use crate::model::global::{LOGLIK_SCALE_FLOOR, up_part};
use crate::model::likelihood::NodeState;
use crate::model::merge::EffLeaf;
use crate::model::place::{PlacementParams, place};
use crate::search::polytomy::{CentreStar, splice_star};
use crate::search::star::StarParams;
use crate::search::{Leaves, leaf_words, settled_down, split_fingerprint_with, tree_loglik};
use crate::tree::{NO_NODE, Tree};
use crate::utils::kernels::prune_general;
use crate::utils::rng::SplitMix64;
use crate::utils::simd::prune_binary;
use crate::utils::traits::{BonsaiFloat, narrow, wide};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cell::OnceCell;

////////////////
// Parameters //
////////////////

/// Default for [`SprParams::max_rounds`].
///
/// A runaway guard and not a working limit. Every accepted move raises the tree
/// loglikelihood by more than [`acceptance_floor`] and the loglikelihood is
/// bounded above, so the sweeps terminate on their own; this only bounds how
/// long it can take to notice. One round performs every improving move it
/// finds, not one, so the count needed is small: simulated fixtures from 16 to
/// 64 leaves need two to four rounds and never more, and a hundred is more than
/// an order of magnitude of headroom on that.
///
/// That argument is only ever as good as the floor behind it. An absolute floor
/// lets a large tree accept rounding noise forever and cycle, at which point
/// this cap is the only thing that ends the run; see
/// [`DEFAULT_MIN_RELATIVE_GAIN`]. With the scale-relative floor the cap does
/// not bind, and it stays where it is as the guard it was written to be.
const DEFAULT_MAX_ROUNDS: usize = 100;

/// Default for [`SprParams::min_relative_gain`], as a fraction of `|L|`.
///
/// The star primitive's `StarParams::min_gain` is a floor on a *merge score*,
/// a sum over the `p` features of one pair, magnitude `O(p)` and rounding
/// floor `1.1e-16` per feature. This module accepts on a *whole-tree*
/// loglikelihood, a sum over every internal node of the tree, magnitude
/// `O(n p)`. The absolute constant is right for the first quantity and wrong
/// for the second, and the wrongness grows with the problem. On realistic data
/// at ten thousand cells, where `|L|` is around `1e7`, an absolute floor let
/// SPR accept one likelihood-neutral move of about 32 ulps of the sum in every
/// round to the cap, cycling with period two and spending most of the step on
/// it.
///
/// So the floor scales with the magnitude of the quantity it gates, in the
/// same shape [`crate::model::global::optimise_branch_lengths`] already uses:
/// `min_relative_gain * max(|L|, LOGLIK_SCALE_FLOOR)`, the floor there to stop
/// the test collapsing when the loglikelihood passes through zero (SPEC.md
/// section 3.2 leaves it defined only up to an additive constant).
///
/// `1e-12` is the tolerance `test_the_incremental_loglik_matches_a_fresh_prune`
/// already pins the proposal's score to, which is the right number by
/// construction: a candidate is accepted on an incrementally assembled figure,
/// and there is no sense in accepting a gain smaller than the disagreement
/// between that figure and the fresh prune it stands in for. Against the
/// observed noise it is 200x of headroom (`1.1e-5` against `5.6e-8` at 10k)
/// and against a real move it is six orders of margin, since accepted SPR
/// gains at 10k average about 15 nats and the smallest measured on any rung of
/// the subsample ladder was `2e-5`. At 512 cells the floor is `5e-7`, at 5,000
/// `5.6e-6`; below about 1,000 cells `StarParams::min_gain` is the binding one
/// and nothing changes.
const DEFAULT_MIN_RELATIVE_GAIN: f64 = 1e-12;

/// Fewest candidates a sweep proposes in parallel before deciding any of them.
///
/// A chunk is proposed against one tree and decided in order; the first
/// accepted move invalidates the rest of its chunk, which is proposed again
/// against the new tree. So the chunk bounds the work an acceptance throws
/// away, and it halves on every acceptance down to this. Ours, chosen by
/// measurement: the same tree comes out at every floor, and the wall time is
/// flat below eight while the wasted proposals keep falling, so there is
/// nothing to buy by going lower.
///
/// Below eight the chunk is too small to keep the pool busy where acceptances
/// are dense, and above it the discarded proposals cost more than they save.
const PROPOSAL_CHUNK_MIN: usize = 8;

/// Most candidates a sweep proposes in parallel before deciding any of them.
///
/// Doubles from [`PROPOSAL_CHUNK_MIN`] on every chunk that accepts nothing,
/// which on a settled tree is every chunk. Bounds memory rather than waste: a
/// candidate is a whole arena, so a chunk holds this many trees at once.
const PROPOSAL_CHUNK_MAX: usize = 256;

/// Which subtree the sweep considers next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneOrder {
    /// Descending length of the branch above the subtree, ties going to the
    /// lower node index.
    ///
    /// The paper's default, and the reasoning in SPEC.md section 9.3 is
    /// sound: a long upstream branch means the subtree sits far from its
    /// parent, so it is the one the current topology is least sure of.
    ///
    /// **It holds up, mildly.** Measured against a random order: both reach a
    /// Robinson-Foulds distance of zero from the generating tree on every
    /// fixture, so the ordering does not decide whether the search arrives. What
    /// it decides is how cheaply and how high, and the margin grows with the
    /// tree while being lost in the noise on a small one.
    LongestBranch,
    /// A uniform shuffle of the eligible subtrees, from [`SprParams::seed`].
    Random,
}

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
    /// is also an absolute floor on the loglikelihood improvement that will be
    /// accepted as a move, but it is a merge-score floor and on any tree worth
    /// searching [`SprParams::min_relative_gain`] is the binding one; see
    /// [`acceptance_floor`].
    pub star: StarParams,
    /// Smallest improvement in the whole-tree loglikelihood that will be
    /// accepted as a move, as a fraction of `max(|L|, 1)`.
    ///
    /// Strictly greater than, and taken together with `star.min_gain` as a
    /// maximum, so raising either raises the floor. See
    /// [`DEFAULT_MIN_RELATIVE_GAIN`] for why an absolute floor alone is not
    /// enough.
    pub min_relative_gain: f64,
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
        }
    }
}

///////////////
// Threshold //
///////////////

/// Smallest gain, in nats, that this round will accept as a move.
///
/// Two floors, whichever is larger. `star.min_gain` is absolute and is the
/// primitive's merge-score floor, kept so that a caller can still raise the
/// bar by hand. `min_relative_gain` scales with the magnitude of the thing
/// actually being compared, which is a whole-tree loglikelihood and grows as
/// `O(n p)`; see [`DEFAULT_MIN_RELATIVE_GAIN`] for what happens without it.
/// The `max(|L|, LOGLIK_SCALE_FLOOR)` is
/// [`crate::model::global::optimise_branch_lengths`]'s guard against a
/// loglikelihood that happens to sit near zero.
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

////////////
// Output //
////////////

/// What one accepted move did.
#[derive(Clone, Copy, Debug)]
pub struct SprGain {
    /// The subtree that was pruned, as a node of the tree the move started
    /// from. Renumbering makes it meaningless in the tree that came out.
    pub pruned: u32,
    /// Loglikelihood gain of the whole move, in nats: the difference of two
    /// independent [`NodeState::prune`] calls, so it covers the regraft and the
    /// polytomy resolution that followed it.
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

impl SprResult {
    /// Number of moves performed.
    ///
    /// ### Returns
    ///
    /// The move count.
    #[inline]
    pub fn n_moves(&self) -> usize {
        self.gains.len()
    }
}

///////////////////////
// Arena rebuilding  //
///////////////////////

/// Renumber a parent array into the arena invariant and build the tree.
///
/// Anything the walk from `root` does not reach is dropped, which is how both a
/// detached subtree and a suppressed degree-two node leave the arena. Kept
/// leaves are compacted into `0..k` in ascending original order, and kept
/// internal nodes are numbered by height and then by original index, which is
/// exactly the order [`Tree::from_parents`] relabels into. Its relabelling is
/// therefore the identity, which is what makes the map returned here a map of
/// the tree that comes back rather than of the array that went in;
/// `test_the_node_map_survives_the_arena` pins that.
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
fn assemble(
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

    // One post-order walk settles both reachability and height. A cycle in the
    // input cannot be reached from the root, which has no parent and is
    // therefore nobody's child, so the walk always terminates.
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
    // Numbered by height and then by index, as a counting sort: the walk over
    // `n_leaves..n` is already in index order, so bucketing by height keeps
    // it within each height.
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
    let tree = Tree::from_parents(new_parent, new_branch, n_kept_leaves)?;
    Ok((tree, new_id))
}

/////////////
// Pruning //
/////////////

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

/// A detached subtree and the tree left behind.
///
/// The remaining tree is a real arena with its own leaf numbering, because
/// [`place`] needs a [`Tree`]. The original-space arrays are kept alongside it,
/// because the regraft has to be assembled in the numbering the detached
/// subtree is still expressed in.
///
/// The leaf data is **not** copied over. It used to be, to build a
/// [`NodeState`] for the remaining tree, and that copy was `O(n p)` per
/// candidate subtree for a state whose rows are almost all equal to rows the
/// original tree already holds; [`LazyRows`] reads those instead.
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
    let Some(par) = tree.parent(x).filter(|_| can_prune(tree, x)) else {
        return Ok(None);
    };
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
        // A degree-two internal node: join its two remaining neighbours, which
        // are its parent and its one surviving child, with the summed branch.
        (Some(above), &[s]) => {
            parent[s as usize] = above;
            branch[s as usize] += branch[par as usize];
            parent[par as usize] = NO_NODE;
        }
        // A degree-two root, whose two remaining neighbours are both children.
        // Joining them means making one of them the root; a leaf cannot take
        // that job, and when both are leaves the degree-two root is the tree's
        // only edge and has to stay.
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

//////////////////
// Settled rows //
//////////////////

/// One node's settled row, means and precisions.
///
/// Owned and boxed rather than written into a slab, because the beam search's
/// provider hands out a borrow that has to stay valid for the whole search
/// while later rows are still being formed behind it.
type Row<T> = (Box<[T]>, Box<[T]>);

/// A tree's settled rows, computed only where they are read.
///
/// One proposal used to settle two whole trees, the one the cut leaves behind
/// and the one the regraft builds, and to collapse every node of the first into
/// an effective leaf for the beam search to score against. Five `O(n p)` sweeps
/// per candidate subtree and `O(n)` candidate subtrees a round is where step
/// 5's quadratic term lived.
///
/// Almost none of it is read. The beam search scores a few dozen nodes of a
/// balanced tree, not all of them ([`PlacementParams::tolerance`] records 13 of
/// 127 and 18 of 511), and the resolution that follows reads a handful of rows
/// at one node. So the rows are computed on demand and remembered, and the ones
/// nobody asks for are never formed.
///
/// ### Why on-demand is the same arithmetic and not an approximation of it
///
/// **Down rows.** Cutting a subtree out and hanging it back somewhere else
/// leaves most subtrees alone, and a node whose subtree did not change keeps
/// the down row it had bit for bit, because [`NodeState::prune`] would walk the
/// same children in the same order with the same branch lengths and the kernels
/// are deterministic. `inherited` marks those and points them at the rows of
/// the tree the move started from; the rest are the ancestors of the cut and of
/// the attachment, `O(depth)` of them, and are recomputed here bottom-up
/// through the same kernels [`NodeState::prune`] dispatches to.
///
/// The test deciding which is which compares the children one for one and in
/// order, so a node the arena renumbered into a different child order fails it
/// and is recomputed. That is the conservative direction, and it is what makes
/// this safe on the regrafted tree, whose attachment raises its ancestors'
/// heights and so can reorder them among their siblings.
///
/// **Up rows.** [`crate::model::global::UpState::sweep`] writes a node's
/// children from the node's own up row, working down from the root, so the up
/// row at one node depends only on the chain of nodes above it. Filling that
/// chain and nothing else gives the same numbers as sweeping the whole arena,
/// which is why this is a different traversal of the same recursion rather
/// than a different recursion.
/// The chain is filled from the top down and not by recursing, so a ladder does
/// not put its depth on the stack.
///
/// `test_the_lazy_rows_match_a_full_settle` pins every row of both against a
/// full [`NodeState::prune`], a full [`crate::model::global::UpState::sweep`]
/// and a collapse of every node, bit for bit, at every node of both trees a
/// proposal builds. That is the gate this is allowed through on.
struct LazyRows<'a, T> {
    /// The tree the rows describe.
    tree: &'a Tree,
    /// Number of features.
    p: usize,
    /// Down rows of the tree the move started from.
    base: &'a NodeState<T>,
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
    /// Up rows, filled a chain at a time.
    up: Vec<OnceCell<Row<T>>>,
    /// Effective leaves, filled as the beam search asks for them.
    eff: Vec<OnceCell<Row<T>>>,
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
    /// anywhere but to a leaf. [`assemble`] never does, because it numbers the
    /// leaves it keeps out of the leaves it was given, but a row read through a
    /// map that did would silently be the wrong node's.
    fn new(
        tree: &'a Tree,
        to_old: &[u32],
        was: &Tree,
        base: &'a NodeState<T>,
    ) -> Result<Self, BonsaiErrors> {
        let p = base.n_features();
        let n = tree.n_nodes();

        // A node keeps its row when the cut left its whole subtree alone, which
        // is a local test once the children have been tested: same children, in
        // the same order, on the same branches, and each of them keeping its
        // own row. The arena orders children ascending and puts every child
        // below its parent, so one pass over `0..n` settles all of them.
        let mut inherited = vec![NO_NODE; n];
        for v in 0..n {
            let old = to_old[v];
            if old == NO_NODE || old as usize >= was.n_nodes() {
                continue;
            }
            if v < tree.n_leaves() {
                // A leaf's row is its own observation, so it is always
                // inherited and never recomputed; nothing here would know how.
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
            up: (0..n).map(|_| OnceCell::new()).collect(),
            eff: (0..n).map(|_| OnceCell::new()).collect(),
        };

        // Ascending node index is a valid post-order on the arena, so a dirty
        // node's dirty children are already done when it is reached.
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
            // Written aside and copied in, because the children borrow the
            // slab the answer goes into whenever one of them is dirty too.
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

    /// The tree loglikelihood, from the rows.
    ///
    /// The same terms [`NodeState::prune`] sums, read off the base state where
    /// the row was inherited and off the recomputed row where it was not. The
    /// association differs from the level-wise one `prune` uses, so the two
    /// agree to rounding and not to the bit; `test_the_incremental_loglik_matches_a_fresh_prune`
    /// pins how close.
    ///
    /// ### Returns
    ///
    /// The loglikelihood, up to the dropped additive constants.
    fn loglik(&self) -> f64 {
        let mut total = 0.0f64;
        for v in self.tree.n_leaves()..self.tree.n_nodes() {
            let old = self.inherited[v];
            total += if old != NO_NODE {
                self.base.contribution(old)
            } else {
                self.fresh_contrib[self.slot[v] as usize]
            };
        }
        total
    }

    /// The rows as a settled state of the tree they describe.
    ///
    /// Row for row what [`NodeState::prune`] would produce, because every row
    /// either is one of its rows or was formed through the kernels it
    /// dispatches to; a copy rather than a sweep, which is what makes accepting
    /// a move cheaper than proposing one.
    ///
    /// ### Returns
    ///
    /// The state, ready to propose against.
    fn into_state(self) -> NodeState<T> {
        let p = self.p;
        let n = self.tree.n_nodes();
        let mut m = vec![T::zero(); n * p];
        let mut w = vec![T::zero(); n * p];
        let mut contrib = vec![0.0f64; n];
        for v in 0..n {
            let (m_v, w_v) = self.down_row(v as u32);
            m[v * p..(v + 1) * p].copy_from_slice(m_v);
            w[v * p..(v + 1) * p].copy_from_slice(w_v);
            if v >= self.tree.n_leaves() {
                let old = self.inherited[v];
                contrib[v] = if old != NO_NODE {
                    self.base.contribution(old)
                } else {
                    self.fresh_contrib[self.slot[v] as usize]
                };
            }
        }
        NodeState::from_rows(p, self.tree.n_leaves(), m, w, contrib)
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
    fn down_row(&self, node: u32) -> (&[T], &[T]) {
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
    fn up_row(&self, node: u32) -> &Row<T> {
        let mut chain: Vec<u32> = Vec::new();
        let mut here = node;
        while self.up[here as usize].get().is_none() {
            chain.push(here);
            match self.tree.parent(here) {
                None => break,
                Some(a) => here = a,
            }
        }
        for &v in chain.iter().rev() {
            let row = self.compute_up(v);
            let _ = self.up[v as usize].set(row);
        }
        // Filled by the loop above, so the initialiser never runs; it is here
        // because `get` returns an option and this module does not unwrap.
        self.up[node as usize].get_or_init(|| self.compute_up(node))
    }

    /// One node's up row, from its parent's.
    ///
    /// A transcription of [`crate::model::global::UpState::sweep`]'s per-node
    /// step, for one child rather than for all of them at once. Both arms are
    /// the same expressions in the same order, which is what makes the answer
    /// the same bits.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree whose parent's up row is settled
    ///
    /// ### Returns
    ///
    /// The node's up row.
    fn compute_up(&self, node: u32) -> Row<T> {
        let p = self.p;
        let Some(a) = self.tree.parent(node) else {
            // The root has nothing outside it.
            return (
                vec![T::zero(); p].into_boxed_slice(),
                vec![T::zero(); p].into_boxed_slice(),
            );
        };
        let mut m_out = vec![T::zero(); p];
        let mut w_out = vec![T::zero(); p];

        let is_root = self.tree.parent(a).is_none();
        let t_a = self.tree.branch(a);
        let above = self.up_row(a);
        let (up_m, up_w) = (&above.0, &above.1);
        let kids = self.tree.children(a);
        let t_c = self.tree.branch(node);
        let (m_c, w_c) = self.down_row(node);

        if kids.len() == 2 {
            let other = if kids[0] == node { kids[1] } else { kids[0] };
            let t_o = self.tree.branch(other);
            let (m_o, w_o) = self.down_row(other);
            for g in 0..p {
                let (w_up, m_up) = up_part(is_root, t_a, up_w[g], up_m[g]);
                let wo = wide(w_o[g]);
                let wdo = wo / (1.0 + t_o * wo);
                let mo = wide(m_o[g]);
                let tot = wdo + w_up;
                w_out[g] = narrow(tot);
                m_out[g] = narrow(mo + (m_up - mo) * (w_up / tot));
            }
            return (m_out.into_boxed_slice(), w_out.into_boxed_slice());
        }

        let (m_a, w_a) = self.down_row(a);
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
        (m_out.into_boxed_slice(), w_out.into_boxed_slice())
    }

    /// The whole tree collapsed onto one node.
    ///
    /// Two rows meet at a node: the subtree below it, which
    /// [`NodeState::prune`] leaves in place, and everything outside that
    /// subtree, which the up sweep leaves sitting at the node's *parent* and
    /// which therefore has to be diffused down the branch above the node
    /// before the two can be combined. The root has no up part at all, so its row is
    /// its down row untouched, and that is guarded on the root rather than on a
    /// zero up precision because the root's own branch entry is not part of the
    /// tree and is allowed to hold anything at all, a NaN included.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the pruned tree
    ///
    /// ### Returns
    ///
    /// The effective leaf the beam search scores an attachment against.
    fn eff_leaf(&self, node: u32) -> EffLeaf<'_, T> {
        let rows = self.eff[node as usize].get_or_init(|| {
            let p = self.p;
            let is_root = self.tree.parent(node).is_none();
            let t = self.tree.branch(node);
            let (m_down, w_down) = self.down_row(node);
            let above = self.up_row(node);
            let (m_up, w_up) = (&above.0, &above.1);
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
        });
        EffLeaf {
            m: &rows.0,
            w: &rows.1,
        }
    }
}

/// The star around a node, read off rows rather than off a settled tree.
///
/// The same star [`centre_star`] builds, member for member and in the same
/// order, which matters because the primitive's pair scan and its tie-breaking
/// both read the member order.
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
fn lazy_centre_star<T: BonsaiFloat>(
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

/// One node's down row, from wherever it lives.
///
/// A free function rather than a method so the constructor can call it while it
/// is still filling the slab the dirty rows live in.
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

/// Hang a detached subtree back onto the remaining tree below `target`.
///
/// Attaching below an internal node is one more child. Attaching below a leaf
/// is not, because the arena has no data-carrying internal node: a new node
/// takes the leaf's place in the tree and the leaf hangs off it on a
/// zero-length branch, which puts the leaf's data at exactly the point the
/// attachment was scored at and is the same unrooted tree.
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
/// index of each of its nodes, or the error the arena failed with.
///
/// The fresh node an attachment below a leaf creates has no original index and
/// is [`NO_NODE`] in the map, which is what [`LazyRows`] reads as a node the
/// move made and has to settle.
fn regraft(
    pruned: &Pruned,
    x: u32,
    target: u32,
    branch: f64,
    n_leaves: usize,
) -> Result<(Tree, u32, Vec<u32>), BonsaiErrors> {
    let mut par = pruned.parent.clone();
    let mut len = pruned.branch.clone();
    // A root always has children, so it is never a leaf and this arm never
    // moves the root.
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
    let (tree, map) = assemble(&par, &len, pruned.root, n_leaves)?;
    let mut to_old = vec![NO_NODE; tree.n_nodes()];
    for (old, &new) in map.iter().enumerate() {
        if new != NO_NODE {
            to_old[new as usize] = old as u32;
        }
    }
    let centre = map[centre as usize];
    Ok((tree, centre, to_old))
}

/// A candidate move, proposed and scored but not yet decided.
struct Proposal {
    /// The tree the move produces.
    tree: Tree,
    /// Per node of `tree`, the node of the tree the move started from whose
    /// leaf set it shares, or [`NO_NODE`]; leaves map to themselves.
    to_old: Vec<u32>,
    /// The subtree that was pruned, as a node of the tree the move started
    /// from.
    pruned: u32,
    /// Loglikelihood of `tree`, from [`LazyRows::loglik`].
    loglik: f64,
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
/// A proposal whose splits match the current tree's is not a move at all, only
/// a reoptimisation of the branches the cut and the regraft touched, and is
/// dropped here; see the module docs for what accepting those costs. The rest
/// are scored by [`LazyRows::loglik`], which reads the current tree's rows
/// wherever the candidate's subtrees are the current tree's. Which they are
/// is settled by leaf set: the candidate's nodes are matched to the current
/// tree's by [`leaf_words`], and [`LazyRows`] then checks children and
/// branches one for one before trusting the match.
///
/// A proposal never touches the leaf data. Everything it reads is either the
/// arena or a row of the tree it starts from, which is what
/// [`crate::search::spr::LazyRows`] is for: the subtree is summarised by its
/// own down row, which pruning does not touch, and the two trees the move
/// builds are settled only at the nodes their own rows are asked for. The attachment
/// is followed by the polytomy resolution SPEC.md section 7.3 asks for, at the
/// attachment point and nowhere else: pruning only ever lowers a node's degree
/// and the ancestors a resolution creates are binary, so the attachment point
/// is the only node whose degree the move raised. That resolution reads six
/// rows and used to be handed them by settling the whole attached tree; see
/// [`attachment_star`] for where they come from instead.
///
/// ### Params
///
/// * `tree` - The tree; not modified
/// * `down` - Down rows, settled against `tree`
/// * `x` - Node to prune
/// * `params` - Knobs
/// * `here` - Split fingerprint of `tree`
/// * `by_word` - [`word_index`] of `tree`
///
/// ### Returns
///
/// The scored candidate, `None` if `x` may not be pruned or the move changes
/// no split, or the error the placement, the primitive or the arena failed
/// with.
fn propose<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    x: u32,
    params: &SprParams,
    here: u64,
    by_word: &FxHashMap<u64, u32>,
) -> Result<Option<Proposal>, BonsaiErrors> {
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
    let (attached, centre, to_old) = regraft(&pruned, x, target, best.branch, tree.n_leaves())?;

    // Settling the attached tree was the single most expensive thing a proposal
    // did, and the only thing it fed was this resolution. Neither is needed:
    // whether there is a resolution to do is a question about the centre's
    // degree, which the arena answers on its own, and the handful of rows it
    // reads if there is are the ones the regraft left alone plus the centre's
    // own ancestors.
    let members = attached.children(centre).len() + usize::from(attached.parent(centre).is_some());
    let candidate = if members <= crate::search::polytomy::RESOLVED_STAR_MEMBERS {
        attached
    } else {
        let attached_rows = LazyRows::new(&attached, &to_old, tree, down)?;
        let star = lazy_centre_star(&attached, &attached_rows, centre)?;
        splice_star(&attached, &star, Some(params.star))?.tree
    };

    let word = leaf_words(&candidate);
    let same = split_fingerprint_with(&candidate, &word) == here;
    if same {
        return Ok(None);
    }

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
    let loglik = LazyRows::new(&candidate, &to_old, tree, down)?.loglik();
    Ok(Some(Proposal {
        tree: candidate,
        to_old,
        pruned: x,
        loglik,
    }))
}

////////////
// Rounds //

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
            // Fisher-Yates downwards, so the number of draws is a function of
            // the candidate count alone and nothing depends on the thread
            // count.
            for i in (1..nodes.len()).rev() {
                let j = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
                nodes.swap(i, j);
            }
        }
    }
    nodes.iter().map(|&x| word[x as usize]).collect()
}

/// One sweep over the candidate subtrees, performing every improving move.
///
/// The order is fixed against the tree the sweep starts from, but the tree
/// changes under it: an accepted move renumbers the arena, so each candidate is
/// looked up by its [`leaf_words`] entry in the tree as it stands, and one
/// whose word no longer names a node has been swallowed by an earlier move and
/// is skipped. Everything the scan reads off the tree is settled once and
/// resettled only where a move actually moved it.
///
/// A candidate is scored by [`LazyRows::loglik`] on the tree it would produce,
/// which sums the same per-node terms a fresh [`NodeState::prune`] would and
/// recomputes only the ones the move changed. An accepted move is therefore an
/// improvement in the quantity that matters and the sweep is monotone in it.
/// The one figure that is not incremental is the result's, which is a fresh
/// prune of the tree the sweep finished on.
///
/// ### Chunks
///
/// The decisions are sequential and the proposals are not. A candidate must
/// be decided against the tree as it stands when its turn comes, and an
/// accepted move changes that tree, so a candidate proposed before an earlier
/// one was decided may have been proposed against the wrong tree. What makes
/// parallel proposal safe is refusing to use those: candidates are proposed a
/// chunk at a time against one tree, decided in order, and at the first
/// acceptance the rest of the chunk is discarded and proposed again against
/// the new tree. Every candidate that is decided was proposed against the tree
/// it is decided against, which is the sequential sweep's invariant, so the
/// result does not depend on the chunk size or the thread count. The chunk
/// halves on an acceptance and doubles on a clean chunk, between
/// [`PROPOSAL_CHUNK_MIN`] and [`PROPOSAL_CHUNK_MAX`], so the work an
/// acceptance discards tracks how often acceptances come.
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
    let params = params.unwrap_or_default();
    let mut rng = SplitMix64::new(params.seed);
    let mut tree = tree.clone();
    let mut gains = Vec::new();

    // All of these describe the tree as it stands, and a rejected candidate
    // leaves it exactly as it stands, so they are settled once and again only
    // when a move is accepted.
    let (mut down, mut best) = settled_down(&tree, leaves)?;
    let mut word = leaf_words(&tree);
    let mut here = split_fingerprint_with(&tree, &word);
    let mut by_word = word_index(&word);

    let order = candidate_order(&tree, &params, &mut rng);
    let mut next = 0usize;
    let mut chunk = PROPOSAL_CHUNK_MIN;
    while next < order.len() {
        let end = (next + chunk).min(order.len());
        let proposals: Vec<Option<Proposal>> = order[next..end]
            .par_iter()
            .map(|want| match by_word.get(want) {
                Some(&x) => propose(&tree, &down, x, &params, here, &by_word),
                None => Ok(None),
            })
            .collect::<Result<_, _>>()?;

        let mut accepted = None;
        for (k, proposal) in proposals.into_iter().enumerate() {
            let Some(proposal) = proposal else {
                continue;
            };
            if proposal.loglik <= best + acceptance_floor(&params, best) {
                continue;
            }
            gains.push(SprGain {
                pruned: proposal.pruned,
                gain: proposal.loglik - best,
            });
            best = proposal.loglik;
            let fresh = LazyRows::new(&proposal.tree, &proposal.to_old, &tree, &down)?.into_state();
            down = fresh;
            tree = proposal.tree;
            word = leaf_words(&tree);
            here = split_fingerprint_with(&tree, &word);
            by_word = word_index(&word);
            accepted = Some(k);
            break;
        }

        // The rest of a chunk that accepted was proposed against the tree
        // before the move, so it is proposed again against the tree after it.
        match accepted {
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

    let loglik = tree_loglik(&tree, leaves)?;
    Ok(SprResult {
        tree,
        loglik,
        gains,
        rounds: 1,
    })
}

/// Search step 5: sweep until a sweep finds nothing.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults
///
/// ### Returns
///
/// The tree the sweeps finished on, whose loglikelihood is never below the
/// input's, or the error the placement, the primitive or the arena failed with.
pub fn spr<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<SprParams>,
) -> Result<SprResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let mut out = SprResult {
        tree: tree.clone(),
        loglik: tree_loglik(tree, leaves)?,
        gains: Vec::new(),
        rounds: 0,
    };

    while out.rounds < params.max_rounds {
        // The random order has to differ between rounds, or the second round
        // retries the first round's order on a tree that has moved under it.
        let round = spr_round(
            &out.tree,
            leaves,
            Some(SprParams {
                seed: params.seed.wrapping_add(out.rounds as u64),
                ..params
            }),
        )?;
        out.rounds += 1;
        if round.gains.is_empty() {
            break;
        }
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
    /// The independent reference [`LazyRows`] is checked against, and nothing
    /// else: the search reads a few dozen of these rows per candidate subtree,
    /// and forming all `n` of them was a large part of what made step 5
    /// quadratic.
    ///
    /// Two rows meet at a node: the subtree below it, which
    /// [`NodeState::prune`] leaves in place, and everything outside that
    /// subtree, which the up sweep leaves sitting at the node's *parent* and
    /// which therefore has to be diffused down the branch above the node
    /// before the two can be combined. The root has no up-part at all, so its
    /// row is its down row untouched.
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
                // Guarded on the root rather than on a zero up precision: the
                // root's own branch entry is not part of the tree and is allowed to
                // hold anything at all, a NaN included.
                let w_above = if is_root {
                    0.0
                } else {
                    let w = wide(w_up[g]);
                    w / (1.0 + t * w)
                };
                let below = wide(w_down[g]);
                let total = below + w_above;
                let m = wide(m_down[g]);
                // A convex combination, so the result is pinned between the two
                // means it interpolates and cannot cancel.
                means[base + g] = narrow(m + (wide(m_up[g]) - m) * (w_above / total));
                precisions[base + g] = narrow(total);
            }
        }
        (means, precisions)
    }

    /// Settle a tree's down and up rows, over the whole arena.
    ///
    /// The independent reference [`LazyRows`] is checked against, and nothing else:
    /// the search reads a few dozen of these rows per candidate subtree and
    /// sweeping all `n` of them, twice per proposal, was most of what made step 5
    /// quadratic.
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
    /// The generating tree with its branch lengths optimised is not that shape.
    /// The merge scan numbers the arena its own way and leaves polytomies
    /// behind that step 3 does not always clear, and both of those are what the
    /// proposals here have to cope with.
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
        let mut candidates = EllipsoidBounds::new(KnnCandidates::new(None), None);
        let (tree, _) = star_tree_with(
            Star {
                means: leaves.means,
                precisions: leaves.precisions,
                branch: &star.branches()[..n_leaves],
                n_features: p,
            },
            None,
            &mut candidates,
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
    /// [`prune_subtree`] used to gather this and no longer does, because the
    /// search does not settle the remaining tree any more. The tests that
    /// settle it as an independent reference still need it.
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
        // The one that matters. Every round is scored by an independent
        // `NodeState::prune` of the tree it produced, never by an incremental
        // figure, and the start is a deliberately wrong topology so that there
        // is plenty for the sweep to do.
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
                // Relative, not absolute: both sides are sums of magnitude
                // `O(n p)`, so an absolute tolerance is a fixture-size
                // tolerance and stops meaning anything on real data.
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
            let out = spr(&start, leaves, None).expect("spr");
            let after = tree_loglik(&out.tree, leaves).expect("loglik");
            assert!(after >= before, "seed {seed}: {before} fell to {after}");
            assert_relative_eq!(after, out.loglik, max_relative = 1e-12);
            assert!(out.rounds >= 1);
        }
    }

    #[test]
    fn test_prune_and_regraft_where_it_came_from_is_the_original_tree() {
        // The mechanics on their own: detach a subtree whose parent survives
        // the cut, then put it straight back with the branch it had. Anything
        // wrong in the detach, the compaction or the reassembly shows up here
        // as a changed branch length or a changed split.
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
        // The gate the whole optimisation rests on. Every row `LazyRows`
        // hands out has to be the row a full settle of the remaining tree
        // followed by a full collapse would hand out, bit for bit at every
        // node, on shapes with polytomies and without, and for both storage
        // types. Anything looser and the beam search could pick a different
        // attachment point and step 5 would quietly find different trees.
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

                // And again on the tree the regraft builds, which is settled
                // from the same base and whose ancestors the arena is free to
                // renumber.
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
            // The inherited rows are the point, but the recomputed ones are
            // where a mistake would live, so the fixtures have to reach them.
            assert!(dirty_seen > 0, "no row was ever recomputed");
        }
    }

    #[test]
    fn test_the_lazy_rows_only_form_what_is_asked_for() {
        // The saving is the rows that are never formed, so it is worth a test
        // of its own that they are not: a proposal on a balanced tree of 512
        // nodes must touch a small fraction of them, not all of them. This is
        // the measurement `PlacementParams::tolerance` records from the other
        // side, and it is what turned the sweep's `O(n p)` per candidate into
        // something that grows with the depth and the beam instead.
        let p = 24usize;
        let (data, w) = dataset(256, p, 17);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&data.tree, leaves);
        let down = settled_down(&tree, leaves).expect("down").0;

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
            total += rows.eff.len();
            formed += rows.eff.iter().filter(|c| c.get().is_some()).count();
        }
        assert!(total > 0);
        // Not an invariant, a measurement: the beam forms a small fraction of
        // the arena. A regression here is a regression in the running time even
        // though every answer is still right, which no other test would catch.
        assert!(
            formed * 10 < total,
            "the beam formed {formed} of {total} effective leaves"
        );
    }

    #[test]
    fn test_the_polytomy_threshold_matches_the_primitive() {
        // `propose` decides from the arena alone whether an attachment left a
        // polytomy, rather than settling the attached tree and asking the star
        // it builds. Both now read the same `RESOLVED_STAR_MEMBERS`, so the
        // threshold can no longer drift, but the two decision paths are still
        // independent and this is what says they agree.
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
        // The gate on the whole optimisation. The star the resolution is handed
        // has to be the star a freshly settled attached tree gives, bit for bit
        // and not merely close: the primitive picks its pair by a strict
        // comparison, so a difference in the last place can pick a different
        // pair and hand back a different tree.
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
        // Pruning leaf 0 leaves node 5 with one child, which the arena cannot
        // hold. The suppression has to join leaf 1 to node 6 with `b1 + b5`;
        // the reference tree here is that join written out by hand.
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

        // Leaves 1, 2, 3, 4 become 0, 1, 2, 3; node 6 becomes 4 and node 7 the
        // root. Leaf 0 of the reference carries `b1 + b5`.
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
        // The root's own slot is not a branch of the tree and is not compared.
        let last = reference.n_nodes() - 1;
        assert_eq!(pruned.tree.branches()[..last], reference.branches()[..last]);
        assert_relative_eq!(
            tree_loglik(&pruned.tree, rest).expect("loglik"),
            tree_loglik(&reference, rest).expect("loglik"),
            epsilon = 1e-12
        );

        // The test bites: dropping the suppressed node's own branch changes the
        // answer, so the assertion above is not passing on topology alone.
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
        // Pruning a child of a three-way root leaves the root with two
        // children, which scores correctly but can never be pruned from again.
        // It is joined the same way, by making the internal child the root.
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

        // Four leaves, two internal nodes: the old root is gone rather than
        // left behind with two children.
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
        // Asserted directly: the beam search runs on `pruned.tree`, so what has
        // to be true is that no node of that tree maps back to anything inside
        // the subtree. Checked at every eligible subtree of a real fixture.
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
            // Nothing else went missing either: the remaining tree holds every
            // node outside the subtree bar at most the one the suppression
            // removed.
            let lost = tree.n_nodes() - inside.len() - pruned.tree.n_nodes();
            assert!(lost <= 1, "pruning {x} lost {lost} nodes");
            checked += 1;
        }
        assert!(checked > 0, "the fixture had no eligible subtree");
    }

    #[test]
    fn test_the_node_map_survives_the_arena() {
        // `assemble` numbers its input so that `Tree::from_parents` relabels
        // nothing, which is the only reason the map it returns describes the
        // tree that came back. If that ever stops being true this fails.
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
        // The seam between this module and `model::place`, checked end to end
        // and not by inspection. SPEC.md section 7.1 says the loglikelihood of
        // the regrafted tree is the remaining tree's, plus the subtree's, plus
        // the attachment score, and only the last of those depends on where the
        // subtree lands. So the difference between the two has to be the same
        // constant at every candidate target, leaf targets with their
        // zero-length insertion included.
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
        // A ladder is about as wrong as a starting topology gets for data
        // generated on a balanced tree. Branch lengths first, as the spec's
        // step order has it: the interchanges of step 6 found the same, that a
        // topology search reading unoptimised branch lengths is reading noise.
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
            let out = spr(&start, leaves, None).expect("spr");
            let after = robinson_foulds(&out.tree, &data.tree).expect("rf");
            // The distance goes to zero on every fixture here. The assertion
            // is the measurement rather than a hedge around it, so a regression
            // that merely improves the tree shows up here instead of passing
            // quietly.
            assert_eq!(
                after, 0,
                "seed {seed}: Robinson-Foulds {before} fell only to {after}"
            );
            assert!(out.rounds <= 5, "seed {seed} took {} rounds", out.rounds);
        }
    }

    #[test]
    fn test_the_two_orders_reach_the_same_topology() {
        // The paper's ordering is inherited rather than derived, so it is
        // checked rather than assumed. Both orders recover the generating tree;
        // see `PruneOrder::LongestBranch` for what separates them, which is
        // move count and the final loglikelihood rather than whether they get
        // there.
        let (p, n) = (256usize, 32usize);
        for seed in [1u64, 2] {
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let ordered = spr(&start, leaves, None).expect("spr");
            let random = spr(
                &start,
                leaves,
                Some(SprParams {
                    order: PruneOrder::Random,
                    seed: 11,
                    ..SprParams::default()
                }),
            )
            .expect("spr");
            assert_eq!(robinson_foulds(&ordered.tree, &data.tree).expect("rf"), 0);
            assert_eq!(robinson_foulds(&random.tree, &data.tree).expect("rf"), 0);
            assert!(ordered.n_moves() <= random.n_moves());
        }
    }

    #[test]
    fn test_a_tree_too_small_for_a_legal_move() {
        // Two leaves under one root: both are children of a degree-two root, so
        // nothing may be pruned and the sweep has nothing to do.
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
        let out = spr(&tree, leaves, None).expect("spr");
        assert_eq!(out.n_moves(), 0);
        assert_eq!(out.rounds, 1);
        assert_eq!(splits(&out.tree), splits(&tree));
    }

    #[test]
    fn test_a_star_tree() {
        // Every leaf of a star is prunable and every regraft lands on a root
        // that is still a polytomy, so each move drags the section 9.2
        // resolution in behind it. What matters is that it stays monotone and
        // comes back a tree.
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

        let out = spr(&star, leaves, None).expect("spr");
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
            )
            .expect("propose") else {
                continue;
            };
            assert_eq!(candidate.tree.n_leaves(), tree.n_leaves());
            // Regrafting is a proposal, not an acceptance, so it is allowed to
            // come back worse; what it may not come back is malformed.
            let _ = tree_loglik(&candidate.tree, leaves).expect("loglik");
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_the_incremental_loglik_matches_a_fresh_prune() {
        // Every candidate a round proposes is scored off the current tree's
        // rows plus the few the move changed, and accepted on that figure. So
        // the figure has to be the fresh prune's to rounding, and the state an
        // acceptance builds from the same rows has to be the fresh prune's to
        // the bit. A ladder start on a small tree makes nearly every candidate
        // a real move, so the filter passes most of them through.
        let (p, n) = (96usize, 32usize);
        let (data, w) = dataset(n, p, 5);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let (down, _, _) = settle(&tree, leaves).expect("settle");
        let word = leaf_words(&tree);
        let here = split_fingerprint_with(&tree, &word);
        let by_word = word_index(&word);

        let mut checked = 0usize;
        for x in 0..tree.n_nodes() as u32 {
            let Some(proposal) =
                propose(&tree, &down, x, &SprParams::default(), here, &by_word).expect("propose")
            else {
                continue;
            };
            let (fresh, loglik) = settled_down(&proposal.tree, leaves).expect("settle");
            assert_relative_eq!(proposal.loglik, loglik, max_relative = 1e-12);
            let built = LazyRows::new(&proposal.tree, &proposal.to_old, &tree, &down)
                .expect("rows")
                .into_state();
            for v in 0..proposal.tree.n_nodes() as u32 {
                assert_eq!(
                    built.means(v),
                    fresh.means(v),
                    "means at node {v} of candidate {x}"
                );
                assert_eq!(built.precisions(v), fresh.precisions(v));
            }
            checked += 1;
        }
        assert!(checked > n, "only {checked} candidates exercised");
    }

    #[test]
    fn test_the_sweep_is_deterministic_whatever_the_thread_count() {
        // Proposals run a chunk at a time in parallel and an acceptance throws
        // the rest of its chunk away, so the sweep has to come out the same
        // however the chunks were scheduled. A ladder is the fixture with real
        // work in it: tens of accepted moves, most of them inside a chunk.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let reference = spr_round(&start, leaves, None).expect("round");
        // Some sixty candidates and a chunk of thirty-two, so this many moves
        // cannot all fall on chunk boundaries.
        assert!(
            reference.n_moves() >= 8,
            "the fixture has to accept inside chunks, got {} moves",
            reference.n_moves()
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
            assert_eq!(got.n_moves(), reference.n_moves());
            for (a, b) in got.gains.iter().zip(&reference.gains) {
                assert_eq!(a.pruned, b.pruned);
                assert_eq!(a.gain.to_bits(), b.gain.to_bits());
            }
        }
    }

    #[test]
    fn test_a_sweep_that_improves_nothing_changes_nothing() {
        // The generating tree with its branch lengths at the step 4 optimum:
        // there is no subtree whose regraft beats it, so the sweep has to come
        // back with the tree it was given rather than with a reshuffle worth
        // nothing.
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
            assert_eq!(out.n_moves(), 0, "seed {seed} moved a tree it should not");
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
                pool.install(|| spr(&start, leaves, Some(params)).expect("spr"))
            })
            .collect();

        for run in &runs[1..] {
            assert_eq!(run.tree.branches(), runs[0].tree.branches());
            assert_eq!(splits(&run.tree), splits(&runs[0].tree));
            assert_eq!(run.loglik.to_bits(), runs[0].loglik.to_bits());
            assert_eq!(run.n_moves(), runs[0].n_moves());
        }
    }

    /// Magnitude of the whole-tree loglikelihood on the realistic dataset an
    /// absolute floor failed on, ten thousand cells by a few thousand genes.
    /// Used as the extrapolation target, so the scaling test is checked against
    /// the regime that broke rather than the one it runs in.
    const REALISTIC_LOGLIK: f64 = 1.1e7;

    /// Worst gain accepted on that dataset that turned out to be rounding.
    const REALISTIC_NOISE: f64 = 5.96e-8;

    #[test]
    fn test_the_acceptance_floor_outgrows_the_loglikelihood_rounding_floor() {
        // A candidate is accepted on `proposal.loglik - best`, and both sides
        // are sums over every internal node, so the smallest difference the
        // arithmetic can resolve grows with `|L|`, which is `O(n p)`. An
        // absolute floor therefore has a size above which it sits below the
        // noise, and the search accepts neutral topology changes for ever.
        // That is not hypothetical: with the floor at `StarParams::min_gain`
        // alone it happened at 10,000 cells, and it was the cap on the rounds
        // that ended the run.
        //
        // Fixtures at this size cannot reach that magnitude, so the noise is
        // measured over a range of sizes, its growth law is checked, and the
        // floor is then tested against the noise extrapolated to the magnitude
        // that broke. That extrapolation is the part a 64-leaf test does not
        // have and the reason this one exists.
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
            let word = leaf_words(&tree);
            let here = split_fingerprint_with(&tree, &word);
            let by_word = word_index(&word);

            // The noise is the disagreement between the incrementally
            // assembled figure a candidate is accepted on and a fresh prune of
            // the very same tree. Anything below it is not a gain, it is which
            // way the last bit fell.
            let mut noise = 0.0f64;
            let mut seen = 0usize;
            for x in 0..tree.n_nodes() as u32 {
                let Some(proposal) =
                    propose(&tree, &down, x, &params, here, &by_word).expect("propose")
                else {
                    continue;
                };
                let (_, fresh) = settled_down(&proposal.tree, leaves).expect("settle");
                noise = noise.max((proposal.loglik - fresh).abs());
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

        // Relative noise over the four rungs grew as the square root of `|L|`,
        // which is the random walk an unordered `f64` sum accumulates: 1.2e-16
        // at |L| = 4.7e2 to 6.2e-16 at 1.1e4, a factor of 5 over a factor of
        // 25. Pin the law rather than the numbers, then extrapolate on it.
        //
        // A rung can come out at exactly zero when both sums round the same
        // way, as the smallest one does on Windows. Zero means below one ulp,
        // not no noise, so the base is floored at unit roundoff rather than
        // divided by.
        let (l0, rel0) = rungs[0];
        let rel0 = rel0.max(f64::EPSILON / 2.0);
        let (l1, rel1) = *rungs.last().expect("rungs");
        assert!(
            rel1 / rel0 < 4.0 * (l1 / l0).sqrt(),
            "relative noise grew faster than the square root: {rel0:.3e} to {rel1:.3e}"
        );
        // Ten, not a hundred, because the projection is itself pessimistic:
        // it lands on 2.2e-7 where the value measured on the real 10k dataset
        // was 6.0e-8, so a small fixture over-predicts the noise by about 4x
        // and the margin below is against that inflated figure. The hundredfold
        // check is made directly against the measurement, next.
        let projected = rel1 * (REALISTIC_LOGLIK / l1).sqrt() * REALISTIC_LOGLIK;
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) > 10.0 * projected,
            "extrapolated to |L| = {REALISTIC_LOGLIK:.3e}: floor {:.3e} against noise \
             {projected:.3e}",
            acceptance_floor(&params, -REALISTIC_LOGLIK)
        );

        // And against what was actually observed there, rather than projected.
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) > 100.0 * REALISTIC_NOISE,
            "floor {:.3e} against the measured 10k noise {REALISTIC_NOISE:.3e}",
            acceptance_floor(&params, -REALISTIC_LOGLIK)
        );

        // The floor has to stay far below a move that carries information. The
        // smallest gain accepted on any rung of the subsample ladder was 2e-5
        // nats at |L| = 1.7e6, so scale that down as the floor scales and check
        // the margin holds.
        assert!(
            acceptance_floor(&params, -REALISTIC_LOGLIK) < 2e-5 * (REALISTIC_LOGLIK / 1.7e6),
            "the floor would reject a real move"
        );
    }

    #[test]
    fn test_the_sweeps_reach_a_fixed_point_rather_than_cycling() {
        // The failure the floor allows is not a crash, it is a cycle: every
        // round accepts one neutral move, the tree changes, the loglikelihood
        // does not, and the run ends on the cap. So the test is that `spr`
        // stops on its own and that a second run from its output is the
        // identity, at several sizes rather than one.
        for (n, p) in [(16usize, 64usize), (32, 128), (64, 256)] {
            let (data, w) = dataset(n, p, 7);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = searched(n, leaves);
            let params = SprParams::default();
            let first = spr(&start, leaves, Some(params)).expect("spr");
            assert!(
                first.rounds < params.max_rounds,
                "{n} by {p} ran to the cap at {} rounds",
                first.rounds
            );
            let again = spr(&first.tree, leaves, Some(params)).expect("spr");
            assert_eq!(
                again.n_moves(),
                0,
                "{n} by {p} kept moving after it stopped"
            );
            assert_eq!(again.tree.branches(), first.tree.branches());
            assert_eq!(again.loglik.to_bits(), first.loglik.to_bits());
        }
    }
}
