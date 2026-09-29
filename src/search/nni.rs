//! Nearest-neighbour interchanges, generalised to polytomies
//! (SPEC.md section 9.4).
//!
//! The move is three lines: pick an internal edge `k-l` with both ends
//! internal, delete `k` and move all of its subtrees onto `l`, then run the
//! star primitive on `l`. For four subtrees that is exactly the classical
//! interchange, and `test_a_classical_interchange_reconnects_four_subtrees`
//! pins it.
//!
//! ### Why the collapse costs nothing
//!
//! Deleting `k` changes the tree only strictly inside `l`'s subtree and
//! strictly above `k`'s children. So the effective leaf of every subtree the
//! star needs is already in the *original* tree's settled rows: `l`'s remaining
//! children and `k`'s children keep their down rows, and `l`'s upstream side
//! keeps its up row. Nothing is repruned to propose a move, and one settled
//! pair of sweeps serves every edge of a round. What the collapse does change
//! is the branch to the centre for each of `k`'s children, which becomes
//! `t_c + t_k`: diffusion times add along a path, so that is the length that
//! leaves each subtree where it was.
//!
//! ### Two phases
//!
//! The random phase samples the pair to merge inside the star rather than
//! taking the best ([`StarSelection::Weighted`]) and accepts the result
//! unconditionally, which is meant to escape a local optimum. How much it
//! actually escapes depends hard on the feature count, and at the counts this
//! crate expects the answer is "not much": see [`StarSelection::Weighted`] for
//! the measurement. The greedy phase scores an interchange at every eligible
//! edge, performs the best, and repeats until none improves the tree.
//!
//! **Monotonicity.** The greedy phase never lowers the tree loglikelihood: a
//! move is accepted only when its exact gain over the current tree clears
//! [`StarParams::min_gain`], and that gain is the difference of two whole-tree
//! loglikelihoods computed without sweeping either of them (see
//! [`collapse_delta`]). The random phase gives no such guarantee and is not
//! meant to; the collapse alone can lose a split that the resampled star does
//! not put back.
//!
//! The gain is a *difference* of whole-tree loglikelihoods but it is never
//! formed as one. Every term the two trees share cancels symbolically rather
//! than numerically: the merge gains are `O(p)` sums over one pair each, and
//! [`collapse_delta`] is three `O(p)` peels over the members of one star. So
//! the rounding floor of an interchange gain is `O(p eps)` and not
//! `O(n p eps)`, and [`StarParams::min_gain`] is the right floor for it at any
//! `n`. [`crate::search::spr`] scores its candidates the other way, on the
//! whole-tree figure, and needs a floor that scales with `n p`. Measured on
//! realistic data at ten thousand cells, every accepted interchange here gained
//! well clear of a nat and none was rounding.
//!
//! ### This is a topology search and only a topology search
//!
//! A collapse and re-resolution that puts the same subtrees back where they
//! were is not an interchange at all: it is a reoptimisation of the three
//! branches the star primitive creates at `l`, and it nearly always gains a
//! little. Accepting those turns the greedy phase into an extremely expensive
//! branch-length descent: started from the generating tree itself it runs for
//! hundreds of rounds with the Robinson-Foulds distance to the truth pinned at
//! zero throughout. So a proposal is discarded unless it changes the splits;
//! branch lengths are search steps 4 and 7, which do the same job globally and
//! for a fraction of the cost.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::search::live::{LiveTree, MoveEdit, Topology};
use crate::search::polytomy::{CentreStar, Splice, splice_edits, splice_star};
use crate::search::spr::{Row, RowStore, UpSide, assemble, root_up, up_step};
use crate::search::star::{StarParams, StarResult, StarSelection, resolve_star};
use crate::search::{Leaves, leaf_words, leaves_below, settle, settled_down, tree_loglik};
use crate::tree::{NO_NODE, Tree};
use crate::utils::kernels::prune_general;
use crate::utils::rng::SplitMix64;
use crate::utils::simd::prune_binary;
use crate::utils::traits::BonsaiFloat;
use crate::utils::verbosity::Verbosity;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::time::Instant;

////////////////
// Parameters //
////////////////

/// Smallest star an interchange can do anything with.
///
/// One more than the three members the star primitive stops at: a collapse that
/// leaves three members has nothing to merge, and splicing it back would only
/// throw away the split that `k` carried. Those edges are skipped rather than
/// proposed and rejected.
const MIN_INTERCHANGE_MEMBERS: usize = 4;

/// Default for [`NniParams::max_rounds`].
///
/// A runaway guard and not a working limit. Every accepted greedy move raises
/// the tree loglikelihood by more than [`StarParams::min_gain`] and the
/// loglikelihood is bounded above, so the phase terminates on its own; this
/// only bounds how long it can take to notice.
///
/// One round performs one move, so the requirement is the number of
/// interchanges between the starting topology and the local optimum, and that
/// grows with the leaf count. Measured from a ladder at the step 4 optimum,
/// running to a Robinson-Foulds distance of zero from the generating tree, it
/// comes out a shade under one round per leaf. The default therefore covers
/// roughly ten thousand leaves; beyond that a caller should raise it rather
/// than accept a truncated search.
const DEFAULT_MAX_ROUNDS: usize = 10_000;

/// Default for [`NniParams::n_random`].
///
/// Zero: the random phase is opt-in. It is a diversification budget traded
/// against wall time, nothing in SPEC.md fixes one, and any constant here would
/// be a number this crate invented and then had to defend.
///
/// It is also weaker than it looks at the feature counts this crate expects.
/// The sampling weight is a softmax over tree loglikelihoods, whose gaps are
/// `O(p)` nats, so it concentrates on the greedy pick as `p` grows. Measured
/// over seeds at a fixed leaf count, the fraction that move the tree off its
/// starting topology at all falls away by a couple of hundred features. So a
/// budget buys real diversification on small feature sets and mostly buys
/// branch-length reoptimisation on large ones.
const DEFAULT_RANDOM_MOVES: usize = 0;

/// Default for [`NniParams::n_restarts`].
///
/// Zero: one random phase, if any, then one greedy phase, which is the
/// composition SPEC.md section 9.4 describes. Measured against iterated local
/// search; see [`nni`].
const DEFAULT_RESTARTS: usize = 0;

/// Default for [`NniParams::temperature`].
///
/// One is the specification's distribution: weights proportional to the
/// likelihood of the resulting tree. Ours only in the sense of being the
/// neutral setting; see [`nni_random`].
pub const DEFAULT_RANDOM_TEMPERATURE: f64 = 1.0;

/// Default for [`NniApprox::rescore_radius`].
///
/// Measured 2026-09-25, steps 5 to 8 from the same step-4 tree with the
/// default SPR, scored against the generating tree, on balanced,
/// random-branch and unbalanced trees at noise 0.4, 1.0 and 1.6 at 4,096 by
/// 1,000 and on the four Sanity-preprocessed configurations. At radius five
/// the finished tree matched the exact phase on all thirteen: loglikelihood,
/// Robinson-Foulds and distance recovery to the last printed digit. Step 6
/// went 7.6 s to 1.5 s, 12.0 s to 2.1 s and 13.2 s to 2.3 s on the three noisy
/// synthetic trees, 29.6 s to 5.9 s at 5,000 cells and 99.4 s to 16.5 s at
/// 10,000. On step 6 alone, radius three lost 7.5 nats at 5,000 cells and
/// radius two 0.01; five was identical at both sizes.
const DEFAULT_RESCORE_RADIUS: usize = 5;

/// Knobs of the approximate greedy phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NniApprox {
    /// After a move, rescore only the edges within this many edges of a clade
    /// the move created, and keep every other edge's gain from the round it
    /// was last scored in.
    ///
    /// Lazy greedy evaluation (Minoux, *Optimization Techniques*, 1978): the
    /// leading cached gain is rescored on the current tree before it is taken,
    /// and taken only if it still beats the runner-up, so every accepted move
    /// is scored exactly. What a stale gain can do is hide an edge that has
    /// become improving; a full scan runs whenever the cache has nothing left,
    /// so the phase still stops only on a tree where no edge improves.
    pub rescore_radius: usize,
}

impl Default for NniApprox {
    /// Every approximation at its default.
    ///
    /// ### Returns
    ///
    /// The default knobs.
    fn default() -> Self {
        Self {
            rescore_radius: DEFAULT_RESCORE_RADIUS,
        }
    }
}

/// The specified greedy phase, or this crate's approximation of it.
///
/// Same shape as [`crate::search::spr::SprSearch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NniSearch {
    /// Score every eligible edge every round and take the best, as SPEC.md
    /// section 9.4 specifies.
    Exact,
    /// Lazy rescoring, see [`NniApprox`]. The default: the same finished
    /// tree as [`NniSearch::Exact`] on every dataset measured, and five to
    /// eight times faster on the noisy ones.
    Approximate(NniApprox),
}

impl Default for NniSearch {
    /// [`NniSearch::Approximate`] at its measured defaults.
    ///
    /// ### Returns
    ///
    /// The default search.
    fn default() -> Self {
        Self::Approximate(NniApprox::default())
    }
}

/// Tuning knobs for search step 6.
#[derive(Clone, Copy, Debug)]
pub struct NniParams {
    /// Star primitive knobs. The greedy phase uses these as they stand; the
    /// random phase overrides [`StarParams::selection`] with its own seed per
    /// move and leaves everything else alone.
    pub star: StarParams,
    /// Number of randomised moves performed before the greedy phase.
    pub n_random: usize,
    /// Seed for the random phase, unused when `n_random` is zero.
    pub seed: u64,
    /// Cap on greedy rounds.
    pub max_rounds: usize,
    /// Number of perturb-and-climb repeats after the first greedy phase, each
    /// of `n_random` random moves from the best tree so far followed by a
    /// greedy phase, keeping the best tree seen. Zero runs the two phases once
    /// each, in the order the specification gives them.
    pub n_restarts: usize,
    /// Softmax temperature of the random phase's pair draw.
    pub temperature: f64,
    /// Whether the greedy phase rescores every edge every round.
    pub search: NniSearch,
}

impl Default for NniParams {
    /// The default star knobs, no random phase and `DEFAULT_MAX_ROUNDS`.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            star: StarParams::default(),
            n_random: DEFAULT_RANDOM_MOVES,
            seed: 0,
            max_rounds: DEFAULT_MAX_ROUNDS,
            n_restarts: DEFAULT_RESTARTS,
            temperature: DEFAULT_RANDOM_TEMPERATURE,
            search: NniSearch::default(),
        }
    }
}

////////////
// Output //
////////////

/// What a run of the interchanges did.
#[derive(Clone, Debug)]
pub struct NniResult {
    /// The tree the phase finished on.
    pub tree: Tree,
    /// Its loglikelihood, from [`NodeState::prune`] on the tree itself.
    ///
    /// The exact greedy phase settles the tree at the top of every round and
    /// stops on a round that finds no move, so what comes back is that round's
    /// own sweep; a run truncated by [`NniParams::max_rounds`] returns the last
    /// sweep plus the accepted gains instead. The approximate phase sweeps once
    /// at the end.
    pub loglik: f64,
    /// Number of moves performed.
    pub n_moves: usize,
    /// Number of greedy rounds, the last of which found no improving move.
    /// Zero for a run of the random phase alone.
    pub rounds: usize,
}

/// What scanning one edge produced: its exact gain when it changes a split,
/// minus infinity otherwise, and the proposal when that gain clears the floor.
type ScannedEdge<T> = (f64, Option<(f64, u32, CentreStar<T>)>);

//////////
// Rows //
//////////

/// Where an interchange reads its effective leaves from.
///
/// A settled pair of sweeps, or [`LazyRows`] over a store kept current move by
/// move. Both give the same bits for every row ([`LazyRows`] is gated on that
/// against a full settle), so which one a caller hands in changes the cost and
/// never the answer.
trait EdgeRows<T> {
    /// Down row of a node: its subtree collapsed onto it.
    ///
    /// ### Params
    ///
    /// * `node` - Node of the tree the rows describe
    ///
    /// ### Returns
    ///
    /// Effective means and precisions.
    fn down(&self, node: u32) -> (&[T], &[T]);

    /// Up row of a node: everything outside its subtree, at its parent.
    ///
    /// ### Params
    ///
    /// * `node` - Non-root node of the tree the rows describe
    ///
    /// ### Returns
    ///
    /// Effective means and precisions.
    fn up(&self, node: u32) -> (&[T], &[T]);
}

/// A settled pair of sweeps, as [`settle`] returns them.
struct Settled<'a, T> {
    /// Down rows.
    down: &'a NodeState<T>,
    /// Up rows.
    up: &'a UpState<T>,
}

impl<T: BonsaiFloat> EdgeRows<T> for Settled<'_, T> {
    fn down(&self, node: u32) -> (&[T], &[T]) {
        (self.down.means(node), self.down.precisions(node))
    }

    fn up(&self, node: u32) -> (&[T], &[T]) {
        (self.up.means(node), self.up.precisions(node))
    }
}

/// Down rows from a store kept current move by move, up rows formed for the
/// round.
struct LiveRows<'a, T> {
    /// Down rows by node id.
    store: &'a RowStore<T>,
    /// Up rows of the nodes the round reads, by node id.
    up: &'a FxHashMap<u32, Row<T>>,
}

impl<T: BonsaiFloat> EdgeRows<T> for LiveRows<'_, T> {
    fn down(&self, node: u32) -> (&[T], &[T]) {
        (self.store.means(node), self.store.precisions(node))
    }

    fn up(&self, node: u32) -> (&[T], &[T]) {
        let row = &self.up[&node];
        (&row.0, &row.1)
    }
}

///////////////
// One move //
///////////////

/// Size of the star an interchange at `k` would build.
///
/// ### Params
///
/// * `tree` - The tree
/// * `k` - The node that would be deleted, the lower end of the edge
///
/// ### Returns
///
/// The member count, or `None` if the edge is not eligible: `k` is the root or
/// a leaf, or the collapse leaves too few members to merge.
fn interchange_members(tree: &impl Topology, k: u32) -> Option<usize> {
    let l = tree.parent(k)?;
    if tree.children(k).is_empty() {
        return None;
    }
    let n =
        tree.children(l).len() - 1 + tree.children(k).len() + usize::from(tree.parent(l).is_some());
    (n >= MIN_INTERCHANGE_MEMBERS).then_some(n)
}

/// Build the star at `l` that deleting `k` leaves behind.
///
/// The members are `l`'s other children on their own branches, then `k`'s
/// children on `t_c + t_k`, then `l`'s upstream side. All of them are read off
/// the *original* tree's settled rows; see the module docs for why that is
/// sound.
///
/// ### Params
///
/// * `tree` - The tree
/// * `rows` - Its down and up rows
/// * `k` - The node to delete
///
/// ### Returns
///
/// The star, or `None` if the edge is not eligible.
fn collapsed_star<T: BonsaiFloat>(
    tree: &impl Topology,
    rows: &impl EdgeRows<T>,
    k: u32,
) -> Option<CentreStar<T>> {
    let n = interchange_members(tree, k)?;
    let l = tree.parent(k)?;
    let p = rows.down(k).0.len();
    let t_k = tree.branch(k);
    let above = tree.parent(l);

    let mut star = CentreStar {
        centre: l,
        member_nodes: Vec::with_capacity(n),
        has_upstream: above.is_some(),
        deleted: vec![k],
        means: Vec::with_capacity(n * p),
        precisions: Vec::with_capacity(n * p),
        branch: Vec::with_capacity(n),
        n_features: p,
    };

    let push = |node: u32, branch: f64, star: &mut CentreStar<T>| {
        let (m, w) = rows.down(node);
        star.member_nodes.push(node);
        star.means.extend_from_slice(m);
        star.precisions.extend_from_slice(w);
        star.branch.push(branch);
    };

    for &child in tree.children(l) {
        if child != k {
            push(child, tree.branch(child), &mut star);
        }
    }
    for &child in tree.children(k) {
        push(child, tree.branch(child) + t_k, &mut star);
    }
    if let Some(par) = above {
        let (m, w) = rows.up(l);
        star.member_nodes.push(par);
        star.means.extend_from_slice(m);
        star.precisions.extend_from_slice(w);
        star.branch.push(tree.branch(l));
    }
    Some(star)
}

/// Propose the interchange at one internal edge.
///
/// ### Params
///
/// * `tree` - The tree
/// * `down` - Down rows, settled against this tree
/// * `up` - Up rows, settled against the same
/// * `k` - Lower end of the edge, the node that is deleted
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The proposed tree, or `None` if the edge is not eligible, or the error the
/// primitive or the arena failed with.
///
/// The [`Splice::gain`] that comes back is measured against the *collapsed*
/// tree and not against `tree`, because the collapse happened before the star
/// was scored. Callers that need the gain of the move itself take the
/// difference of two [`NodeState::prune`] calls, which is what both phases do.
pub fn interchange_at<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
    k: u32,
    params: Option<StarParams>,
) -> Result<Option<Splice>, BonsaiErrors> {
    match collapsed_star(tree, &Settled { down, up }, k) {
        None => Ok(None),
        Some(star) => Ok(Some(splice_star(tree, &star, params)?)),
    }
}

////////////////
// Exact gain //
////////////////

/// Scratch for the local peels, reused across the candidates of a round.
///
/// [`prune_general`] writes the peeled node's own effective leaf as well as
/// returning its contribution. Nothing here reads the leaf, so the two
/// destination rows exist only to be overwritten.
struct PeelScratch<T> {
    /// Effective means of the peeled node, written and discarded.
    means: Vec<T>,
    /// Its effective precisions, the same.
    precisions: Vec<T>,
    /// The kernel's own scratch, `p` per child, grown on demand.
    work: Vec<f64>,
}

impl<T: BonsaiFloat> PeelScratch<T> {
    /// Allocate for a given feature count.
    ///
    /// ### Params
    ///
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// The scratch.
    fn new(p: usize) -> Self {
        Self {
            means: vec![T::default(); p],
            precisions: vec![T::default(); p],
            work: Vec::new(),
        }
    }

    /// Loglikelihood contribution of one node, given its children's effective
    /// leaves and the branches down to them.
    ///
    /// ### Params
    ///
    /// * `children` - Effective means, effective precisions and branch length
    ///   of each child
    ///
    /// ### Returns
    ///
    /// The node's contribution, zero for a node with fewer than two children:
    /// such a node passes its child's effective leaf straight up and adds
    /// nothing to the loglikelihood.
    fn peel(&mut self, children: &[(&[T], &[T], f64)]) -> f64 {
        if children.len() < 2 {
            return 0.0;
        }
        let p = self.means.len();
        let need = p * children.len();
        if self.work.len() < need {
            self.work.resize(need, 0.0);
        }
        prune_general(
            children,
            &mut self.means,
            &mut self.precisions,
            &mut self.work[..need],
        )
    }
}

/// What the collapse alone did to the loglikelihood.
///
/// [`Splice::gain`] is exact against the *collapsed* tree, so the gain of the
/// interchange itself is that plus this. Deleting `k` changes the tree only at
/// `l`: rooted there, the loglikelihood is the contributions of the members'
/// own subtrees, plus the contribution of everything outside, plus the
/// contributions of the nodes of the local topology. The first two are
/// identical either side of the collapse and cancel, so what is left is three
/// peels over a handful of members and not two sweeps over the tree.
///
/// Before, the local topology is `k` peeling its children and `l` peeling its
/// own children and its upstream side. After, it is `l` peeling the collapsed
/// star's members and nothing else.
///
/// **The kernel is [`prune_general`] on both sides, deliberately.**
/// [`NodeState::prune`] would use the binary kernel for a binary `k`, and the
/// two agree only to rounding. Nothing here is ever differenced against the
/// tree's own sweep, only against another peel of this routine's, so the
/// difference is exact in the algebra and loses nothing to the mismatch.
///
/// ### Params
///
/// * `tree` - The tree
/// * `rows` - Its down and up rows
/// * `k` - The node the collapse deletes
/// * `l` - Its parent, the centre of the star
/// * `star` - The collapsed star, from [`collapsed_star`]
/// * `scratch` - Peel scratch, reused across candidates
///
/// ### Returns
///
/// The loglikelihood of the collapsed tree less that of `tree`, in nats.
fn collapse_delta<T: BonsaiFloat>(
    tree: &impl Topology,
    rows: &impl EdgeRows<T>,
    k: u32,
    l: u32,
    star: &CentreStar<T>,
    scratch: &mut PeelScratch<T>,
) -> f64 {
    let p = star.n_features;
    let after: Vec<(&[T], &[T], f64)> = (0..star.branch.len())
        .map(|i| {
            (
                &star.means[i * p..(i + 1) * p],
                &star.precisions[i * p..(i + 1) * p],
                star.branch[i],
            )
        })
        .collect();
    let after = scratch.peel(&after);

    let below = |node: u32| {
        let (m, w) = rows.down(node);
        (m, w, tree.branch(node))
    };
    let at_k: Vec<(&[T], &[T], f64)> = tree.children(k).iter().map(|&c| below(c)).collect();
    let mut at_l: Vec<(&[T], &[T], f64)> = tree.children(l).iter().map(|&c| below(c)).collect();
    if tree.parent(l).is_some() {
        let (m, w) = rows.up(l);
        at_l.push((m, w, tree.branch(l)));
    }
    after - scratch.peel(&at_k) - scratch.peel(&at_l)
}

/////////////////////////
// Topology comparison //
/////////////////////////

/// Whether a resolved star puts back exactly the split the deleted edge
/// carried, and so proposes no interchange at all.
///
/// This is the structural form of the filter the module docs argue for, and it
/// has to reject exactly what a split fingerprint of the spliced tree would.
/// The two agree because the star region's splits are enumerable. Every member
/// contributes the split "my leaves against the rest" on the edge above it, in
/// the tree the star came from and in every tree the primitive can build from
/// it. What differs is one split per internal node of the local topology: the
/// edge `k-l` before, one edge per ancestor after. So the proposal changes
/// nothing iff its ancestors carry the same bipartitions of the member set as
/// `k` did.
///
/// Distinct ancestors are nested and so carry distinct bipartitions, which
/// leaves exactly two ways to match: the one ancestor covers `k`'s children, or
/// it covers all the other members. Both give the same *unrooted* split, and
/// [`crate::search::split_fingerprint`] canonicalises a split against its
/// complement, so both have to be rejected. Membership is decided on two counts
/// rather than on a set: a subset of the members that has `|K|` members of
/// which `|K|` are in `K` is `K`, and one with `m - |K|` members of which none
/// are in `K` is its complement.
///
/// The triviality test is the fingerprint's: a split with fewer than two leaves
/// on a side is carried by every tree over these leaves, so it is not part of
/// the comparison. It can only bite where `k`'s own edge is trivial, which
/// needs `l` to be a root of degree two with a leaf on its other side.
///
/// ### Params
///
/// * `result` - What the primitive built from the collapsed star
/// * `k_lo` - First member index that is a child of `k`
/// * `k_hi` - One past the last
/// * `member_leaves` - Leaves standing behind each member, upstream included
/// * `n_leaves` - Leaves in the whole tree
///
/// ### Returns
///
/// True when the spliced tree would have the same splits as the tree the star
/// was built from.
fn rebuilds_the_same_splits<T>(
    result: &StarResult<T>,
    k_lo: usize,
    k_hi: usize,
    member_leaves: &[usize],
    n_leaves: usize,
) -> bool {
    let n_members = result.n_members;
    let n_local = result.parent.len();
    let mut members = vec![0usize; n_local];
    let mut in_k = vec![0usize; n_local];
    let mut leaves = vec![0usize; n_local];
    for i in 0..n_members {
        members[i] = 1;
        in_k[i] = usize::from(i >= k_lo && i < k_hi);
        leaves[i] = member_leaves[i];
    }
    for (j, merge) in result.merges.iter().enumerate() {
        let a = n_members + j;
        for child in [merge.left as usize, merge.right as usize] {
            members[a] += members[child];
            in_k[a] += in_k[child];
            leaves[a] += leaves[child];
        }
    }

    let k_members = k_hi - k_lo;
    let k_leaves: usize = member_leaves[k_lo..k_hi].iter().sum();
    let was_a_split = usize::from(k_leaves.min(n_leaves - k_leaves) >= 2);

    let mut splits = 0usize;
    let mut all_match = true;
    for j in 0..result.merges.len() {
        let a = n_members + j;
        if leaves[a].min(n_leaves - leaves[a]) < 2 {
            continue;
        }
        splits += 1;
        let is_k = members[a] == k_members && in_k[a] == k_members;
        let is_complement = members[a] == n_members - k_members && in_k[a] == 0;
        all_match &= is_k || is_complement;
    }
    splits == was_a_split && all_match
}

/// Score the interchange at one edge against the current tree's rows.
///
/// ### Params
///
/// * `tree` - The tree
/// * `rows` - Its down and up rows
/// * `below` - [`leaves_below`] of the tree
/// * `k` - Lower end of the edge
/// * `star_params` - Star primitive knobs
/// * `scratch` - Peel scratch, reused across edges
///
/// ### Returns
///
/// See [`ScannedEdge`]; or the error the primitive failed with.
fn scan_edge<T: BonsaiFloat>(
    tree: &impl Topology,
    rows: &impl EdgeRows<T>,
    below: &[usize],
    k: u32,
    star_params: StarParams,
    scratch: &mut PeelScratch<T>,
) -> Result<ScannedEdge<T>, BonsaiErrors> {
    let nothing = (f64::NEG_INFINITY, None);
    let Some(l) = tree.parent(k) else {
        return Ok(nothing);
    };
    let Some(star) = collapsed_star(tree, rows, k) else {
        return Ok(nothing);
    };
    let n_leaves = tree.n_leaves();
    let result = resolve_star(star.view(), Some(star_params))?;
    if result.merges.is_empty() {
        return Ok(nothing);
    }
    // A proposal that puts the same subtrees back where they were is not an
    // interchange: it is a reoptimisation of the three branches the star
    // primitive creates at `l`. Those nearly always gain a little, and taking
    // them turns the phase into branch-length descent that steps 4 and 7 do
    // properly and far more cheaply. Started from the generating tree itself,
    // accepting them runs hundreds of rounds with the Robinson-Foulds distance
    // pinned at zero throughout: every one of those rounds is branch lengths
    // and none is topology.
    let k_lo = tree.children(l).len() - 1;
    let k_hi = k_lo + tree.children(k).len();
    let last = star.member_nodes.len() - 1;
    let member_leaves: Vec<usize> = star
        .member_nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            if star.has_upstream && i == last {
                n_leaves - below[l as usize]
            } else {
                below[node as usize]
            }
        })
        .collect();
    if rebuilds_the_same_splits(&result, k_lo, k_hi, &member_leaves, n_leaves) {
        return Ok(nothing);
    }

    let gain: f64 = result.merges.iter().map(|x| x.gain).sum::<f64>()
        + collapse_delta(tree, rows, k, l, &star, scratch);
    if gain > star_params.min_gain {
        Ok((gain, Some((gain, k, star))))
    } else {
        Ok((gain, None))
    }
}

/// Perform an interchange: resolve its star and splice the result in.
///
/// Numbered by [`assemble`] rather than by
/// [`crate::search::polytomy::splice_star`]'s post-order rebuild: a level keeps
/// the order its nodes had, the ancestors the splice made last, so a move
/// reorders only the nodes whose height it changed. That is what lets the lazy
/// phase keep its tree up to date locally.
///
/// ### Params
///
/// * `tree` - The tree the star was built from
/// * `star` - The collapsed star
/// * `params` - Star primitive knobs
///
/// ### Returns
///
/// The spliced tree and, per node of it, its node in `tree` or [`NO_NODE`] for
/// one the splice made; or the error the primitive or the arena failed with.
fn perform<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    params: StarParams,
) -> Result<(Tree, Vec<u32>), BonsaiErrors> {
    let result = resolve_star(star.view(), Some(params))?;
    let n = tree.n_nodes();
    let n_made = result.parent.len() - star.member_nodes.len();
    let mut parent: Vec<u32> = (0..n as u32)
        .map(|v| tree.parent(v).unwrap_or(NO_NODE))
        .chain(std::iter::repeat_n(NO_NODE, n_made))
        .collect();
    let mut branch = tree.branches().to_vec();
    branch.resize(n + n_made, 0.0);
    for &k in &star.deleted {
        parent[k as usize] = NO_NODE;
    }
    for (v, up, t) in splice_edits(star, &result, n as u32) {
        parent[v as usize] = up;
        branch[v as usize] = t;
    }
    let (next, map) = assemble(&parent, &branch, tree.root(), tree.n_leaves())?;
    let mut to_old = vec![NO_NODE; next.n_nodes()];
    for (old, &new) in map.iter().enumerate().take(n) {
        if new != NO_NODE {
            to_old[new as usize] = old as u32;
        }
    }
    Ok((next, to_old))
}

/////////////
// Phases //
/////////////

/// The random phase: `n_random` interchanges with the merge sampled rather than
/// chosen, accepted whatever they do to the tree.
///
/// The edge is drawn uniformly from the eligible ones and the star's own seed
/// is drawn from the same stream, so the whole phase is a function of
/// [`NniParams::seed`] and the tree. Nothing here reduces over rayon, and the
/// sampling inside the star is done over a fixed candidate order, so the result
/// does not depend on the thread count.
///
/// ### Params
///
/// * `tree` - Tree to move away from; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults, whose `n_random` is zero
///
/// ### Returns
///
/// The tree the phase finished on, which may be worse than the one it started
/// from, or the error the primitive or the arena failed with.
pub fn nni_random<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let mut rng = SplitMix64::new(params.seed);
    let mut tree = tree.clone();
    let mut n_moves = 0usize;

    for _ in 0..params.n_random {
        let eligible: Vec<u32> = tree
            .internal_postorder()
            .filter(|&k| interchange_members(&tree, k).is_some())
            .collect();
        if eligible.is_empty() {
            break;
        }
        let draw = (rng.uniform() * eligible.len() as f64) as usize;
        let k = eligible[draw.min(eligible.len() - 1)];
        let star = StarParams {
            selection: StarSelection::Weighted {
                seed: rng.next_u64(),
                temperature: params.temperature,
            },
            ..params.star
        };

        let (down, up, _) = settle(&tree, leaves)?;
        if let Some(spliced) = interchange_at(&tree, &down, &up, k, Some(star))? {
            tree = spliced.tree;
            n_moves += 1;
        }
    }

    let loglik = tree_loglik(&tree, leaves)?;
    Ok(NniResult {
        tree,
        loglik,
        n_moves,
        rounds: 0,
    })
}

/// The greedy phase: score an interchange at every eligible edge, perform the
/// best, repeat until none improves the tree.
///
/// Every candidate is scored by the exact loglikelihood gain of the tree it
/// would produce, so the accepted move is an improvement in the quantity that
/// actually matters rather than in the star primitive's local gain, which is
/// measured against the collapsed tree and not against this one. That makes the
/// phase monotone by construction. A candidate whose splits match the current
/// tree's is discarded before it is scored; see the module docs.
///
/// Edges are visited in ascending node order and ties go to the lower node, so
/// the round's winner is fixed. The scan is sequential, and it is the one place
/// in the search that could be parallel and is not: every candidate reads the
/// same settled rows and writes nothing, so only the reduction over gains would
/// need ordering. It has not been worth it since the topology filter took a
/// round to linear in the leaf count.
///
/// ### A round is linear in the leaf count
///
/// Nothing whole-tree happens per candidate. The exact gain is
/// [`Splice::gain`] plus [`collapse_delta`], both of which read a handful of
/// members; the topology filter is [`rebuilds_the_same_splits`], which counts
/// members under the star's own ancestors instead of splicing a tree and
/// fingerprinting it. Only the winner is ever spliced, once, at the end of the
/// round. What is left per round is the one settling sweep every candidate
/// reads its rows from, and the star primitive itself once per edge.
///
/// A round is linear in the leaf count on this arrangement, against
/// superlinear when the filter spliced a tree per candidate.
///
/// **What was expensive was not what it looked like.** The `O(n p)` re-prune
/// per candidate is nearly free at this point in the search, because the
/// topology filter discards almost everything before it: near convergence at
/// most one eligible edge per round changes a split at all. The cost was the
/// filter itself, which spliced a whole tree and fingerprinted it to answer a
/// question about one star's ancestors. Scoring on the exact gain rather than
/// on a re-prune is also what makes the *other* end of the search cheap, where
/// the tree is far from converged and a large fraction of proposals do change
/// a split.
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
/// error the primitive or the arena failed with.
pub fn nni_greedy<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    if let NniSearch::Approximate(approx) = params.search {
        return nni_lazy(tree, leaves, params, approx.rescore_radius);
    }
    let mut tree = tree.clone();
    let mut best: Option<f64> = None;
    let mut n_moves = 0usize;
    let mut rounds = 0usize;

    while rounds < params.max_rounds {
        rounds += 1;
        let (down, up, loglik) = settle(&tree, leaves)?;
        best = Some(loglik);
        let below = leaves_below(&tree);

        // Every edge is scored against the same settled rows and nothing in the
        // scan writes to the tree, so the candidates are independent and the
        // round is parallel. The reduction keeps only the running best rather
        // than collecting, because a `CentreStar` carries its members' rows and
        // one per edge would be gigabytes at atlas scale.
        //
        // **Determinism.** The sequential scan took the first strict maximum in
        // `internal_postorder`, which the arena invariant makes ascending node
        // order, so the reduction breaks ties on the lower node id and the
        // winner is the same at any thread count. No float is summed across
        // candidates, so there is nothing else for the order to change.
        let edges: Vec<u32> = tree.internal_postorder().collect();
        let winner = edges
            .par_iter()
            .map_init(
                || PeelScratch::<T>::new(leaves.n_features),
                |scratch, &k| {
                    let rows = Settled {
                        down: &down,
                        up: &up,
                    };
                    scan_edge(&tree, &rows, &below, k, params.star, scratch).map(|(_, p)| p)
                },
            )
            .try_reduce(
                || None,
                |a, b| {
                    Ok(match (a, b) {
                        (None, other) | (other, None) => other,
                        (Some(x), Some(y)) => {
                            if y.0 > x.0 || (y.0 == x.0 && y.1 < x.1) {
                                Some(y)
                            } else {
                                Some(x)
                            }
                        }
                    })
                },
            )?;

        match winner {
            None => break,
            Some((gain, _, star)) => {
                tree = perform(&tree, &star, params.star)?.0;
                best = Some(loglik + gain);
                n_moves += 1;
            }
        }
    }

    let loglik = match best {
        Some(loglik) => loglik,
        // Only reachable at `max_rounds` zero, where the loop never ran.
        None => tree_loglik(&tree, leaves)?,
    };
    Ok(NniResult {
        tree,
        loglik,
        n_moves,
        rounds,
    })
}

/// A cached gain as a sort key, highest first.
#[derive(Clone, Copy, Debug)]
struct Desc(f64);

impl PartialEq for Desc {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Desc {}

impl PartialOrd for Desc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Desc {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}

/// The cached gains of the tree's edges, ordered so that the leader and the
/// runner-up are read off the front rather than found by a pass over the tree.
#[derive(Default)]
struct Leaders {
    /// Gain and lower end of every edge with a cached gain, highest first.
    order: BTreeSet<(Desc, u32)>,
    /// The gain each edge is filed under.
    gain: FxHashMap<u32, f64>,
}

impl Leaders {
    /// File an edge under a gain, replacing what it had.
    ///
    /// ### Params
    ///
    /// * `k` - Lower end of the edge
    /// * `g` - Its gain
    fn set(&mut self, k: u32, g: f64) {
        if let Some(old) = self.gain.insert(k, g) {
            self.order.remove(&(Desc(old), k));
        }
        self.order.insert((Desc(g), k));
    }

    /// Forget an edge.
    ///
    /// ### Params
    ///
    /// * `k` - Lower end of the edge
    fn remove(&mut self, k: u32) {
        if let Some(old) = self.gain.remove(&k) {
            self.order.remove(&(Desc(old), k));
        }
    }

    /// Forget every edge.
    fn clear(&mut self) {
        self.order.clear();
        self.gain.clear();
    }

    /// The leading edge and the best gain of the rest.
    ///
    /// Ties on the leading gain go to the edge first in arena order, as a
    /// pass over the arena would find it, and make the runner-up equal to the
    /// leader. An edge with no cached gain counts as minus infinity.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree, for arena order
    ///
    /// ### Returns
    ///
    /// The leader's gain, its lower end, and the runner-up's gain; `None` if
    /// nothing is cached.
    fn lead(&self, tree: &LiveTree) -> Option<(f64, u32, f64)> {
        let mut it = self.order.iter();
        let &(Desc(top), first) = it.next()?;
        let (mut lead, mut second) = (first, f64::NEG_INFINITY);
        for &(Desc(g), k) in it {
            if g != top {
                second = second.max(g);
                break;
            }
            second = top;
            if tree.order(k) < tree.order(lead) {
                lead = k;
            }
        }
        Some((top, lead, second))
    }
}

/// One node's up row, from its parent's.
///
/// [`crate::search::spr::LazyRows`]'s expressions over the live tree, so the
/// bits are the up sweep's.
///
/// ### Params
///
/// * `tree` - The tree
/// * `store` - Its down rows
/// * `c` - The node
/// * `above` - Its parent's up row, `None` at the root
///
/// ### Returns
///
/// The node's up row.
fn up_row_of<T: BonsaiFloat>(
    tree: &LiveTree,
    store: &RowStore<T>,
    c: u32,
    above: Option<&Row<T>>,
) -> Row<T> {
    let Some(a) = tree.parent(c) else {
        return root_up(store.n_features());
    };
    let above = above.expect("a parent's up row is formed before its children's");
    let kids = tree.children(a);
    let side = if kids.len() == 2 {
        let o = if kids[0] == c { kids[1] } else { kids[0] };
        UpSide::Sibling(tree.branch(o), store.means(o), store.precisions(o))
    } else {
        UpSide::Parent(
            store.means(a),
            store.precisions(a),
            tree.branch(c),
            store.means(c),
            store.precisions(c),
        )
    };
    up_step(
        tree.parent(a).is_none(),
        tree.branch(a),
        (&above.0, &above.1),
        side,
    )
}

/// Form the up rows of some nodes and of every ancestor of them.
///
/// ### Params
///
/// * `tree` - The tree
/// * `store` - Its down rows
/// * `nodes` - The nodes whose up rows are wanted
/// * `up` - Rows formed so far this round, extended
fn fill_up<T: BonsaiFloat>(
    tree: &LiveTree,
    store: &RowStore<T>,
    nodes: impl Iterator<Item = u32>,
    up: &mut FxHashMap<u32, Row<T>>,
) {
    let mut need: Vec<u32> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for v in nodes {
        let mut u = v;
        while !up.contains_key(&u) && seen.insert(u) {
            need.push(u);
            match tree.parent(u) {
                Some(a) => u = a,
                None => break,
            }
        }
    }
    // A parent is taller than its children, so tallest first is top-down.
    need.sort_unstable_by_key(|&u| std::cmp::Reverse(tree.height(u)));
    for u in need {
        let row = up_row_of(tree, store, u, tree.parent(u).map(|a| &up[&a]));
        up.insert(u, row);
    }
}

/// Form the up row of every internal node, a level at a time.
///
/// ### Params
///
/// * `tree` - The tree
/// * `store` - Its down rows
///
/// ### Returns
///
/// Up rows by node id.
fn fill_up_all<T: BonsaiFloat>(tree: &LiveTree, store: &RowStore<T>) -> FxHashMap<u32, Row<T>> {
    let mut up: FxHashMap<u32, Row<T>> = FxHashMap::default();
    for h in (1..=tree.n_levels() as u32).rev() {
        let nodes = tree.level_nodes(h);
        let rows: Vec<(u32, Row<T>)> = nodes
            .par_iter()
            .map(|&c| {
                (
                    c,
                    up_row_of(tree, store, c, tree.parent(c).map(|a| &up[&a])),
                )
            })
            .collect();
        up.extend(rows);
    }
    up
}

/// What the lazy phase keeps about the tree between moves, by node id.
struct Lazy<T> {
    /// The tree.
    tree: LiveTree,
    /// Its down rows.
    store: RowStore<T>,
    /// Leaf words.
    word: Vec<u64>,
    /// Leaf counts.
    below: Vec<usize>,
    /// Node per leaf word.
    by_word: FxHashMap<u64, u32>,
}

impl<T: BonsaiFloat> Lazy<T> {
    /// Perform an interchange in place, as [`perform`] numbers it.
    ///
    /// The collapse deletes `k` and the splice hangs the star's members under
    /// the ancestors the primitive made; only those ancestors, the centre and
    /// the path above it change their rows or their heights. Leaf sets move
    /// between them: with the upstream side merged, an ancestor takes over the
    /// centre's old leaf set and the centre the deleted node's.
    ///
    /// ### Params
    ///
    /// * `star` - The collapsed star at the edge above `k`
    /// * `k` - The node the collapse deletes
    /// * `params` - Star primitive knobs
    ///
    /// ### Returns
    ///
    /// The nodes whose rows the move changed or made, and the ones among them
    /// whose leaf set the tree did not have; or the error the primitive failed
    /// with.
    fn perform(
        &mut self,
        star: &CentreStar<T>,
        k: u32,
        params: StarParams,
    ) -> Result<(Vec<u32>, Vec<u32>), BonsaiErrors> {
        let result = resolve_star(star.view(), Some(params))?;
        let tree = &self.tree;
        let l = star.centre;
        let first_new = tree.id_space() as u32 + 1;
        let is_new = |v: u32| v >= first_new;
        let n_made = result.parent.len() - star.member_nodes.len();
        let edits = splice_edits(star, &result, first_new);

        let mut parent: FxHashMap<u32, u32> = FxHashMap::default();
        let mut branch: FxHashMap<u32, f64> = FxHashMap::default();
        let mut children: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        children.insert(
            l,
            tree.children(l)
                .iter()
                .copied()
                .filter(|&c| c != k)
                .collect(),
        );
        for &(v, up, t) in &edits {
            let old = if is_new(v) {
                None
            } else {
                parent.get(&v).copied().or_else(|| tree.parent(v))
            };
            if old != Some(up) {
                if let Some(op) = old.filter(|&op| op != k) {
                    children
                        .entry(op)
                        .or_insert_with(|| tree.children(op).to_vec())
                        .retain(|&c| c != v);
                }
                children
                    .entry(up)
                    .or_insert_with(|| {
                        if is_new(up) {
                            Vec::new()
                        } else {
                            tree.children(up).to_vec()
                        }
                    })
                    .push(v);
                parent.insert(v, up);
            }
            branch.insert(v, t);
        }
        let par_of = |v: u32| parent.get(&v).copied().or_else(|| tree.parent(v));
        let kids_of = |v: u32| -> &[u32] {
            match children.get(&v) {
                Some(kids) => kids,
                None => tree.children(v),
            }
        };

        // The ancestors made, the centre and its ancestors, in post-order.
        let mut changed: FxHashSet<u32> = (first_new..first_new + n_made as u32).collect();
        let mut u = Some(l);
        while let Some(v) = u {
            changed.insert(v);
            u = par_of(v);
        }
        let mut order: Vec<u32> = Vec::with_capacity(changed.len());
        let mut stack = vec![(tree.root(), false)];
        while let Some((v, expanded)) = stack.pop() {
            if expanded {
                order.push(v);
                continue;
            }
            stack.push((v, true));
            for &c in kids_of(v) {
                if changed.contains(&c) {
                    stack.push((c, false));
                }
            }
        }
        debug_assert_eq!(order.len(), changed.len());
        let mut height: FxHashMap<u32, u32> = FxHashMap::default();
        for &v in &order {
            let h = kids_of(v)
                .iter()
                .map(|&c| height.get(&c).copied().unwrap_or_else(|| tree.height(c)))
                .max()
                .unwrap_or(0)
                + 1;
            height.insert(v, h);
        }
        // Children in the arena order of the result: height, then the order
        // before, the ancestors made last.
        let key = |c: u32| {
            let h = height.get(&c).copied().unwrap_or_else(|| tree.height(c));
            if is_new(c) {
                (h, (u32::MAX, c))
            } else {
                (h, tree.order(c))
            }
        };
        let sorted: FxHashMap<u32, Vec<u32>> = order
            .iter()
            .map(|&v| {
                let mut kids = kids_of(v).to_vec();
                kids.sort_unstable_by_key(|&c| key(c));
                (v, kids)
            })
            .collect();

        // Rows bottom-up, through the kernels the prune dispatches to.
        let p = self.store.n_features();
        let mut rows: FxHashMap<u32, (Vec<T>, Vec<T>, f64)> = FxHashMap::default();
        let mut work: Vec<f64> = Vec::new();
        for &v in &order {
            let kids = &sorted[&v];
            let children_rows: Vec<(&[T], &[T], f64)> = kids
                .iter()
                .map(|&c| {
                    let t = branch.get(&c).copied().unwrap_or_else(|| tree.branch(c));
                    match rows.get(&c) {
                        Some(r) => (r.0.as_slice(), r.1.as_slice(), t),
                        None => (self.store.means(c), self.store.precisions(c), t),
                    }
                })
                .collect();
            let mut m_out = vec![T::zero(); p];
            let mut w_out = vec![T::zero(); p];
            let contrib = if children_rows.len() == 2 {
                let (a, b) = (children_rows[0], children_rows[1]);
                prune_binary(a.0, a.1, a.2, b.0, b.1, b.2, &mut m_out, &mut w_out)
            } else {
                let need = p * children_rows.len();
                if work.len() < need {
                    work.resize(need, 0.0);
                }
                prune_general(&children_rows, &mut m_out, &mut w_out, &mut work[..need])
            };
            rows.insert(v, (m_out, w_out, contrib));
        }

        // Read before the move: the id it frees can go to a node it makes, and
        // the centre can take over the deleted node's leaf set.
        let old_words: Vec<u64> = order
            .iter()
            .filter(|&&v| !is_new(v))
            .map(|&v| self.word[v as usize])
            .chain(std::iter::once(self.word[k as usize]))
            .collect();
        let before_height = FxHashMap::default();
        let real = self.tree.apply(MoveEdit {
            parent: &parent,
            children: &sorted,
            branch: &branch,
            height: &height,
            before_height: &before_height,
            suppressed: Some(k),
            n_made: Some(n_made),
        });
        let r = |v: u32| real.get(&v).copied().unwrap_or(v);
        for (&v, row) in &rows {
            self.store.write_row(r(v), &row.0, &row.1, row.2);
        }
        let space = self.tree.id_space();
        self.word.resize(space, 0);
        self.below.resize(space, 0);
        for &v in &order {
            let rv = r(v);
            let (mut w, mut c) = (0u64, 0usize);
            for &kid in self.tree.children(rv) {
                w = w.wrapping_add(self.word[kid as usize]);
                c += self.below[kid as usize];
            }
            self.word[rv as usize] = w;
            self.below[rv as usize] = c;
        }
        let changed: Vec<u32> = order.iter().map(|&v| r(v)).collect();
        let fresh: Vec<u32> = changed
            .iter()
            .copied()
            .filter(|&v| !self.by_word.contains_key(&self.word[v as usize]))
            .collect();
        for &v in &changed {
            self.by_word.insert(self.word[v as usize], v);
        }
        for w in old_words {
            if let Some(&v) = self.by_word.get(&w)
                && !(self.tree.contains(v) && self.word[v as usize] == w)
            {
                self.by_word.remove(&w);
            }
        }
        Ok((changed, fresh))
    }
}

/// The greedy phase with cached gains, [`NniSearch::Approximate`].
///
/// Round one scores every edge. After a move, only the edges within `radius`
/// of the clades the move created lose their cached gain and are rescored; the
/// rest keep the gain from the round they were last scored in. The move itself
/// is chosen lazily: the leading cached gain is rescored on the current tree
/// and taken only if it still beats the runner-up, otherwise its fresh gain goes
/// back in the cache and the next leader is tried. When the cache holds nothing
/// improving, the next round is a full scan, and a full scan that finds
/// nothing ends the phase, which is the same stopping rule as the exact phase.
///
/// The tree is a [`LiveTree`] for the length of the phase, so a move changes
/// the rows, words and counts of the nodes on the paths it touched and nothing
/// else. Down rows are settled once and kept in a [`RowStore`]; up rows are
/// formed per round along the chains the scanned edges read, all of them on a
/// full scan. Both are the sweep's own bits, so the moves are the ones a settle
/// per round would make.
///
/// Edges are keyed by [`leaf_words`] of their lower end, as the cache is.
/// Everything that decides a move is sequential and the parallel scans collect
/// in edge order, so the result does not depend on the thread count.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs
/// * `radius` - See [`NniApprox::rescore_radius`]
///
/// ### Returns
///
/// As [`nni_greedy`].
fn nni_lazy<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: NniParams,
    radius: usize,
) -> Result<NniResult, BonsaiErrors> {
    let min_gain = params.star.min_gain;
    let mut n_moves = 0usize;
    let mut rounds = 0usize;
    let mut cache: FxHashMap<u64, f64> = FxHashMap::default();
    let mut leaders = Leaders::default();
    let mut full = true;
    let mut pending: Vec<u32> = Vec::new();

    let (down, _) = settled_down(tree, leaves)?;
    let word = leaf_words(tree);
    let mut lazy = Lazy {
        store: RowStore::from_state(&down, tree.n_nodes()),
        tree: LiveTree::from_tree(tree),
        by_word: word
            .iter()
            .enumerate()
            .map(|(v, &w)| (w, v as u32))
            .collect(),
        below: leaves_below(tree),
        word,
    };
    drop(down);

    while rounds < params.max_rounds {
        rounds += 1;
        let live = &lazy.tree;
        let edges: Vec<u32> = if full {
            cache.clear();
            leaders.clear();
            pending.clear();
            (1..=live.n_levels() as u32)
                .flat_map(|h| live.level_nodes(h))
                .collect()
        } else {
            let mut edges: Vec<u32> = pending
                .drain(..)
                .filter(|&v| {
                    v as usize >= live.n_leaves()
                        && live.contains(v)
                        && !cache.contains_key(&lazy.word[v as usize])
                })
                .collect();
            edges.sort_unstable();
            edges.dedup();
            edges
        };
        let mut up = if full {
            fill_up_all(live, &lazy.store)
        } else {
            let mut up = FxHashMap::default();
            fill_up(
                live,
                &lazy.store,
                edges.iter().filter_map(|&k| live.parent(k)),
                &mut up,
            );
            up
        };

        let scanned: Vec<(u32, f64)> = {
            let rows = LiveRows {
                store: &lazy.store,
                up: &up,
            };
            edges
                .par_iter()
                .map_init(
                    || PeelScratch::<T>::new(leaves.n_features),
                    |scratch, &k| {
                        scan_edge(live, &rows, &lazy.below, k, params.star, scratch)
                            .map(|(gain, _)| (k, gain))
                    },
                )
                .collect::<Result<_, _>>()?
        };
        for &(k, gain) in &scanned {
            cache.insert(lazy.word[k as usize], gain);
            leaders.set(k, gain);
        }

        let mut scratch = PeelScratch::<T>::new(leaves.n_features);
        let winner = loop {
            let Some((_, k, second)) = leaders.lead(live).filter(|&(g, _, _)| g > min_gain) else {
                break None;
            };
            fill_up(live, &lazy.store, live.parent(k).into_iter(), &mut up);
            let rows = LiveRows {
                store: &lazy.store,
                up: &up,
            };
            let (gain, proposal) =
                scan_edge(live, &rows, &lazy.below, k, params.star, &mut scratch)?;
            cache.insert(lazy.word[k as usize], gain);
            leaders.set(k, gain);
            if let Some(found) = proposal.filter(|p| p.0 >= second) {
                break Some(found);
            }
        };
        drop(up);

        match winner {
            None if full => break,
            None => full = true,
            Some((_, k, star)) => {
                #[cfg(debug_assertions)]
                let expected = expected_interchange(&lazy.tree, leaves, k, params.star)?;
                leaders.remove(k);
                let (changed, fresh) = lazy.perform(&star, k, params.star)?;
                #[cfg(debug_assertions)]
                check_interchange(&lazy, &expected, leaves)?;
                n_moves += 1;
                let mut stale: Vec<u32> = Vec::new();
                near(&lazy.tree, &fresh, radius, &mut stale);
                for &v in &stale {
                    cache.remove(&lazy.word[v as usize]);
                    leaders.remove(v);
                }
                // A gain is the leaf set's, as the cache is keyed, so a node
                // whose leaf set changed takes the gain filed under its new one.
                for &v in &changed {
                    leaders.remove(v);
                    if let Some(&g) = cache.get(&lazy.word[v as usize]) {
                        leaders.set(v, g);
                    }
                }
                pending.extend(stale);
                pending.extend(changed);
                full = false;
            }
        }
    }

    // One sweep at the end rather than one a round: the rows' own sum agrees
    // with it only to rounding, and callers compare against sweeps.
    let tree = lazy.tree.to_tree()?.0;
    let loglik = tree_loglik(&tree, leaves)?;
    Ok(NniResult {
        tree,
        loglik,
        n_moves,
        rounds,
    })
}

/// Every node within `radius` edges of some nodes, the nodes included.
///
/// [`mark_near_new_clades`] over the live tree, from the created clades
/// alone, so it touches only the neighbourhood it marks.
///
/// ### Params
///
/// * `tree` - The tree
/// * `seeds` - The nodes to start from
/// * `radius` - How many edges out to go
/// * `out` - Receives the nodes reached, each once
fn near(tree: &LiveTree, seeds: &[u32], radius: usize, out: &mut Vec<u32>) {
    let mut dist: FxHashMap<u32, usize> = FxHashMap::default();
    let mut queue = std::collections::VecDeque::new();
    for &v in seeds {
        if dist.insert(v, 0).is_none() {
            queue.push_back(v);
        }
    }
    while let Some(v) = queue.pop_front() {
        out.push(v);
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

/// Debug builds: the tree [`perform`] builds from an interchange on the live
/// tree numbered as an arena.
///
/// ### Params
///
/// * `tree` - The live tree
/// * `leaves` - The leaf data
/// * `k` - The node the interchange deletes
/// * `params` - Star primitive knobs
///
/// ### Returns
///
/// The arena tree the built path makes.
#[cfg(debug_assertions)]
fn expected_interchange<T: BonsaiFloat>(
    tree: &LiveTree,
    leaves: Leaves<'_, T>,
    k: u32,
    params: StarParams,
) -> Result<Tree, BonsaiErrors> {
    let (arena, _, arena_of) = tree.to_tree()?;
    let (down, up, _) = settle(&arena, leaves)?;
    let rows = Settled {
        down: &down,
        up: &up,
    };
    let star = collapsed_star(&arena, &rows, arena_of[k as usize])
        .expect("the live tree found this edge eligible");
    Ok(perform(&arena, &star, params)?.0)
}

/// Debug builds: the live tree after an interchange against the built path,
/// and its rows, words and counts against a fresh settle.
///
/// ### Params
///
/// * `lazy` - The state after the move
/// * `expected` - The built path's tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// `Ok` if they agree; panics otherwise.
#[cfg(debug_assertions)]
fn check_interchange<T: BonsaiFloat>(
    lazy: &Lazy<T>,
    expected: &Tree,
    leaves: Leaves<'_, T>,
) -> Result<(), BonsaiErrors> {
    let (got, id_of, _) = lazy.tree.to_tree()?;
    assert_eq!(got.n_nodes(), expected.n_nodes(), "node count");
    for v in 0..got.n_nodes() as u32 {
        assert_eq!(got.parent(v), expected.parent(v), "parent of {v}");
        assert_eq!(
            got.branch(v).to_bits(),
            expected.branch(v).to_bits(),
            "branch of {v}"
        );
    }
    let (fresh, _) = settled_down(&got, leaves)?;
    let word = leaf_words(&got);
    let below = leaves_below(&got);
    for v in 0..got.n_nodes() {
        let id = id_of[v];
        let same = |a: &[T], b: &[T]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| x.to_f64().map(f64::to_bits) == y.to_f64().map(f64::to_bits))
        };
        assert!(
            same(lazy.store.means(id), fresh.means(v as u32)),
            "means of {v}"
        );
        assert!(
            same(lazy.store.precisions(id), fresh.precisions(v as u32)),
            "precisions of {v}"
        );
        assert_eq!(lazy.word[id as usize], word[v], "word of {v}");
        assert_eq!(lazy.below[id as usize], below[v], "count of {v}");
        assert_eq!(lazy.by_word.get(&word[v]), Some(&id), "index of {v}");
    }
    Ok(())
}

/// Search step 6: the random phase, then the greedy phase.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults, which skip the random phase
/// * `verbosity` - [`Verbosity::Detailed`] prints one line per climb
///
/// ### Returns
///
/// The tree both phases finished on, with `n_moves` counting the moves of both,
/// or the error the primitive or the arena failed with.
pub fn nni<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
    verbosity: Verbosity,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let started = Instant::now();
    let random = nni_random(tree, leaves, Some(params))?;
    let mut best = nni_greedy(&random.tree, leaves, Some(params))?;
    best.n_moves += random.n_moves;
    if verbosity.detailed_verbosity() {
        println!(
            "    climb: {} moves over {} rounds, loglik {:.6e} ({:.2?})",
            best.n_moves,
            best.rounds,
            best.loglik,
            started.elapsed()
        );
    }

    // Iterated local search: perturb the best tree so far, climb, keep the
    // better of the two. Each restart draws from its own seed so that the
    // walks differ, and the whole thing is still a function of `params.seed`.
    let mut n_moves = best.n_moves;
    for restart in 0..params.n_restarts {
        let started = Instant::now();
        let perturbed = nni_random(
            &best.tree,
            leaves,
            Some(NniParams {
                seed: params.seed ^ (restart as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                ..params
            }),
        )?;
        let climbed = nni_greedy(&perturbed.tree, leaves, Some(params))?;
        n_moves += perturbed.n_moves + climbed.n_moves;
        if verbosity.detailed_verbosity() {
            println!(
                "    restart {} / {}: loglik {:.6e}, best {:.6e} ({:.2?})",
                restart + 1,
                params.n_restarts,
                climbed.loglik,
                best.loglik.max(climbed.loglik),
                started.elapsed()
            );
        }
        if climbed.loglik > best.loglik {
            best = climbed;
        }
    }
    best.n_moves = n_moves;
    Ok(best)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::global::optimise_branch_lengths;
    use crate::search::polytomy::resolve_polytomies;
    use crate::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary, splits};
    use crate::tree::{NO_NODE, simulate::SimulatedData};
    use std::collections::HashSet;

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

    /// Leaves below every node of a tree.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    ///
    /// ### Returns
    ///
    /// One sorted leaf set per node.
    fn leaf_sets(tree: &Tree) -> Vec<Vec<u32>> {
        let mut sets: Vec<Vec<u32>> = vec![Vec::new(); tree.n_nodes()];
        for leaf in 0..tree.n_leaves() {
            sets[leaf].push(leaf as u32);
        }
        for node in tree.internal_postorder() {
            let mut here: Vec<u32> = tree
                .children(node)
                .iter()
                .flat_map(|&c| sets[c as usize].clone())
                .collect();
            here.sort_unstable();
            sets[node as usize] = here;
        }
        sets
    }

    /// Canonicalise a leaf set into a split key the way `splits` does.
    ///
    /// ### Params
    ///
    /// * `side` - One side of the bipartition
    /// * `n_leaves` - Total leaf count
    ///
    /// ### Returns
    ///
    /// The side that does not hold leaf zero, sorted.
    fn canonical(side: &[u32], n_leaves: usize) -> Vec<u32> {
        let holds_zero = side.contains(&0);
        let mut out: Vec<u32> = if holds_zero {
            (0..n_leaves as u32)
                .filter(|leaf| !side.contains(leaf))
                .collect()
        } else {
            side.to_vec()
        };
        out.sort_unstable();
        out
    }

    /// A starting tree with its branch lengths optimised, as search step 4
    /// leaves them.
    ///
    /// The interchanges of step 6 run after the global branch-length
    /// optimisation of step 4, and they are a topology search: run them on a
    /// tree whose branches are all at their default and the landscape they read
    /// is dominated by the branch lengths being wrong rather than by the shape.
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

    #[test]
    fn test_a_classical_interchange_reconnects_four_subtrees() {
        // A binary tree, an internal edge with an internal node at both ends,
        // and no polytomy anywhere: the generalised move must collapse to the
        // textbook one. The four subtrees admit exactly three unrooted
        // topologies, and the tree that comes back has to be one of them.
        let (p, n) = (128usize, 16usize);
        let (data, w) = dataset(n, p, 4);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = &data.tree;
        let (down, up, _) = settle(tree, leaves).expect("settle");
        let sets = leaf_sets(tree);
        let base = splits(tree);

        let mut checked = 0usize;
        for k in tree.internal_postorder() {
            if interchange_members(tree, k) != Some(4) {
                continue;
            }
            let kids = tree.children(k);
            let l = tree.parent(k).expect("k is not the root");
            let sibling = tree
                .children(l)
                .iter()
                .copied()
                .find(|&c| c != k)
                .expect("l has another child");
            let a = &sets[kids[0] as usize];
            let b = &sets[kids[1] as usize];
            let c = &sets[sibling as usize];

            let split_k = canonical(&sets[k as usize], n);
            let mut without = base.clone();
            assert!(without.remove(&split_k), "the edge at {k} was not a split");

            // The three reconnections: A with B, which is what is already
            // there, A with C, and A with the upstream side. Each is one split
            // swapped for another, everything else untouched.
            let allowed: Vec<Vec<u32>> = [b, c]
                .iter()
                .map(|other| {
                    let mut side = a.clone();
                    side.extend_from_slice(other);
                    canonical(&side, n)
                })
                .chain(std::iter::once(canonical(a, n)))
                .collect();

            let spliced = interchange_at(tree, &down, &up, k, None)
                .expect("interchange")
                .expect("eligible edge");
            assert_eq!(spliced.tree.n_nodes(), tree.n_nodes());
            for node in spliced.tree.internal_postorder() {
                assert_eq!(
                    spliced.tree.children(node).len(),
                    2,
                    "the interchange at {k} left a polytomy"
                );
            }

            let got = splits(&spliced.tree);
            let extra: Vec<Vec<u32>> = got.difference(&without).cloned().collect();
            assert_eq!(
                extra.len(),
                1,
                "the interchange at {k} changed more than one split"
            );
            assert!(
                allowed.contains(&extra[0]),
                "the interchange at {k} produced {:?}, not one of the three reconnections {allowed:?}",
                extra[0]
            );
            assert_eq!(
                got.len(),
                base.len(),
                "the interchange at {k} changed the split count"
            );
            checked += 1;
        }
        assert!(checked > 0, "the fixture had no eligible internal edge");
    }

    /// Trees to walk every eligible edge of: a binary one and one carrying
    /// polytomies, so that both the two-child and the many-child collapse are
    /// exercised.
    ///
    /// ### Params
    ///
    /// * `n` - Number of leaves
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The trees.
    fn fixtures(n: usize, leaves: Leaves<'_, f64>) -> Vec<Tree> {
        let ladder = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let star = Tree::from_parents(parent, vec![0.5; n + 1], n).expect("star tree");
        let resolved = resolve_polytomies(&star, leaves, None)
            .expect("resolve")
            .tree;
        vec![ladder, resolved]
    }

    #[test]
    fn test_the_exact_gain_is_what_a_full_prune_reports() {
        // The candidate score. `Splice::gain` is measured against the collapsed
        // tree, `collapse_delta` supplies the rest, and nothing whole-tree is
        // computed to reach it. If the two do not add up to the difference of
        // two prunes then the greedy phase is choosing on a different quantity
        // from the one it claims to.
        let mut worst = 0.0f64;
        let mut checked = 0usize;
        for seed in [1u64, 2, 3] {
            let (p, n) = (64usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            for tree in fixtures(n, leaves) {
                let (down, up, before) = settle(&tree, leaves).expect("settle");
                let mut scratch = PeelScratch::<f64>::new(p);
                for k in tree.internal_postorder() {
                    let Some(l) = tree.parent(k) else { continue };
                    let Some(star) = collapsed_star(
                        &tree,
                        &Settled {
                            down: &down,
                            up: &up,
                        },
                        k,
                    ) else {
                        continue;
                    };
                    let spliced = splice_star(&tree, &star, None).expect("splice");
                    if spliced.n_merges == 0 {
                        continue;
                    }
                    let want = tree_loglik(&spliced.tree, leaves).expect("loglik") - before;
                    let got = spliced.gain
                        + collapse_delta(
                            &tree,
                            &Settled {
                                down: &down,
                                up: &up,
                            },
                            k,
                            l,
                            &star,
                            &mut scratch,
                        );
                    worst = worst.max((got - want).abs());
                    checked += 1;
                }
            }
        }
        println!("{checked} edges, worst absolute disagreement {worst:e} nats");
        assert!(checked > 100, "only {checked} edges exercised");
        // The worst disagreement measured here is `O(1e-13)` nats against tree
        // loglikelihoods of `O(1e3)`. The bound sits three orders above that and
        // one below the `min_gain` a move has to clear, which is what it has to
        // stay under for the search's answer not to turn on it.
        assert!(worst < 1e-10, "worst disagreement {worst:e} nats");
    }

    #[test]
    fn test_the_structural_filter_rejects_exactly_what_a_fingerprint_would() {
        // The greedy phase's topology filter used to splice a tree and
        // fingerprint it, which is `O(n)` per candidate. `rebuilds_the_same_splits`
        // reads the star's own ancestors instead, and the two have to agree
        // edge for edge: a filter that is merely nearly the same one silently
        // changes both the search's answer and whether it terminates.
        let (mut same, mut different) = (0usize, 0usize);
        for seed in [1u64, 2, 3] {
            let (p, n) = (64usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            for tree in fixtures(n, leaves) {
                let (down, up, _) = settle(&tree, leaves).expect("settle");
                let here = crate::search::split_fingerprint(&tree);
                let below = leaves_below(&tree);
                for k in tree.internal_postorder() {
                    let Some(l) = tree.parent(k) else { continue };
                    let Some(star) = collapsed_star(
                        &tree,
                        &Settled {
                            down: &down,
                            up: &up,
                        },
                        k,
                    ) else {
                        continue;
                    };
                    let result = resolve_star(star.view(), None).expect("resolve");
                    if result.merges.is_empty() {
                        continue;
                    }
                    let k_lo = tree.children(l).len() - 1;
                    let k_hi = k_lo + tree.children(k).len();
                    let last = star.member_nodes.len() - 1;
                    let member_leaves: Vec<usize> = star
                        .member_nodes
                        .iter()
                        .enumerate()
                        .map(|(i, &node)| {
                            if star.has_upstream && i == last {
                                tree.n_leaves() - below[l as usize]
                            } else {
                                below[node as usize]
                            }
                        })
                        .collect();
                    let structural = rebuilds_the_same_splits(
                        &result,
                        k_lo,
                        k_hi,
                        &member_leaves,
                        tree.n_leaves(),
                    );

                    let spliced = splice_star(&tree, &star, None).expect("splice");
                    let by_fingerprint = crate::search::split_fingerprint(&spliced.tree) == here;
                    assert_eq!(
                        structural, by_fingerprint,
                        "seed {seed}, edge {k}: the structural filter said {structural} and the \
                         fingerprint said {by_fingerprint}"
                    );
                    if structural {
                        same += 1;
                    } else {
                        different += 1;
                    }
                }
            }
        }
        // A filter that never fires and one that always does are both useless,
        // so both outcomes have to be represented.
        println!("{same} proposals rebuilt the same splits, {different} did not");
        assert!(
            same > 0 && different > 0,
            "{same} same, {different} different"
        );
    }

    #[test]
    fn test_the_greedy_phase_never_lowers_the_loglikelihood() {
        // Monotonicity, from a deliberately wrong starting topology so that
        // there is plenty for the phase to do.
        for seed in [1u64, 2, 3] {
            let (p, n) = (128usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = tree_loglik(&start, leaves).expect("loglik");
            let out = nni_greedy(&start, leaves, None).expect("greedy");
            let after = tree_loglik(&out.tree, leaves).expect("loglik");

            assert!(
                after >= before - 1e-9,
                "seed {seed}: {before} fell to {after}"
            );
            approx::assert_relative_eq!(out.loglik, after, max_relative = 1e-12);
            assert!(out.rounds < NniParams::default().max_rounds);
        }
    }

    #[test]
    fn test_the_greedy_phase_stops_at_a_fixed_point() {
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 6);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let once = nni_greedy(&start, leaves, None).expect("greedy");
        let twice = nni_greedy(&once.tree, leaves, None).expect("greedy");
        assert_eq!(twice.n_moves, 0);
        assert_eq!(twice.rounds, 1);
        assert_eq!(splits(&twice.tree), splits(&once.tree));
    }

    #[test]
    fn test_the_greedy_phase_does_not_chase_branch_lengths() {
        // Started on the generating tree with its branch lengths optimised,
        // there is no topology left to find. Every proposal from here rebuilds
        // the same splits and only reoptimises the three branches the star
        // primitive creates, so every one must be discarded. Without that
        // filter this ran for over two hundred rounds at a Robinson-Foulds
        // distance of zero throughout.
        let (p, n) = (256usize, 32usize);
        let (data, w) = dataset(n, p, 5);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&data.tree, leaves);
        let out = nni_greedy(&start, leaves, None).expect("greedy");
        println!(
            "from the truth: {} moves over {} rounds, RF {}",
            out.n_moves,
            out.rounds,
            robinson_foulds(&out.tree, &data.tree).expect("rf")
        );
        assert!(
            out.rounds < 10,
            "{} rounds from a tree with nothing to find",
            out.rounds
        );
    }

    #[test]
    fn test_the_greedy_phase_recovers_the_generating_topology() {
        // The recovery test. Start from a ladder, which shares almost nothing
        // with the tree the data came off, put its branch lengths where search
        // step 4 would, and show the distance to truth falls. The numbers are
        // printed rather than pinned to a threshold beyond "it must fall":
        // what is being tested is the direction.
        for seed in [1u64, 2, 3] {
            let (p, n) = (256usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = robinson_foulds(&start, &data.tree).expect("rf");
            let out = nni_greedy(&start, leaves, None).expect("greedy");
            let after = robinson_foulds(&out.tree, &data.tree).expect("rf");
            println!(
                "seed {seed}: RF {before} -> {after} of a possible {} in {} moves over {} rounds",
                2 * (n - 3),
                out.n_moves,
                out.rounds
            );
            assert!(after < before, "seed {seed}: RF went {before} -> {after}");
        }
    }

    #[test]
    fn test_the_greedy_phase_recovers_what_the_random_phase_costs() {
        // The random phase is not monotone by construction: a collapse can lose
        // a split that the resampled star does not put back. What is claimed is
        // that the greedy phase which follows recovers at least the tree the
        // pair started from.
        let (p, n) = (256usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        // Start at a local optimum of the greedy phase, so that the random
        // phase has something to escape from.
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let settled = nni_greedy(&start, leaves, None).expect("greedy");

        for seed in [1u64, 7, 19] {
            let params = NniParams {
                n_random: n / 4,
                seed,
                ..NniParams::default()
            };
            let random = nni_random(&settled.tree, leaves, Some(params)).expect("random");
            let out = nni(&settled.tree, leaves, Some(params), Verbosity::Quiet).expect("nni");
            println!(
                "seed {seed}: start {:.3} (RF to truth {}), after {} random moves {:.3} \
                 (RF to the start {}), after greedy {:.3}",
                settled.loglik,
                robinson_foulds(&settled.tree, &data.tree).expect("rf"),
                random.n_moves,
                random.loglik,
                robinson_foulds(&random.tree, &settled.tree).expect("rf"),
                out.loglik
            );
            assert!(
                out.loglik >= settled.loglik - 1e-6,
                "seed {seed}: random-then-greedy left {} against a start of {}",
                out.loglik,
                settled.loglik
            );
        }
    }

    #[test]
    fn test_the_random_phase_is_deterministic_whatever_the_thread_count() {
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 8);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let params = NniParams {
            n_random: 12,
            seed: 0x2545_F491_4F6C_DD1D,
            ..NniParams::default()
        };
        let reference = nni_random(&data.tree, leaves, Some(params)).expect("random");

        for threads in [1usize, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got =
                pool.install(|| nni_random(&data.tree, leaves, Some(params)).expect("random"));
            assert_eq!(
                splits(&got.tree),
                splits(&reference.tree),
                "topology moved at {threads} threads"
            );
            assert_eq!(got.tree.branches(), reference.tree.branches());
            assert_eq!(got.loglik.to_bits(), reference.loglik.to_bits());
            assert_eq!(got.n_moves, reference.n_moves);
        }
    }

    #[test]
    fn test_the_greedy_phase_is_deterministic_whatever_the_thread_count() {
        // The round scores every eligible edge in parallel and reduces to one
        // winner, so a tie broken by arrival order rather than by node id would
        // make the search thread-dependent. A ladder is the fixture that puts
        // real work in front of the phase: it is as far from the generating
        // tree as the simulator gets and takes tens of moves to fix.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let reference = nni_greedy(&start, leaves, None).expect("greedy");
        assert!(
            reference.n_moves > 0,
            "the fixture has to give the phase something to do"
        );

        for threads in [1usize, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| nni_greedy(&start, leaves, None).expect("greedy"));
            assert_eq!(
                splits(&got.tree),
                splits(&reference.tree),
                "topology moved at {threads} threads"
            );
            assert_eq!(got.tree.branches(), reference.tree.branches());
            assert_eq!(got.loglik.to_bits(), reference.loglik.to_bits());
            assert_eq!(got.n_moves, reference.n_moves);
            assert_eq!(got.rounds, reference.rounds);
        }
    }

    /// Greedy knobs with the given search.
    ///
    /// ### Params
    ///
    /// * `search` - Exact or approximate
    ///
    /// ### Returns
    ///
    /// The default knobs with that search.
    fn with(search: NniSearch) -> Option<NniParams> {
        Some(NniParams {
            search,
            ..NniParams::default()
        })
    }

    #[test]
    fn test_a_rescore_radius_wider_than_the_tree_makes_the_exact_moves() {
        // With every cached gain thrown away after every move, each round
        // rescores every edge on the current rows, so the leader is the exact
        // phase's winner and its fresh rescore reproduces it. The only
        // difference is one extra full scan at the end to confirm.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let exact = nni_greedy(&start, leaves, with(NniSearch::Exact)).expect("exact");
        let wide = nni_greedy(
            &start,
            leaves,
            with(NniSearch::Approximate(NniApprox {
                rescore_radius: 4 * n,
            })),
        )
        .expect("lazy");
        assert!(exact.n_moves > 0);
        assert_eq!(wide.n_moves, exact.n_moves);
        assert_eq!(splits(&wide.tree), splits(&exact.tree));
        assert_eq!(wide.loglik.to_bits(), exact.loglik.to_bits());
    }

    #[test]
    fn test_the_lazy_phase_stops_only_where_the_exact_phase_would() {
        // Stale gains may hide an improving edge for a while, never for good:
        // the phase ends only on a full scan that finds nothing. So the exact
        // phase, handed the lazy phase's tree, must find no move, and the
        // loglikelihood must be the tree's own.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let before = tree_loglik(&start, leaves).expect("loglik");
        for radius in [1usize, 2, 3] {
            let lazy = nni_greedy(
                &start,
                leaves,
                with(NniSearch::Approximate(NniApprox {
                    rescore_radius: radius,
                })),
            )
            .expect("lazy");
            assert!(lazy.loglik > before, "radius {radius}");
            approx::assert_relative_eq!(
                lazy.loglik,
                tree_loglik(&lazy.tree, leaves).expect("loglik"),
                max_relative = 1e-12
            );
            let again = nni_greedy(&lazy.tree, leaves, with(NniSearch::Exact)).expect("exact");
            assert_eq!(again.n_moves, 0, "radius {radius} stopped early");
        }
    }

    #[test]
    fn test_a_different_seed_gives_a_different_walk() {
        // Otherwise the random phase is randomised in name only.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 8);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let walk = |seed: u64| {
            let params = NniParams {
                n_random: 12,
                seed,
                ..NniParams::default()
            };
            nni_random(&data.tree, leaves, Some(params)).expect("random")
        };
        let seen: HashSet<Vec<Vec<u32>>> = (0..6u64)
            .map(|s| {
                let mut keys: Vec<Vec<u32>> = splits(&walk(s).tree).into_iter().collect();
                keys.sort();
                keys
            })
            .collect();
        assert!(seen.len() > 1, "six seeds all produced the same tree");
    }

    #[test]
    fn test_a_star_tree_has_no_eligible_edge() {
        // Every leaf hangs off the root, so there is no internal edge at all.
        let n = 12usize;
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let tree = Tree::from_parents(parent, vec![0.5; n + 1], n).expect("star tree");
        for node in tree.internal_postorder() {
            assert_eq!(interchange_members(&tree, node), None);
        }

        let p = 64usize;
        let (data, w) = dataset(16, p, 21);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let out = nni_greedy(&tree, leaves, None).expect("greedy");
        assert_eq!(out.n_moves, 0);
        assert_eq!(out.rounds, 1);
    }

    #[test]
    fn test_a_tree_too_small_for_an_interchange() {
        // Four leaves on a balanced binary tree. The root has two children, so
        // collapsing either of them leaves a three-member star and there is
        // nothing to interchange.
        let (p, n) = (64usize, 4usize);
        let (data, w) = dataset(16, p, 15);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let tree = Tree::balanced_binary(n, 1.0).expect("balanced");
        for node in tree.internal_postorder() {
            assert_eq!(interchange_members(&tree, node), None);
        }
        let out = nni_greedy(&tree, leaves, None).expect("greedy");
        assert_eq!(out.n_moves, 0);
    }

    #[test]
    fn test_interchanges_survive_a_polytomy() {
        // The generalised move has to work on a tree that is not binary, which
        // is the whole point of SPEC.md section 9.4. Resolve the polytomies of
        // a star tree first so the fixture is a real search state, then leave
        // one node unresolved by hand.
        let (p, n) = (128usize, 24usize);
        let (data, w) = dataset(32, p, 12);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let start = Tree::from_parents(parent, vec![0.5; n + 1], n).expect("star tree");
        let resolved = resolve_polytomies(&start, leaves, None).expect("resolve");

        let before = tree_loglik(&resolved.tree, leaves).expect("loglik");
        let out = nni_greedy(&resolved.tree, leaves, None).expect("greedy");
        assert!(out.loglik >= before - 1e-9);
    }
}
