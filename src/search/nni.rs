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
//! whole-tree figure, and needs a floor that scales with `n p`; measured
//! 2026-09-13 at 10,000 cells by 2,767 genes, every one of the 89 accepted
//! interchanges gained between 0.08 and 26 nats and none was rounding.
//!
//! ### This is a topology search and only a topology search
//!
//! A collapse and re-resolution that puts the same subtrees back where they
//! were is not an interchange at all: it is a reoptimisation of the three
//! branches the star primitive creates at `l`, and it nearly always gains a
//! little. Accepting those turns the greedy phase into an extremely expensive
//! branch-length descent. Measured 2026-08-31, starting from the generating
//! tree itself at 32 to 64 leaves and 256 features, that ran for 104 to 239
//! rounds with the Robinson-Foulds distance to the truth pinned at zero
//! throughout. So a proposal is discarded unless it changes the tree's splits;
//! branch lengths are search steps 4 and 7, which do the same job globally and
//! for a fraction of the cost.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::search::polytomy::{CentreStar, Splice, splice_star};
use crate::search::star::{StarParams, StarResult, StarSelection, resolve_star};
use crate::search::{Leaves, leaves_below, settle, tree_loglik};
use crate::tree::Tree;
use crate::utils::kernels::prune_general;
use crate::utils::rng::SplitMix64;
use crate::utils::traits::BonsaiFloat;
use rayon::prelude::*;

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
/// grows with the leaf count. Measured 2026-08-31 on simulated data at 256
/// features, starting from a ladder with its branch lengths already at the step
/// 4 optimum and running to a Robinson-Foulds distance of zero from the
/// generating tree: 23 rounds at 32 leaves and 53 at 64, which is a shade under
/// one round per leaf. The default therefore covers roughly ten thousand
/// leaves; beyond that a caller should raise it rather than accept a truncated
/// search.
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
/// 2026-08-31 at 32 leaves, eight moves and eight seeds: at 8 features all
/// eight seeds moved the tree off its starting topology, at 32 features four,
/// and at 128 and 512 features two and three. So a budget buys real
/// diversification on small feature sets and mostly buys branch-length
/// reoptimisation on large ones.
const DEFAULT_RANDOM_MOVES: usize = 0;

/// Default for [`NniParams::n_restarts`].
///
/// Zero: one random phase, if any, then one greedy phase, which is the
/// composition SPEC.md section 9.4 describes. Measured against iterated local
/// search on 2026-09-13; see [`nni`].
const DEFAULT_RESTARTS: usize = 0;

/// Default for [`NniParams::temperature`].
///
/// One is the specification's distribution: weights proportional to the
/// likelihood of the resulting tree. Measured 2026-09-13; see [`nni_random`].
pub const DEFAULT_RANDOM_TEMPERATURE: f64 = 1.0;

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
    /// The greedy phase settles the tree at the top of every round and stops on
    /// a round that finds no move, so what comes back is that round's own
    /// sweep. Only a run truncated by [`NniParams::max_rounds`] returns the
    /// last sweep plus the accepted gains instead.
    pub loglik: f64,
    /// Number of moves performed.
    pub n_moves: usize,
    /// Number of greedy rounds, the last of which found no improving move.
    /// Zero for a run of the random phase alone.
    pub rounds: usize,
    /// What every greedy round saw, in order. Empty for the random phase.
    pub trace: Vec<NniRound>,
}

/// What scanning one edge produced: that edge's contribution to the round's
/// tallies, and the proposal itself when it yielded one worth taking.
type ScannedEdge<T> = (NniRound, Option<(f64, u32, CentreStar<T>)>);

/// Where one greedy round's candidates fell out.
///
/// The counts nest: every edge is either eligible or not, every eligible edge
/// either produced a proposal or the primitive merged nothing, every proposal
/// either changed a split or rebuilt the tree it came from, and every changed
/// proposal either cleared [`StarParams::min_gain`] or did not. Kept because
/// "the phase found nothing" and "the phase rejected everything it found" look
/// identical from outside and call for opposite fixes.
#[derive(Clone, Copy, Debug, Default)]
pub struct NniRound {
    /// Edges whose collapse leaves a star of at least four members.
    pub eligible: usize,
    /// Of those, edges where the primitive merged at least one pair.
    pub proposed: usize,
    /// Of those, proposals whose splits differ from the current tree's.
    pub changed: usize,
    /// Of those, proposals whose exact gain clears the floor.
    pub improving: usize,
    /// The best exact gain among the changed proposals, whether or not it
    /// cleared the floor; `NEG_INFINITY` when nothing changed a split.
    pub best_gain: f64,
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
fn interchange_members(tree: &Tree, k: u32) -> Option<usize> {
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
/// * `down` - Down rows, settled by [`NodeState::prune`] against this tree
/// * `up` - Up rows, settled by [`UpState::sweep`] against the same
/// * `k` - The node to delete
///
/// ### Returns
///
/// The star, or `None` if the edge is not eligible.
fn collapsed_star<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
    k: u32,
) -> Option<CentreStar<T>> {
    let n = interchange_members(tree, k)?;
    let l = tree.parent(k)?;
    let p = down.n_features();
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
        star.member_nodes.push(node);
        star.means.extend_from_slice(down.means(node));
        star.precisions.extend_from_slice(down.precisions(node));
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
        star.member_nodes.push(par);
        star.means.extend_from_slice(up.means(l));
        star.precisions.extend_from_slice(up.precisions(l));
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
    match collapsed_star(tree, down, up, k) {
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
/// * `down` - Down rows, settled against this tree
/// * `up` - Up rows, settled against the same
/// * `k` - The node the collapse deletes
/// * `l` - Its parent, the centre of the star
/// * `star` - The collapsed star, from [`collapsed_star`]
/// * `scratch` - Peel scratch, reused across candidates
///
/// ### Returns
///
/// The loglikelihood of the collapsed tree less that of `tree`, in nats.
fn collapse_delta<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
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

    let below = |node: u32| (down.means(node), down.precisions(node), tree.branch(node));
    let at_k: Vec<(&[T], &[T], f64)> = tree.children(k).iter().map(|&c| below(c)).collect();
    let mut at_l: Vec<(&[T], &[T], f64)> = tree.children(l).iter().map(|&c| below(c)).collect();
    if tree.parent(l).is_some() {
        at_l.push((up.means(l), up.precisions(l), tree.branch(l)));
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
        trace: Vec::new(),
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
/// Measured 2026-09-06 at 200 features on the tree search step 5 leaves
/// behind, on the two sizes where the phase happened to run exactly one round
/// so that the move count cannot confound the timing: **0.238 s at 1024 leaves
/// and 1.979 s at 4096 before, `n^1.53`, against 0.125 s and 0.512 s after,
/// `n^1.02`.** The whole phase was `n^3.05` in `benches/steps.rs` because the
/// round count grows on top of that.
///
/// **What was expensive was not what it looked like.** The `O(n p)` re-prune
/// per candidate is nearly free at this point in the search, because the
/// topology filter discards almost everything before it: counted over the same
/// runs, every one of the 1021 to 4093 eligible edges produced merges and
/// between zero and one of them per round changed a split. The quadratic was
/// the filter itself, which spliced a whole tree and fingerprinted it to answer
/// a question about one star's ancestors. Scoring on the exact gain rather than
/// on a re-prune is what makes the *other* end of the search cheap, where the
/// tree is far from converged and a large fraction of proposals do change a
/// split: over the forty fixtures of 32 to 256 leaves and 16 to 256 features
/// used for the identical-tree check, the longest of which takes 240 moves from
/// a ladder, the two phases together are 1.44 times faster.
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
    let mut tree = tree.clone();
    let mut best: Option<f64> = None;
    let mut n_moves = 0usize;
    let mut rounds = 0usize;
    let mut trace: Vec<NniRound> = Vec::new();

    while rounds < params.max_rounds {
        rounds += 1;
        let (down, up, loglik) = settle(&tree, leaves)?;
        best = Some(loglik);
        let below = leaves_below(&tree);
        let n_leaves = tree.n_leaves();

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
        //
        // The counts ride along with the winner. They are integers and a max,
        // so the split order cannot change them.
        let edges: Vec<u32> = tree.internal_postorder().collect();
        let (round, winner) = edges
            .par_iter()
            .map_init(
                || PeelScratch::<T>::new(leaves.n_features),
                |scratch, &k| -> Result<ScannedEdge<T>, BonsaiErrors> {
                    let mut seen = NniRound {
                        best_gain: f64::NEG_INFINITY,
                        ..NniRound::default()
                    };
                    let Some(l) = tree.parent(k) else {
                        return Ok((seen, None));
                    };
                    let Some(star) = collapsed_star(&tree, &down, &up, k) else {
                        return Ok((seen, None));
                    };
                    seen.eligible = 1;
                    let result = resolve_star(star.view(), Some(params.star))?;
                    if result.merges.is_empty() {
                        return Ok((seen, None));
                    }
                    seen.proposed = 1;
                    // A proposal that puts the same subtrees back where they were is
                    // not an interchange: it is a reoptimisation of the three branches
                    // the star primitive creates at `l`. Those nearly always gain a
                    // little, and taking them turns the phase into branch-length
                    // descent that steps 4 and 7 do properly and far more cheaply.
                    // Measured 2026-08-31: from the generating tree itself, at 32 to 64
                    // leaves and 256 features, accepting them ran 104 to 239 rounds
                    // with the Robinson-Foulds distance pinned at zero throughout, so
                    // every one of those rounds was branch lengths and none was
                    // topology.
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
                        return Ok((seen, None));
                    }
                    seen.changed = 1;

                    let gain: f64 = result.merges.iter().map(|x| x.gain).sum::<f64>()
                        + collapse_delta(&tree, &down, &up, k, l, &star, scratch);
                    seen.best_gain = gain;
                    if gain > params.star.min_gain {
                        seen.improving = 1;
                        Ok((seen, Some((gain, k, star))))
                    } else {
                        Ok((seen, None))
                    }
                },
            )
            .try_reduce(
                || {
                    (
                        NniRound {
                            best_gain: f64::NEG_INFINITY,
                            ..NniRound::default()
                        },
                        None,
                    )
                },
                |(ra, a), (rb, b)| {
                    let round = NniRound {
                        eligible: ra.eligible + rb.eligible,
                        proposed: ra.proposed + rb.proposed,
                        changed: ra.changed + rb.changed,
                        improving: ra.improving + rb.improving,
                        best_gain: ra.best_gain.max(rb.best_gain),
                    };
                    let winner = match (a, b) {
                        (None, other) | (other, None) => other,
                        (Some(x), Some(y)) => {
                            if y.0 > x.0 || (y.0 == x.0 && y.1 < x.1) {
                                Some(y)
                            } else {
                                Some(x)
                            }
                        }
                    };
                    Ok((round, winner))
                },
            )?;
        trace.push(round);

        match winner {
            None => break,
            Some((gain, _, star)) => {
                tree = splice_star(&tree, &star, Some(params.star))?.tree;
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
        trace,
    })
}

/// Search step 6: the random phase, then the greedy phase.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults, which skip the random phase
///
/// ### Returns
///
/// The tree both phases finished on, with `n_moves` counting the moves of both,
/// or the error the primitive or the arena failed with.
pub fn nni<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let random = nni_random(tree, leaves, Some(params))?;
    let mut best = nni_greedy(&random.tree, leaves, Some(params))?;
    best.n_moves += random.n_moves;

    // Iterated local search: perturb the best tree so far, climb, keep the
    // better of the two. Each restart draws from its own seed so that the
    // walks differ, and the whole thing is still a function of `params.seed`.
    let mut n_moves = best.n_moves;
    for restart in 0..params.n_restarts {
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
                    let Some(star) = collapsed_star(&tree, &down, &up, k) else {
                        continue;
                    };
                    let spliced = splice_star(&tree, &star, None).expect("splice");
                    if spliced.n_merges == 0 {
                        continue;
                    }
                    let want = tree_loglik(&spliced.tree, leaves).expect("loglik") - before;
                    let got =
                        spliced.gain + collapse_delta(&tree, &down, &up, k, l, &star, &mut scratch);
                    worst = worst.max((got - want).abs());
                    checked += 1;
                }
            }
        }
        println!("{checked} edges, worst absolute disagreement {worst:e} nats");
        assert!(checked > 100, "only {checked} edges exercised");
        // Measured 2026-09-06 at 3.0e-13 nats over these 174 edges, against
        // tree loglikelihoods of `O(1e3)`. The bound is set three orders above
        // that and one below the `min_gain` a move has to clear, which is what
        // it has to stay under for the search's answer not to turn on it.
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
                    let Some(star) = collapsed_star(&tree, &down, &up, k) else {
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
            let out = nni(&settled.tree, leaves, Some(params)).expect("nni");
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
