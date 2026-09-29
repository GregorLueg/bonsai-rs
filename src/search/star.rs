//! The star primitive of SPEC.md section 9.1.
//!
//! Given a centre and the star of effective leaves hanging off it, score every
//! candidate pair with [`crate::model::merge::score_merge`], insert an ancestor
//! above the pair [`StarSelection`] picks, summarise it as an effective leaf
//! (SPEC.md section 4) so the rest is a star again, and repeat. Stops at three
//! members or when no pair gives a gain worth taking.
//!
//! Drives search step 2 (centre is the root, members are every leaf), step 3
//! (centre is a polytomy node) and step 6 (after an NNI edge deletion). A centre
//! with a parent passes its upstream side in as one more member, so three
//! members always means "resolved".
//!
//! Candidate pairs come from a [`CandidatePairs`] provider: [`AllPairs`] here,
//! the `k`-NN restriction of section 11 in [`crate::search::candidates`], the
//! upper bounds of section 10 in [`crate::search::bounds`]. [`Round`] and
//! [`CandidatePairs::bounds`] are the seams those need.

use crate::errors::BonsaiErrors;
use crate::model::merge::{EffLeaf, MergeParams, MergeScore, MergeScratch, score_merge};
use crate::tree::{NO_NODE, Tree};
use crate::utils::rng::SplitMix64;
use crate::utils::traits::{BonsaiFloat, narrow, wide};
use crate::utils::verbosity::{Verbosity, report_decile_progress};
use rayon::prelude::*;
use std::time::Instant;

////////////////
// Parameters //
////////////////

/// Number of members at which the centre is resolved and the primitive stops.
///
/// SPEC.md section 9.1. A degree-three node is fully resolved in an unrooted
/// tree; a fourth merge would only insert a degree-two node.
const MIN_CENTRE_MEMBERS: usize = 3;

/// Default for [`StarParams::min_gain`], in nats.
///
/// Ours. Exactly-zero-gain merges score `7.1e-15` at 64 features and `-3.6e-12`
/// at 32768, i.e. `1.1e-16` per feature (one `f64` rounding of an `O(p)` sum),
/// so `1e-9` clears the floor by 900x at ten thousand features. See
/// `test_the_default_min_gain_clears_the_zero_gain_floor`.
///
/// A merge-score floor only, for sums of magnitude `O(p)`. Not valid for a
/// whole-tree loglikelihood of magnitude `O(n p)`; [`crate::search::spr`]
/// carries its own scale-relative floor.
const DEFAULT_MIN_GAIN: f64 = 1e-9;

/// How many bound-ordered pairs [`walk_bounded`] scores before it rechecks the
/// stopping rule.
///
/// Ours, measured: wall time is flat above 16 while pairs scored keeps growing,
/// and 4 is too small to fill the pool. Does not change the answer at any value.
/// Public because rounds offering fewer pairs than this always walk all of
/// them, which the online ellipsoid sizing of SPEC.md section 10.5 must not read
/// as "the bounds pruned nothing".
pub const BOUND_WALK_CHUNK: usize = 16;

/// Rounds between exact recomputations of the centre's effective leaf when
/// [`StarParams::incremental_centre`] is on.
const CENTRE_EXACT_EVERY: usize = 32;

/// Fewest candidate pairs a scan spreads over the thread pool.
///
/// Ours, measured: search step 5 resolves six-pair stars from inside a parallel
/// loop, where a parallel scan costs several times the sequential one.
const PAR_PAIRS_MIN: usize = 64;

/// How the primitive picks the pair to merge in a round.
///
/// SPEC.md section 9.4. The greedy rule drives search steps 2 and 3 and is the
/// default; the weighted rule is the random phase of the nearest-neighbour
/// interchanges.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum StarSelection {
    /// Take the highest-scoring pair of the round.
    #[default]
    Greedy,
    /// Sample a pair with probability proportional to the likelihood of the
    /// tree the merge would produce.
    ///
    /// A softmax over the gains, round maximum subtracted first.
    ///
    /// **Deviation.** The specification samples over every pair; this samples
    /// over pairs clearing [`StarParams::min_gain`] and stops when none do, so
    /// the stopping rule matches greedy mode.
    ///
    /// Materialises one score per pair, so it wants the small stars of an
    /// interchange. Gaps are `O(p)` nats, so at high feature counts it
    /// concentrates on the greedy pick (measured through
    /// [`crate::search::nni::nni_random`]: seeds that leave the starting
    /// topology vanish within a couple of hundred features).
    Weighted {
        /// Seed of the splitmix64 stream, one draw per round after scoring, so
        /// the pick does not depend on the thread count.
        seed: u64,
        /// Softmax temperature; `1.0` is the specification's distribution,
        /// larger flattens it. See
        /// [`crate::search::nni::DEFAULT_RANDOM_TEMPERATURE`].
        temperature: f64,
    },
}

/// Tuning knobs for the star primitive.
#[derive(Clone, Copy, Debug)]
pub struct StarParams {
    /// Smallest loglikelihood gain, in nats, that will be accepted as a merge.
    ///
    /// Strictly greater than. Absolute, not scaled by the feature count: raise
    /// it at unusually large `p`; see `DEFAULT_MIN_GAIN`.
    pub min_gain: f64,
    /// Which pair of the round is merged.
    pub selection: StarSelection,
    /// Branch-length solve knobs handed to [`score_merge`].
    pub merge: MergeParams,
    /// Update the centre's effective leaf by removing the merged pair and adding
    /// the ancestor (`O(p)`, the peel of section 8.1 plus the ancestor's
    /// contribution), rather than re-accumulating over every member (`O(n p)` a
    /// round).
    ///
    /// Off by default. It only pays once the bounds of section 10 cut the scan
    /// to a handful of pairs, and it is a silent numerical change to a path
    /// every search step uses. Drift is measured safe (`O(1e-14)` relative in
    /// the precision, `O(1e-12)` in the mean, identical trees on every fixture;
    /// see `test_the_incremental_centre_leaf_does_not_drift`), and an exact
    /// recompute every `CENTRE_EXACT_EVERY` rounds caps it. The shipped pipeline
    /// composes the bounds but leaves this off; revisit with a measurement.
    pub incremental_centre: bool,
}

impl Default for StarParams {
    /// `DEFAULT_MIN_GAIN`, greedy selection and the default [`MergeParams`].
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            min_gain: DEFAULT_MIN_GAIN,
            selection: StarSelection::Greedy,
            merge: MergeParams::default(),
            incremental_centre: false,
        }
    }
}

/////////////////////
// Candidate pairs //
/////////////////////

/// One round's read-only view of the star, as a candidate provider sees it.
///
/// Both slabs are indexed by *node id*, row-major with stride `n_features`, and
/// cover every node created so far (members and swallowed children), so a
/// provider can hold state keyed by node id. Precisions are not diffusion
/// corrected (section 4); `branch` carries what the correction needs.
///
/// A geometry-only provider such as [`crate::search::candidates::KnnCandidates`]
/// reads the first four fields. The rest serve providers that score pairs
/// (section 10); see [`Round::score_pair`].
#[derive(Clone, Copy, Debug)]
pub struct Round<'a, T> {
    /// Node ids of the current star members, ascending.
    pub members: &'a [u32],
    /// Effective means of every node created so far, row-major `[node][g]`.
    pub means: &'a [T],
    /// Effective precisions, same layout.
    pub precisions: &'a [T],
    /// Row stride of both slabs, that is, the number of features.
    pub n_features: usize,
    /// Branch length from each node to its ancestor, or to the centre, indexed
    /// by node id.
    pub branch: &'a [f64],
    /// The centre's own effective means over the current members, length
    /// `n_features`. This is `M[g,r]` of SPEC.md section 8.1.
    pub centre_means: &'a [f64],
    /// The centre's own effective precisions, `W[g,r]`, same length.
    pub centre_precisions: &'a [f64],
    /// Branch-length solve knobs the scan uses; a provider scoring pairs must
    /// pass these through to match the scan bit for bit.
    pub merge: MergeParams,
    /// Gain of the merge accepted in the previous round, or negative infinity
    /// in the first round.
    ///
    /// Not a sound bound on this round's best: the root moves between rounds and
    /// a later merge can beat an earlier one.
    pub best_gain: f64,
    /// How many pairs the previous round's scan actually scored, or zero in the
    /// first round.
    ///
    /// Read by the online ellipsoid sizing of SPEC.md section 10.5.
    pub scored_last_round: usize,
}

impl<'a, T: BonsaiFloat> Round<'a, T> {
    /// Score one candidate pair exactly as the scan will.
    ///
    /// For the upper-bound machinery of SPEC.md section 10. The peel of section
    /// 8.1 is left in `scratch` for callers differentiating against the centre.
    ///
    /// ### Params
    ///
    /// * `i` - Position of the first member of the pair in `members`
    /// * `j` - Position of the second
    /// * `scratch` - Reusable buffers, sized to `n_features`
    ///
    /// ### Returns
    ///
    /// The gain and the three optimised branch lengths, or the error the
    /// branch-length solve failed with.
    pub fn score_pair(
        &self,
        i: usize,
        j: usize,
        scratch: &mut PairScratch<T>,
    ) -> Result<MergeScore, BonsaiErrors> {
        let work = Working {
            m: self.means,
            w: self.precisions,
            branch: self.branch,
            mc: self.centre_means,
            wc: self.centre_precisions,
            p: self.n_features,
        };
        score_pair_raw(&work, self.members, i, j, self.merge, scratch)
    }
}

/// Source of the candidate pairs a round of the scan considers.
///
/// The seam for SPEC.md sections 10 and 11. Called once per round and notified
/// after every merge, from one thread in round order, so a provider needs no
/// locking.
pub trait CandidatePairs<T: BonsaiFloat> {
    /// Fill `out` with the pairs to score this round.
    ///
    /// ### Params
    ///
    /// * `round` - Read-only view of the current round
    /// * `out` - Destination, cleared by the caller, holding positions into
    ///   `round.members` with the smaller position first
    ///
    /// ### Returns
    ///
    /// Nothing, or the error the provider failed with. An empty `out` stops the
    /// primitive, so a provider with nothing left to offer just returns.
    fn candidates(
        &mut self,
        round: Round<'_, T>,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<(), BonsaiErrors>;

    /// Notification that a merge has been accepted.
    ///
    /// Called after the ancestor is summarised and put back into the star. The
    /// default does nothing.
    ///
    /// ### Params
    ///
    /// * `merge` - The merge that was performed
    /// * `ancestor` - The new ancestor's row of the means and precisions slabs
    fn merged(&mut self, merge: &StarMerge, ancestor: EffLeaf<'_, T>) {
        let _ = (merge, ancestor);
    }

    /// Upper bounds on the gains of the pairs the last [`CandidatePairs::candidates`]
    /// call emitted, in that order.
    ///
    /// SPEC.md section 10.4. When `Some`, the greedy scan walks the pairs in
    /// order and stops once the best true gain exceeds the next bound. The slice
    /// must match `out` in length and **must be non-increasing**, or the walk
    /// stops on a pair that was not the best. `None`, the default, scores every
    /// pair.
    ///
    /// ### Returns
    ///
    /// The bounds, or `None` for a provider that does not maintain them.
    fn bounds(&self) -> Option<&[f64]> {
        None
    }
}

/// Every pair of the current members.
///
/// The exhaustive provider, the primitive's default.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllPairs;

impl<T: BonsaiFloat> CandidatePairs<T> for AllPairs {
    /// All `n * (n - 1) / 2` pairs, in ascending lexicographic order.
    ///
    /// ### Params
    ///
    /// * `round` - Read-only view of the current round
    /// * `out` - Destination for the pairs
    ///
    /// ### Returns
    ///
    /// Nothing; the exhaustive enumeration cannot fail.
    fn candidates(
        &mut self,
        round: Round<'_, T>,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<(), BonsaiErrors> {
        let n = round.members.len();
        for i in 0..n {
            for j in (i + 1)..n {
                out.push((i, j));
            }
        }
        Ok(())
    }
}

///////////////////
// Input, output //
///////////////////

/// The star handed to the primitive: effective leaves and their branches to a
/// common centre.
///
/// Means and precisions are row-major `[member][feature]`, as in
/// [`crate::model::likelihood::NodeState`]. Precisions are not diffusion
/// corrected (SPEC.md section 4); `branch` carries the correction's input.
#[derive(Clone, Copy, Debug)]
pub struct Star<'a, T> {
    /// Effective means, `[member][feature]`, row-major.
    pub means: &'a [T],
    /// Effective precisions, same layout.
    pub precisions: &'a [T],
    /// Branch length from each member to the centre.
    pub branch: &'a [f64],
    /// Number of features.
    pub n_features: usize,
}

/// One merge performed by the primitive.
#[derive(Clone, Copy, Debug)]
pub struct StarMerge {
    /// The member that became the ancestor's first child.
    pub left: u32,
    /// The member that became its second child.
    pub right: u32,
    /// Node id of the ancestor that was created.
    pub ancestor: u32,
    /// Optimised branch from the ancestor down to `left`.
    pub t_left: f64,
    /// Optimised branch from the ancestor down to `right`.
    pub t_right: f64,
    /// Optimised branch from the ancestor up to the centre, at the moment it
    /// was created. Overwritten in [`StarResult::branch`] if the ancestor is
    /// itself merged later, which is why it is recorded here.
    pub t_centre: f64,
    /// Loglikelihood gain of this merge, from SPEC.md section 8.3.
    pub gain: f64,
}

/// What the primitive built.
///
/// Node ids run `0..n_members` for the input members and `n_members..` for the
/// ancestors, in creation order, so every parent index exceeds its children's.
#[derive(Clone, Debug)]
pub struct StarResult<T> {
    /// Number of members the star started with.
    pub n_members: usize,
    /// Parent of each node, [`NO_NODE`] for anything still on the centre.
    pub parent: Vec<u32>,
    /// Branch above each node: to its ancestor, or to the centre.
    pub branch: Vec<f64>,
    /// The merges, in the order they were performed.
    pub merges: Vec<StarMerge>,
    /// Members still attached to the centre, ascending.
    pub centre_children: Vec<u32>,
    /// Effective means of the ancestors, row-major `[ancestor][feature]`.
    pub ancestor_means: Vec<T>,
    /// Effective precisions of the ancestors, same layout.
    pub ancestor_precisions: Vec<T>,
}

//////////////
// The scan //
//////////////

/// The best pair found in one round of the scan.
///
/// Ordered by gain, then by node ids ascending: a total order, so the reduction
/// does not depend on how rayon splits the work.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// Loglikelihood gain of the merge.
    gain: f64,
    /// Node id of the first member.
    left: u32,
    /// Node id of the second member.
    right: u32,
    /// Optimised branch from the new ancestor to `left`.
    t_ak: f64,
    /// Optimised branch from the new ancestor to `right`.
    t_al: f64,
    /// Optimised branch from the new ancestor to the centre.
    t_ar: f64,
}

impl Candidate {
    /// The identity of the reduction: worse than any real candidate.
    const NONE: Self = Self {
        gain: f64::NEG_INFINITY,
        left: NO_NODE,
        right: NO_NODE,
        t_ak: 0.0,
        t_al: 0.0,
        t_ar: 0.0,
    };

    /// Pick the better of two candidates.
    ///
    /// ### Params
    ///
    /// * `other` - The candidate to compare against
    ///
    /// ### Returns
    ///
    /// The one with the larger gain, the smaller pair of node ids breaking a
    /// tie.
    #[inline]
    fn better(self, other: Self) -> Self {
        if other.gain > self.gain {
            other
        } else if self.gain > other.gain {
            self
        } else if (other.left, other.right) < (self.left, self.right) {
            other
        } else {
            self
        }
    }
}

/// One round's read-only view of everything the pair scan reads, rebuilt each
/// round as the arrays grow.
#[derive(Clone, Copy, Debug)]
struct Working<'a, T> {
    /// Effective means of every node created so far, row-major.
    m: &'a [T],
    /// Effective precisions, same layout.
    w: &'a [T],
    /// Branch from each node to its ancestor, or to the centre.
    branch: &'a [f64],
    /// The centre's own effective means, length `p`.
    mc: &'a [f64],
    /// The centre's own effective precisions, length `p`.
    wc: &'a [f64],
    /// Number of features.
    p: usize,
}

/// The centre's own effective leaf, over the members currently attached to it.
///
/// SPEC.md section 4 over the whole star. The mean accumulates as a running
/// weighted average, as in [`crate::utils::kernels::prune_general`], so partial
/// values stay in the convex hull of the member means.
///
/// `O(n p)` a round; see [`update_centre_leaf`] and
/// [`StarParams::incremental_centre`].
///
/// ### Params
///
/// * `m` - Effective means of every node, row-major
/// * `w` - Effective precisions of every node, row-major
/// * `branch` - Branch from each node to its parent or to the centre
/// * `members` - Nodes currently attached to the centre
/// * `p` - Number of features
/// * `mc` - Destination for the centre's effective means, length `p`
/// * `wc` - Destination for the centre's effective precisions, length `p`
fn centre_leaf<T: BonsaiFloat>(
    m: &[T],
    w: &[T],
    branch: &[f64],
    members: &[u32],
    p: usize,
    mc: &mut [f64],
    wc: &mut [f64],
) {
    mc.fill(0.0);
    wc.fill(0.0);
    for &member in members {
        let base = member as usize * p;
        let t = branch[member as usize];
        for g in 0..p {
            let wi = wide(w[base + g]);
            let wd = wi / (1.0 + t * wi);
            wc[g] += wd;
            mc[g] += (wide(m[base + g]) - mc[g]) * (wd / wc[g]);
        }
    }
}

/// Move the centre's effective leaf across one merge, in `O(p)`.
///
/// Two subtractions and one addition instead of an `O(n p)` re-accumulation.
/// The subtractions are the peel of SPEC.md section 8.1 and lose digits the same
/// way; the caller caps drift with a periodic exact recompute. See
/// [`StarParams::incremental_centre`].
///
/// Call *before* the children's branches are overwritten with those to their
/// new ancestor.
///
/// ### Params
///
/// * `m` - Effective means of every node, row-major
/// * `w` - Effective precisions of every node, row-major
/// * `t_rk` - Branch the first child had to the centre
/// * `t_rl` - Branch the second child had to the centre
/// * `k` - Node id of the first child
/// * `l` - Node id of the second child
/// * `a` - Node id of the ancestor, already summarised into `m` and `w`
/// * `t_ar` - Branch from the ancestor to the centre
/// * `p` - Number of features
/// * `mc` - The centre's effective means, updated in place
/// * `wc` - The centre's effective precisions, updated in place
#[allow(clippy::too_many_arguments)]
fn update_centre_leaf<T: BonsaiFloat>(
    m: &[T],
    w: &[T],
    t_rk: f64,
    t_rl: f64,
    k: usize,
    l: usize,
    a: usize,
    t_ar: f64,
    p: usize,
    mc: &mut [f64],
    wc: &mut [f64],
) {
    let (bk, bl, ba) = (k * p, l * p, a * p);
    for g in 0..p {
        let wk = wide(w[bk + g]);
        let wl = wide(w[bl + g]);
        let wa = wide(w[ba + g]);
        let wdk = wk / (1.0 + t_rk * wk);
        let wdl = wl / (1.0 + t_rl * wl);
        let wda = wa / (1.0 + t_ar * wa);

        let total =
            wc[g] * mc[g] - wdk * wide(m[bk + g]) - wdl * wide(m[bl + g]) + wda * wide(m[ba + g]);
        wc[g] += wda - wdk - wdl;
        mc[g] = total / wc[g];
    }
}

/// Peel a pair off the centre's effective leaf to get the rest of the star.
///
/// SPEC.md section 8.1. `O(p)` per pair by subtraction rather than `O(n p)` by
/// re-accumulating over the other members.
///
/// **Conditioning.** If one member carries most of the centre's precision, `WR`
/// is a small difference of large numbers and loses digits (the mean worse
/// still). Relative error in the remainder's mean is about
/// `2^-53 * W_centre / W_R`: `2.5e-1` at dominance `1e15`, total loss at `1e16`
/// (`test_the_peel_loses_the_remainder_when_one_member_dominates`). The
/// diffusion correction caps `wd` at `1/t`, and late-round `W_centre / W_R` is
/// of order `n`, so real stars stay clear; only zero-length-branch fixtures
/// reach the band. Accumulation is in `f64` regardless of storage type.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `i` - First member of the pair
/// * `j` - Second member of the pair
/// * `m_r` - Destination for the remainder's means, length `p`
/// * `w_r` - Destination for the remainder's precisions, length `p`
fn peel<T: BonsaiFloat>(work: &Working<'_, T>, i: usize, j: usize, m_r: &mut [T], w_r: &mut [T]) {
    let p = work.p;
    let (bi, bj) = (i * p, j * p);
    let (ti, tj) = (work.branch[i], work.branch[j]);
    for g in 0..p {
        let wi = wide(work.w[bi + g]);
        let wj = wide(work.w[bj + g]);
        let wdi = wi / (1.0 + ti * wi);
        let wdj = wj / (1.0 + tj * wj);
        let wr = work.wc[g] - wdi - wdj;
        let mr =
            (work.mc[g] * work.wc[g] - wdi * wide(work.m[bi + g]) - wdj * wide(work.m[bj + g]))
                / wr;
        m_r[g] = narrow(mr);
        w_r[g] = narrow(wr);
    }
}

/// One worker's reusable buffers for the pair scan.
///
/// Allocated once per worker through `map_init`. Public for providers that
/// score pairs themselves (SPEC.md section 10).
pub struct PairScratch<T> {
    /// Branch-length solve scratch.
    merge: MergeScratch,
    /// The remainder's means, from the peel of SPEC.md section 8.1.
    pub m_r: Vec<T>,
    /// The remainder's precisions, from the same peel.
    pub w_r: Vec<T>,
}

impl<T: BonsaiFloat> PairScratch<T> {
    /// Allocate for a given feature count.
    ///
    /// ### Params
    ///
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// The scratch.
    pub fn new(p: usize) -> Self {
        Self {
            merge: MergeScratch::new(p),
            m_r: vec![T::zero(); p],
            w_r: vec![T::zero(); p],
        }
    }
}

/// Peel a pair off the centre and score the merge.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `a` - Position of the first member of the pair
/// * `b` - Position of the second
/// * `merge` - Branch-length solve knobs
/// * `scratch` - Reusable buffers, left holding the peel
///
/// ### Returns
///
/// The gain and the three optimised branch lengths, or the error the
/// branch-length solve failed with.
fn score_pair_raw<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    a: usize,
    b: usize,
    merge: MergeParams,
    scratch: &mut PairScratch<T>,
) -> Result<MergeScore, BonsaiErrors> {
    let p = work.p;
    let (i, j) = (members[a] as usize, members[b] as usize);
    peel(work, i, j, &mut scratch.m_r, &mut scratch.w_r);

    score_merge(
        EffLeaf {
            m: &work.m[i * p..i * p + p],
            w: &work.w[i * p..i * p + p],
        },
        EffLeaf {
            m: &work.m[j * p..j * p + p],
            w: &work.w[j * p..j * p + p],
        },
        EffLeaf {
            m: &scratch.m_r,
            w: &scratch.w_r,
        },
        work.branch[i],
        work.branch[j],
        Some(merge),
        &mut scratch.merge,
    )
}

/// What one scan found: the best pair, and how many were dropped.
///
/// `dropped` is summed, which is associative and commutative, so the reduction
/// is independent of how rayon splits the work.
#[derive(Clone, Copy, Debug)]
struct Scan {
    /// Best candidate seen, [`Candidate::NONE`] if there was none.
    best: Candidate,
    /// Pairs whose branch-length solve diverged and were dropped.
    dropped: usize,
}

impl Scan {
    /// The identity of the reduction: nothing seen, nothing dropped.
    const EMPTY: Self = Self {
        best: Candidate::NONE,
        dropped: 0,
    };

    /// One pair whose branch-length solve diverged.
    const DROPPED: Self = Self {
        best: Candidate::NONE,
        dropped: 1,
    };

    /// Combine two scans.
    ///
    /// ### Params
    ///
    /// * `other` - The scan to fold in
    ///
    /// ### Returns
    ///
    /// The better candidate and the summed drop count.
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            best: self.best.better(other.best),
            dropped: self.dropped + other.dropped,
        }
    }
}

/// Score one candidate pair.
///
/// A non-finite gain becomes [`Candidate::NONE`], and a diverged solve is
/// dropped and counted, so one pathological pair cannot stop the primitive.
/// Propagating the error instead would also make it depend on the thread count,
/// since `try_reduce` short-circuits on whichever worker fails first.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `a` - Position of the first member of the pair
/// * `b` - Position of the second
/// * `merge` - Branch-length solve knobs
/// * `scratch` - Reusable buffers
///
/// ### Returns
///
/// The scan of this one pair, or the error the branch-length solve failed with
/// where that error is not a divergence.
fn score_pair<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    a: usize,
    b: usize,
    merge: MergeParams,
    scratch: &mut PairScratch<T>,
) -> Result<Scan, BonsaiErrors> {
    let score = match score_pair_raw(work, members, a, b, merge, scratch) {
        Ok(score) => score,
        Err(BonsaiErrors::RootFindDiverged { .. }) => return Ok(Scan::DROPPED),
        Err(other) => return Err(other),
    };

    Ok(Scan {
        best: if score.gain.is_finite() {
            Candidate {
                gain: score.gain,
                left: members[a],
                right: members[b],
                t_ak: score.t_ak,
                t_al: score.t_al,
                t_ar: score.t_ar,
            }
        } else {
            Candidate::NONE
        },
        dropped: 0,
    })
}

/// Recover the error that a round of nothing but diverged pairs swallowed.
///
/// Called when every pair in a round was dropped. Rescores the first pair
/// sequentially, a fixed choice, so the error is independent of the thread
/// count.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `pairs` - The round's candidate pairs, non-empty
/// * `merge` - Branch-length solve knobs
///
/// ### Returns
///
/// The error the first pair fails with.
fn diverged_round<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    pairs: &[(usize, usize)],
    merge: MergeParams,
) -> BonsaiErrors {
    let (a, b) = pairs[0];
    let mut scratch = PairScratch::new(work.p);
    match score_pair_raw(work, members, a, b, merge, &mut scratch) {
        Err(error) => error,
        Ok(_) => BonsaiErrors::RootFindDiverged {
            max_iter: 0,
            last_step: f64::NAN,
        },
    }
}

/// Score every candidate pair and return the best.
///
/// Parallel map and reduce over a total order on `(gain, node ids)`, so the
/// winner does not depend on the thread count.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `pairs` - Candidate pairs, as positions into `members`
/// * `merge` - Branch-length solve knobs
///
/// ### Returns
///
/// The scan, whose best is [`Candidate::NONE`] if there were no scoreable
/// pairs, or the error the branch-length solve failed with.
fn scan_pairs<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    pairs: &[(usize, usize)],
    merge: MergeParams,
) -> Result<Scan, BonsaiErrors> {
    let p = work.p;
    if pairs.len() < PAR_PAIRS_MIN {
        let mut scratch = PairScratch::new(p);
        let mut scan = Scan::EMPTY;
        for &(a, b) in pairs {
            scan = scan.merge(score_pair(work, members, a, b, merge, &mut scratch)?);
        }
        return Ok(scan);
    }
    pairs
        .par_iter()
        .map_init(
            || PairScratch::new(p),
            |scratch, &(a, b)| score_pair(work, members, a, b, merge, scratch),
        )
        .try_reduce(|| Scan::EMPTY, |x, y| Ok(x.merge(y)))
}

/// Walk a bound-ordered candidate list, stopping once the best is provably
/// found.
///
/// SPEC.md section 10.4 step 2. `bounds` is non-increasing and each entry is an
/// upper bound on its pair's gain, so once the best true gain seen exceeds the
/// next entry every remaining pair is beaten and the walk can stop.
///
/// **The comparison is strict.** With `>=` a pair tying the incumbent could go
/// unscored, and [`Candidate::better`] breaks ties on node ids, so the winner
/// would depend on where the walk stopped. Strict `>` recovers the exhaustive
/// answer exactly.
///
/// Consumed in index-fixed chunks of [`BOUND_WALK_CHUNK`], so each chunk scans in
/// parallel and the result is the same at any thread count.
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `pairs` - Candidate pairs, as positions into `members`, in bound order
/// * `bounds` - Upper bound per pair, non-increasing
/// * `merge` - Branch-length solve knobs
///
/// ### Returns
///
/// The scan and how many pairs were scored, or the error the branch-length
/// solve failed with.
fn walk_bounded<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    pairs: &[(usize, usize)],
    bounds: &[f64],
    merge: MergeParams,
) -> Result<(Scan, usize), BonsaiErrors> {
    let mut scan = Scan::EMPTY;
    let mut done = 0usize;
    while done < pairs.len() {
        if scan.best.gain > bounds[done] {
            break;
        }
        let end = (done + BOUND_WALK_CHUNK).min(pairs.len());
        scan = scan.merge(scan_pairs(work, members, &pairs[done..end], merge)?);
        done = end;
    }
    Ok((scan, done))
}

/// Score every candidate pair and sample one in proportion to the likelihood of
/// the tree its merge would produce.
///
/// SPEC.md section 9.4. Scores are collected in `pairs` order and the softmax
/// and draw run sequentially, so the pick depends only on the seed and the star.
/// Only pairs clearing `min_gain` are eligible; see [`StarSelection::Weighted`].
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `pairs` - Candidate pairs, as positions into `members`
/// * `merge` - Branch-length solve knobs
/// * `min_gain` - Floor a pair must clear to be eligible
/// * `temperature` - Divisor on the gains before the exponential
/// * `rng` - Stream the round's single draw is taken from
///
/// ### Returns
///
/// The sampled candidate, or [`Candidate::NONE`] if no pair was eligible, or
/// the error the branch-length solve failed with.
fn sample_pair<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    pairs: &[(usize, usize)],
    merge: MergeParams,
    min_gain: f64,
    temperature: f64,
    rng: &mut SplitMix64,
) -> Result<Candidate, BonsaiErrors> {
    let p = work.p;
    let scored: Vec<Candidate> = if pairs.len() < PAR_PAIRS_MIN {
        let mut scratch = PairScratch::new(p);
        pairs
            .iter()
            .map(|&(a, b)| {
                score_pair(work, members, a, b, merge, &mut scratch).map(|scan| scan.best)
            })
            .collect::<Result<Vec<_>, BonsaiErrors>>()?
    } else {
        pairs
            .par_iter()
            .map_init(
                || PairScratch::new(p),
                |scratch, &(a, b)| {
                    score_pair(work, members, a, b, merge, scratch).map(|scan| scan.best)
                },
            )
            .collect::<Result<Vec<_>, BonsaiErrors>>()?
    };

    let eligible = |c: &&Candidate| c.gain > min_gain;
    let top = scored
        .iter()
        .filter(eligible)
        .fold(f64::NEG_INFINITY, |acc, c| acc.max(c.gain));
    if !top.is_finite() {
        return Ok(Candidate::NONE);
    }

    // Subtract the maximum: gains are `O(p)` nats apart and would overflow.
    let total: f64 = scored
        .iter()
        .filter(eligible)
        .map(|c| ((c.gain - top) / temperature).exp())
        .sum();
    let target = rng.uniform() * total;
    let mut acc = 0.0f64;
    for c in scored.iter().filter(eligible) {
        acc += ((c.gain - top) / temperature).exp();
        if acc >= target {
            return Ok(*c);
        }
    }
    // The cumulative sum can fall a rounding short of `total`.
    Ok(scored
        .iter()
        .rev()
        .find(eligible)
        .copied()
        .unwrap_or(Candidate::NONE))
}

/// Summarise a merged pair as one effective leaf, SPEC.md section 4.
///
/// ### Params
///
/// * `m` - Effective means of every node, row-major, appended to
/// * `w` - Effective precisions of every node, row-major, appended to
/// * `k` - First child
/// * `l` - Second child
/// * `t_ak` - Branch from the new ancestor to `k`
/// * `t_al` - Branch from the new ancestor to `l`
/// * `p` - Number of features
fn push_ancestor<T: BonsaiFloat>(
    m: &mut Vec<T>,
    w: &mut Vec<T>,
    k: usize,
    l: usize,
    t_ak: f64,
    t_al: f64,
    p: usize,
) {
    let (bk, bl) = (k * p, l * p);
    for g in 0..p {
        let wk = wide(w[bk + g]);
        let wl = wide(w[bl + g]);
        let wdk = wk / (1.0 + t_ak * wk);
        let wdl = wl / (1.0 + t_al * wl);
        let wa = wdk + wdl;
        // Convex combination: the mean stays between the children's and cannot
        // cancel, which matters as error compounds up the tree.
        let mk = wide(m[bk + g]);
        let ma = mk + (wide(m[bl + g]) - mk) * (wdl / wa);
        m.push(narrow(ma));
        w.push(narrow(wa));
    }
}

///////////////////
// The primitive //
///////////////////

/// Run the star primitive with the exhaustive candidate provider.
///
/// SPEC.md section 9.1. See [`resolve_star_with`] for the general form.
///
/// ### Params
///
/// * `star` - The members and their branches to the centre
/// * `params` - Tuning knobs, or `None` for the defaults
///
/// ### Returns
///
/// The topology that was built, or `MalformedTree` if the star is
/// ill-described, or the error the branch-length solve failed with.
pub fn resolve_star<T: BonsaiFloat>(
    star: Star<'_, T>,
    params: Option<StarParams>,
) -> Result<StarResult<T>, BonsaiErrors> {
    resolve_star_with(star, params, &mut AllPairs, Verbosity::Quiet)
}

/// Run the star primitive.
///
/// Each round scores the candidate pairs, inserts an ancestor above the one
/// [`StarSelection`] picks and puts it back in the star in place of the two
/// members it swallowed. Stops at `MIN_CENTRE_MEMBERS` members or when no pair
/// clears [`StarParams::min_gain`]. Only the three branches a merge creates are
/// optimised (SPEC.md section 8.4); the rest is left to search steps 4 and 7.
///
/// ### Params
///
/// * `star` - The members and their branches to the centre
/// * `params` - Tuning knobs, or `None` for the defaults
/// * `candidates` - Which pairs to consider each round
/// * `verbosity` - [`Verbosity::Detailed`] prints progress at every tenth of
///   the possible merges
///
/// ### Returns
///
/// The topology that was built, or `MalformedTree` if the star is
/// ill-described, or the error the branch-length solve failed with.
pub fn resolve_star_with<T: BonsaiFloat, C: CandidatePairs<T>>(
    star: Star<'_, T>,
    params: Option<StarParams>,
    candidates: &mut C,
    verbosity: Verbosity,
) -> Result<StarResult<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let p = star.n_features;
    let n = star.branch.len();

    if p == 0 || n < 2 {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("a star needs at least two members and one feature, got {n} by {p}"),
        });
    }
    if star.means.len() != n * p || star.precisions.len() != n * p {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "{n} members by {p} features needs {} entries, got {} means and {} precisions",
                n * p,
                star.means.len(),
                star.precisions.len()
            ),
        });
    }

    // A non-finite input would otherwise yield `Ok` with an unresolved star:
    // every gain maps to `Candidate::NONE` and the loop exits normally.
    for (i, &value) in star.means.iter().enumerate() {
        if !wide(value).is_finite() {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!(
                    "member {} has a non-finite mean at feature {}",
                    i / p,
                    i % p
                ),
            });
        }
    }
    for (i, &value) in star.precisions.iter().enumerate() {
        let value = wide(value);
        if !value.is_finite() || value <= 0.0 {
            return Err(BonsaiErrors::NonPositiveSd {
                value,
                cell: i / p,
                feature: i % p,
            });
        }
    }
    for (i, &len) in star.branch.iter().enumerate() {
        if !len.is_finite() || len < 0.0 {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!(
                    "member {i} has branch length {len} to the centre: branch lengths are \
                     diffusion times and must be finite and non-negative"
                ),
            });
        }
    }

    // At most `n - MIN_CENTRE_MEMBERS` merges, each shrinking the star by one.
    let max_merges = n.saturating_sub(MIN_CENTRE_MEMBERS);
    let mut m: Vec<T> = Vec::with_capacity((n + max_merges) * p);
    let mut w: Vec<T> = Vec::with_capacity((n + max_merges) * p);
    m.extend_from_slice(star.means);
    w.extend_from_slice(star.precisions);

    let mut parent = vec![NO_NODE; n];
    let mut branch = star.branch.to_vec();
    let mut merges: Vec<StarMerge> = Vec::with_capacity(max_merges);
    let mut members: Vec<u32> = (0..n as u32).collect();
    let started = Instant::now();

    let mut mc = vec![0.0f64; p];
    let mut wc = vec![0.0f64; p];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut best_gain = f64::NEG_INFINITY;
    let mut scored_last_round = 0usize;
    let mut since_exact_centre = usize::MAX;
    let mut rng = SplitMix64::new(match params.selection {
        StarSelection::Greedy => 0,
        StarSelection::Weighted { seed, .. } => seed,
    });

    while members.len() > MIN_CENTRE_MEMBERS {
        if !params.incremental_centre || since_exact_centre >= CENTRE_EXACT_EVERY {
            centre_leaf(&m, &w, &branch, &members, p, &mut mc, &mut wc);
            since_exact_centre = 0;
        }

        pairs.clear();
        candidates.candidates(
            Round {
                members: &members,
                means: &m,
                precisions: &w,
                n_features: p,
                branch: &branch,
                centre_means: &mc,
                centre_precisions: &wc,
                merge: params.merge,
                best_gain,
                scored_last_round,
            },
            &mut pairs,
        )?;
        if pairs.is_empty() {
            break;
        }

        let work = Working {
            m: &m,
            w: &w,
            branch: &branch,
            mc: &mc,
            wc: &wc,
            p,
        };
        let best = match params.selection {
            // Weighted selection needs every score, so bounds are ignored.
            StarSelection::Greedy => {
                let (scan, done) = match candidates.bounds() {
                    Some(bounds) if bounds.len() == pairs.len() => {
                        walk_bounded(&work, &members, &pairs, bounds, params.merge)?
                    }
                    _ => (
                        scan_pairs(&work, &members, &pairs, params.merge)?,
                        pairs.len(),
                    ),
                };
                scored_last_round = done;
                // Every scored pair diverged: surface the solver error.
                if scan.dropped == done && done > 0 {
                    return Err(diverged_round(&work, &members, &pairs, params.merge));
                }
                scan.best
            }
            StarSelection::Weighted { temperature, .. } => {
                scored_last_round = pairs.len();
                sample_pair(
                    &work,
                    &members,
                    &pairs,
                    params.merge,
                    params.min_gain,
                    temperature,
                    &mut rng,
                )?
            }
        };
        if best.gain <= params.min_gain {
            break;
        }

        let (k, l) = (best.left as usize, best.right as usize);
        let ancestor = parent.len() as u32;
        push_ancestor(&mut m, &mut w, k, l, best.t_ak, best.t_al, p);
        if params.incremental_centre {
            update_centre_leaf(
                &m,
                &w,
                branch[k],
                branch[l],
                k,
                l,
                ancestor as usize,
                best.t_ar,
                p,
                &mut mc,
                &mut wc,
            );
            since_exact_centre += 1;
        }
        parent[k] = ancestor;
        parent[l] = ancestor;
        branch[k] = best.t_ak;
        branch[l] = best.t_al;
        parent.push(NO_NODE);
        branch.push(best.t_ar);

        // The ancestor's id exceeds every member's, so appending keeps
        // `members` ascending.
        members.retain(|&x| x != best.left && x != best.right);
        members.push(ancestor);

        let merge = StarMerge {
            left: best.left,
            right: best.right,
            ancestor,
            t_left: best.t_ak,
            t_right: best.t_al,
            t_centre: best.t_ar,
            gain: best.gain,
        };
        let base = ancestor as usize * p;
        candidates.merged(
            &merge,
            EffLeaf {
                m: &m[base..base + p],
                w: &w[base..base + p],
            },
        );
        best_gain = best.gain;
        merges.push(merge);
        if verbosity.detailed_verbosity() {
            let done = merges.len();
            report_decile_progress(done, done - 1, max_merges, "merges", started.elapsed());
        }
    }

    let ancestor_means = m.split_off(n * p);
    let ancestor_precisions = w.split_off(n * p);

    Ok(StarResult {
        n_members: n,
        parent,
        branch,
        merges,
        centre_children: members,
        ancestor_means,
        ancestor_precisions,
    })
}

/// Run the primitive and assemble the result into a [`Tree`].
///
/// Step 2 of the search (SPEC.md section 9): the members are the leaves, the
/// centre becomes the root, and whatever is still attached to the centre when
/// the primitive stops becomes the root's children.
///
/// The tree's internal nodes are relabelled into level order by
/// [`Tree::from_parents`], so they do **not** line up with the ancestor ids in
/// a [`StarResult`]. A caller that needs both should use [`resolve_star`] and
/// keep its own indexing, or recover the effective leaves from the finished
/// tree with [`crate::model::likelihood::NodeState::prune`].
///
/// ### Params
///
/// * `star` - The leaves and their branches to the root
/// * `params` - Tuning knobs, or `None` for the defaults
///
/// ### Returns
///
/// The tree and the total loglikelihood gain over the star it started from, or
/// the error the primitive or the arena failed with.
pub fn star_tree<T: BonsaiFloat>(
    star: Star<'_, T>,
    params: Option<StarParams>,
) -> Result<(Tree, f64), BonsaiErrors> {
    star_tree_with(star, params, &mut AllPairs, Verbosity::Quiet)
}

/// Resolve a star into a tree, choosing which candidate pairs are considered.
///
/// The exhaustive scan of [`star_tree`] is `O(n^3 p)` over a star.
/// [`crate::search::candidates::KnnCandidates`] and
/// [`crate::search::bounds::EllipsoidBounds`] avoid that and compose, bounds
/// outermost:
///
/// ```ignore
/// let mut provider = EllipsoidBounds::new(KnnCandidates::new(None));
/// let (tree, gain) = star_tree_with(star, None, &mut provider)?;
/// ```
///
/// ### Params
///
/// * `star` - The members and their branches to the centre
/// * `params` - Star knobs, or `None` for the defaults
/// * `candidates` - Which pairs to consider each round
/// * `verbosity` - As [`resolve_star_with`]
///
/// ### Returns
///
/// The tree and the summed gain of its merges.
pub fn star_tree_with<T: BonsaiFloat, C: CandidatePairs<T>>(
    star: Star<'_, T>,
    params: Option<StarParams>,
    candidates: &mut C,
    verbosity: Verbosity,
) -> Result<(Tree, f64), BonsaiErrors> {
    let result = resolve_star_with(star, params, candidates, verbosity)?;
    let gain: f64 = result.merges.iter().map(|x| x.gain).sum();

    let root = result.parent.len() as u32;
    let mut parent = result.parent;
    for slot in parent.iter_mut() {
        if *slot == NO_NODE {
            *slot = root;
        }
    }
    parent.push(NO_NODE);

    let mut branch = result.branch;
    branch.push(0.0);

    Ok((Tree::from_parents(parent, branch, result.n_members)?, gain))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::likelihood::NodeState;
    use crate::tree::simulate::splits;
    use crate::utils::rng::SplitMix64;
    use approx::assert_relative_eq;

    /// Worst relative deviation of the incremental centre from the exact one.
    ///
    /// Replays a star's merges twice over the same topology, once maintaining
    /// the centre by [`update_centre_leaf`] and once recomputing it, and
    /// compares the two at every round. Driving both over the *same* merges is
    /// the point: it isolates the arithmetic from any topology change it might
    /// otherwise cause.
    ///
    /// ### Params
    ///
    /// * `n` - Members
    /// * `p` - Features
    /// * `seed` - Random seed
    /// * `every` - Rounds between exact recomputes, zero for never
    ///
    /// ### Returns
    ///
    /// The worst relative deviation of the precision and of the mean.
    fn centre_drift(n: usize, p: usize, seed: u64, every: usize) -> (f64, f64) {
        let mut rng = SplitMix64::new(seed);
        let groups = (n / 4).max(2);
        let centres: Vec<Vec<f64>> = (0..groups)
            .map(|_| (0..p).map(|_| 4.0 * (rng.uniform() - 0.5)).collect())
            .collect();
        let mut m = Vec::new();
        let mut w = Vec::new();
        for i in 0..n {
            for g in 0..p {
                m.push(centres[i % groups][g] + 0.35 * (rng.uniform() - 0.5));
                w.push(0.5 + 2.0 * rng.uniform());
            }
        }
        let branch: Vec<f64> = (0..n).map(|_| 0.05 + 0.2 * rng.uniform()).collect();

        let base = resolve_star(
            Star {
                means: &m,
                precisions: &w,
                branch: &branch,
                n_features: p,
            },
            None,
        )
        .expect("resolve");

        let mut mm = m.clone();
        let mut ww = w.clone();
        let mut br = branch.clone();
        let mut members: Vec<u32> = (0..n as u32).collect();
        let (mut mc, mut wc) = (vec![0.0; p], vec![0.0; p]);
        let (mut exact_m, mut exact_w) = (vec![0.0; p], vec![0.0; p]);
        let (mut worst_w, mut worst_m) = (0.0f64, 0.0f64);
        let mut since = usize::MAX;
        for merge in &base.merges {
            if since == usize::MAX || (every > 0 && since >= every) {
                centre_leaf(&mm, &ww, &br, &members, p, &mut mc, &mut wc);
                since = 0;
            }
            centre_leaf(&mm, &ww, &br, &members, p, &mut exact_m, &mut exact_w);
            for g in 0..p {
                worst_w = worst_w.max((wc[g] - exact_w[g]).abs() / exact_w[g].abs());
                worst_m = worst_m.max((mc[g] - exact_m[g]).abs() / exact_m[g].abs().max(1e-3));
            }

            let (k, l) = (merge.left as usize, merge.right as usize);
            let (t_rk, t_rl) = (br[k], br[l]);
            push_ancestor(&mut mm, &mut ww, k, l, merge.t_left, merge.t_right, p);
            let a = br.len();
            update_centre_leaf(
                &mm,
                &ww,
                t_rk,
                t_rl,
                k,
                l,
                a,
                merge.t_centre,
                p,
                &mut mc,
                &mut wc,
            );
            since += 1;
            br[k] = merge.t_left;
            br[l] = merge.t_right;
            br.push(merge.t_centre);
            members.retain(|&x| x != merge.left && x != merge.right);
            members.push(a as u32);
        }
        (worst_w, worst_m)
    }

    /// The incremental centre update stays inside the gains' own rounding.
    ///
    /// Worst relative deviation from the exact recompute at any round of the
    /// star, over eight seeds:
    ///
    /// | members | features | recompute | precision | mean |
    /// |---|---|---|---|---|
    /// | 64 | 100 | never | 9.3e-15 | 1.5e-12 |
    /// | 64 | 100 | every 32 | 2.6e-15 | 1.0e-12 |
    /// | 128 | 200 | never | 1.5e-14 | 2.4e-12 |
    /// | 128 | 200 | every 32 | 3.2e-15 | 7.2e-13 |
    ///
    /// The mean drifts three orders further than the precision, which is what
    /// [`peel`] warns about: it is a difference of large numbers divided by
    /// another difference of large numbers, and the precision is only the
    /// first half of that. Both are far inside the `1e-16` per feature that a
    /// merge gain itself rounds to, and neither grows enough over a star to
    /// need the periodic recompute. The recompute is kept anyway: it costs one
    /// `O(n p)` pass in thirty-two and it is what stops the drift being
    /// unbounded in the star sizes nobody has run yet.
    #[test]
    fn test_the_incremental_centre_leaf_does_not_drift() {
        for &(n, p) in &[(64usize, 100usize), (128, 200)] {
            for every in [0usize, CENTRE_EXACT_EVERY] {
                let (mut worst_w, mut worst_m) = (0.0f64, 0.0f64);
                for seed in [1u64, 2, 3, 4, 5, 6, 7, 8] {
                    let (a, b) = centre_drift(n, p, seed, every);
                    worst_w = worst_w.max(a);
                    worst_m = worst_m.max(b);
                }
                assert!(
                    worst_w < 1e-12 && worst_m < 1e-10,
                    "{n} by {p}, recompute every {every}: drift {worst_w:e} in the precision \
                     and {worst_m:e} in the mean, an order past what was measured"
                );
            }
        }
    }

    /// Leaves drawn in `n_clusters` tight groups, far apart from each other.
    ///
    /// ### Params
    ///
    /// * `n_clusters` - Number of groups
    /// * `per_cluster` - Leaves in each group
    /// * `p` - Number of features
    /// * `spread` - Within-group jitter on the means
    /// * `separation` - Distance between consecutive group centres
    ///
    /// ### Returns
    ///
    /// Row-major means and precisions.
    fn clustered(
        n_clusters: usize,
        per_cluster: usize,
        p: usize,
        spread: f64,
        separation: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(0x2545_F491_4F6C_DD1D);
        // Each centre gets its own direction. Putting them on a line instead
        // makes the middle cluster coincide with the precision-weighted mean of
        // the outer two, so the remainder it would be peeled against sits
        // exactly on top of it and no branch separates them. That is the model
        // behaving correctly on a degenerate fixture, not a merge failing.
        let centres: Vec<Vec<f64>> = (0..n_clusters)
            .map(|_| (0..p).map(|_| separation * (rng.uniform() - 0.5)).collect())
            .collect();

        let mut m = Vec::with_capacity(n_clusters * per_cluster * p);
        let mut w = Vec::with_capacity(n_clusters * per_cluster * p);
        for c in 0..n_clusters {
            for _ in 0..per_cluster {
                for g in 0..p {
                    m.push(centres[c][g] + spread * (rng.uniform() - 0.5));
                    w.push(0.5 + rng.uniform());
                }
            }
        }
        (m, w)
    }

    /// The star that the tests hand to the primitive.
    ///
    /// ### Params
    ///
    /// * `m` - Means, row-major
    /// * `w` - Precisions, row-major
    /// * `t` - Branches to the centre
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// The star.
    fn star<'a>(m: &'a [f64], w: &'a [f64], t: &'a [f64], p: usize) -> Star<'a, f64> {
        Star {
            means: m,
            precisions: w,
            branch: t,
            n_features: p,
        }
    }

    /// Rebuild the tree as it stood after the first `n_merges` merges.
    ///
    /// The members keep their original branches to the centre until they are
    /// merged, which is why [`StarMerge::t_centre`] is recorded: an ancestor
    /// that is later swallowed has its entry in [`StarResult::branch`]
    /// overwritten.
    ///
    /// ### Params
    ///
    /// * `result` - What the primitive built
    /// * `t0` - The branches to the centre the star started with
    /// * `n_merges` - How many merges to apply
    ///
    /// ### Returns
    ///
    /// The tree, rooted at the centre.
    fn replay(result: &StarResult<f64>, t0: &[f64], n_merges: usize) -> Tree {
        let n = result.n_members;
        let root = (n + n_merges) as u32;
        let mut parent = vec![root; n + n_merges];
        let mut branch = vec![0.0f64; n + n_merges];
        branch[..n].copy_from_slice(t0);

        for merge in &result.merges[..n_merges] {
            parent[merge.left as usize] = merge.ancestor;
            parent[merge.right as usize] = merge.ancestor;
            branch[merge.left as usize] = merge.t_left;
            branch[merge.right as usize] = merge.t_right;
            branch[merge.ancestor as usize] = merge.t_centre;
        }
        parent.push(NO_NODE);
        branch.push(0.0);
        Tree::from_parents(parent, branch, n).expect("replayed tree is malformed")
    }

    /// Loglikelihood of a tree over the given leaves, computed independently of
    /// anything in this module.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `m` - Leaf means, row-major
    /// * `w` - Leaf precisions, row-major
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// The tree loglikelihood.
    fn loglik(tree: &Tree, m: &[f64], w: &[f64], p: usize) -> f64 {
        let mut state = NodeState::new(tree.n_nodes(), p, m, w).expect("state");
        state.prune(tree)
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

    #[test]
    fn test_degenerate_input_errors_rather_than_returning_an_unresolved_star() {
        // Regression. A single non-finite value
        // anywhere made every candidate gain non-finite, so the round found no
        // best pair, the loop exited normally, and the caller got `Ok` with
        // zero merges and no diagnostic. On a 10k by 20k input matrix one stray
        // NaN would have produced a star tree and reported success.
        let (n, p) = (12usize, 8usize);
        let base_means = vec![0.0f64; n * p];
        let base_precisions = vec![1.0f64; n * p];
        let base_branch = vec![0.5f64; n];

        let run = |m: &[f64], w: &[f64], b: &[f64]| {
            resolve_star::<f64>(
                Star {
                    means: m,
                    precisions: w,
                    branch: b,
                    n_features: p,
                },
                None,
            )
        };

        for bad in [f64::NAN, f64::INFINITY] {
            let mut means = base_means.clone();
            means[5] = bad;
            assert!(
                run(&means, &base_precisions, &base_branch).is_err(),
                "a {bad} mean was accepted"
            );

            let mut branch = base_branch.clone();
            branch[2] = bad;
            assert!(
                run(&base_means, &base_precisions, &branch).is_err(),
                "a {bad} branch was accepted"
            );
        }

        // Zero precision is infinite variance, which the peel turns into an
        // infinity rather than a large number.
        let mut precisions = base_precisions.clone();
        precisions[3] = 0.0;
        assert!(
            run(&base_means, &precisions, &base_branch).is_err(),
            "a zero precision was accepted"
        );

        // A negative branch was previously accepted and merged, unvalidated.
        let mut branch = base_branch.clone();
        branch[4] = -0.4;
        assert!(
            run(&base_means, &base_precisions, &branch).is_err(),
            "a negative branch was accepted"
        );

        // The well-formed star still works, so the checks are not over-eager.
        assert!(run(&base_means, &base_precisions, &base_branch).is_ok());
    }

    #[test]
    fn test_loglikelihood_never_decreases_across_a_merge() {
        // The test that matters. The primitive claims a gain from a closed-form
        // three-leaf expression; `NodeState::prune` knows nothing about any of
        // that and scores the whole tree from the leaves up. A sign error or a
        // mis-peeled remainder shows up here as a merge that made the tree
        // worse.
        let (p, n) = (40usize, 14usize);
        let (m, w) = clustered(3, 5, p, 0.6, 4.0);
        let (m, w) = (&m[..n * p], &w[..n * p]);
        let t0: Vec<f64> = (0..n).map(|i| 0.3 + 0.05 * i as f64).collect();

        let result = resolve_star(star(m, w, &t0, p), None).expect("resolve");
        assert!(!result.merges.is_empty());

        let mut previous = loglik(&replay(&result, &t0, 0), m, w, p);
        for j in 1..=result.merges.len() {
            let now = loglik(&replay(&result, &t0, j), m, w, p);
            assert!(
                now >= previous - 1e-9,
                "merge {j} dropped the loglikelihood from {previous} to {now}"
            );
            previous = now;
        }
    }

    #[test]
    fn test_claimed_gain_is_the_real_gain() {
        // Every merge's reported gain must equal the difference of the two
        // whole-tree loglikelihoods it sits between. This is the same check as
        // `model::merge`'s bottom test, extended from one hand-built four-leaf
        // star to the whole sequence of trees the primitive actually builds.
        let (p, n) = (32usize, 12usize);
        let (m, w) = clustered(3, 4, p, 0.5, 3.5);
        let t0 = vec![0.45f64; n];

        let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
        assert!(result.merges.len() >= 3);

        let mut previous = loglik(&replay(&result, &t0, 0), &m, &w, p);
        for (j, merge) in result.merges.iter().enumerate() {
            let now = loglik(&replay(&result, &t0, j + 1), &m, &w, p);
            assert_relative_eq!(merge.gain, now - previous, max_relative = 1e-8);
            previous = now;
        }
    }

    #[test]
    fn test_separated_clusters_become_clades() {
        // Tight groups a long way apart. Each must come out as its own clade,
        // which is a far stronger statement than the primitive merely
        // terminating: a greedy scan that peels the remainder wrongly still
        // terminates, it just builds nonsense.
        //
        // Checked as splits of the *unrooted* tree, which is what `splits`
        // computes. That is not pedantry: once the star is down to four members
        // the last merge has two spellings of one unrooted tree, joining the
        // first two members or the other two, differing only in which of the
        // centre's degree-three nodes it sits on. Which one the greedy picks is
        // arbitrary, so a rooted clade check would reject half of the correct
        // answers.
        for (n_clusters, per) in [(3usize, 4usize), (4, 4), (3, 6)] {
            let p = 24usize;
            let n = n_clusters * per;
            let (m, w) = clustered(n_clusters, per, p, 0.3, 12.0);
            let t0 = vec![0.5f64; n];

            let (tree, gain) = star_tree(star(&m, &w, &t0, p), None).expect("star tree");
            assert!(gain > 0.0);

            let got = splits(&tree);
            for c in 0..n_clusters {
                // `splits` keys each bipartition by the side without leaf zero,
                // so the cluster holding leaf zero is looked up by its
                // complement. Clusters are contiguous blocks of leaves, so that
                // is the only one that needs it.
                let want: Vec<u32> = if c == 0 {
                    (per as u32..n as u32).collect()
                } else {
                    ((c * per) as u32..((c + 1) * per) as u32).collect()
                };
                assert!(
                    got.contains(&want),
                    "cluster {c} of {n_clusters} ({want:?}) is not a split of the tree; \
                     the splits were {got:?}"
                );
            }
        }
    }

    #[test]
    fn test_same_tree_whatever_the_thread_count() {
        // The pair scan reduces in whatever order rayon splits the work, so the
        // selected pair has to be decided by a total order over (gain, ids) and
        // not by whichever thread finished first.
        let (p, n) = (28usize, 13usize);
        let (m, w) = clustered(3, 5, p, 0.7, 3.0);
        let (m, w) = (&m[..n * p], &w[..n * p]);
        let t0: Vec<f64> = (0..n).map(|i| 0.2 + 0.07 * (i % 4) as f64).collect();

        let reference = resolve_star(star(m, w, &t0, p), None).expect("resolve");
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            for _ in 0..3 {
                let got = pool.install(|| resolve_star(star(m, w, &t0, p), None).expect("resolve"));
                assert_eq!(
                    got.parent, reference.parent,
                    "topology moved at {threads} threads"
                );
                assert_eq!(
                    got.branch, reference.branch,
                    "branch lengths moved at {threads} threads"
                );
                assert_eq!(got.centre_children, reference.centre_children);
                let gains: Vec<f64> = got.merges.iter().map(|x| x.gain).collect();
                let want: Vec<f64> = reference.merges.iter().map(|x| x.gain).collect();
                assert_eq!(gains, want, "gains moved at {threads} threads");
            }
        }
    }

    #[test]
    fn test_stops_with_three_members_at_the_centre() {
        let (p, n) = (20usize, 11usize);
        let (m, w) = clustered(3, 4, p, 0.5, 5.0);
        let (m, w) = (&m[..n * p], &w[..n * p]);
        let t0 = vec![0.4f64; n];

        let result = resolve_star(star(m, w, &t0, p), None).expect("resolve");
        assert_eq!(result.centre_children.len(), MIN_CENTRE_MEMBERS);
        assert_eq!(result.merges.len(), n - MIN_CENTRE_MEMBERS);

        let (tree, _) = star_tree(star(m, w, &t0, p), None).expect("star tree");
        assert_eq!(tree.children(tree.root()).len(), MIN_CENTRE_MEMBERS);
    }

    #[test]
    fn test_stops_when_no_pair_helps() {
        // Members that are exactly identical and already sit on zero-length
        // branches leave nothing for a merge to gain: the ancestor it would
        // insert is a node of degree three whose three branches are all zero.
        let (p, n) = (48usize, 8usize);
        let row: Vec<f64> = (0..p).map(|g| (g as f64 * 0.17).sin()).collect();
        let prec: Vec<f64> = (0..p)
            .map(|g| 0.9 + 0.2 * (g as f64 * 0.11).cos())
            .collect();
        let m: Vec<f64> = (0..n).flat_map(|_| row.clone()).collect();
        let w: Vec<f64> = (0..n).flat_map(|_| prec.clone()).collect();
        let t0 = vec![0.0f64; n];

        let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
        assert!(
            result.merges.is_empty(),
            "took a merge worth {:?} nats on identical members",
            result.merges.first().map(|x| x.gain)
        );
        assert_eq!(result.centre_children.len(), n);
    }

    #[test]
    fn test_the_default_min_gain_clears_the_zero_gain_floor() {
        /// Measured magnitude of the zero-gain floor, per feature.
        ///
        /// One `f64` rounding of an `O(p)` sum, so the floor tracks the
        /// feature count. `7.1e-15` at 64 features and `-3.6e-12` at 32768 both
        /// come to `1.1e-16` per feature; this is that with an order of
        /// headroom.
        const FLOOR_PER_FEATURE: f64 = 1e-15;

        // Identical members on zero-length branches have a true merge gain of
        // exactly zero, so whatever `score_merge` returns for them is the
        // rounding floor `DEFAULT_MIN_GAIN` has to sit above. Swept over three
        // decades of feature count, because the floor grows with the number of
        // per-feature terms the score sums and nothing else. This is what pins
        // the constant.
        for p in [64usize, 2048, 32768] {
            let n = 4usize;
            let row: Vec<f64> = (0..p).map(|g| (g as f64 * 0.13).sin()).collect();
            let prec: Vec<f64> = (0..p)
                .map(|g| 0.7 + 0.3 * (g as f64 * 0.09).cos())
                .collect();
            let m: Vec<f64> = (0..n).flat_map(|_| row.clone()).collect();
            let w: Vec<f64> = (0..n).flat_map(|_| prec.clone()).collect();
            let branch = vec![0.0f64; n];
            let members: Vec<u32> = (0..n as u32).collect();

            let mut mc = vec![0.0f64; p];
            let mut wc = vec![0.0f64; p];
            centre_leaf(&m, &w, &branch, &members, p, &mut mc, &mut wc);
            let mut pairs = Vec::new();
            AllPairs
                .candidates(
                    Round {
                        members: &members,
                        means: &m,
                        precisions: &w,
                        n_features: p,
                        branch: &branch,
                        centre_means: &mc,
                        centre_precisions: &wc,
                        merge: MergeParams::default(),
                        best_gain: f64::NEG_INFINITY,
                        scored_last_round: 0,
                    },
                    &mut pairs,
                )
                .expect("candidates");
            let work = Working {
                m: &m,
                w: &w,
                branch: &branch,
                mc: &mc,
                wc: &wc,
                p,
            };
            let best = scan_pairs(&work, &members, &pairs, MergeParams::default())
                .expect("scan")
                .best;

            let bound = FLOOR_PER_FEATURE * p as f64;
            assert!(
                best.gain.abs() < bound,
                "the floor at {p} features is {:e}, above the {bound:e} the constant assumes",
                best.gain
            );
            assert!(
                bound < DEFAULT_MIN_GAIN,
                "at {p} features the floor reaches {bound:e} and the default min gain is {:e}",
                DEFAULT_MIN_GAIN
            );
        }
    }

    /// A star whose feature 0 makes the peel of some or all pairs unscoreable.
    ///
    /// With `poison_all` the whole feature carries `f64::MAX`, so the centre's
    /// precision there is `+inf` and every peel leaves `inf` behind, whose mean
    /// is `NaN`. Otherwise only members 0 and 1 carry weight in that feature and
    /// the rest carry `1e-300`, which the centre's sum swallows exactly, so
    /// peeling that one pair leaves a remainder of exactly zero precision and no
    /// other pair is affected. Every input is finite and positive, so the star
    /// passes validation and the pathology is genuinely in the arithmetic.
    ///
    /// ### Params
    ///
    /// * `n` - Number of members
    /// * `p` - Number of features
    /// * `poison_all` - Whether every pair is unscoreable rather than one
    ///
    /// ### Returns
    ///
    /// The means, the precisions and the branches to the centre.
    fn poisoned_peel(n: usize, p: usize, poison_all: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut m = Vec::with_capacity(n * p);
        let mut w = Vec::with_capacity(n * p);
        for i in 0..n {
            for g in 0..p {
                // Well separated, so the root branch of a merge has a positive
                // bracket and the solve is entered rather than short-circuited.
                m.push(6.0 * i as f64 + (g as f64 * 0.37).sin());
                w.push(match (g, poison_all) {
                    (0, true) => f64::MAX,
                    (0, false) if i < 2 => 1.0,
                    (0, false) => 1e-300,
                    _ => 1.0 + 0.3 * (i as f64 + g as f64).cos(),
                });
            }
        }
        (m, w, vec![0.0f64; n])
    }

    #[test]
    fn test_the_peel_loses_the_remainder_when_one_member_dominates() {
        // The fixture the doc comment on `peel` asks for. Nothing here is a
        // bug: the peel is a subtraction and this is
        // what a subtraction does. What the test pins is where the loss starts
        // and that it is monotone in the dominance, so a change to the
        // accumulation that moved either would be caught.
        let (p, n) = (8usize, 6usize);
        let mut previous = 0.0f64;
        for (ratio, bound, non_positive) in [
            (1e9f64, 1e-6f64, false),
            (1e13, 1e-2, false),
            (1e15, 1e0, false),
            (1e16, f64::INFINITY, true),
        ] {
            let mut m = Vec::with_capacity(n * p);
            let mut w = Vec::with_capacity(n * p);
            for i in 0..n {
                for g in 0..p {
                    m.push((i as f64 * 0.7 + g as f64 * 0.13).sin());
                    w.push(if i == 0 { ratio } else { 1.0 });
                }
            }
            let t0 = vec![0.0f64; n];
            let members: Vec<u32> = (0..n as u32).collect();
            let (mut mc, mut wc) = (vec![0.0f64; p], vec![0.0f64; p]);
            centre_leaf(&m, &w, &t0, &members, p, &mut mc, &mut wc);
            let work = Working {
                m: &m,
                w: &w,
                branch: &t0,
                mc: &mc,
                wc: &wc,
                p,
            };

            // Peel the pair holding the dominant member, then re-accumulate the
            // same remainder exactly and compare.
            let (mut m_r, mut w_r) = (vec![0.0f64; p], vec![0.0f64; p]);
            peel(&work, 0, 1, &mut m_r, &mut w_r);
            let rest: Vec<u32> = (2..n as u32).collect();
            let (mut m_e, mut w_e) = (vec![0.0f64; p], vec![0.0f64; p]);
            centre_leaf(&m, &w, &t0, &rest, p, &mut m_e, &mut w_e);

            let mut worst = 0.0f64;
            let mut seen_non_positive = false;
            for g in 0..p {
                seen_non_positive |= w_r[g] <= 0.0;
                worst = worst.max((m_r[g] - m_e[g]).abs() / m_e[g].abs().max(1e-30));
            }
            assert_eq!(
                seen_non_positive, non_positive,
                "at a dominance of {ratio:e} the remainder's precision changed sign class"
            );
            assert!(
                worst < bound,
                "at a dominance of {ratio:e} the remainder's mean was off by {worst:e}"
            );
            assert!(
                worst >= previous,
                "the loss stopped being monotone in the dominance at {ratio:e}"
            );
            previous = worst;
        }
    }

    #[test]
    fn test_one_diverged_pair_does_not_stop_the_star() {
        // A non-finite gain is dropped because letting it win would stop the
        // whole primitive; a diverged
        // branch-length solve used to propagate out of the scan's `try_reduce`
        // and stop the primitive anyway, which is the same outcome by another
        // route. It is now dropped like any other unscoreable pair.
        let (p, n) = (5usize, 5usize);
        let (m, w, t0) = poisoned_peel(n, p, false);

        // The fixture only means anything if that pair really does diverge.
        let members: Vec<u32> = (0..n as u32).collect();
        let (mut mc, mut wc) = (vec![0.0f64; p], vec![0.0f64; p]);
        centre_leaf(&m, &w, &t0, &members, p, &mut mc, &mut wc);
        let work = Working {
            m: &m,
            w: &w,
            branch: &t0,
            mc: &mc,
            wc: &wc,
            p,
        };
        let mut scratch = PairScratch::new(p);
        assert!(
            matches!(
                score_pair_raw(&work, &members, 0, 1, MergeParams::default(), &mut scratch),
                Err(BonsaiErrors::RootFindDiverged { .. })
            ),
            "the fixture no longer diverges on the pair it was built around"
        );
        assert!(
            score_pair_raw(&work, &members, 0, 2, MergeParams::default(), &mut scratch).is_ok(),
            "the fixture poisoned more than the one pair"
        );

        let result =
            resolve_star(star(&m, &w, &t0, p), None).expect("one bad pair stopped the star");
        assert_eq!(result.centre_children.len(), MIN_CENTRE_MEMBERS);
        assert_eq!(result.merges.len(), n - MIN_CENTRE_MEMBERS);
        for merge in &result.merges {
            assert!(merge.gain.is_finite() && merge.gain > 0.0);
        }
    }

    #[test]
    fn test_a_round_of_nothing_but_diverged_pairs_is_an_error() {
        // The other half of N9: dropping every pair silently would hand back an
        // unresolved star and call it success, which is what the blocking B2
        // fix exists to prevent. The error is recovered from the first pair, so
        // it is the same error whatever the thread count.
        let (p, n) = (5usize, 5usize);
        let (m, w, t0) = poisoned_peel(n, p, true);
        let out = resolve_star(star(&m, &w, &t0, p), None);
        assert!(
            matches!(out, Err(BonsaiErrors::RootFindDiverged { .. })),
            "a wholly unscoreable round returned {:?}",
            out.map(|r| r.merges.len())
        );
    }

    #[test]
    fn test_identical_members_on_real_branches_collapse_to_zero_length() {
        // The same members, but hanging off non-zero branches. Now a merge does
        // help, because it deletes those branches: identical data want no
        // separation at all. Every branch the primitive creates must be zero.
        let (p, n) = (32usize, 7usize);
        let row: Vec<f64> = (0..p).map(|g| (g as f64 * 0.23).cos()).collect();
        let m: Vec<f64> = (0..n).flat_map(|_| row.clone()).collect();
        let w = vec![1.0f64; n * p];
        let t0 = vec![0.6f64; n];

        let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
        assert_eq!(result.centre_children.len(), MIN_CENTRE_MEMBERS);
        for merge in &result.merges {
            assert_eq!(merge.t_left, 0.0);
            assert_eq!(merge.t_right, 0.0);
            assert_eq!(merge.t_centre, 0.0);
            assert!(merge.gain > 0.0);
        }
    }

    #[test]
    fn test_ancestor_effective_leaves_match_the_pruning_recursion() {
        // The effective leaf carried forward for a new ancestor is what every
        // later round scores against, so it has to be exactly what the pruning
        // recursion would compute for that node in the finished tree.
        let (p, n) = (36usize, 12usize);
        let (m, w) = clustered(3, 4, p, 0.6, 4.5);
        let t0: Vec<f64> = (0..n).map(|i| 0.25 + 0.03 * i as f64).collect();

        let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
        let n_merges = result.merges.len();
        let tree = replay(&result, &t0, n_merges);
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).expect("state");
        state.prune(&tree);

        // `from_parents` relabels the internal nodes, so match them up by the
        // set of leaves below them rather than by index.
        let sets = leaf_sets(&tree);
        let mut own: Vec<Vec<u32>> = vec![Vec::new(); n + n_merges];
        for leaf in 0..n {
            own[leaf].push(leaf as u32);
        }
        for merge in &result.merges {
            let mut here = own[merge.left as usize].clone();
            here.extend_from_slice(&own[merge.right as usize]);
            here.sort_unstable();
            own[merge.ancestor as usize] = here;
        }

        for (a, merge) in result.merges.iter().enumerate() {
            let want = &own[merge.ancestor as usize];
            let node = sets
                .iter()
                .position(|s| s == want)
                .expect("ancestor has no counterpart in the finished tree");
            let node = node as u32;
            for g in 0..p {
                assert_relative_eq!(
                    result.ancestor_precisions[a * p + g],
                    state.precisions(node)[g],
                    max_relative = 1e-12
                );
                // Absolute on the means: everything downstream consumes squared
                // differences of them, so a mean near zero carrying large
                // relative error is fine.
                assert_relative_eq!(
                    result.ancestor_means[a * p + g],
                    state.means(node)[g],
                    epsilon = 1e-11
                );
            }
        }
    }

    #[test]
    fn test_two_and_three_members_do_nothing() {
        let p = 16usize;
        for n in [2usize, 3] {
            let (m, w) = clustered(n, 1, p, 0.3, 5.0);
            let t0 = vec![0.5f64; n];
            let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
            assert!(result.merges.is_empty(), "{n} members should not merge");
            assert_eq!(result.centre_children.len(), n);
            assert!(result.ancestor_means.is_empty());

            let (tree, gain) = star_tree(star(&m, &w, &t0, p), None).expect("star tree");
            assert_eq!(gain, 0.0);
            assert_eq!(tree.n_nodes(), n + 1);
        }
    }

    #[test]
    fn test_one_distant_member_ends_up_alone_on_a_long_branch() {
        // Eight members close together and one a long way off.
        //
        // The outlier is not left at the centre, and expecting it to be was
        // wrong: this star's branches have not been optimised, which is search
        // step 1, so the outlier sits at 0.4 where the data put it at about
        // 1600, and the largest gain available anywhere in the first round is
        // the merge that finally gives it that branch. Its partner comes out
        // on a zero-length branch, which is exactly the polytomy SPEC.md
        // section 9.2 exists to go back and resolve. What is pinned here is
        // that the outlier ends up isolated on a branch orders of magnitude
        // longer than anything else, rather than dragging a near member out
        // of the group with it.
        let (p, n) = (24usize, 9usize);
        let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15);
        let base: Vec<f64> = (0..p).map(|g| (g as f64 * 0.19).sin()).collect();
        let mut m = Vec::with_capacity(n * p);
        let mut w = Vec::with_capacity(n * p);
        for i in 0..n {
            for g in 0..p {
                let far = if i == n - 1 { 40.0 } else { 0.0 };
                m.push(base[g] + far + 0.3 * (rng.uniform() - 0.5));
                w.push(0.8 + 0.4 * rng.uniform());
            }
        }
        let t0 = vec![0.4f64; n];

        let result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");
        let outlier = result.branch[n - 1];
        assert!(
            outlier > 1e2,
            "the distant member kept a short branch: {outlier}"
        );
        for (i, &t) in result.branch[..n - 1].iter().enumerate() {
            assert!(
                t < 1e-2 * outlier,
                "near member {i} was separated by {t} against the outlier's {outlier}"
            );
        }
    }

    #[test]
    fn test_rejects_an_ill_described_star() {
        let p = 8usize;
        let m = vec![0.0f64; 3 * p];
        let w = vec![1.0f64; 3 * p];
        let t0 = vec![0.5f64; 3];

        let short = Star {
            means: &m[..2 * p],
            precisions: &w,
            branch: &t0,
            n_features: p,
        };
        assert!(matches!(
            resolve_star(short, None),
            Err(BonsaiErrors::MalformedTree { .. })
        ));

        let lonely = star(&m[..p], &w[..p], &t0[..1], p);
        assert!(matches!(
            resolve_star(lonely, None),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_candidate_provider_restricts_the_scan() {
        /// A provider that only ever offers consecutive members, standing in
        /// for the `k`-nearest-neighbour restriction of SPEC.md section 11.
        struct Consecutive;

        impl CandidatePairs<f64> for Consecutive {
            /// ### Params
            ///
            /// * `round` - Current round
            /// * `out` - Destination for the pairs
            ///
            /// ### Returns
            ///
            /// Nothing.
            fn candidates(
                &mut self,
                round: Round<'_, f64>,
                out: &mut Vec<(usize, usize)>,
            ) -> Result<(), BonsaiErrors> {
                for i in 0..round.members.len().saturating_sub(1) {
                    out.push((i, i + 1));
                }
                Ok(())
            }
        }

        let (p, n) = (20usize, 10usize);
        let (m, w) = clustered(5, 2, p, 0.4, 4.0);
        let t0 = vec![0.5f64; n];

        let restricted = resolve_star_with(
            star(&m, &w, &t0, p),
            None,
            &mut Consecutive,
            Verbosity::Quiet,
        )
        .expect("resolve");
        let exhaustive = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");

        // The clusters are consecutive in index, so the restriction still finds
        // them; what it cannot do is beat the exhaustive scan.
        assert_eq!(restricted.centre_children.len(), MIN_CENTRE_MEMBERS);
        let a: f64 = restricted.merges.iter().map(|x| x.gain).sum();
        let b: f64 = exhaustive.merges.iter().map(|x| x.gain).sum();
        assert!(a <= b + 1e-9, "restricted scan gained {a} against {b}");
    }

    #[test]
    fn test_f32_storage_builds_the_same_topology() {
        let (p, n) = (32usize, 12usize);
        let (m, w) = clustered(3, 4, p, 0.5, 5.0);
        let t0 = vec![0.4f64; n];

        let wide_result = resolve_star(star(&m, &w, &t0, p), None).expect("resolve");

        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let narrow_result = resolve_star(
            Star {
                means: &m32,
                precisions: &w32,
                branch: &t0,
                n_features: p,
            },
            None,
        )
        .expect("resolve");

        assert_eq!(narrow_result.parent, wide_result.parent);
    }
}
