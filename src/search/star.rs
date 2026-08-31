//! The star primitive of SPEC.md section 9.1.
//!
//! Given a centre and the star of effective leaves hanging off it, score every
//! candidate pair with [`crate::model::merge::score_merge`], insert an ancestor
//! above the pair [`StarSelection`] picks, summarise that ancestor as an effective leaf
//! (SPEC.md section 4) so the remaining structure is a star again, and repeat.
//! Stop when the centre has three members left or no pair gives a gain worth
//! taking.
//!
//! **One routine drives three of the seven search steps.** It is step 2 with
//! the centre being the root and the members being every leaf, step 3 with the
//! centre being a polytomy node, and step 6 after an NNI edge deletion. So it
//! is written over "a set of effective leaves with branch lengths to a common
//! centre" and knows nothing about where they came from. A centre that has a
//! parent passes its upstream side in as one more member, which is what makes
//! stopping at three members mean "the centre is resolved" in every case.
//!
//! ### What this module does not do
//!
//! Candidate pairs come from a [`CandidatePairs`] provider, and the only one
//! here is [`AllPairs`]. The `k`-nearest-neighbour restriction of SPEC.md
//! section 11 lives in [`crate::search::candidates`]; the upper-bound machinery
//! of section 10 fits behind the same trait and is separate work.

use crate::errors::BonsaiErrors;
use crate::model::merge::{EffLeaf, MergeParams, MergeScratch, score_merge};
use crate::tree::{NO_NODE, Tree};
use crate::utils::rng::SplitMix64;
use crate::utils::traits::{BonsaiFloat, narrow, wide};
use rayon::prelude::*;

////////////////
// Parameters //
////////////////

/// Number of members at which the centre is resolved and the primitive stops.
///
/// SPEC.md section 9.1. Three members around a centre is a degree-three node,
/// which is fully resolved in an unrooted tree; a fourth merge would only
/// insert a node of degree two, which cannot change the likelihood.
const MIN_CENTRE_MEMBERS: usize = 3;

/// Default for [`StarParams::min_gain`], in nats.
///
/// A merge whose gain is indistinguishable from zero is not worth making, and
/// accepting one costs a round of the scan while pretending to have learnt
/// something. The floor it has to clear was measured on 2026-08-27: for
/// members that are exactly identical on zero-length branches, where the true
/// gain is exactly zero, `score_merge` returns `7.1e-15` at 64 features and
/// `-3.6e-12` at 32768, both of which are `1.1e-16` per feature. That is one
/// `f64` rounding of a sum whose magnitude is `O(p)`, which is what it should
/// be, so the floor scales with the feature count and not with anything else.
///
/// `1e-9` therefore clears the floor by four orders of magnitude at the ten
/// thousand features this crate expects and still by one at a million, while
/// sitting far below any gain that carries information: a real merge gain is
/// `O(p)` nats. See `test_the_default_min_gain_clears_the_zero_gain_floor`.
const DEFAULT_MIN_GAIN: f64 = 1e-9;

/// How the primitive picks the pair to merge in a round.
///
/// SPEC.md section 9.4. The greedy rule drives search steps 2 and 3 and is the
/// default; the weighted rule is the random phase of the nearest-neighbour
/// interchanges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StarSelection {
    /// Take the highest-scoring pair of the round.
    #[default]
    Greedy,
    /// Sample a pair with probability proportional to the likelihood of the
    /// tree the merge would produce.
    ///
    /// The candidates differ from the resulting tree loglikelihoods by the
    /// current tree's own loglikelihood, which is common to the round, so a
    /// softmax over the gains is the same distribution as a softmax over the
    /// resulting tree loglikelihoods. The round maximum is subtracted before
    /// exponentiating.
    ///
    /// **Deviation.** The specification samples over every pair; this samples
    /// over the pairs that clear [`StarParams::min_gain`] and stops when none
    /// do, so that the primitive's stopping rule is the same in both modes. A
    /// pair below the floor carries softmax weight `exp(-O(p))` against the
    /// round's best, so nothing measurable is given up.
    ///
    /// **Cost.** Unlike the greedy rule this materialises one score per
    /// candidate pair rather than reducing them as they are produced, so it
    /// wants the small stars of an interchange and not a whole-dataset star.
    ///
    /// **How random this actually is.** The weights are a softmax over
    /// quantities whose gaps are `O(p)` nats, so the distribution concentrates
    /// on the greedy pick as the feature count grows. That is the specification
    /// taken literally and not a shortcut: measured 2026-08-31 through
    /// [`crate::search::nni::nni_random`] at 32 leaves over eight seeds, eight
    /// of eight seeds moved the tree off its starting topology at 8 features,
    /// four of eight at 32, and two and three of eight at 128 and 512. A caller
    /// relying on this to escape a local optimum at ten thousand features
    /// should expect it to behave close to greedy.
    Weighted {
        /// Seed of the splitmix64 stream the draws come from.
        ///
        /// One draw per round, taken after the pairs have been scored and
        /// ordered, so the sampled pair does not depend on the thread count.
        seed: u64,
    },
}

/// Tuning knobs for the star primitive.
#[derive(Clone, Copy, Debug)]
pub struct StarParams {
    /// Smallest loglikelihood gain, in nats, that will be accepted as a merge.
    ///
    /// Strictly greater than: a merge is taken only when its gain exceeds this.
    /// Absolute rather than scaled by the feature count, so a caller running at
    /// an unusually large `p` should raise it; see `DEFAULT_MIN_GAIN`.
    pub min_gain: f64,
    /// Which pair of the round is merged.
    pub selection: StarSelection,
    /// Branch-length solve knobs handed to [`score_merge`].
    pub merge: MergeParams,
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
        }
    }
}

/////////////////////
// Candidate pairs //
/////////////////////

/// One round's read-only view of the star, as a candidate provider sees it.
///
/// Both slabs are indexed by *node id*, not by position in `members`, and are
/// row-major with stride `n_features`. They cover every node created so far,
/// members and swallowed children alike, which is what lets a provider hold
/// state keyed by node id across rounds. Precisions are not diffusion
/// corrected; the branch to the centre is not part of this view, because
/// nothing that chooses *which* pairs to consider needs it.
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
    /// Gain of the merge accepted in the previous round, or negative infinity
    /// in the first round.
    ///
    /// Every round accepts the best pair it scored, so this is also the largest
    /// gain seen so far in the round before this one. It is the incumbent the
    /// upper bounds of SPEC.md section 10 prune against, and it is deliberately
    /// *not* a sound bound on this round's best: the root moves between rounds
    /// and a later merge can beat an earlier one.
    pub best_gain: f64,
}

/// Source of the candidate pairs a round of the scan considers.
///
/// The seam for SPEC.md sections 10 and 11. Scoring every pair is `O(n^2 p)`
/// per round; restricting to a `k`-nearest-neighbour graph or to the pairs
/// whose upper bound still beats the incumbent replaces this implementation
/// without the primitive changing.
///
/// Providers are called once per round and notified after every merge, so they
/// can maintain state incrementally rather than rediscovering what changed.
/// Both halves are `&mut self` and the primitive calls them from one thread, in
/// round order, so a provider needs no interior mutability and no locking.
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
    /// Called after the ancestor has been summarised as an effective leaf and
    /// put back into the star, so `ancestor` is exactly what later rounds will
    /// score against. The default does nothing, which is right for any provider
    /// that keeps no state between rounds.
    ///
    /// ### Params
    ///
    /// * `merge` - The merge that was performed
    /// * `ancestor` - The new ancestor's row of the means and precisions slabs
    fn merged(&mut self, merge: &StarMerge, ancestor: EffLeaf<'_, T>) {
        let _ = (merge, ancestor);
    }
}

/// Every pair of the current members.
///
/// The exhaustive provider, and the one the primitive defaults to.
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
/// Means and precisions are row-major `[member][feature]`, matching the layout
/// of [`crate::model::likelihood::NodeState`]. Precisions are *not* diffusion
/// corrected: the branch to the centre is carried separately in `branch`, and
/// the correction of SPEC.md section 4 is applied where it is needed.
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
/// Node ids run `0..n_members` for the members that were handed in and
/// `n_members..` for the ancestors that were created, in creation order. That
/// numbering already satisfies the arena invariant: an ancestor's children are
/// always created before it, so every parent index exceeds its children's.
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
/// Ordered by gain, then by the pair's node ids ascending. That total order is
/// what makes the reduction independent of how rayon happened to split the
/// work: two candidates can only tie on the gain, never on the ids.
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

/// One round's read-only view of everything the pair scan reads.
///
/// The arrays grow as ancestors are created, so this is rebuilt each round
/// rather than held across the whole run.
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
/// weighted average rather than as a ratio of sums, for the same conditioning
/// reason as in [`crate::utils::kernels::prune_general`]: every partial value
/// stays inside the convex hull of the member means.
///
/// Recomputed from scratch once per round rather than updated in place, which
/// makes the primitive `O(n^2 p)` overall in the recompute alone. That is the
/// same order as the exhaustive pair scan it sits inside, so it costs nothing
/// today; once candidate restriction cuts the scan to `O(n k p)` this becomes
/// the dominant term and wants an incremental update, at the cost of drift
/// accumulating over the whole run.
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

/// Peel a pair off the centre's effective leaf to get the rest of the star.
///
/// SPEC.md section 8.1. `O(p)` per pair by subtraction, rather than `O(n p)` by
/// re-accumulating over the other members, which is the whole reason a merge
/// scan is affordable.
///
/// **Conditioning.** Both lines difference quantities of similar magnitude. If
/// one member carries most of the centre's precision then `WR` is a small
/// difference of large numbers and loses significant digits, and the mean is
/// worse because the subtraction happens in the numerator *and* the small `WR`
/// then divides it. The accumulation is in `f64` regardless of storage type,
/// which keeps this out of the way for the star sizes and precision ranges
/// tested so far. If a fixture ever shows it biting, that is a finding to
/// report, not something to paper over by changing the maths: the re-accumulated
/// form is a different cost class.
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
/// Allocated once per worker through `map_init` rather than once per pair.
struct PairScratch<T> {
    /// Branch-length solve scratch.
    merge: MergeScratch,
    /// The remainder's means, from the peel.
    m_r: Vec<T>,
    /// The remainder's precisions, from the peel.
    w_r: Vec<T>,
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
    fn new(p: usize) -> Self {
        Self {
            merge: MergeScratch::new(p),
            m_r: vec![T::zero(); p],
            w_r: vec![T::zero(); p],
        }
    }
}

/// Score one candidate pair.
///
/// A pair whose gain comes back non-finite becomes [`Candidate::NONE`] rather
/// than a selectable candidate. That is a numerical pathology in the
/// branch-length solve for that pair alone, and letting it win would stop the
/// whole primitive.
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
/// The candidate, or the error the branch-length solve failed with.
fn score_pair<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    a: usize,
    b: usize,
    merge: MergeParams,
    scratch: &mut PairScratch<T>,
) -> Result<Candidate, BonsaiErrors> {
    let p = work.p;
    let (i, j) = (members[a] as usize, members[b] as usize);
    peel(work, i, j, &mut scratch.m_r, &mut scratch.w_r);

    let score = score_merge(
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
    )?;

    Ok(if score.gain.is_finite() {
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
    })
}

/// Score every candidate pair and return the best.
///
/// Read-only over the working set, so the scan is a parallel map and reduce.
/// Each worker allocates its scratch and its remainder buffers once through
/// `map_init` rather than once per pair; the reduction is a total order over
/// `(gain, node ids)`, so the winner does not depend on the thread count.
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
/// The best candidate, or [`Candidate::NONE`] if there were no scoreable
/// pairs, or the error the branch-length solve failed with.
fn scan_pairs<T: BonsaiFloat>(
    work: &Working<'_, T>,
    members: &[u32],
    pairs: &[(usize, usize)],
    merge: MergeParams,
) -> Result<Candidate, BonsaiErrors> {
    let p = work.p;
    pairs
        .par_iter()
        .map_init(
            || PairScratch::new(p),
            |scratch, &(a, b)| score_pair(work, members, a, b, merge, scratch),
        )
        .try_reduce(|| Candidate::NONE, |x, y| Ok(x.better(y)))
}

/// Score every candidate pair and sample one in proportion to the likelihood of
/// the tree its merge would produce.
///
/// SPEC.md section 9.4. The scores are collected in `pairs` order, which rayon
/// preserves for an indexed iterator, and the softmax and the cumulative draw
/// then run sequentially over that fixed order. So the sampled pair is a
/// function of the seed and the star alone and not of which worker finished
/// first, which is what a `RAYON_NUM_THREADS` sweep in the tests pins.
///
/// Only pairs clearing `min_gain` are eligible, so this has the same stopping
/// rule as the greedy scan; see [`StarSelection::Weighted`].
///
/// ### Params
///
/// * `work` - The round's working set
/// * `members` - Nodes currently attached to the centre
/// * `pairs` - Candidate pairs, as positions into `members`
/// * `merge` - Branch-length solve knobs
/// * `min_gain` - Floor a pair must clear to be eligible
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
    rng: &mut SplitMix64,
) -> Result<Candidate, BonsaiErrors> {
    let p = work.p;
    let scored: Vec<Candidate> = pairs
        .par_iter()
        .map_init(
            || PairScratch::new(p),
            |scratch, &(a, b)| score_pair(work, members, a, b, merge, scratch),
        )
        .collect::<Result<Vec<_>, BonsaiErrors>>()?;

    let eligible = |c: &&Candidate| c.gain > min_gain;
    let top = scored
        .iter()
        .filter(eligible)
        .fold(f64::NEG_INFINITY, |acc, c| acc.max(c.gain));
    if !top.is_finite() {
        return Ok(Candidate::NONE);
    }

    // Softmax over loglikelihoods, so the maximum comes off before the
    // exponential: a round's gains are `O(p)` nats apart and would otherwise
    // overflow at a few hundred features.
    let total: f64 = scored
        .iter()
        .filter(eligible)
        .map(|c| (c.gain - top).exp())
        .sum();
    let target = rng.uniform() * total;
    let mut acc = 0.0f64;
    for c in scored.iter().filter(eligible) {
        acc += (c.gain - top).exp();
        if acc >= target {
            return Ok(*c);
        }
    }
    // The cumulative sum can fall a rounding short of `total`. Nothing was
    // sampled then, so take the last eligible pair rather than nothing.
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
        // Convex combination, so the ancestor's mean is pinned between its two
        // children's and cannot cancel. It feeds straight back into the next
        // round, so error here compounds up the tree.
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
    resolve_star_with(star, params, &mut AllPairs)
}

/// Run the star primitive.
///
/// Each round scores the candidate pairs, inserts an ancestor above the one
/// [`StarSelection`] picks, summarises it as an effective leaf, and puts it back in the star in
/// place of the two members it swallowed. Stops at `MIN_CENTRE_MEMBERS`
/// members or when no pair clears [`StarParams::min_gain`].
///
/// The branch lengths of members other than the merged pair are untouched: a
/// merge only optimises the three branches it creates (SPEC.md section 8.4),
/// and the rest of the tree is reoptimised by search steps 4 and 7.
///
/// ### Params
///
/// * `star` - The members and their branches to the centre
/// * `params` - Tuning knobs, or `None` for the defaults
/// * `candidates` - Which pairs to consider each round
///
/// ### Returns
///
/// The topology that was built, or `MalformedTree` if the star is
/// ill-described, or the error the branch-length solve failed with.
pub fn resolve_star_with<T: BonsaiFloat, C: CandidatePairs<T>>(
    star: Star<'_, T>,
    params: Option<StarParams>,
    candidates: &mut C,
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

    // Reject a star the arithmetic cannot score, rather than discovering it one
    // pair at a time. A non-finite mean or precision makes every candidate gain
    // non-finite; those map to `Candidate::NONE`, the round finds no best pair,
    // and the loop exits normally. The caller then gets `Ok` with an unresolved
    // star and no indication that anything went wrong, which for a single `NaN`
    // in a large input matrix is the worst possible failure mode.
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

    let mut mc = vec![0.0f64; p];
    let mut wc = vec![0.0f64; p];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut best_gain = f64::NEG_INFINITY;
    let mut rng = SplitMix64::new(match params.selection {
        StarSelection::Greedy => 0,
        StarSelection::Weighted { seed } => seed,
    });

    while members.len() > MIN_CENTRE_MEMBERS {
        centre_leaf(&m, &w, &branch, &members, p, &mut mc, &mut wc);

        pairs.clear();
        candidates.candidates(
            Round {
                members: &members,
                means: &m,
                precisions: &w,
                n_features: p,
                best_gain,
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
        // `Candidate::NONE` carries minus infinity, so an empty or entirely
        // non-finite round falls out of the loop here too.
        let best = match params.selection {
            StarSelection::Greedy => scan_pairs(&work, &members, &pairs, params.merge)?,
            StarSelection::Weighted { .. } => sample_pair(
                &work,
                &members,
                &pairs,
                params.merge,
                params.min_gain,
                &mut rng,
            )?,
        };
        if best.gain <= params.min_gain {
            break;
        }

        let (k, l) = (best.left as usize, best.right as usize);
        let ancestor = parent.len() as u32;
        push_ancestor(&mut m, &mut w, k, l, best.t_ak, best.t_al, p);
        parent[k] = ancestor;
        parent[l] = ancestor;
        branch[k] = best.t_ak;
        branch[l] = best.t_al;
        parent.push(NO_NODE);
        branch.push(best.t_ar);

        // The ancestor's id exceeds every member's, so appending keeps the
        // membership ascending and the tie-break in `Candidate::better` is a
        // tie-break on position as well as on id.
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
    let result = resolve_star(star, params)?;
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
        // Regression, adversarial review 2026-08-27. A single non-finite value
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
        // The load-bearing test. The primitive claims a gain from a closed-form
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
        /// 2026-08-27: one `f64` rounding of an `O(p)` sum, so the floor
        /// tracks the feature count. `7.1e-15` at 64 features and `-3.6e-12`
        /// at 32768 both come to `1.1e-16` per feature; this is that with an
        /// order of headroom.
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
                        best_gain: f64::NEG_INFINITY,
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
            let best = scan_pairs(&work, &members, &pairs, MergeParams::default()).expect("scan");

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

        let restricted =
            resolve_star_with(star(&m, &w, &t0, p), None, &mut Consecutive).expect("resolve");
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
