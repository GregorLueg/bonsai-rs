//! Upper bounds on merge scores
//!
//! **Not an optimisation.** The naive search is `O(n^3 p)`: every round scores
//! every pair, because a merge moves the root and the root enters every other
//! pair's score through the peel of SPEC.md section 8.1. The candidate
//! restriction of [`crate::search::candidates`] takes out one factor of `n`.
//! This module takes out most of another, by not rescoring pairs that provably
//! cannot win.
//!
//! ### What the bound is
//!
//! A pair's gain `dL` depends on the centre only through the peeled remainder
//! `(MR, WR)`, and that in turn only through the centre's own effective leaf
//! `(M_r, W_r)`. Linearise `dL` in a movement `(xi_mu, xi_w)` of the centre and
//! maximise the linear part over the ellipsoid of plausible movements that
//! SPEC.md section 10.3 derives. What comes out is two numbers per pair: the
//! gain where it was scored, and how fast that gain can rise per unit of centre
//! movement.
//!
//! **Deviation: the bound tracks the centre rather than assuming the worst.**
//! Section 10.4 draws a fixed ellipsoid, adds its whole slack to every gain,
//! and asks each round whether the centre is still inside. But the maximum of a
//! linear form over a ball of radius `R` is `R` times its maximum over the unit
//! ball, so the same information gives an exact bound at whatever radius the
//! centre has actually reached. So the ellipsoid is not a region to be inside;
//! it is a metric, the distance travelled in it is accumulated round by round,
//! and a pair's bound is its gain plus that distance times its unit slack. A
//! pair scored last round is bounded to last round's gain and not to the worst
//! case of a whole schedule of merges, which is where most of the pruning
//! comes from. It also means a pair scored mid-generation needs no special
//! treatment: its bound is its own gain in the round it was scored in.
//!
//! `nsteps` then decides one thing only: how far the centre may travel before
//! the *linearisation* is redone. See [`EllipsoidBoundsParams::nsteps`].
//!
//! ### Honesty about what this is not
//!
//! It is not a strict mathematical bound. The linearisation can underestimate.
//! [`EllipsoidBoundsParams::verify`] turns that from an argument into a
//! measurement: it scores every offered pair every round, counts how often a
//! true gain exceeds its recorded bound, and replays the primitive's walk to
//! count how often that cost the answer. On this crate's fixtures the first
//! number is about one in a thousand and the second is zero. Both are
//! tabulated, with the diagnosis, in [`EllipsoidBoundsParams::default`].
//!
//! ### The derivative
//!
//! The flattened `d(dL)/d(W[g,r])` of the SI is a seven-term expression and is
//! not transcribed here. It is assembled instead by the chain rule through the
//! peel, which is four one-line partials, with `d(dL)/d(MR)` and
//! `d(dL)/d(WR)` read off SPEC.md section 8.3. Every one of them is pinned
//! against central differences in this module's tests.

use crate::errors::BonsaiErrors;
use crate::model::merge::EffLeaf;
use crate::search::star::{BOUND_WALK_CHUNK, CandidatePairs, PairScratch, Round, StarMerge};
use crate::utils::traits::{BonsaiFloat, wide};
use rayon::prelude::*;
use std::collections::HashMap;

///////////////
// Constants //
///////////////

/// Default for [`EllipsoidBoundsParams::nsteps`].
///
/// See [`EllipsoidBoundsParams::default`] for the measurement that pins it.
const DEFAULT_NSTEPS: f64 = 48.0;

/// Smallest value the online sizing will shrink `nsteps` to.
///
/// One merge moves the centre about two to three units of the metric, so below
/// this every round redraws and the bounds pay for themselves twice over while
/// pruning nothing: at `nsteps = 1` every round redraws and the star scores
/// more pairs than the exhaustive scan.
const NSTEPS_MIN: f64 = 1.0;

/// Largest value the online sizing will grow `nsteps` to.
///
/// A pure runaway guard. The schedule is a hill-climb on a bowl and settles far
/// below this on every fixture measured; the cap is here so that a star whose
/// costs never cross cannot wander somewhere the linearisation is meaningless.
const NSTEPS_MAX: f64 = 512.0;

/// Weight of the newest round in the two running cost averages.
///
/// The schedule compares what redraws cost against what the walk costs, and
/// both are spiky: a redraw round scores thousands of pairs and the round after
/// it scores a chunk. Averaging over roughly ten rounds is enough to see past
/// that without lagging a whole star behind.
const COST_DECAY: f64 = 0.1;

/// Multiplier applied to `nsteps` on a round where the walk is the larger cost.
///
/// Paired with `GROW` at the reciprocal rate. Both are gentle because the
/// schedule moves every round: at five per cent a step it takes fourteen
/// consecutive rounds to move `nsteps` by a factor of two, which is slow enough
/// that the two cost averages settle rather than chase each other.
const SHRINK: f64 = 0.95;

/// Multiplier applied to `nsteps` on a round where redrawing is.
const GROW: f64 = 1.05;

/// Chunks a round must offer before its cost is taken as a signal.
///
/// The walk cannot stop inside a chunk, so a round offering less than one chunk
/// always scores everything and reads as maximally deep whatever the bounds
/// did. The tail of every star is such a round. Four chunks is where the signal
/// stops being dominated by the granularity.
const ADAPT_MIN_CHUNKS: usize = 4;

///////////////////////////
// EllipsoidBoundsParams //
///////////////////////////

/// Tuning knobs for [`EllipsoidBounds`].
///
/// None of these change the answer. `nsteps` and the online schedule trade
/// bound tightness against how often the linearisation is redrawn; `verify` is
/// a measurement mode.
#[derive(Clone, Copy, Debug)]
pub struct EllipsoidBoundsParams {
    /// How far the centre may travel before every bound is redrawn, in the
    /// metric of SPEC.md section 10.3.
    ///
    /// One merge moves the centre about two to three of these units, so this is
    /// `nsteps` in the section's sense to within that constant.
    ///
    /// **What it does and does not control.** It does not control how loose a
    /// bound is: a bound grows exactly in step with the distance the centre has
    /// actually travelled since the pair was scored, so a pair scored last
    /// round is bounded tightly whatever this is set to. What it controls is
    /// how far the *linearisation* is stretched before it is redone, and
    /// therefore the trade between redrawing often and walking deep. The total
    /// is a bowl; see [`EllipsoidBoundsParams::default`].
    pub nsteps: f64,
    /// Move `nsteps` towards the size that costs least (SPEC.md section 10.5).
    pub adapt: bool,
    /// Score every offered pair every round and check it against its recorded
    /// bound.
    ///
    /// Costs the whole saving, so this is for tests and for the measurements in
    /// [`EllipsoidBoundsParams::default`], not for production. The pairs
    /// emitted and their order are unchanged, so a verified run builds the same
    /// tree as an unverified one.
    pub verify: bool,
}

impl Default for EllipsoidBoundsParams {
    /// `nsteps = 48`, online sizing on, verification off.
    ///
    /// ### Where these came from
    ///
    /// Ours, chosen by measurement over clustered stars of 128 members by 200
    /// features, four seeds, wrapping [`crate::search::star::AllPairs`] so the
    /// comparison is the `349,500` pairs a star scores exhaustively in 1.68
    /// seconds. "Bounded" is the pairs the provider scored to build bounds,
    /// "walked" the pairs the primitive's walk scored, and "scored" their sum,
    /// which is the whole cost.
    ///
    /// | `nsteps` | redraws | bounded | walked | scored | fraction | seconds |
    /// |---|---|---|---|---|---|---|
    /// | 6 | 25.2 | 93,644 | 2,633 | 96,276 | 0.275 | 0.586 |
    /// | 12 | 14.0 | 56,134 | 5,045 | 61,179 | 0.175 | 0.411 |
    /// | 24 | 7.8 | 35,459 | 8,413 | 43,872 | 0.126 | 0.336 |
    /// | 48 | 4.2 | 24,963 | 10,573 | 35,536 | 0.102 | 0.289 |
    /// | 96 | 3.0 | 19,893 | 19,297 | 39,190 | 0.112 | 0.357 |
    /// | 192 | 2.0 | 17,351 | 42,157 | 59,508 | 0.170 | 0.594 |
    ///
    /// A bowl with its floor at 48, where the two halves are within a factor of
    /// two of each other. That is a factor of ten in pairs and six in wall
    /// time, and it widens with the star: the exhaustive term is `O(n^3 p)` and
    /// this one is much flatter.
    ///
    /// ### How it composes, and how it scales
    ///
    /// Same fixtures at 200 features, defaults throughout, four seeds. The
    /// `k`-nearest-neighbour restriction of SPEC.md section 11 is the other
    /// half of the acceleration and the two multiply:
    ///
    /// | members | exhaustive pairs | bounds only | with section 11 | seconds, exhaustive to both |
    /// |---|---|---|---|---|
    /// | 64 | 43,676 | 9,046 | 4,512 | 0.219 to 0.041 |
    /// | 128 | 349,500 | 37,618 | 14,480 | 1.683 to 0.133 |
    /// | 256 | 2,796,156 | 144,116 | 46,605 | 13.619 to 0.457 |
    ///
    /// Sixty times fewer pairs and thirty times less wall time at 256 members,
    /// growing with `n`. The wall-time factor lags the pair-count factor
    /// because the walk is a sequence of small parallel scans while the
    /// exhaustive scan is one large one; see
    /// [`crate::search::star::BOUND_WALK_CHUNK`].
    ///
    /// ### The online schedule
    ///
    /// Its value is not the couple of per cent it buys at the right starting
    /// point. It is what a wrong starting point costs:
    ///
    /// | start | fixed there | adaptive from there |
    /// |---|---|---|
    /// | 12 | 0.175 | 0.109 |
    /// | 48 | 0.102 | 0.108 |
    /// | 192 | 0.170 | 0.137 |
    ///
    /// A default measured on one star size is a guess at another, and the
    /// schedule halves what that guess can cost. At the right value it is
    /// slightly worse than the fixed setting, because it spends the first
    /// rounds finding it.
    ///
    /// ### Bound violations
    ///
    /// The linearisation is not a strict bound and SPEC.md section 10.2 says
    /// so, so this is measured rather than assumed. With `verify` on, four
    /// seeds, every offered pair checked against its recorded bound every
    /// round:
    ///
    /// | members | features | checks | `nsteps` | violations | worst, nats | worst / best gain | misses |
    /// |---|---|---|---|---|---|---|---|
    /// | 48 | 64 | 73,680 | 1 | 0 | 0 | 0 | 0 |
    /// | 48 | 64 | 73,680 | 3 | 63 | 0.45 | 0.7% | 0 |
    /// | 48 | 64 | 73,680 | 12 | 96 | 2.08 | 3.6% | 0 |
    /// | 48 | 64 | 73,680 | 48 | 67 | 2.08 | 3.6% | 0 |
    /// | 48 | 64 | 73,680 | 768 | 90 | 2.08 | 3.6% | 0 |
    /// | 96 | 128 | 589,744 | 3 | 185 | 0.53 | 0.5% | 0 |
    /// | 96 | 128 | 589,744 | 12 | 234 | 1.41 | 1.2% | 0 |
    /// | 96 | 128 | 589,744 | 48 | 107 | 2.16 | 1.9% | 0 |
    /// | 96 | 128 | 589,744 | 768 | 82 | 2.17 | 1.9% | 0 |
    ///
    /// **Violations happen: about one check in a thousand, by up to two nats
    /// against best gains of sixty to a hundred and seventy.** They appear the
    /// moment the centre travels further than about one merge and then
    /// plateau, which is the signature of a second-order term and not of a
    /// wrong derivative: at `nsteps <= 1` every round redraws, every bound is
    /// its own pair's gain, and the count is exactly zero. The plateau is
    /// because the walk rescores the top of the list every round and resets
    /// those bounds, so only pairs nobody is looking at drift far.
    ///
    /// The suspect is the peel. `WR = W_r - Wd_k - Wd_l` is linear and exact,
    /// but `MR` is a ratio whose denominator is that difference, so for a pair
    /// carrying much of the centre's precision the second derivative is large.
    /// [`crate::search::star::peel`] flags the same conditioning for the same
    /// reason.
    ///
    /// **What it costs is nothing, and that is also measured.** The "misses"
    /// column replays the primitive's walk over the true gains and counts the
    /// rounds where it would have stopped on a pair that was not the round's
    /// best. It is zero everywhere: a violated bound has never yet belonged to
    /// a pair anybody was going to pick. `test_bounded_search_matches_the_exhaustive_scan`
    /// is the same claim made structurally, over four star sizes, three seeds
    /// and four ellipsoid sizes.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            nsteps: DEFAULT_NSTEPS,
            adapt: true,
            verify: false,
        }
    }
}

///////////////
// Reporting //
///////////////

/// What a run of [`EllipsoidBounds`] cost and whether the bounds held.
///
/// The saving is the whole justification for the module, so it is reported
/// rather than asserted. Cumulative over the provider's whole life, so an
/// instance reused across several stars reports their total.
#[derive(Clone, Copy, Debug, Default)]
pub struct EllipsoidBoundsStats {
    /// Rounds the provider was asked for candidates.
    pub rounds: usize,
    /// Rounds in which every bound had to be recomputed.
    pub refreshes: usize,
    /// Candidate pairs offered, summed over rounds. What an exhaustive scan
    /// would have scored.
    pub offered: usize,
    /// Pairs the provider scored to build a bound.
    pub bounded: usize,
    /// Pairs the primitive's bound-ordered walk scored.
    ///
    /// Reported to the provider one round late, since it is the only channel
    /// the primitive has, so the last round of a star is not counted. That is
    /// one round in `n` and it is not corrected for.
    pub walked: usize,
    /// Bound checks made under [`EllipsoidBoundsParams::verify`].
    pub checked: usize,
    /// Checks in which the true gain exceeded the recorded bound.
    pub violations: usize,
    /// Largest `true gain - bound` seen, or zero if none was positive.
    pub worst_violation: f64,
    /// Rounds in which the bound-ordered walk would have stopped on a pair
    /// that was not the round's true best.
    ///
    /// The only consequence a violation can have. A bound that is exceeded by
    /// a pair nobody was going to pick costs nothing; this counts the times it
    /// cost the answer.
    pub misses: usize,
}

impl EllipsoidBoundsStats {
    /// Pairs scored, over pairs an exhaustive scan would have scored.
    ///
    /// ### Returns
    ///
    /// The fraction, or one if nothing was offered.
    pub fn work_fraction(&self) -> f64 {
        if self.offered == 0 {
            1.0
        } else {
            (self.bounded + self.walked) as f64 / self.offered as f64
        }
    }
}

/////////////////
// The kernels //
/////////////////

/// One feature of one candidate pair, as the derivative sees it.
///
/// Grouped rather than passed loose because the derivative needs thirteen
/// numbers and an argument list that long is a transcription hazard, which is
/// the exact failure this module exists to avoid.
#[derive(Clone, Copy, Debug)]
struct Feature {
    /// Effective mean of the first child.
    m_k: f64,
    /// Effective precision of the first child.
    w_k: f64,
    /// Effective mean of the second child.
    m_l: f64,
    /// Effective precision of the second child.
    w_l: f64,
    /// Effective mean of the peeled remainder, `MR` of SPEC.md section 8.1.
    m_rem: f64,
    /// Effective precision of the remainder, `WR`.
    w_rem: f64,
    /// The centre's own effective mean, `M[g,r]`.
    m_c: f64,
    /// The centre's own effective precision, `W[g,r]`.
    w_c: f64,
}

/// The branch lengths a merge score is evaluated at.
#[derive(Clone, Copy, Debug)]
struct Branches {
    /// Existing branch from the centre to the first child.
    t_rk: f64,
    /// Existing branch from the centre to the second child.
    t_rl: f64,
    /// New branch from the ancestor to the first child.
    t_ak: f64,
    /// New branch from the ancestor to the second child.
    t_al: f64,
    /// New branch from the ancestor to the centre.
    t_ar: f64,
}

/// Sensitivity of one feature's contribution to the merge score to the two
/// remainder quantities.
///
/// SPEC.md section 8.3 differentiated with respect to `MR` and `WR`, holding
/// the pair's own effective leaves and all five branch lengths fixed. Both
/// stars contribute: `MR` enters through the two squared separations
/// `d_kR` and `d_lR`, which are shared, and `WR` enters as `O3` before the
/// merge and through `A3 = 1/(t_ar + 1/WR)` after it.
///
/// **Why the branch lengths may be held fixed.** `t_ak`, `t_al` and `t_ar` are
/// the optimum of SPEC.md section 8.4 and so are functions of `MR` and `WR`,
/// but the score is stationary in them there, so their movement contributes at
/// second order. `t_rk` and `t_rl` are properties of the existing tree and do
/// not move at all. The total `k`-to-`l` length is fixed by stage one, which
/// never looks at the remainder.
///
/// The stationarity is not unconditional. `t_ar` can rest at zero and the split
/// at either end of its bracket, and at a boundary the score is stationary only
/// for movements that keep it there. A movement that frees it improves the true
/// score by more than this predicts, which is one of the two things that can
/// make a bound too small; the other is the curvature of the peel. See
/// [`EllipsoidBoundsParams::default`] for what the two of them together
/// actually cost, which on the fixtures here is nothing.
///
/// ### Params
///
/// * `f` - The feature's effective leaves
/// * `b` - The branch lengths
///
/// ### Returns
///
/// `d(dL)/d(MR)` and `d(dL)/d(WR)` for this feature.
fn remainder_partials(f: Feature, b: Branches) -> (f64, f64) {
    let (ck, cl) = (1.0 / f.w_k, 1.0 / f.w_l);

    // Before the merge: a three-leaf star on the centre. The remainder sits on
    // the centre itself, so it takes no diffusion correction.
    let o1 = 1.0 / (b.t_rk + ck);
    let o2 = 1.0 / (b.t_rl + cl);
    let o3 = f.w_rem;
    // After: the same three leaves on the new ancestor, with the remainder now
    // a branch away.
    let a1 = 1.0 / (b.t_ak + ck);
    let a2 = 1.0 / (b.t_al + cl);
    let a3 = 1.0 / (b.t_ar + 1.0 / f.w_rem);

    let (dk, dl) = (f.m_k - f.m_rem, f.m_l - f.m_rem);
    let d_kl = (f.m_k - f.m_l) * (f.m_k - f.m_l);
    let (d_kr, d_lr) = (dk * dk, dl * dl);

    let sa = a1 + a2 + a3;
    let qa = a1 * a2 * d_kl + a1 * a3 * d_kr + a2 * a3 * d_lr;
    let so = o1 + o2 + o3;
    let qo = o1 * o2 * d_kl + o1 * o3 * d_kr + o2 * o3 * d_lr;

    // The derivative of `star3` with respect to its third precision, which is
    // the same expression `model::merge`'s `g3` carries.
    let g_a3 = 1.0 / a3 - 1.0 / sa - (a1 * d_kr + a2 * d_lr) / sa + qa / (sa * sa);
    let g_o3 = 1.0 / o3 - 1.0 / so - (o1 * d_kr + o2 * d_lr) / so + qo / (so * so);

    // d(A3)/d(WR) = A3^2 / WR^2, from A3 = 1/(t_ar + 1/WR). d(O3)/d(WR) = 1.
    // The `O` half enters the score with the opposite sign.
    let d_w = 0.5 * (g_a3 * a3 * a3 / (f.w_rem * f.w_rem) - g_o3);

    // Only the two separations depend on MR, and d(d_kR)/d(MR) = -2*(M_k - MR).
    let d_m = (dk * (a1 * a3 / sa - o1 * o3 / so)) + (dl * (a2 * a3 / sa - o2 * o3 / so));

    (d_m, d_w)
}

/// Sensitivity of one feature's contribution to the merge score to the
/// *centre's* own effective leaf.
///
/// The chain rule through the peel of SPEC.md section 8.1. With
/// `WR = W_r - Wd_k - Wd_l` and `MR = (M_r*W_r - Wd_k*M_k - Wd_l*M_l) / WR`,
/// and the pair's own quantities held fixed, the four partials are
///
/// ```text
/// d(MR)/d(M_r) = W_r / WR        d(MR)/d(W_r) = (M_r - MR) / WR
/// d(WR)/d(M_r) = 0               d(WR)/d(W_r) = 1
/// ```
///
/// The second one is `d/dW_r [(M_r*W_r - c)/(W_r - c')]`, which collapses to
/// `(M_r*WR - MR*WR)/WR^2` because the numerator is `MR*WR` by definition.
///
/// ### Params
///
/// * `f` - The feature's effective leaves
/// * `b` - The branch lengths
///
/// ### Returns
///
/// `d(dL)/d(M[g,r])` and `d(dL)/d(W[g,r])` for this feature.
fn centre_partials(f: Feature, b: Branches) -> (f64, f64) {
    let (d_mr, d_wr) = remainder_partials(f, b);
    let d_m = d_mr * (f.w_c / f.w_rem);
    let d_w = d_mr * ((f.m_c - f.m_rem) / f.w_rem) + d_wr;
    (d_m, d_w)
}

/// Per-feature scales that define the metric the centre's movement is measured
/// in.
///
/// SPEC.md section 10.3, S41, with `nsteps` factored out. The centre is a
/// precision-weighted mean over `nc` children; one merge removes two and adds
/// one, moving the position by about `1/sqrt(nc * W[g,r])` and the precision by
/// about `W[g,r]/nc`. Dividing a movement by these turns the two ellipsoids of
/// S41 into unit balls, which is S42.
///
/// **Deviation: the feature count belongs in the scales.** S41 writes them
/// without it, which makes them per-feature sizes while the ellipsoid they
/// define is a norm over all `p` features. A movement of exactly one merge's
/// size in *every* feature then sits at radius `sqrt(p)` rather than at one, so
/// the unit as written means a merge only at a single feature. Multiplying the
/// mean scale by `sqrt(p)` and the precision scale by `sqrt(p)` fixes both, and
/// is what is written below. The unit is then one merge's movement whatever the
/// feature count, which is what makes a measured default carry from one dataset
/// to another. At 200 features the unscaled form is out by `sqrt(200)` in the
/// scale, so by 200 in the `nsteps` that would have to be asked for.
///
/// Measured on this crate's fixtures, one merge moves the centre about three of
/// these units in the mean and two in the precision, so the derivation is right
/// to a small constant and not to the digit. That constant is why
/// [`EllipsoidBoundsParams::nsteps`] is measured rather than set to a merge
/// count.
///
/// ### Params
///
/// * `w_c` - The centre's effective precision for this feature
/// * `nc` - Members the centre had when the metric was fixed
/// * `root_p` - Square root of the feature count
///
/// ### Returns
///
/// The scale in the mean and the scale in the precision.
#[inline]
fn metric_scales(w_c: f64, nc: f64, root_p: f64) -> (f64, f64) {
    (root_p * (1.0 / (nc * w_c)).sqrt(), root_p * w_c / nc)
}

/// Largest first-order increase in the gain per unit of centre movement.
///
/// Rescaling the ellipsoid to a unit ball turns the linearised change into a
/// dot product, and the maximum of a dot product over a unit ball is the
/// vector's norm (SPEC.md section 10.3, S42 and S45). The two ellipsoids are
/// independent, so they are returned separately and the caller adds them after
/// weighting each by how far the centre has actually moved in that coordinate.
///
/// **Deviation.** S45 is written as a sum of absolute values, which is the
/// maximum over the *box* that circumscribes the ellipsoid, not over the
/// ellipsoid itself; the prose either side of it says "the vector's norm",
/// which is the Euclidean one, and that is what is taken here. The difference
/// is not cosmetic: the two differ by up to `sqrt(p)`, so at a few thousand
/// features the box form inflates every bound by a factor of fifty and prunes
/// nothing.
///
/// ### Params
///
/// * `d_m` - Per-feature `d(dL)/d(M[g,r])`
/// * `d_w` - Per-feature `d(dL)/d(W[g,r])`
/// * `metric_w` - The centre's effective precisions when the metric was fixed
/// * `nc` - Members the centre had then
///
/// ### Returns
///
/// The slack per unit of movement in the mean and in the precision, both
/// non-negative.
fn unit_slack(d_m: &[f64], d_w: &[f64], metric_w: &[f64], nc: f64) -> (f64, f64) {
    let mut acc_m = 0.0f64;
    let mut acc_w = 0.0f64;
    let root_p = (metric_w.len() as f64).sqrt();
    for g in 0..metric_w.len() {
        let (s_m, s_w) = metric_scales(metric_w[g], nc, root_p);
        let a = d_m[g] * s_m;
        let b = d_w[g] * s_w;
        acc_m += a * a;
        acc_w += b * b;
    }
    (acc_m.sqrt(), acc_w.sqrt())
}

/// How far the centre moved between two rounds, in the metric.
///
/// Accumulated round by round rather than measured against the anchor, because
/// what a bound needs is the distance from the centre *the pair was scored at*,
/// and pairs are scored in different rounds. Summing the steps bounds every
/// such distance at once by the triangle inequality, and does it far more
/// tightly than adding two distances from a common anchor would.
///
/// ### Params
///
/// * `m_c` - The centre's effective means now
/// * `w_c` - The centre's effective precisions now
/// * `m_prev` - The same, one round ago
/// * `w_prev` - The same, one round ago
/// * `metric_w` - The centre's effective precisions when the metric was fixed
/// * `nc` - Members the centre had then
///
/// ### Returns
///
/// The step length in the mean and in the precision.
fn metric_step(
    m_c: &[f64],
    w_c: &[f64],
    m_prev: &[f64],
    w_prev: &[f64],
    metric_w: &[f64],
    nc: f64,
) -> (f64, f64) {
    let mut acc_m = 0.0f64;
    let mut acc_w = 0.0f64;
    let root_p = (metric_w.len() as f64).sqrt();
    for g in 0..metric_w.len() {
        let (s_m, s_w) = metric_scales(metric_w[g], nc, root_p);
        let a = (m_c[g] - m_prev[g]) / s_m;
        let b = (w_c[g] - w_prev[g]) / s_w;
        acc_m += a * a;
        acc_w += b * b;
    }
    (acc_m.sqrt(), acc_w.sqrt())
}

//////////////////
// The provider //
//////////////////

/// What is remembered about one pair between rounds.
///
/// The bound at any later round is
/// `gain + (path_m - born_m) * slack_m + (path_w - born_w) * slack_w`, because
/// the maximum of a linear form over a ball of radius `R` is `R` times its
/// maximum over the unit ball. So the bound tightens itself: it is exactly the
/// gain in the round the pair was scored in, and grows only as fast as the
/// centre actually travels.
#[derive(Clone, Copy, Debug)]
struct Bound {
    /// The gain at the centre the pair was scored at.
    gain: f64,
    /// Slack per unit of movement of the centre's mean.
    slack_m: f64,
    /// Slack per unit of movement of the centre's precision.
    slack_w: f64,
    /// Distance the centre had already travelled in the mean when this was
    /// scored.
    born_m: f64,
    /// The same in the precision.
    born_w: f64,
}

impl Bound {
    /// The bound at a given point on the centre's path.
    ///
    /// ### Params
    ///
    /// * `path_m` - Distance travelled in the mean since the metric was fixed
    /// * `path_w` - The same in the precision
    ///
    /// ### Returns
    ///
    /// The upper bound on the pair's gain now.
    #[inline]
    fn at(&self, path_m: f64, path_w: f64) -> f64 {
        if self.gain == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        self.gain + (path_m - self.born_m) * self.slack_m + (path_w - self.born_w) * self.slack_w
    }
}

/// One worker's buffers for building a bound.
struct BoundScratch<T> {
    /// The pair scan's own scratch, which also carries the peel.
    pair: PairScratch<T>,
    /// Per-feature `d(dL)/d(M[g,r])`.
    d_m: Vec<f64>,
    /// Per-feature `d(dL)/d(W[g,r])`.
    d_w: Vec<f64>,
}

impl<T: BonsaiFloat> BoundScratch<T> {
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
            pair: PairScratch::new(p),
            d_m: vec![0.0; p],
            d_w: vec![0.0; p],
        }
    }
}

/// One pair's true gain and how fast that gain can rise as the centre moves.
///
/// ### Params
///
/// * `round` - The round's view of the star
/// * `i` - Position of the first member of the pair
/// * `j` - Position of the second
/// * `metric_w` - The centre's effective precisions when the metric was fixed
/// * `nc` - Members the centre had then
/// * `scratch` - Reusable buffers
///
/// ### Returns
///
/// The gain at the current centre and the two unit slacks, or the error the
/// branch-length solve failed with. A pair whose gain is not finite comes back
/// with a negative infinity gain and zero slack, which sorts it out of the way
/// exactly as [`crate::search::star::AllPairs`] would drop it.
fn bound_pair<T: BonsaiFloat>(
    round: &Round<'_, T>,
    i: usize,
    j: usize,
    metric_w: &[f64],
    nc: f64,
    scratch: &mut BoundScratch<T>,
) -> Result<Bound, BonsaiErrors> {
    let p = round.n_features;
    let score = round.score_pair(i, j, &mut scratch.pair)?;
    if !score.gain.is_finite() {
        return Ok(Bound {
            gain: f64::NEG_INFINITY,
            slack_m: 0.0,
            slack_w: 0.0,
            born_m: 0.0,
            born_w: 0.0,
        });
    }

    let (node_k, node_l) = (round.members[i] as usize, round.members[j] as usize);
    let b = Branches {
        t_rk: round.branch[node_k],
        t_rl: round.branch[node_l],
        t_ak: score.t_ak,
        t_al: score.t_al,
        t_ar: score.t_ar,
    };
    let (bk, bl) = (node_k * p, node_l * p);

    for g in 0..p {
        let f = Feature {
            m_k: wide(round.means[bk + g]),
            w_k: wide(round.precisions[bk + g]),
            m_l: wide(round.means[bl + g]),
            w_l: wide(round.precisions[bl + g]),
            m_rem: wide(scratch.pair.m_r[g]),
            w_rem: wide(scratch.pair.w_r[g]),
            m_c: round.centre_means[g],
            w_c: round.centre_precisions[g],
        };
        let (d_m, d_w) = centre_partials(f, b);
        scratch.d_m[g] = d_m;
        scratch.d_w[g] = d_w;
    }

    let (slack_m, slack_w) = unit_slack(&scratch.d_m, &scratch.d_w, metric_w, nc);
    Ok(Bound {
        gain: score.gain,
        // A slack that came out non-finite would silently stop pruning the
        // wrong way round, so it becomes infinite rather than being trusted.
        slack_m: if slack_m.is_finite() {
            slack_m
        } else {
            f64::INFINITY
        },
        slack_w: if slack_w.is_finite() {
            slack_w
        } else {
            f64::INFINITY
        },
        born_m: 0.0,
        born_w: 0.0,
    })
}

/// Candidate pairs ordered by an upper bound on their merge score.
///
/// Wraps another provider, which decides *which* pairs exist; this decides in
/// what order they are worth scoring and when the answer is already known. So
/// it composes with [`crate::search::candidates::KnnCandidates`], which is how
/// it is meant to be run: section 11 removes one factor of `n` and section 10
/// most of another.
///
/// ### How a round goes
///
/// 1. Ask the inner provider for the live pairs.
/// 2. Add the step the centre took since last round to the running distance.
/// 3. If that distance has passed `nsteps`, throw the table away and fix a new
///    metric here.
/// 4. Any pair with no entry, which is every pair after a redraw and the new
///    ancestor's pairs otherwise, is scored now and given one.
/// 5. Emit the pairs sorted by their bound at the current distance, descending.
///
/// A pair scored this round is emitted at exactly its own gain, because the
/// distance from where it was scored is zero. That is what stops a redraw round
/// scoring everything twice: the whole table is exact, the top of the emitted
/// list is the round's true best, and the primitive's walk stops in its first
/// chunk.
///
/// ### Pairs whose entry is missing
///
/// A pair the table has no entry for is emitted with an infinite bound rather
/// than a guess, so it is always scored. Nothing produces one today, since
/// anything the inner provider offers is either in the table or scored above,
/// but the alternative to being explicit about it is a silent prune.
#[derive(Clone, Debug)]
pub struct EllipsoidBounds<C> {
    /// Which pairs exist.
    inner: C,
    /// Tuning knobs.
    params: EllipsoidBoundsParams,
    /// How far the centre may travel before the linearisation is redrawn,
    /// which the online schedule moves.
    nsteps: f64,
    /// The centre's effective precisions when the metric was fixed.
    metric_w: Vec<f64>,
    /// Members the centre had then.
    metric_nc: f64,
    /// The centre's effective means one round ago.
    prev_m: Vec<f64>,
    /// The centre's effective precisions one round ago.
    prev_w: Vec<f64>,
    /// Distance the centre has travelled in the mean since the metric was
    /// fixed, and in the precision.
    path_m: f64,
    /// Distance travelled in the precision.
    path_w: f64,
    /// Whether a metric has been fixed for the current star.
    anchored: bool,
    /// Running average of the pairs a round scores to build bounds.
    cost_bounded: f64,
    /// Running average of the pairs a round's walk scores.
    cost_walked: f64,
    /// What is known about each live pair, keyed by node ids, smaller first.
    table: HashMap<(u32, u32), Bound>,
    /// Table being built for the next round, so dead pairs fall out for free.
    next_table: HashMap<(u32, u32), Bound>,
    /// The bounds of the pairs emitted this round, in emitted order.
    emitted: Vec<f64>,
    /// The inner provider's pairs, before reordering.
    raw: Vec<(usize, usize)>,
    /// Emission order being built: bound, then node ids for the tie-break.
    order: Vec<(f64, u32, u32, usize, usize)>,
    /// What the run cost and whether the bounds held.
    stats: EllipsoidBoundsStats,
}

impl<C> EllipsoidBounds<C> {
    /// A provider over a fresh ellipsoid.
    ///
    /// ### Params
    ///
    /// * `inner` - Provider deciding which pairs exist
    /// * `params` - Tuning knobs, or `None` for
    ///   [`EllipsoidBoundsParams::default`]
    ///
    /// ### Returns
    ///
    /// The provider. The first round of a star anchors it.
    pub fn new(inner: C, params: Option<EllipsoidBoundsParams>) -> Self {
        let params = params.unwrap_or_default();
        Self {
            inner,
            params,
            nsteps: params.nsteps,
            metric_w: Vec::new(),
            metric_nc: 0.0,
            prev_m: Vec::new(),
            prev_w: Vec::new(),
            path_m: 0.0,
            path_w: 0.0,
            anchored: false,
            cost_bounded: 0.0,
            cost_walked: 0.0,
            table: HashMap::new(),
            next_table: HashMap::new(),
            emitted: Vec::new(),
            raw: Vec::new(),
            order: Vec::new(),
            stats: EllipsoidBoundsStats::default(),
        }
    }

    /// What the run has cost so far.
    ///
    /// ### Returns
    ///
    /// The statistics.
    pub fn stats(&self) -> EllipsoidBoundsStats {
        self.stats
    }

    /// The ellipsoid size the online schedule has settled on.
    ///
    /// ### Returns
    ///
    /// The current `nsteps`.
    pub fn nsteps(&self) -> f64 {
        self.nsteps
    }

    /// Move `nsteps` towards the size that costs least.
    ///
    /// SPEC.md section 10.5 reads the depth of the walk: deep means the bounds
    /// are too loose, shallow means a looser one would do. **Depth alone is the
    /// wrong signal and measurement says so.** The walk cannot stop inside a
    /// chunk, so on a star of any size it bottoms out at one chunk and stays
    /// there, the schedule reads every round as shallow, and `nsteps` grows
    /// until it hits whatever cap it was given: a depth-only schedule ends
    /// pinned at the cap from every starting value, so the cap and not the
    /// schedule is choosing the answer.
    ///
    /// What is missing is the other half of the trade. A smaller ellipsoid
    /// redraws more often and walks less; a larger one does the reverse; the
    /// total is a bowl. So both costs are tracked as running averages and the
    /// schedule walks downhill on their difference, which for a trade-off of
    /// this shape puts the resting point at the bottom of the bowl. It lands at
    /// `nsteps` near 60 on the fixtures in
    /// [`EllipsoidBoundsParams::default`], where the fixed-value sweep has its
    /// floor at 48.
    ///
    /// Rounds offering less than `ADAPT_MIN_CHUNKS` chunks are ignored: the
    /// tail of every star is such a round and it reads as maximally deep
    /// whatever the bounds did.
    ///
    /// ### Params
    ///
    /// * `bounded` - Pairs this round scored to build bounds
    /// * `walked` - Pairs the previous round's walk scored
    /// * `offered` - Pairs this round offered
    fn adapt(&mut self, bounded: usize, walked: usize, offered: usize) {
        if !self.params.adapt || offered < ADAPT_MIN_CHUNKS * BOUND_WALK_CHUNK {
            return;
        }
        self.cost_bounded += COST_DECAY * (bounded as f64 - self.cost_bounded);
        self.cost_walked += COST_DECAY * (walked as f64 - self.cost_walked);
        self.nsteps = if self.cost_bounded > self.cost_walked {
            (self.nsteps * GROW).min(NSTEPS_MAX)
        } else {
            (self.nsteps * SHRINK).max(NSTEPS_MIN)
        };
    }
}

impl<T: BonsaiFloat, C: CandidatePairs<T>> CandidatePairs<T> for EllipsoidBounds<C> {
    /// The inner provider's pairs, reordered by upper bound, descending.
    ///
    /// ### Params
    ///
    /// * `round` - Read-only view of the current round
    /// * `out` - Destination for the pairs
    ///
    /// ### Returns
    ///
    /// Nothing, or the error the inner provider or the branch-length solve
    /// failed with.
    fn candidates(
        &mut self,
        round: Round<'_, T>,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<(), BonsaiErrors> {
        self.raw.clear();
        self.inner.candidates(round, &mut self.raw)?;
        self.emitted.clear();
        if self.raw.is_empty() {
            return Ok(());
        }

        let p = round.n_features;
        let nc = round.members.len() as f64;
        self.stats.rounds += 1;
        self.stats.offered += self.raw.len();
        self.stats.walked += round.scored_last_round;

        // The first round of a star, whoever owned this provider before. No
        // ancestor exists yet exactly then, and node ids restart with the star,
        // so nothing carried over from a previous one may be reused. The
        // feature count is checked too, since a provider handed a differently
        // shaped star would otherwise index a metric of the wrong length.
        let n_nodes = round.means.len() / p;
        let fresh_star =
            !self.anchored || self.metric_w.len() != p || round.members.len() == n_nodes;

        if !fresh_star {
            let (step_m, step_w) = metric_step(
                round.centre_means,
                round.centre_precisions,
                &self.prev_m,
                &self.prev_w,
                &self.metric_w,
                self.metric_nc,
            );
            self.path_m += step_m;
            self.path_w += step_w;
        }

        // The linearisation is redrawn once the centre has travelled further
        // than the schedule allows. That is the only thing `nsteps` decides:
        // the bounds themselves are exact in the distance travelled, so a
        // longer leash costs accuracy of the linearisation and nothing else.
        let stale = self.path_m.max(self.path_w) >= self.nsteps;
        if fresh_star || stale {
            self.table.clear();
            self.metric_w.clear();
            self.metric_w.extend_from_slice(round.centre_precisions);
            self.metric_nc = nc;
            self.path_m = 0.0;
            self.path_w = 0.0;
            self.anchored = true;
            self.stats.refreshes += 1;
        }
        self.prev_m.clear();
        self.prev_m.extend_from_slice(round.centre_means);
        self.prev_w.clear();
        self.prev_w.extend_from_slice(round.centre_precisions);

        // Everything without an entry is scored now: every pair after a
        // redraw, and the new ancestor's pairs otherwise.
        let missing: Vec<(usize, usize)> = self
            .raw
            .iter()
            .copied()
            .filter(|&(i, j)| {
                !self
                    .table
                    .contains_key(&(round.members[i], round.members[j]))
            })
            .collect();

        let metric_w = std::mem::take(&mut self.metric_w);
        let metric_nc = self.metric_nc;
        let scored: Result<Vec<Bound>, BonsaiErrors> = missing
            .par_iter()
            .map_init(
                || BoundScratch::new(p),
                |scratch, &(i, j)| bound_pair(&round, i, j, &metric_w, metric_nc, scratch),
            )
            .collect();
        self.metric_w = metric_w;
        let scored = scored?;
        self.stats.bounded += scored.len();
        self.adapt(scored.len(), round.scored_last_round, self.raw.len());

        for (&(i, j), &bound) in missing.iter().zip(scored.iter()) {
            self.table.insert(
                (round.members[i], round.members[j]),
                Bound {
                    born_m: self.path_m,
                    born_w: self.path_w,
                    ..bound
                },
            );
        }

        self.order.clear();
        self.next_table.clear();
        for &(i, j) in &self.raw {
            let key = (round.members[i], round.members[j]);
            // A pair the table cannot account for must always be scored, so it
            // is emitted at the top with no bound at all rather than pruned.
            let bound = self.table.get(&key).copied();
            if let Some(bound) = bound {
                self.next_table.insert(key, bound);
            }
            let emit = bound.map_or(f64::INFINITY, |b| b.at(self.path_m, self.path_w));
            self.order.push((emit, key.0, key.1, i, j));
        }
        std::mem::swap(&mut self.table, &mut self.next_table);

        // Descending on the bound, ascending on the node ids, so the emitted
        // order is a function of the star alone.
        self.order
            .sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        for &(bound, _, _, i, j) in &self.order {
            out.push((i, j));
            self.emitted.push(bound);
        }

        if self.params.verify {
            self.check(&round)?;
        }
        Ok(())
    }

    /// Forward the merge to the inner provider.
    ///
    /// The bound table needs nothing here: the inner provider stops offering
    /// pairs that mention either child, and the table is rebuilt from what is
    /// offered, so dead entries fall out on their own.
    ///
    /// ### Params
    ///
    /// * `merge` - The merge that was performed
    /// * `ancestor` - The new ancestor's effective leaf
    fn merged(&mut self, merge: &StarMerge, ancestor: EffLeaf<'_, T>) {
        self.inner.merged(merge, ancestor);
    }

    /// The bounds of this round's emitted pairs.
    ///
    /// ### Returns
    ///
    /// The bounds, in emitted order and non-increasing.
    fn bounds(&self) -> Option<&[f64]> {
        Some(&self.emitted)
    }
}

impl<C> EllipsoidBounds<C> {
    /// Score every emitted pair and check it against its recorded bound.
    ///
    /// The property the whole scheme rests on, measured rather than assumed.
    /// Only runs under [`EllipsoidBoundsParams::verify`], and does not change
    /// what is emitted.
    ///
    /// ### Params
    ///
    /// * `round` - The round's view of the star
    ///
    /// ### Returns
    ///
    /// Nothing, or the error the branch-length solve failed with.
    fn check<T: BonsaiFloat>(&mut self, round: &Round<'_, T>) -> Result<(), BonsaiErrors> {
        let p = round.n_features;
        // Non-finite gains become minus infinity, which is what the primitive
        // does with them, so the replay below orders pairs the same way it
        // would and no comparison ever sees a `NaN`.
        let gains: Vec<f64> = self
            .order
            .par_iter()
            .map_init(
                || PairScratch::new(p),
                |scratch, &(_, _, _, i, j)| {
                    round.score_pair(i, j, scratch).map(|score| {
                        if score.gain.is_finite() {
                            score.gain
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                },
            )
            .collect::<Result<Vec<_>, BonsaiErrors>>()?;

        for (gain, &bound) in gains.iter().zip(self.emitted.iter()) {
            if *gain == f64::NEG_INFINITY {
                continue;
            }
            self.stats.checked += 1;
            let excess = gain - bound;
            if excess > 0.0 {
                self.stats.violations += 1;
                self.stats.worst_violation = self.stats.worst_violation.max(excess);
            }
        }

        // Replay the primitive's walk over the emitted order and see whether it
        // would have stopped on the round's true best. This is the only thing a
        // violation can actually cost.
        let key = |i: usize| {
            (
                gains[i],
                std::cmp::Reverse((self.order[i].1, self.order[i].2)),
            )
        };
        let better = |a: usize, b: usize| if key(b) > key(a) { b } else { a };
        let mut best = 0usize;
        let mut done = 0usize;
        while done < gains.len() {
            if gains[best] > self.emitted[done] && done > 0 {
                break;
            }
            let end = (done + BOUND_WALK_CHUNK).min(gains.len());
            for i in done..end {
                best = better(best, i);
            }
            done = end;
        }
        let truth = (0..gains.len()).fold(0usize, better);
        if key(best) != key(truth) {
            self.stats.misses += 1;
        }
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::merge::{MergeParams, MergeScratch, gain_at, score_merge};
    use crate::search::candidates::{KnnCandidates, KnnCandidatesParams};
    use crate::search::star::{AllPairs, Star, StarParams, StarResult, resolve_star_with};
    use crate::utils::rng::SplitMix64;

    /// A star of clustered members with unequal precisions and branches.
    ///
    /// ### Params
    ///
    /// * `n` - Members
    /// * `p` - Features
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Means, precisions and branch lengths to the centre.
    fn star_fixture(n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let n_groups = (n / 4).max(2);
        let centres: Vec<Vec<f64>> = (0..n_groups)
            .map(|_| (0..p).map(|_| 4.0 * (rng.uniform() - 0.5)).collect())
            .collect();

        let mut m = Vec::with_capacity(n * p);
        let mut w = Vec::with_capacity(n * p);
        for i in 0..n {
            let c = &centres[i % n_groups];
            for g in 0..p {
                m.push(c[g] + 0.35 * (rng.uniform() - 0.5));
                w.push(0.5 + 2.0 * rng.uniform());
            }
        }
        let branch: Vec<f64> = (0..n).map(|_| 0.05 + 0.2 * rng.uniform()).collect();
        (m, w, branch)
    }

    /// The centre's effective leaf over a whole star.
    ///
    /// ### Params
    ///
    /// * `m` - Means, row-major
    /// * `w` - Precisions, row-major
    /// * `branch` - Branches to the centre
    /// * `p` - Features
    ///
    /// ### Returns
    ///
    /// The centre's means and precisions.
    fn centre(m: &[f64], w: &[f64], branch: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut mc = vec![0.0; p];
        let mut wc = vec![0.0; p];
        for (i, &t) in branch.iter().enumerate() {
            for g in 0..p {
                let wi = w[i * p + g];
                let wd = wi / (1.0 + t * wi);
                wc[g] += wd;
                mc[g] += (m[i * p + g] - mc[g]) * (wd / wc[g]);
            }
        }
        (mc, wc)
    }

    /// The gain of one pair at an explicitly given centre, branch lengths held
    /// fixed.
    ///
    /// The function the central-difference tests differentiate: it runs the
    /// peel of SPEC.md section 8.1 and then the score of section 8.3 with no
    /// reoptimisation, which is exactly what the analytic partials hold fixed.
    ///
    /// ### Params
    ///
    /// * `f` - Per-feature effective leaves, the remainder fields ignored
    /// * `b` - The branch lengths
    /// * `wd_k` - Diffusion-corrected precision of the first child
    /// * `wd_l` - Diffusion-corrected precision of the second child
    ///
    /// ### Returns
    ///
    /// The gain.
    fn gain_at_centre(f: &[Feature], b: Branches, wd_k: &[f64], wd_l: &[f64]) -> f64 {
        let p = f.len();
        let (mut m_k, mut w_k) = (vec![0.0; p], vec![0.0; p]);
        let (mut m_l, mut w_l) = (vec![0.0; p], vec![0.0; p]);
        let (mut m_r, mut w_r) = (vec![0.0; p], vec![0.0; p]);
        for g in 0..p {
            m_k[g] = f[g].m_k;
            w_k[g] = f[g].w_k;
            m_l[g] = f[g].m_l;
            w_l[g] = f[g].w_l;
            let wr = f[g].w_c - wd_k[g] - wd_l[g];
            w_r[g] = wr;
            m_r[g] = (f[g].m_c * f[g].w_c - wd_k[g] * f[g].m_k - wd_l[g] * f[g].m_l) / wr;
        }
        let mut scratch = MergeScratch::new(p);
        scratch.prepare(
            EffLeaf { m: &m_k, w: &w_k },
            EffLeaf { m: &m_l, w: &w_l },
            EffLeaf { m: &m_r, w: &w_r },
            b.t_rk,
            b.t_rl,
        );
        gain_at(b.t_ak + b.t_al, b.t_ak, b.t_ar, &scratch)
    }

    /// A three-leaf configuration to differentiate, with its peel.
    ///
    /// ### Params
    ///
    /// * `p` - Features
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// The per-feature state, the branch lengths and the two children's
    /// diffusion-corrected precisions.
    fn differentiable(p: usize, seed: u64) -> (Vec<Feature>, Branches, Vec<f64>, Vec<f64>) {
        let (m, w, branch) = star_fixture(9, p, seed);
        let (mc, wc) = centre(&m, &w, &branch, p);
        let b = Branches {
            t_rk: branch[0],
            t_rl: branch[1],
            t_ak: 0.13,
            t_al: 0.21,
            t_ar: 0.09,
        };

        let mut wd_k = vec![0.0; p];
        let mut wd_l = vec![0.0; p];
        let mut f = Vec::with_capacity(p);
        for g in 0..p {
            wd_k[g] = w[g] / (1.0 + b.t_rk * w[g]);
            wd_l[g] = w[p + g] / (1.0 + b.t_rl * w[p + g]);
            let wr = wc[g] - wd_k[g] - wd_l[g];
            f.push(Feature {
                m_k: m[g],
                w_k: w[g],
                m_l: m[p + g],
                w_l: w[p + g],
                m_rem: (mc[g] * wc[g] - wd_k[g] * m[g] - wd_l[g] * m[p + g]) / wr,
                w_rem: wr,
                m_c: mc[g],
                w_c: wc[g],
            });
        }
        (f, b, wd_k, wd_l)
    }

    /// Both partials of SPEC.md section 10.2 against central differences.
    #[test]
    fn test_centre_partials_match_central_differences() {
        let p = 24;
        let (f, b, wd_k, wd_l) = differentiable(p, 0x5EED_0001);

        for g in 0..p {
            let (d_m, d_w) = centre_partials(f[g], b);

            let h_m = 1e-6 * f[g].m_c.abs().max(1.0);
            let mut up = f.clone();
            let mut dn = f.clone();
            up[g].m_c += h_m;
            dn[g].m_c -= h_m;
            let fd_m = (gain_at_centre(&up, b, &wd_k, &wd_l)
                - gain_at_centre(&dn, b, &wd_k, &wd_l))
                / (2.0 * h_m);

            let h_w = 1e-6 * f[g].w_c;
            let mut up = f.clone();
            let mut dn = f.clone();
            up[g].w_c += h_w;
            dn[g].w_c -= h_w;
            let fd_w = (gain_at_centre(&up, b, &wd_k, &wd_l)
                - gain_at_centre(&dn, b, &wd_k, &wd_l))
                / (2.0 * h_w);

            let tol_m = 1e-6 * fd_m.abs().max(1e-4);
            let tol_w = 1e-6 * fd_w.abs().max(1e-4);
            assert!(
                (d_m - fd_m).abs() < tol_m,
                "feature {g}: d(dL)/d(M_r) analytic {d_m:e} against finite difference {fd_m:e}"
            );
            assert!(
                (d_w - fd_w).abs() < tol_w,
                "feature {g}: d(dL)/d(W_r) analytic {d_w:e} against finite difference {fd_w:e}"
            );
        }
    }

    /// The four peel partials of SPEC.md section 10.2, each on its own.
    ///
    /// `centre_partials` composes them with `remainder_partials`, so a sign
    /// error in one can be hidden by a sign error in another. This pins the
    /// peel by itself.
    #[test]
    fn test_peel_partials_match_central_differences() {
        let p = 8;
        let (f, b, wd_k, wd_l) = differentiable(p, 0x5EED_0002);
        let _ = b;

        for g in 0..p {
            let (m_c, w_c) = (f[g].m_c, f[g].w_c);
            let (ck, cl) = (wd_k[g], wd_l[g]);
            let (m_k, m_l) = (f[g].m_k, f[g].m_l);
            let peel = |m_r: f64, w_r: f64| {
                let wr = w_r - ck - cl;
                ((m_r * w_r - ck * m_k - cl * m_l) / wr, wr)
            };

            let (mr, wr) = peel(m_c, w_c);
            let h_m = 1e-6 * m_c.abs().max(1.0);
            let h_w = 1e-6 * w_c;

            let d_mr_d_mr = (peel(m_c + h_m, w_c).0 - peel(m_c - h_m, w_c).0) / (2.0 * h_m);
            let d_mr_d_wr = (peel(m_c, w_c + h_w).0 - peel(m_c, w_c - h_w).0) / (2.0 * h_w);
            let d_wr_d_mr = (peel(m_c + h_m, w_c).1 - peel(m_c - h_m, w_c).1) / (2.0 * h_m);
            let d_wr_d_wr = (peel(m_c, w_c + h_w).1 - peel(m_c, w_c - h_w).1) / (2.0 * h_w);

            assert!((d_mr_d_mr - w_c / wr).abs() < 1e-6 * (w_c / wr));
            assert!((d_mr_d_wr - (m_c - mr) / wr).abs() < 1e-6 * ((m_c - mr) / wr).abs().max(1e-6));
            assert!(d_wr_d_mr.abs() < 1e-9);
            assert!((d_wr_d_wr - 1.0).abs() < 1e-9);
        }
    }

    /// The slack is never negative, so the bound never sits below the gain it
    /// was scored at, and it grows with the distance travelled.
    #[test]
    fn test_slack_is_non_negative_and_grows_with_distance() {
        let p = 16;
        let (f, b, _, _) = differentiable(p, 0x5EED_0003);
        let (mut d_m, mut d_w, mut w_c) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
        for g in 0..p {
            let (a, b_) = centre_partials(f[g], b);
            d_m[g] = a;
            d_w[g] = b_;
            w_c[g] = f[g].w_c;
        }
        let (slack_m, slack_w) = unit_slack(&d_m, &d_w, &w_c, 9.0);
        assert!(slack_m >= 0.0 && slack_w >= 0.0);
        let b = Bound {
            gain: 1.0,
            slack_m,
            slack_w,
            born_m: 0.0,
            born_w: 0.0,
        };
        assert_eq!(b.at(0.0, 0.0), 1.0);
        assert!(b.at(1.0, 1.0) > 1.0);
        assert!(b.at(4.0, 4.0) > b.at(1.0, 1.0));
    }

    /// Run a star with a given provider.
    ///
    /// ### Params
    ///
    /// * `m` - Means, row-major
    /// * `w` - Precisions, row-major
    /// * `branch` - Branches to the centre
    /// * `p` - Features
    /// * `params` - Star knobs
    /// * `provider` - Candidate provider
    ///
    /// ### Returns
    ///
    /// The result.
    fn run<C: CandidatePairs<f64>>(
        m: &[f64],
        w: &[f64],
        branch: &[f64],
        p: usize,
        params: StarParams,
        provider: &mut C,
    ) -> StarResult<f64> {
        resolve_star_with(
            Star {
                means: m,
                precisions: w,
                branch,
                n_features: p,
            },
            Some(params),
            provider,
        )
        .expect("star")
    }

    /// Two results describe the same tree with the same branch lengths and
    /// gains.
    ///
    /// ### Params
    ///
    /// * `a` - First result
    /// * `b` - Second result
    /// * `what` - Label for the failure message
    fn assert_same(a: &StarResult<f64>, b: &StarResult<f64>, what: &str) {
        assert_eq!(a.parent, b.parent, "{what}: parent arrays differ");
        assert_eq!(
            a.centre_children, b.centre_children,
            "{what}: centre children differ"
        );
        assert_eq!(
            a.merges.len(),
            b.merges.len(),
            "{what}: merge counts differ"
        );
        for (i, (x, y)) in a.merges.iter().zip(b.merges.iter()).enumerate() {
            assert_eq!((x.left, x.right), (y.left, y.right), "{what}: merge {i}");
            assert_eq!(x.gain, y.gain, "{what}: merge {i} gain");
            assert_eq!(x.t_left, y.t_left, "{what}: merge {i} left branch");
            assert_eq!(x.t_right, y.t_right, "{what}: merge {i} right branch");
            assert_eq!(x.t_centre, y.t_centre, "{what}: merge {i} centre branch");
        }
        assert_eq!(a.branch, b.branch, "{what}: branch lengths differ");
    }

    /// The gate: bounds must reproduce the exhaustive scan exactly.
    ///
    /// This is not the approximation that SPEC.md section 11 is. A pair the
    /// bounds skip is a pair that provably could not win, so any divergence is
    /// a defect.
    #[test]
    fn test_bounded_search_matches_the_exhaustive_scan() {
        for &(n, p) in &[(8usize, 12usize), (17, 40), (32, 24), (48, 64)] {
            for seed in [0x11u64, 0x22, 0x33] {
                for nsteps in [1.0f64, 4.0, 16.0, 64.0] {
                    let (m, w, branch) = star_fixture(n, p, seed);
                    let params = StarParams::default();
                    let base = run(&m, &w, &branch, p, params, &mut AllPairs);
                    let mut bounded = EllipsoidBounds::new(
                        AllPairs,
                        Some(EllipsoidBoundsParams {
                            nsteps,
                            adapt: false,
                            verify: false,
                        }),
                    );
                    let got = run(&m, &w, &branch, p, params, &mut bounded);
                    assert_same(
                        &base,
                        &got,
                        &format!("n {n}, p {p}, seed {seed:x}, nsteps {nsteps}"),
                    );
                    assert!(
                        bounded.stats().refreshes >= 1,
                        "n {n} nsteps {nsteps}: never anchored"
                    );
                }
            }
        }
    }

    /// The online schedule must not change the answer either.
    #[test]
    fn test_adaptive_sizing_matches_the_exhaustive_scan() {
        for seed in [0xA1u64, 0xB2, 0xC3] {
            let (n, p) = (40, 32);
            let (m, w, branch) = star_fixture(n, p, seed);
            let params = StarParams::default();
            let base = run(&m, &w, &branch, p, params, &mut AllPairs);
            let mut bounded = EllipsoidBounds::new(AllPairs, None);
            let got = run(&m, &w, &branch, p, params, &mut bounded);
            assert_same(&base, &got, &format!("adaptive, seed {seed:x}"));
        }
    }

    /// Composed with the candidate restriction, which is how production runs.
    #[test]
    fn test_composes_with_the_neighbour_restriction() {
        for seed in [0xD1u64, 0xE2] {
            let (n, p) = (36, 32);
            let (m, w, branch) = star_fixture(n, p, seed);
            let params = StarParams::default();
            let knn = KnnCandidatesParams {
                k: 8,
                rebuild_every: 8,
            };
            let base = run(
                &m,
                &w,
                &branch,
                p,
                params,
                &mut KnnCandidates::new(Some(knn)),
            );
            let mut bounded =
                EllipsoidBounds::new(KnnCandidates::new(Some(knn)), Some(Default::default()));
            let got = run(&m, &w, &branch, p, params, &mut bounded);
            assert_same(&base, &got, &format!("knn composition, seed {seed:x}"));
            assert!(bounded.stats().work_fraction() < 1.0);
        }
    }

    /// The property the whole scheme rests on, quantified rather than assumed.
    ///
    /// **Violations are real and this test does not pretend otherwise.** The
    /// rate and the diagnosis are in [`EllipsoidBoundsParams::default`]. What
    /// is asserted here is the two things that must hold: they are rare, and
    /// they never change which pair the walk picks. A regression that made the
    /// bound wrong rather than merely soft would break the second long before
    /// it broke the first.
    #[test]
    fn test_no_true_gain_exceeds_its_bound() {
        let mut checked = 0usize;
        let mut violations = 0usize;
        let mut misses = 0usize;
        let mut worst = 0.0f64;
        for &(n, p) in &[(24usize, 24usize), (40, 48)] {
            for seed in [0x71u64, 0x82, 0x93] {
                for nsteps in [1.0f64, 8.0, 64.0, 512.0] {
                    let (m, w, branch) = star_fixture(n, p, seed);
                    let mut bounded = EllipsoidBounds::new(
                        AllPairs,
                        Some(EllipsoidBoundsParams {
                            nsteps,
                            adapt: false,
                            verify: true,
                        }),
                    );
                    run(&m, &w, &branch, p, StarParams::default(), &mut bounded);
                    let s = bounded.stats();
                    checked += s.checked;
                    violations += s.violations;
                    misses += s.misses;
                    worst = worst.max(s.worst_violation);
                }
            }
        }
        assert!(checked > 10_000, "only {checked} bound checks made");
        assert_eq!(
            misses, 0,
            "the walk would have stopped on the wrong pair in {misses} rounds"
        );
        assert!(
            violations * 100 < checked,
            "{violations} of {checked} true gains exceeded their bound, worst by {worst:e}: \
             that is over one per cent and far past the one in a thousand measured"
        );
    }

    /// Redrawing every round makes the bounds exact, so nothing can exceed one.
    ///
    /// The control for the test above: it separates "the linearisation is
    /// stretched too far" from "the derivative is wrong".
    #[test]
    fn test_bounds_are_exact_when_they_are_never_stretched() {
        for seed in [0x71u64, 0x82, 0x93] {
            let (n, p) = (32, 40);
            let (m, w, branch) = star_fixture(n, p, seed);
            let mut bounded = EllipsoidBounds::new(
                AllPairs,
                Some(EllipsoidBoundsParams {
                    nsteps: 0.0,
                    adapt: false,
                    verify: true,
                }),
            );
            run(&m, &w, &branch, p, StarParams::default(), &mut bounded);
            let s = bounded.stats();
            assert_eq!(s.refreshes, s.rounds);
            assert_eq!(
                s.violations, 0,
                "{} of {} exceeded a bound that was redrawn every round",
                s.violations, s.checked
            );
        }
    }

    /// The refresh path has to be exercised, not accidentally never taken.
    ///
    /// A tiny ellipsoid leaves it every round; a huge one never does. Both must
    /// build the same tree, and the small one must actually refresh.
    #[test]
    fn test_the_refresh_path_is_taken_and_is_correct() {
        let (n, p) = (32, 32);
        let (m, w, branch) = star_fixture(n, p, 0x4444);
        let params = StarParams::default();
        let base = run(&m, &w, &branch, p, params, &mut AllPairs);

        let mut tight = EllipsoidBounds::new(
            AllPairs,
            Some(EllipsoidBoundsParams {
                nsteps: 1e-6,
                adapt: false,
                verify: false,
            }),
        );
        let got = run(&m, &w, &branch, p, params, &mut tight);
        assert_same(&base, &got, "tight ellipsoid");
        let s = tight.stats();
        assert_eq!(
            s.refreshes, s.rounds,
            "a tiny ellipsoid must refresh always"
        );

        let mut loose = EllipsoidBounds::new(
            AllPairs,
            Some(EllipsoidBoundsParams {
                nsteps: 1e12,
                adapt: false,
                verify: false,
            }),
        );
        let got = run(&m, &w, &branch, p, params, &mut loose);
        assert_same(&base, &got, "loose ellipsoid");
        assert_eq!(
            loose.stats().refreshes,
            1,
            "a huge ellipsoid must anchor once"
        );
    }

    /// The saving, which is the entire justification for the module.
    #[test]
    fn test_the_bounds_cut_the_pairs_scored() {
        // Loose against the numbers in `EllipsoidBoundsParams::default`, which
        // are 0.207 at 64 members and 0.108 at 128 with more features. This is
        // a regression guard on the saving existing at all, not a restatement
        // of the measurement.
        for &(n, want) in &[(48usize, 0.45f64), (96, 0.30), (192, 0.15)] {
            let p = 64;
            let (m, w, branch) = star_fixture(n, p, 0x9001);
            let mut bounded = EllipsoidBounds::new(AllPairs, None);
            run(&m, &w, &branch, p, StarParams::default(), &mut bounded);
            let s = bounded.stats();
            assert!(
                s.work_fraction() < want,
                "{n} members: scored {} of {} offered, fraction {:.3}, wanted under {want}",
                s.bounded + s.walked,
                s.offered,
                s.work_fraction()
            );
        }
    }

    /// The winner cannot depend on how rayon split the work.
    #[test]
    fn test_deterministic_under_thread_counts() {
        let (n, p) = (40, 32);
        let (m, w, branch) = star_fixture(n, p, 0x1357);
        let mut reference: Option<StarResult<f64>> = None;
        for threads in [1usize, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let got = pool.install(|| {
                let mut bounded = EllipsoidBounds::new(AllPairs, None);
                run(&m, &w, &branch, p, StarParams::default(), &mut bounded)
            });
            match &reference {
                None => reference = Some(got),
                Some(base) => assert_same(base, &got, &format!("{threads} threads")),
            }
        }
    }

    /// One instance driven over two different stars in a row.
    ///
    /// A metric fixed on the first star describes nothing about the second, and
    /// node ids restart, so the provider has to notice. It does not go wrong
    /// quietly if it misses: the second star's first round would be bounded
    /// against a distance travelled by a different centre entirely.
    #[test]
    fn test_one_instance_over_two_stars() {
        let p = 32;
        let mut shared = EllipsoidBounds::new(AllPairs, None);
        for seed in [0x1111u64, 0x2222] {
            for n in [20usize, 28] {
                let (m, w, branch) = star_fixture(n, p, seed);
                let params = StarParams::default();
                let base = run(&m, &w, &branch, p, params, &mut AllPairs);
                let got = run(&m, &w, &branch, p, params, &mut shared);
                assert_same(
                    &base,
                    &got,
                    &format!("reused instance, {n} by seed {seed:x}"),
                );
            }
        }
    }

    /// A star that resolves in one merge, and one that cannot merge at all.
    #[test]
    fn test_degenerate_stars() {
        let p = 8;
        let (m, w, branch) = star_fixture(4, p, 0x2468);
        let params = StarParams::default();
        let base = run(&m, &w, &branch, p, params, &mut AllPairs);
        let mut bounded = EllipsoidBounds::new(AllPairs, None);
        let got = run(&m, &w, &branch, p, params, &mut bounded);
        assert_same(&base, &got, "four members");

        // Three members is already resolved, so no round ever runs.
        let (m, w, branch) = star_fixture(3, p, 0x2468);
        let mut bounded = EllipsoidBounds::new(AllPairs, None);
        let got = run(&m, &w, &branch, p, params, &mut bounded);
        assert!(got.merges.is_empty());
        assert_eq!(bounded.stats().rounds, 0);
    }

    /// Every pair scoring identically is where a non-strict stopping rule would
    /// pick a different winner from the exhaustive scan.
    #[test]
    fn test_identical_members_agree_with_the_exhaustive_scan() {
        let (n, p) = (12, 16);
        let row: Vec<f64> = (0..p).map(|g| (g as f64 * 0.31).sin()).collect();
        let prec: Vec<f64> = (0..p)
            .map(|g| 1.0 + 0.4 * (g as f64 * 0.17).cos())
            .collect();
        let m: Vec<f64> = (0..n).flat_map(|_| row.clone()).collect();
        let w: Vec<f64> = (0..n).flat_map(|_| prec.clone()).collect();
        let branch = vec![0.1f64; n];

        let params = StarParams {
            min_gain: f64::NEG_INFINITY,
            ..StarParams::default()
        };
        let base = run(&m, &w, &branch, p, params, &mut AllPairs);
        let mut bounded = EllipsoidBounds::new(AllPairs, None);
        let got = run(&m, &w, &branch, p, params, &mut bounded);
        assert_same(&base, &got, "identical members");
    }

    /// `f32` storage goes through the same path.
    #[test]
    fn test_narrow_storage_matches_the_exhaustive_scan() {
        let (n, p) = (24, 24);
        let (m, w, branch) = star_fixture(n, p, 0x0F0F);
        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let star = Star {
            means: &m32,
            precisions: &w32,
            branch: &branch,
            n_features: p,
        };
        let params = Some(StarParams::default());
        let base = resolve_star_with(star, params, &mut AllPairs).expect("base");
        let mut bounded = EllipsoidBounds::new(AllPairs, None);
        let got = resolve_star_with(star, params, &mut bounded).expect("bounded");
        assert_eq!(base.parent, got.parent);
        assert_eq!(base.centre_children, got.centre_children);
    }

    /// The incremental centre update must track the exact recompute closely
    /// enough that neither the tree nor the gains move.
    #[test]
    fn test_incremental_centre_matches_the_exact_recompute() {
        for seed in [0x5150u64, 0x6161] {
            let (n, p) = (48, 40);
            let (m, w, branch) = star_fixture(n, p, seed);
            let exact = run(&m, &w, &branch, p, StarParams::default(), &mut AllPairs);
            let params = StarParams {
                incremental_centre: true,
                ..StarParams::default()
            };
            let got = run(&m, &w, &branch, p, params, &mut AllPairs);
            assert_eq!(exact.parent, got.parent, "seed {seed:x}: topology moved");
            for (i, (a, b)) in exact.merges.iter().zip(got.merges.iter()).enumerate() {
                let rel = (a.gain - b.gain).abs() / a.gain.abs().max(1.0);
                assert!(
                    rel < 1e-9,
                    "seed {seed:x}: merge {i} gain drifted by {rel:e}"
                );
            }
        }
    }

    /// The bound and the true gain coincide when the centre has not moved.
    #[test]
    fn test_the_bound_equals_the_gain_before_the_centre_moves() {
        let p = 20;
        let (m, w, branch) = star_fixture(10, p, 0x7777);
        let (mc, wc) = centre(&m, &w, &branch, p);
        let members: Vec<u32> = (0..10u32).collect();
        let round = Round {
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
        };
        let mut scratch = BoundScratch::new(p);
        let b = bound_pair(&round, 0, 1, &wc, 10.0, &mut scratch).expect("bound");
        let gain = b.gain;
        assert_eq!(
            b.at(0.0, 0.0),
            gain,
            "a pair is bounded by its own gain in the round it was scored in"
        );

        // And the gain is the one `score_merge` gives for the same peel.
        let mut merge_scratch = MergeScratch::new(p);
        let (mut m_r, mut w_r) = (vec![0.0; p], vec![0.0; p]);
        for g in 0..p {
            let wk = w[g] / (1.0 + branch[0] * w[g]);
            let wl = w[p + g] / (1.0 + branch[1] * w[p + g]);
            w_r[g] = wc[g] - wk - wl;
            m_r[g] = (mc[g] * wc[g] - wk * m[g] - wl * m[p + g]) / w_r[g];
        }
        let direct = score_merge(
            EffLeaf {
                m: &m[..p],
                w: &w[..p],
            },
            EffLeaf {
                m: &m[p..2 * p],
                w: &w[p..2 * p],
            },
            EffLeaf { m: &m_r, w: &w_r },
            branch[0],
            branch[1],
            None,
            &mut merge_scratch,
        )
        .expect("score");
        assert_eq!(direct.gain, gain);
    }
}
