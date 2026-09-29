//! Upper bounds on merge scores (SPEC.md section 10).
//!
//! A pair's gain depends on the centre only through the peeled remainder
//! `(MR, WR)`, hence through the centre's effective leaf `(M_r, W_r)`. The gain
//! is linearised in the centre's movement and maximised over the ellipsoid of
//! SPEC.md section 10.3, giving per pair its gain and a slack per unit of centre
//! movement. This prunes most rescoring on top of the candidate restriction of
//! [`crate::search::candidates`].
//!
//! **Deviation: the bound tracks the centre.** Section 10.4 adds a fixed
//! ellipsoid's whole slack to every gain. Here the distance the centre actually
//! travelled is accumulated per round and a pair's bound is its gain plus that
//! distance times its unit slack. `nsteps` then only decides when the
//! linearisation is redrawn; see [`DEFAULT_NSTEPS`].
//!
//! Not a strict bound: the linearisation can underestimate.
//!
//! The SI's seven-term derivative is not transcribed; it is assembled by the
//! chain rule through the peel (partials from SPEC.md section 8.3) and pinned
//! against central differences in the tests.

use crate::errors::BonsaiErrors;
use crate::model::merge::EffLeaf;
use crate::search::star::{BOUND_WALK_CHUNK, CandidatePairs, PairScratch, Round, StarMerge};
use crate::utils::traits::{BonsaiFloat, wide};
use rayon::prelude::*;
use std::collections::HashMap;

///////////////
// Constants //
///////////////

/// Where the online schedule starts `nsteps`: how far the centre may travel,
/// in the metric of SPEC.md section 10.3, before every bound is redrawn.
///
/// One merge moves the centre about two to three of these units. It controls
/// how far the linearisation is stretched before it is redone, not how loose a
/// bound is.
///
/// Ours, measured on clustered stars of 128 members by 200 features, four
/// seeds: the fraction of the exhaustive `349,500` pairs scored is 0.275 at 6,
/// 0.102 at 48 and 0.170 at 192, so 48 is the floor of the bowl.
///
/// The linearisation is not a strict bound (SPEC.md section 10.2): about one
/// check in a thousand is violated, by up to two nats, and never by a pair the
/// walk would have picked.
/// `test_bounded_search_matches_the_exhaustive_scan` pins that structurally.
const DEFAULT_NSTEPS: f64 = 48.0;

/// Smallest value the online sizing will shrink `nsteps` to.
///
/// At `nsteps = 1` every round redraws and the star scores more pairs than the
/// exhaustive scan.
const NSTEPS_MIN: f64 = 1.0;

/// Largest value the online sizing will grow `nsteps` to; a runaway guard.
const NSTEPS_MAX: f64 = 512.0;

/// Weight of the newest round in the two running cost averages (about ten
/// rounds).
const COST_DECAY: f64 = 0.1;

/// Multiplier applied to `nsteps` on a round where the walk is the larger cost.
const SHRINK: f64 = 0.95;

/// Multiplier applied to `nsteps` on a round where redrawing is.
const GROW: f64 = 1.05;

/// Chunks a round must offer before its cost is taken as a signal.
///
/// A round offering under one chunk always scores everything (the walk cannot
/// stop inside a chunk), which reads as maximally deep.
const ADAPT_MIN_CHUNKS: usize = 4;

/////////////////
// The kernels //
/////////////////

/// One feature of one candidate pair, as the derivative sees it.
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
/// the pair's own effective leaves and all five branch lengths fixed. The
/// branch lengths are at the optimum of SPEC.md section 8.4, so their movement
/// is second order, except where `t_ar` or the split sit at a bracket end; that
/// and the curvature of the peel are what can make a bound too small (see
/// [`DEFAULT_NSTEPS`]).
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
/// and the pair's own quantities held fixed,
///
/// ```text
/// d(MR)/d(M_r) = W_r / WR        d(MR)/d(W_r) = (M_r - MR) / WR
/// d(WR)/d(M_r) = 0               d(WR)/d(W_r) = 1
/// ```
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

/// Per-feature scales defining the metric the centre's movement is measured in.
///
/// SPEC.md section 10.3, S41, with `nsteps` factored out. One merge moves the
/// centre's position by about `1/sqrt(nc * W[g,r])` and its precision by about
/// `W[g,r]/nc`; dividing by these gives the unit balls of S42.
///
/// **Deviation: the feature count belongs in the scales.** S41 gives
/// per-feature sizes for a norm over all `p` features, so a one-merge movement
/// in every feature would sit at radius `sqrt(p)`. Both scales are multiplied by
/// `sqrt(p)` so the unit is one merge whatever `p`, which is what lets
/// [`DEFAULT_NSTEPS`] carry across datasets.
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
/// The maximum of a dot product over a unit ball is the vector's norm (SPEC.md
/// section 10.3, S42 and S45). The two ellipsoids are independent, so the
/// slacks are returned separately.
///
/// **Deviation.** S45 is a sum of absolute values (the circumscribing box);
/// the Euclidean norm is taken here. The box form inflates bounds by up to
/// `sqrt(p)` and prunes nothing at thousands of features.
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
/// Steps are accumulated round by round: by the triangle inequality the sum
/// bounds the distance from wherever each pair was scored.
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
/// The bound at a later round is
/// `gain + (path_m - born_m) * slack_m + (path_w - born_w) * slack_w`.
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
/// Wraps a provider that decides which pairs exist; this one orders them and
/// lets the walk stop early. Composes with
/// [`crate::search::candidates::KnnCandidates`].
///
/// ### How a round goes
///
/// 1. Ask the inner provider for the live pairs.
/// 2. Add the centre's step since last round to the running distance.
/// 3. If that distance has passed `nsteps`, clear the table and fix a new
///    metric.
/// 4. Score every pair without an entry (all pairs after a redraw, else the
///    new ancestor's) and give it one.
/// 5. Emit pairs by bound at the current distance, descending.
///
/// A pair scored this round is emitted at exactly its own gain. A pair with no
/// entry is emitted with an infinite bound so it is always scored.
#[derive(Clone, Debug)]
pub struct EllipsoidBounds<C> {
    /// Which pairs exist.
    inner: C,
    /// How far the centre may travel before the linearisation is redrawn,
    /// which the online schedule moves.
    nsteps: f64,
    /// Whether the online schedule moves `nsteps`; off only in tests that pin
    /// a size.
    adapt: bool,
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
}

impl<C> EllipsoidBounds<C> {
    /// A provider over a fresh ellipsoid.
    ///
    /// ### Params
    ///
    /// * `inner` - Provider deciding which pairs exist
    ///
    /// ### Returns
    ///
    /// The provider. The first round of a star anchors it.
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            nsteps: DEFAULT_NSTEPS,
            adapt: true,
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
        }
    }

    /// A provider pinned at one ellipsoid size, the online schedule off, so
    /// tests can sweep sizes.
    ///
    /// ### Params
    ///
    /// * `inner` - Provider deciding which pairs exist
    /// * `nsteps` - The size
    ///
    /// ### Returns
    ///
    /// The provider.
    #[cfg(test)]
    fn fixed(inner: C, nsteps: f64) -> Self {
        Self {
            nsteps,
            adapt: false,
            ..Self::new(inner)
        }
    }

    /// Move `nsteps` towards the size that costs least.
    ///
    /// SPEC.md section 10.5 reads the walk depth, but depth alone pins `nsteps`
    /// at the cap (the walk cannot stop inside a chunk). Instead both costs are
    /// tracked as running averages and `nsteps` descends on their difference.
    /// Rounds offering under `ADAPT_MIN_CHUNKS` chunks are ignored.
    ///
    /// ### Params
    ///
    /// * `bounded` - Pairs this round scored to build bounds
    /// * `walked` - Pairs the previous round's walk scored
    /// * `offered` - Pairs this round offered
    fn adapt(&mut self, bounded: usize, walked: usize, offered: usize) {
        if !self.adapt || offered < ADAPT_MIN_CHUNKS * BOUND_WALK_CHUNK {
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

        // A fresh star: nothing from a previous star may be reused (node ids restart).
        // The feature count is checked so a differently shaped star cannot index a stale metric.
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

        // `nsteps` only decides when the linearisation is redrawn; the bounds
        // themselves are exact in the distance travelled.
        let stale = self.path_m.max(self.path_w) >= self.nsteps;
        if fresh_star || stale {
            self.table.clear();
            self.metric_w.clear();
            self.metric_w.extend_from_slice(round.centre_precisions);
            self.metric_nc = nc;
            self.path_m = 0.0;
            self.path_w = 0.0;
            self.anchored = true;
        }
        self.prev_m.clear();
        self.prev_m.extend_from_slice(round.centre_means);
        self.prev_w.clear();
        self.prev_w.extend_from_slice(round.centre_precisions);

        // Pairs without an entry: all of them after a redraw, else the new ancestor's.
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
            // A pair without an entry is emitted at the top, never pruned.
            let bound = self.table.get(&key).copied();
            if let Some(bound) = bound {
                self.next_table.insert(key, bound);
            }
            let emit = bound.map_or(f64::INFINITY, |b| b.at(self.path_m, self.path_w));
            self.order.push((emit, key.0, key.1, i, j));
        }
        std::mem::swap(&mut self.table, &mut self.next_table);

        // Descending on the bound, ascending on node ids: order depends on the star alone.
        self.order
            .sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        for &(bound, _, _, i, j) in &self.order {
            out.push((i, j));
            self.emitted.push(bound);
        }
        Ok(())
    }

    /// Forward the merge to the inner provider. The bound table is rebuilt from
    /// the offered pairs each round, so dead entries fall out on their own.
    ///
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
    use crate::utils::verbosity::Verbosity;

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
            Verbosity::Quiet,
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
                    let mut bounded = EllipsoidBounds::fixed(AllPairs, nsteps);
                    let got = run(&m, &w, &branch, p, params, &mut bounded);
                    assert_same(
                        &base,
                        &got,
                        &format!("n {n}, p {p}, seed {seed:x}, nsteps {nsteps}"),
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
            let mut bounded = EllipsoidBounds::new(AllPairs);
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
            let mut bounded = EllipsoidBounds::new(KnnCandidates::new(Some(knn)));
            let got = run(&m, &w, &branch, p, params, &mut bounded);
            assert_same(&base, &got, &format!("knn composition, seed {seed:x}"));
        }
    }

    /// The refresh path has to be exercised, not accidentally never taken.
    ///
    /// A tiny ellipsoid leaves it every round; a huge one never does. Both must
    /// build the same tree.
    #[test]
    fn test_the_refresh_path_is_taken_and_is_correct() {
        let (n, p) = (32, 32);
        let (m, w, branch) = star_fixture(n, p, 0x4444);
        let params = StarParams::default();
        let base = run(&m, &w, &branch, p, params, &mut AllPairs);

        let mut tight = EllipsoidBounds::fixed(AllPairs, 1e-6);
        let got = run(&m, &w, &branch, p, params, &mut tight);
        assert_same(&base, &got, "tight ellipsoid");

        let mut loose = EllipsoidBounds::fixed(AllPairs, 1e12);
        let got = run(&m, &w, &branch, p, params, &mut loose);
        assert_same(&base, &got, "loose ellipsoid");
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
                let mut bounded = EllipsoidBounds::new(AllPairs);
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
        let mut shared = EllipsoidBounds::new(AllPairs);
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
        let mut bounded = EllipsoidBounds::new(AllPairs);
        let got = run(&m, &w, &branch, p, params, &mut bounded);
        assert_same(&base, &got, "four members");

        // Three members is already resolved, so no round ever runs.
        let (m, w, branch) = star_fixture(3, p, 0x2468);
        let mut bounded = EllipsoidBounds::new(AllPairs);
        let got = run(&m, &w, &branch, p, params, &mut bounded);
        assert!(got.merges.is_empty());
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
        let mut bounded = EllipsoidBounds::new(AllPairs);
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
        let base = resolve_star_with(star, params, &mut AllPairs, Verbosity::Quiet).expect("base");
        let mut bounded = EllipsoidBounds::new(AllPairs);
        let got = resolve_star_with(star, params, &mut bounded, Verbosity::Quiet).expect("bounded");
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
