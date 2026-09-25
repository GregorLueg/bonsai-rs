//! Ingest: turning a measured dataset into what the pruning recursion consumes.
//!
//! Implements SPEC.md sections 3.1, 3.3 and 3.4. Three jobs, in the order a
//! caller meets them:
//!
//! 1. [`from_sanity`] converts Sanity posteriors into likelihood means and
//!    error bars (S5), if that is where the data came from.
//! 2. [`prepare`] estimates the per-feature variance `v[g]` (S8, S9), scores
//!    every feature by signal-to-noise (S6, S10, S11), drops the features that
//!    fail the threshold, and divides the survivors by `sqrt(v[g])` (S21).
//! 3. The result is handed straight to
//!    [`crate::model::likelihood::NodeState::new`].
//!
//! ### Units, which are the thing to get right here
//!
//! Two unit systems meet in this module and confusing them is the most likely
//! silent defect in the crate.
//!
//! * **Raw units** are whatever the caller measured in. `means`, `sds`, the
//!   Sanity posteriors, and every `v[g]` anywhere in this file are raw.
//! * **Transformed units** are raw divided by `sqrt(v[g])`, so the diffusion
//!   prior of SPEC.md section 2 has unit variance per feature and `v[g]`
//!   vanishes from every kernel. Everything downstream of ingest is
//!   transformed.
//!
//! Every field of [`PreparedData`] says which of the two it is in, and the two
//! `transformed_` fields are the only ones that are transformed.
//! [`PreparedData::restore_scale`] is the way back.

use crate::errors::BonsaiErrors;
use crate::utils::kernels::edge_newton;
use crate::utils::traits::{BonsaiFloat, narrow, wide};
use rayon::prelude::*;

///////////////
// Constants //
///////////////

/// Features per chunk of the per-feature pass.
///
/// The input is row-major `[cell][feature]` and the estimator works down a
/// feature column, so columns are gathered a chunk at a time rather than one at
/// a time: 128 consecutive `f32` features span eight cache lines, so a chunk
/// reads each line once and uses all of it, where one feature at a time touches
/// one line per cell per feature.
const INGEST_BLOCK: usize = 128;

/// Iteration budget for the per-feature variance solve.
///
/// Same role as `MAX_NEWTON_ITER` in [`crate::model::branch`]: the bracket
/// halves on every safeguarded step, so bisection alone exhausts any plausible
/// bracket to `f64` precision in under 60 steps. A runaway guard, not a working
/// limit.
const MAX_VARIANCE_ITER: usize = 100;

/// Relative convergence tolerance on `v[g]`.
///
/// Set well below the precision the signal-to-noise ratio needs, so the
/// stopping point of the solve is never what decides whether a feature is
/// retained.
const VARIANCE_TOL: f64 = 1e-12;

/// Default signal-to-noise threshold for retaining a feature.
///
/// One, the paper's threshold (SPEC.md section 3.3), adopted 2026-09-25 so that
/// a default run sees the gene panel the published method would. `S[g]` is the
/// mean ratio of posterior signal variance to measurement error variance, so
/// `S[g] = 1` is the point at which a feature carries as much signal as noise.
/// It was `0.25` before, on the separation measured below; on Sanity-processed
/// Baron data at 5,000 cells that kept 4,591 genes against 2,701 at `1`, and
/// search time is linear in the gene count.
///
/// ### What was measured
///
/// `tree::simulate::simulate_binary` at 400 features, of which half were then
/// replaced by pure noise: the same error bars, but means drawn from those
/// error bars alone with no true position behind them. The two groups' `S`
/// distributions are what the threshold has to separate. `noise_sd` is in
/// transformed units, so `noise_sd = 1` is a feature whose measurement error
/// equals the spread of the data and whose informative features should score
/// `S = 1` by construction.
///
/// ```text
///          informative features          pure-noise features
/// cells    noise_sd  min S   5th pct     v[g] = 0   max S   95th pct
///    64        0.1    91.4    105.2         55%      0.48      0.29
///    64        0.5     3.16     4.03
///    64        1.0     0.50     0.78
///   128        1.0     0.68     0.89         53%      0.31      0.23
///   512        1.0     0.99     1.14         51%      0.15      0.10
/// ```
///
/// The pure-noise columns do not vary with `noise_sd`, because `S` for a
/// feature that is only noise is scale free; they shrink with the cell count
/// like the sampling error of a variance, so about half of those features solve
/// to `v[g] = 0` and drop out on their own and the rest have `S` bounded by
/// roughly `2.7 * sqrt(2/n)`.
///
/// `0.25` sits above the 95th percentile of the pure-noise scores at every cell
/// count tested and below the 5th percentile of the informative scores at every
/// cell count and noise level tested. `1` does not: in the hardest row it
/// discards most of the informative panel, since a feature carrying exactly as
/// much signal as noise scores `1` only in expectation and scatters below it at
/// finite `n`. So on small or noisy datasets a caller should expect `1` to drop
/// informative features, and can pass `0.25` through
/// [`IngestParams::min_signal_to_noise`] to keep them.
///
/// ### On tree recovery
///
/// Recovery of `search::star::star_tree` against `noise_sd`, five seeds per
/// cell, mean Robinson-Foulds:
///
/// | `noise_sd` | 64 leaves, of 122 | 128 leaves, of 250 |
/// |---|---|---|
/// | 0.10 | 0.0 | 0.4 |
/// | 0.25 | 0.4 | 1.2 |
/// | 0.50 | 3.2 | 12.8 |
/// | 1.00 | 32.0 | 99.2 |
/// | 2.00 | 96.0 | 212.4 |
/// | 4.00 | 118.8 | 245.0 |
///
/// So the primitive recovers the tree essentially perfectly below `noise_sd`
/// 0.25 and collapses past 1. The cliff sits where measurement noise equals the
/// per-feature signal variance, which after the transform of section 3.1 is
/// exactly `noise_sd = 1`, i.e. unit signal-to-noise. That is the cliff this
/// filter exists to keep features away from.
///
/// The two are not directly comparable, `S` being a per-feature aggregate
/// against a noise level uniform across features, but the recovery cliff sits
/// nearer `1` than `0.25`, which is the other argument for the paper's value.
pub const DEFAULT_MIN_SIGNAL_TO_NOISE: f64 = 1.0;

/// Largest variance amplification `v / (v - eps^2)` [`from_sanity`] converts.
///
/// Ours. The conversion of SPEC.md section 3.4 multiplies
/// both the posterior mean and the posterior variance by this factor, so it
/// diverges as the posterior approaches the prior and is undefined once
/// `eps^2 >= v`. At the cap the returned error bar is about 32 times the
/// posterior one, which already makes the cell effectively uninformative; the
/// term `w * mu^2` that reaches the pruning recursion grows linearly in the
/// factor, so letting it run to `1e6` would spend six digits of the `f64`
/// accumulation on a cell that says nothing.
pub const MAX_SANITY_AMPLIFICATION: f64 = 1.0e3;

////////////////
// Parameters //
////////////////

/// Tuning knobs for [`prepare`] and [`from_sanity`].
#[derive(Clone, Copy, Debug)]
pub struct IngestParams {
    /// Smallest signal-to-noise ratio `S[g]` (S6) a feature may have and still
    /// be retained. Compared with `>=`. Set to `f64::NEG_INFINITY` to keep
    /// every feature that has a usable variance.
    pub min_signal_to_noise: f64,
    /// Largest `v / (v - eps^2)` [`from_sanity`] will convert. A feature in
    /// which any cell exceeds it is dropped and reported.
    pub max_sanity_amplification: f64,
}

impl Default for IngestParams {
    /// [`DEFAULT_MIN_SIGNAL_TO_NOISE`] and [`MAX_SANITY_AMPLIFICATION`].
    ///
    /// ### Returns
    ///
    /// The default parameter set.
    fn default() -> Self {
        Self {
            min_signal_to_noise: DEFAULT_MIN_SIGNAL_TO_NOISE,
            max_sanity_amplification: MAX_SANITY_AMPLIFICATION,
        }
    }
}

/////////////
// Outputs //
/////////////

/// A dataset ready for the pruning recursion.
///
/// Both matrices are row-major `[cell][retained feature]` with stride
/// `features.len()`, the layout
/// [`crate::model::likelihood::NodeState::new`] takes. The feature axis is the
/// *retained* one, so column `k` is original feature `features[k]`.
#[derive(Clone, Debug)]
pub struct PreparedData<T> {
    /// Means divided by `sqrt(v[g])`. **Transformed units.**
    pub transformed_means: Vec<T>,
    /// Precisions `v[g] / sig[g,i]^2`, that is, one over the square of the
    /// transformed standard deviation. **Transformed units.**
    pub transformed_precisions: Vec<T>,
    /// Original index of each retained feature, ascending. Maps a column of
    /// either matrix back onto the caller's feature axis.
    pub features: Vec<usize>,
    /// The per-feature variance that was divided out, for the retained
    /// features, aligned with `features`. **Raw units**, because that is what
    /// it is a variance of.
    pub variances: Vec<f64>,
    /// Signal-to-noise ratio `S[g]` (S6) of each retained feature, aligned with
    /// `features`. Dimensionless.
    pub signal_to_noise: Vec<f64>,
    /// Number of cells, that is, rows of both matrices.
    pub n_cells: usize,
    /// Number of features the caller supplied, before selection.
    pub n_features_in: usize,
}

impl<T: BonsaiFloat> PreparedData<T> {
    /// Number of retained features, that is, the stride of both matrices.
    ///
    /// ### Returns
    ///
    /// The retained feature count.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Convert positions in transformed units back to raw units.
    ///
    /// The inverse of the scale transform of SPEC.md section 3.1. Takes
    /// anything laid out row-major over the *retained* feature axis, which is
    /// what a row of [`crate::model::likelihood::NodeState::means`] is, and
    /// multiplies each column by `sqrt(v[g])`.
    ///
    /// ### Params
    ///
    /// * `transformed` - Positions in transformed units, row-major with stride
    ///   `self.n_features()`
    ///
    /// ### Returns
    ///
    /// The same positions in raw units, or `ShapeMismatch` if the input is not
    /// a whole number of rows over the retained feature axis.
    pub fn restore_scale(&self, transformed: &[T]) -> Result<Vec<T>, BonsaiErrors> {
        let k = self.n_features();
        if k == 0 || !transformed.len().is_multiple_of(k) {
            return Err(BonsaiErrors::ShapeMismatch {
                mean_cells: transformed.len(),
                mean_features: 1,
                sd_cells: transformed.len() / k.max(1),
                sd_features: k,
            });
        }
        let scale: Vec<f64> = self.variances.iter().map(|v| v.sqrt()).collect();
        Ok(transformed
            .iter()
            .enumerate()
            .map(|(idx, &x)| narrow(wide(x) * scale[idx % k]))
            .collect())
    }
}

/// Likelihood means and error bars recovered from Sanity posteriors.
///
/// Both matrices are row-major `[cell][retained feature]` in **raw units**, so
/// this is exactly the input [`prepare`] takes; pass `variances` straight back
/// in as its `variances` argument, since it has already been subset to the
/// retained features.
#[derive(Clone, Debug)]
pub struct SanityLikelihood<T> {
    /// Likelihood means `mu[g,i]` (S5). **Raw units.**
    pub means: Vec<T>,
    /// Likelihood standard deviations `sig[g,i]` (S5). **Raw units.**
    pub sds: Vec<T>,
    /// Original index of each retained feature, ascending.
    pub features: Vec<usize>,
    /// Sanity's `v[g]` for the retained features, aligned with `features`.
    /// **Raw units.**
    pub variances: Vec<f64>,
    /// Original indices of the features dropped as ill-conditioned, ascending.
    /// A caller who has lost a meaningful share of their genes to the
    /// `eps^2 -> v` limit needs to know, so this is returned rather than
    /// silently repaired.
    pub dropped: Vec<usize>,
    /// Number of cells.
    pub n_cells: usize,
}

////////////////
// Validation //
////////////////

/// Reject any input the estimator cannot work on.
///
/// Permissive input handling here surfaces as a `NaN` loglikelihood a long way
/// downstream, so everything is checked once, up front, over the whole matrix.
/// `!(sd > 0.0)` catches zero, negative and `NaN` in one comparison.
///
/// ### Params
///
/// * `means` - Means, row-major `[cell][feature]`
/// * `sds` - Standard deviations, same layout
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
///
/// ### Returns
///
/// `Ok(())`, or `EmptyInput`, `ShapeMismatch`, `NonPositiveSd` or
/// `NonFiniteMean`.
fn validate<T: BonsaiFloat>(
    means: &[T],
    sds: &[T],
    n_cells: usize,
    n_features: usize,
) -> Result<(), BonsaiErrors> {
    if n_cells == 0 || n_features == 0 {
        return Err(BonsaiErrors::EmptyInput {
            n_cells,
            n_features,
        });
    }
    if means.len() != n_cells * n_features || sds.len() != means.len() {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: means.len() / n_features,
            mean_features: n_features,
            sd_cells: sds.len() / n_features,
            sd_features: n_features,
        });
    }
    for cell in 0..n_cells {
        for feature in 0..n_features {
            let idx = cell * n_features + feature;
            let mu = wide(means[idx]);
            if !mu.is_finite() {
                return Err(BonsaiErrors::NonFiniteMean {
                    value: mu,
                    cell,
                    feature,
                });
            }
            let sd = wide(sds[idx]);
            if !sd.is_finite() || sd <= 0.0 {
                return Err(BonsaiErrors::NonPositiveSd {
                    value: sd,
                    cell,
                    feature,
                });
            }
        }
    }
    Ok(())
}

/// Reject a supplied per-feature variance vector.
///
/// ### Params
///
/// * `variances` - One `v[g]` per feature, raw units
/// * `n_features` - Number of features the matrices have
///
/// ### Returns
///
/// `Ok(())`, or `ShapeMismatch` on a length disagreement, or
/// `NonPositiveVariance`.
fn validate_variances(variances: &[f64], n_features: usize) -> Result<(), BonsaiErrors> {
    if variances.len() != n_features {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: 1,
            mean_features: n_features,
            sd_cells: 1,
            sd_features: variances.len(),
        });
    }
    for (feature, &v) in variances.iter().enumerate() {
        if !v.is_finite() || v <= 0.0 {
            return Err(BonsaiErrors::NonPositiveVariance { value: v, feature });
        }
    }
    Ok(())
}

///////////////////////////
// The per-feature solve //
///////////////////////////

/// Maximum-likelihood feature mean given `v[g]` (S8).
///
/// ```text
/// mubar[g] = (sum_i mu[g,i]/(v+sig[g,i]^2)) / (sum_i 1/(v+sig[g,i]^2))
/// ```
///
/// ### Params
///
/// * `mu` - One feature's means across cells, raw units
/// * `sig2` - The matching squared standard deviations, raw units
/// * `v` - Feature variance to condition on
///
/// ### Returns
///
/// The precision-weighted feature mean. The denominator is a sum of strictly
/// positive terms because `sig2` is validated positive, so it cannot vanish.
fn feature_mean(mu: &[f64], sig2: &[f64], v: f64) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..mu.len() {
        let r = 1.0 / (v + sig2[i]);
        num += mu[i] * r;
        den += r;
    }
    num / den
}

/// Squared deviations from the feature mean at a given `v[g]`.
///
/// ### Params
///
/// * `mu` - One feature's means across cells, raw units
/// * `sig2` - The matching squared standard deviations, raw units
/// * `v` - Feature variance to condition on
/// * `d2` - Destination, length `mu.len()`
fn fill_deviations(mu: &[f64], sig2: &[f64], v: f64, d2: &mut [f64]) {
    let mbar = feature_mean(mu, sig2, v);
    for i in 0..mu.len() {
        let d = mu[i] - mbar;
        d2[i] = d * d;
    }
}

/// Solve the per-feature variance stationarity condition (S9).
///
/// ```text
/// sum_i (1/(v+sig_i^2)) * ((mu_i-mubar(v))^2/(v+sig_i^2) - 1) = 0
/// ```
///
/// This is the branch-length condition of SPEC.md section 6 with
/// `s -> sig_i^2`, `d -> (mu_i-mubar)^2`, `t -> v` and an overall sign flip, so
/// it shares [`edge_newton`] and mirrors
/// [`crate::model::branch::optimise_edge`] step for step. It could not *call*
/// that function: `d` is not a constant here, because `mubar` depends on the
/// unknown, so `d` has to be rebuilt at every iterate.
///
/// Rebuilding it keeps the problem one-dimensional rather than two-dimensional
/// because S8 and S9 are the two partial derivatives of one objective,
/// `-1/2 sum_i [log(v+sig_i^2) + (mu_i-mubar)^2/(v+sig_i^2)]`. S8 zeroes the
/// `mubar` partial, so by the envelope theorem the total derivative of the
/// profiled objective in `v` is exactly the `v` partial with `mubar(v)`
/// substituted. The value `edge_newton` returns is therefore exact; only its
/// *derivative* misses the `dmubar/dv` term, which makes the Newton step a
/// quasi-Newton one. The safeguard that is already there covers that.
///
/// ### Params
///
/// * `mu` - One feature's means across cells, raw units
/// * `sig2` - The matching squared standard deviations, raw units
/// * `d2` - Scratch of length `mu.len()`
///
/// ### Returns
///
/// The maximum-likelihood `v[g]`, exactly zero when the spread of the means is
/// already explained by the error bars alone, that is, when the feature is pure
/// noise. `RootFindDiverged` if the budget is exhausted.
fn solve_variance(mu: &[f64], sig2: &[f64], d2: &mut [f64]) -> Result<f64, BonsaiErrors> {
    fill_deviations(mu, sig2, 0.0, d2);
    if edge_newton(sig2, d2, 0.0).0 >= 0.0 {
        return Ok(0.0);
    }

    // `mubar` lies inside the range of `mu` at every `v`, so `d2[i]` never
    // exceeds the squared range of `mu`. At `v` equal to that range, every
    // `v + sig_i^2` exceeds its `d2[i]` and every term of the condition takes
    // the same sign, so the root is bracketed with no doubling loop and no cap
    // to justify.
    let mut lo_mu = f64::INFINITY;
    let mut hi_mu = f64::NEG_INFINITY;
    for &x in mu {
        lo_mu = lo_mu.min(x);
        hi_mu = hi_mu.max(x);
    }
    let upper = (hi_mu - lo_mu) * (hi_mu - lo_mu);
    // Not finite means the means span a range whose square overflows, which no
    // real dataset does; treating it as no signal drops the feature rather
    // than handing an infinite bracket to the solve.
    if !upper.is_finite() || upper <= 0.0 {
        return Ok(0.0);
    }

    let (mut lo, mut hi) = (0.0f64, upper);
    let mut v = 0.5 * upper;
    let mut step = upper;

    for _ in 0..MAX_VARIANCE_ITER {
        fill_deviations(mu, sig2, v, d2);
        let (f, fp) = edge_newton(sig2, d2, v);
        if f < 0.0 {
            lo = v;
        } else {
            hi = v;
        }

        let newton = v - f / fp;
        let prev = v;
        if newton > lo && newton < hi && (newton - v).abs() < 0.5 * step {
            step = (newton - v).abs();
            v = newton;
        } else {
            step = 0.5 * (hi - lo);
            v = 0.5 * (lo + hi);
        }

        if (v - prev).abs() <= VARIANCE_TOL * v.max(VARIANCE_TOL) {
            return Ok(v);
        }
    }

    Err(BonsaiErrors::RootFindDiverged {
        max_iter: MAX_VARIANCE_ITER,
        last_step: step,
    })
}

/// Signal-to-noise ratio of one feature (S6, S10, S11).
///
/// Substituting the posterior signal `delta[g,i] = v/(v+sig_i^2)*(mu_i-mubar)`
/// and its error bar `eps[g,i]^2 = v*sig_i^2/(v+sig_i^2)` into
/// `S[g] = (1/n) sum_i delta^2/eps^2` collapses to
///
/// ```text
/// S[g] = (1/n) * sum_i v * (mu_i-mubar)^2 / ((v+sig_i^2) * sig_i^2)
/// ```
///
/// with no cancellation and no intermediate array. For a homoscedastic feature
/// this is `v / sig^2`, the ratio of signal variance to noise variance, which
/// is the scale the threshold is set on.
///
/// ### Params
///
/// * `mu` - One feature's means across cells, raw units
/// * `sig2` - The matching squared standard deviations, raw units
/// * `v` - The feature's variance
///
/// ### Returns
///
/// `S[g]`, which is zero for a feature with no estimated variance.
fn signal_to_noise(mu: &[f64], sig2: &[f64], v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let mbar = feature_mean(mu, sig2, v);
    let mut acc = 0.0f64;
    for i in 0..mu.len() {
        let d = mu[i] - mbar;
        acc += v * d * d / ((v + sig2[i]) * sig2[i]);
    }
    acc / mu.len() as f64
}

//////////////////
// The two APIs //
//////////////////

/// Score every feature and, where it was not supplied, estimate its variance.
///
/// The estimator works down feature columns of a row-major `[cell][feature]`
/// matrix, so columns are gathered a chunk of `INGEST_BLOCK` at a time. Within
/// a chunk the features are independent, so rayon fans out over them; the
/// collect is index-ordered and every reduction inside a feature runs in cell
/// order, so the result does not depend on the thread count.
///
/// ### Params
///
/// * `means` - Means, row-major `[cell][feature]`, raw units
/// * `sds` - Standard deviations, same layout, raw units
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `variances` - Supplied `v[g]`, or `None` to estimate them by S9
///
/// ### Returns
///
/// One `v[g]` and one `S[g]` per feature, in the caller's feature order.
fn score_features<T: BonsaiFloat>(
    means: &[T],
    sds: &[T],
    n_cells: usize,
    n_features: usize,
    variances: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), BonsaiErrors> {
    let mut v_out = vec![0.0f64; n_features];
    let mut s_out = vec![0.0f64; n_features];
    let mut mu_col = vec![0.0f64; INGEST_BLOCK * n_cells];
    let mut sig2_col = vec![0.0f64; INGEST_BLOCK * n_cells];

    for lo in (0..n_features).step_by(INGEST_BLOCK) {
        let len = INGEST_BLOCK.min(n_features - lo);
        for cell in 0..n_cells {
            let src = cell * n_features + lo;
            for j in 0..len {
                mu_col[j * n_cells + cell] = wide(means[src + j]);
                let sd = wide(sds[src + j]);
                sig2_col[j * n_cells + cell] = sd * sd;
            }
        }

        let scored: Result<Vec<(f64, f64)>, BonsaiErrors> = (0..len)
            .into_par_iter()
            .map(|j| {
                let mu = &mu_col[j * n_cells..(j + 1) * n_cells];
                let sig2 = &sig2_col[j * n_cells..(j + 1) * n_cells];
                let v = match variances {
                    Some(supplied) => supplied[lo + j],
                    None => {
                        let mut d2 = vec![0.0f64; n_cells];
                        solve_variance(mu, sig2, &mut d2)?
                    }
                };
                Ok((v, signal_to_noise(mu, sig2, v)))
            })
            .collect();

        for (j, (v, s)) in scored?.into_iter().enumerate() {
            v_out[lo + j] = v;
            s_out[lo + j] = s;
        }
    }
    Ok((v_out, s_out))
}

/// Turn a measured mean/SD dataset into transformed leaf data.
///
/// SPEC.md sections 3.3 then 3.1, in that order: score, select, transform. The
/// variance the transform divides by is the one the selection estimated, unless
/// the caller supplied it.
///
/// ### Params
///
/// * `means` - Measured means, row-major `[cell][feature]`, **raw units**
/// * `sds` - Standard deviations on those means, same layout, **raw units**
/// * `n_cells` - Number of cells; must agree with the matrix lengths
/// * `n_features` - Number of features; must agree with the matrix lengths
/// * `variances` - Per-feature `v[g]` in raw units if the caller has them, for
///   instance from Sanity via [`from_sanity`], or `None` to estimate them by S9
/// * `params` - Knobs, or `None` for [`IngestParams::default`]
///
/// ### Returns
///
/// The transformed data and everything needed to map it back, or an error:
/// `EmptyInput` for an empty dataset, `NoFeaturesRetained` for a threshold that
/// keeps nothing, `ShapeMismatch` for disagreeing lengths, `NonPositiveSd` for a
/// standard deviation that is not strictly positive and finite, `NonFiniteMean`
/// for a non-finite mean, `NonPositiveVariance` for a supplied variance that is
/// not positive, and `RootFindDiverged` if the variance solve fails to converge.
pub fn prepare<T: BonsaiFloat>(
    means: &[T],
    sds: &[T],
    n_cells: usize,
    n_features: usize,
    variances: Option<&[f64]>,
    params: Option<IngestParams>,
) -> Result<PreparedData<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    validate(means, sds, n_cells, n_features)?;
    if let Some(v) = variances {
        validate_variances(v, n_features)?;
    }

    let (v_all, s_all) = score_features(means, sds, n_cells, n_features, variances)?;

    // A feature with no variance has nothing to divide by, so it goes whatever
    // the threshold says. `S` is zero there anyway, so only a threshold at
    // negative infinity would otherwise have kept it.
    let kept: Vec<usize> = (0..n_features)
        .filter(|&g| {
            v_all[g] > 0.0
                && v_all[g].is_finite()
                && s_all[g].is_finite()
                && s_all[g] >= params.min_signal_to_noise
        })
        .collect();
    if kept.is_empty() {
        return Err(BonsaiErrors::NoFeaturesRetained {
            n_features,
            threshold: params.min_signal_to_noise,
        });
    }

    let k = kept.len();
    let mut transformed_means = vec![T::zero(); n_cells * k];
    let mut transformed_precisions = vec![T::zero(); n_cells * k];
    // The precision is formed as `v / sig^2` rather than by squaring the
    // transformed standard deviation, which keeps one rounding out of it.
    let inv_sqrt: Vec<f64> = kept.iter().map(|&g| v_all[g].sqrt().recip()).collect();
    for cell in 0..n_cells {
        let src = cell * n_features;
        let dst = cell * k;
        for (slot, &g) in kept.iter().enumerate() {
            let sd = wide(sds[src + g]);
            transformed_means[dst + slot] = narrow(wide(means[src + g]) * inv_sqrt[slot]);
            transformed_precisions[dst + slot] = narrow(v_all[g] / (sd * sd));
        }
    }

    Ok(PreparedData {
        transformed_means,
        transformed_precisions,
        variances: kept.iter().map(|&g| v_all[g]).collect(),
        signal_to_noise: kept.iter().map(|&g| s_all[g]).collect(),
        features: kept,
        n_cells,
        n_features_in: n_features,
    })
}

/// Recover likelihood parameters from Sanity posteriors (S5).
///
/// ```text
/// mu[g,i]    = xstar[g,i] * v[g] / (v[g] - eps[g,i]^2)
/// sig[g,i]^2 = eps[g,i]^2 * v[g] / (v[g] - eps[g,i]^2)
/// ```
///
/// ### The ill-conditioned case
///
/// Both lines share the amplification `v / (v - eps^2)`, which diverges as the
/// posterior widens towards the prior and is meaningless once `eps^2 >= v`.
/// This is not a corner case: it is why the reference grew a Sanity mode
/// returning the posterior-maximising `v[g]` instead of its expectation.
///
/// A feature is **dropped** if any one of its cells exceeds
/// `params.max_sanity_amplification`, and every dropped index is returned in
/// [`SanityLikelihood::dropped`] rather than clamped away. The whole feature
/// rather than the cell, because the model wants a rectangular matrix and
/// because `v[g]` is a per-feature quantity: one saturated posterior says the
/// `v[g]` is not to be trusted for the rest of the column either. Clamping was
/// rejected because it would put a fabricated error bar into the likelihood
/// with no way for the caller to tell; refusing the whole run was rejected
/// because a handful of saturated features in a large panel is normal.
///
/// ### Params
///
/// * `posterior_means` - Sanity's `xstar`, row-major `[cell][feature]`, raw
///   units
/// * `posterior_sds` - Sanity's `eps`, same layout, raw units
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `variances` - Sanity's `v[g]`, one per feature, raw units
/// * `params` - Knobs, or `None` for [`IngestParams::default`]
///
/// ### Returns
///
/// Likelihood means and standard deviations over the surviving features, in raw
/// units and ready for [`prepare`], plus the indices that were dropped. Errors
/// as [`prepare`] does, plus `IllConditionedConversion` when every feature is
/// ill-conditioned.
pub fn from_sanity<T: BonsaiFloat>(
    posterior_means: &[T],
    posterior_sds: &[T],
    n_cells: usize,
    n_features: usize,
    variances: &[f64],
    params: Option<IngestParams>,
) -> Result<SanityLikelihood<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    validate(posterior_means, posterior_sds, n_cells, n_features)?;
    validate_variances(variances, n_features)?;

    // `v - eps^2 >= v / max_amp` is the same test as `amplification <= max_amp`
    // but carries no division, so it stays well defined at `eps^2 >= v` instead
    // of producing an infinity or a negative variance.
    let floor: Vec<f64> = variances
        .iter()
        .map(|v| v / params.max_sanity_amplification)
        .collect();
    let mut ok = vec![true; n_features];
    for cell in 0..n_cells {
        let src = cell * n_features;
        for g in 0..n_features {
            let eps = wide(posterior_sds[src + g]);
            if variances[g] - eps * eps < floor[g] {
                ok[g] = false;
            }
        }
    }

    let kept: Vec<usize> = (0..n_features).filter(|&g| ok[g]).collect();
    let dropped: Vec<usize> = (0..n_features).filter(|&g| !ok[g]).collect();
    if kept.is_empty() {
        return Err(BonsaiErrors::IllConditionedConversion {
            n_features,
            cap: params.max_sanity_amplification,
        });
    }

    let k = kept.len();
    let mut out_means = vec![T::zero(); n_cells * k];
    let mut out_sds = vec![T::zero(); n_cells * k];
    for cell in 0..n_cells {
        let src = cell * n_features;
        let dst = cell * k;
        for (slot, &g) in kept.iter().enumerate() {
            let eps = wide(posterior_sds[src + g]);
            let e2 = eps * eps;
            let amp = variances[g] / (variances[g] - e2);
            out_means[dst + slot] = narrow(wide(posterior_means[src + g]) * amp);
            out_sds[dst + slot] = narrow((e2 * amp).sqrt());
        }
    }

    Ok(SanityLikelihood {
        means: out_means,
        sds: out_sds,
        variances: kept.iter().map(|&g| variances[g]).collect(),
        features: kept,
        dropped,
        n_cells,
    })
}

/// [`from_sanity`] straight from a `sanity-sc-rs` run.
///
/// Takes `log_fold_changes` as `xstar`, **not** the log transcription quotients
/// `m + d_c`. S5 inverts a zero-mean `N(0, v)` prior, so the posterior it
/// undoes is the one on the fold change; handing it `m + d_c` multiplies the
/// gene mean by a per-cell amplification and invents structure.
///
/// Sanity writes gene-major (`g * n_cells + c`); both matrices are transposed
/// to `[cell][gene]` here.
///
/// ### Params
///
/// * `out` - A finished Sanity run, from `sanity` or `sanity_select`
/// * `params` - Knobs, or `None` for [`IngestParams::default`]
///
/// ### Returns
///
/// As [`from_sanity`], except that `features` and `dropped` index the gene axis
/// of the counts Sanity was given, not Sanity's output rows.
#[cfg(feature = "sanity")]
pub fn from_sanity_output<T: BonsaiFloat + sanity_sc_rs::float::SanityFloat>(
    out: &sanity_sc_rs::SanityOutput<T>,
    params: Option<IngestParams>,
) -> Result<SanityLikelihood<T>, BonsaiErrors> {
    let (n_cells, n_genes) = (out.n_cells, out.n_genes);
    let mut means = vec![T::zero(); n_cells * n_genes];
    let mut sds = vec![T::zero(); n_cells * n_genes];
    means
        .par_chunks_mut(n_genes.max(1))
        .zip(sds.par_chunks_mut(n_genes.max(1)))
        .enumerate()
        .for_each(|(cell, (m_row, s_row))| {
            for g in 0..n_genes {
                m_row[g] = out.log_fold_changes[g * n_cells + cell];
                s_row[g] = out.error_bars[g * n_cells + cell];
            }
        });
    let variances: Vec<f64> = out.variance.iter().map(|&v| wide(v)).collect();

    let mut lik = from_sanity(&means, &sds, n_cells, n_genes, &variances, params)?;
    for k in lik.features.iter_mut().chain(lik.dropped.iter_mut()) {
        *k = out.genes[*k];
    }
    Ok(lik)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::likelihood::NodeState;
    use crate::tree::simulate::{SimulationParams, simulate_binary};
    use crate::utils::rng::SplitMix64;
    use approx::assert_relative_eq;

    /// A homoscedastic dataset with a known per-feature signal spread.
    ///
    /// Homoscedastic on purpose: with one `sig` per feature the estimator of
    /// S9 has a closed form, `v[g] = var_n(mu[g]) - sig[g]^2` with `var_n` the
    /// `1/n`-divisor sample variance, so the tests can assert against an exact
    /// answer rather than against a tolerance pulled from the air.
    ///
    /// ### Params
    ///
    /// * `n` - Number of cells
    /// * `spreads` - Standard deviation of the true signal, one per feature
    /// * `sds` - Measurement standard deviation, one per feature
    /// * `seed` - Seed for the stream
    ///
    /// ### Returns
    ///
    /// Row-major `[cell][feature]` means and standard deviations.
    fn homoscedastic(n: usize, spreads: &[f64], sds: &[f64], seed: u64) -> (Vec<f64>, Vec<f64>) {
        let p = spreads.len();
        let mut rng = SplitMix64::new(seed);
        let mut mu = vec![0.0f64; n * p];
        let mut sd = vec![0.0f64; n * p];
        for cell in 0..n {
            for g in 0..p {
                mu[cell * p + g] = spreads[g] * rng.normal() + sds[g] * rng.normal();
                sd[cell * p + g] = sds[g];
            }
        }
        (mu, sd)
    }

    /// The `1/n`-divisor sample variance of one column.
    ///
    /// ### Params
    ///
    /// * `x` - Row-major matrix
    /// * `n` - Number of rows
    /// * `p` - Number of columns
    /// * `g` - Column to measure
    ///
    /// ### Returns
    ///
    /// The column's mean and variance.
    fn column_stats(x: &[f64], n: usize, p: usize, g: usize) -> (f64, f64) {
        let mean: f64 = (0..n).map(|i| x[i * p + g]).sum::<f64>() / n as f64;
        let var: f64 = (0..n)
            .map(|i| (x[i * p + g] - mean) * (x[i * p + g] - mean))
            .sum::<f64>()
            / n as f64;
        (mean, var)
    }

    /// Keep every feature, whatever it scores.
    ///
    /// ### Returns
    ///
    /// Parameters with the selection threshold disabled.
    fn keep_everything() -> IngestParams {
        IngestParams {
            min_signal_to_noise: f64::NEG_INFINITY,
            ..IngestParams::default()
        }
    }

    // -- the scale transform --

    #[test]
    fn test_transform_round_trips_through_restore_scale() {
        let (n, p) = (32, 16);
        let spreads: Vec<f64> = (0..p).map(|g| 0.5 + 0.1 * g as f64).collect();
        let sds = vec![0.2; p];
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0x51ED_2701);
        let v: Vec<f64> = (0..p).map(|g| 0.3 + 0.7 * g as f64).collect();

        let prepared = prepare(&mu, &sd, n, p, Some(&v), Some(keep_everything())).unwrap();
        assert_eq!(prepared.n_features(), p);
        let back = prepared.restore_scale(&prepared.transformed_means).unwrap();
        for k in 0..n * p {
            assert_relative_eq!(back[k], mu[k], max_relative = 1e-14);
        }
    }

    #[test]
    fn test_transform_leaves_the_diffusion_prior_at_unit_variance() {
        // The whole crate's units rest on this. For a homoscedastic feature the
        // estimator makes `var_n(mu) = v + sig^2` exactly, so after dividing by
        // `sqrt(v)` the transformed data has variance `1 + sig_t^2`: signal
        // variance one, plus the measurement noise that is carried separately
        // in the precisions. Subtracting the noise back off must give exactly
        // one, and does so without any tolerance for sampling error because the
        // identity is algebraic rather than statistical.
        let (n, p) = (64, 24);
        let spreads: Vec<f64> = (0..p).map(|g| 0.4 + 0.05 * g as f64).collect();
        let sds: Vec<f64> = (0..p).map(|g| 0.05 + 0.01 * g as f64).collect();
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0xB105_F00D);

        let prepared = prepare(&mu, &sd, n, p, None, Some(keep_everything())).unwrap();
        assert_eq!(prepared.n_features(), p);
        for k in 0..p {
            let (_, var_t) = column_stats(&prepared.transformed_means, n, p, k);
            let noise_t = 1.0 / prepared.transformed_precisions[k];
            assert_relative_eq!(var_t - noise_t, 1.0, max_relative = 1e-9);
        }
    }

    // -- the variance solve --

    #[test]
    fn test_variance_solve_matches_the_analytic_answer() {
        // Homoscedastic, so S8 gives the plain mean and S9 collapses to
        // `v = var_n(mu) - sig^2`, here 7.1875 - 0.25.
        let mu = [1.0f64, 2.0, 4.0, 8.0];
        let sig2 = [0.25f64; 4];
        let mut d2 = [0.0f64; 4];
        let v = solve_variance(&mu, &sig2, &mut d2).unwrap();
        assert_relative_eq!(v, 6.9375, max_relative = 1e-12);
    }

    #[test]
    fn test_stationarity_residual_vanishes_at_the_solution() {
        let (n, p) = (128, 8);
        let spreads: Vec<f64> = (0..p).map(|g| 0.2 * (1 + g) as f64).collect();
        let sds: Vec<f64> = (0..p).map(|g| 0.03 + 0.02 * g as f64).collect();
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0x0BAD_C0DE);

        for g in 0..p {
            let col: Vec<f64> = (0..n).map(|i| mu[i * p + g]).collect();
            let sig2: Vec<f64> = (0..n).map(|i| sd[i * p + g] * sd[i * p + g]).collect();
            let mut d2 = vec![0.0f64; n];
            let v = solve_variance(&col, &sig2, &mut d2).unwrap();
            assert!(v > 0.0, "feature {g} lost its variance");
            fill_deviations(&col, &sig2, v, &mut d2);
            let residual = edge_newton(&sig2, &d2, v).0;
            // Scaled against the residual at the bracket's lower end, so this
            // says "indistinguishable from zero" rather than "small".
            fill_deviations(&col, &sig2, 0.0, &mut d2);
            let scale = edge_newton(&sig2, &d2, 0.0).0.abs().max(1e-30);
            assert!(residual.abs() / scale < 1e-12, "residual {residual:e}");
        }
    }

    #[test]
    fn test_a_feature_with_no_spread_solves_to_exactly_zero() {
        // Identical means: the error bars explain everything, so the condition
        // is negative at the origin and the solve short-circuits.
        let mu = vec![3.0f64; 16];
        let sig2 = vec![0.04f64; 16];
        let mut d2 = vec![0.0f64; 16];
        assert_eq!(solve_variance(&mu, &sig2, &mut d2).unwrap(), 0.0);
        assert_eq!(signal_to_noise(&mu, &sig2, 0.0), 0.0);
    }

    #[test]
    fn test_a_pure_noise_feature_scores_far_below_the_threshold() {
        // A feature that is only measurement noise does not reliably solve to
        // `v = 0`: the sample variance of the means scatters either side of the
        // error bars, and when it lands above them the estimator returns a
        // small positive variance. That is precisely why the threshold is not
        // zero, so the property to assert is that `S` lands well under it, not
        // that it is exactly zero.
        let n = 512;
        let (mu, sd) = homoscedastic(n, &[0.0], &[0.25], 0x1234_5678);
        let sig2: Vec<f64> = sd.iter().map(|s| s * s).collect();
        let mut d2 = vec![0.0f64; n];
        let v = solve_variance(&mu, &sig2, &mut d2).unwrap();
        let s = signal_to_noise(&mu, &sig2, v);
        assert!(
            s < 0.25 * DEFAULT_MIN_SIGNAL_TO_NOISE,
            "pure noise scored {s}"
        );
        assert!(matches!(
            prepare(&mu, &sd, n, 1, None, None),
            Err(BonsaiErrors::NoFeaturesRetained { .. })
        ));
    }

    // -- feature selection --

    #[test]
    fn test_signal_to_noise_ranks_features_by_signal() {
        // Four features on the same noise floor: signal spreads 2.0, 0.5 and
        // 0.1 against `sig = 0.1`, so `S` should land near 400, 25 and 1, and a
        // fourth with no signal at all must come last.
        let n = 512;
        let spreads = [2.0f64, 0.5, 0.1, 0.0];
        let sds = [0.1f64; 4];
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0x5EED_5EED);

        let prepared = prepare(&mu, &sd, n, 4, None, Some(keep_everything())).unwrap();
        // The noise feature never reaches the threshold comparison: its
        // variance solves to zero, and a feature with no variance has nothing
        // to divide by, so it is dropped whatever the threshold says.
        assert_eq!(prepared.features, vec![0, 1, 2]);
        let s = &prepared.signal_to_noise;
        assert!(s[0] > s[1] && s[1] > s[2], "ranking broke: {s:?}");
        assert_relative_eq!(s[0], 400.0, max_relative = 0.15);
        assert_relative_eq!(s[1], 25.0, max_relative = 0.15);
        assert_relative_eq!(s[2], 1.0, max_relative = 0.25);

        // With the threshold in play the weak feature goes too.
        let strict = IngestParams {
            min_signal_to_noise: 10.0,
            ..IngestParams::default()
        };
        let cut = prepare(&mu, &sd, n, 4, None, Some(strict)).unwrap();
        assert_eq!(cut.features, vec![0, 1]);
    }

    // -- the Sanity conversion --

    #[test]
    fn test_sanity_conversion_matches_hand_computation() {
        // v = 4, eps = 1, xstar = 2 gives amplification 4/3, so
        // mu = 8/3 and sig^2 = 4/3.
        let xstar = [2.0f64];
        let eps = [1.0f64];
        let v = [4.0f64];
        let out = from_sanity(&xstar, &eps, 1, 1, &v, None).unwrap();
        assert!(out.dropped.is_empty());
        assert_relative_eq!(out.means[0], 8.0 / 3.0, max_relative = 1e-14);
        assert_relative_eq!(out.sds[0] * out.sds[0], 4.0 / 3.0, max_relative = 1e-14);
    }

    #[test]
    fn test_sanity_drops_the_ill_conditioned_features_and_says_which() {
        // Feature 0 is well conditioned; feature 1 sits just under the prior
        // (amplification 4000, past the cap); feature 2 is past the prior
        // entirely, where the naive formula gives a negative variance.
        let v = [4.0f64, 4.0, 4.0];
        let xstar = [2.0f64, 2.0, 2.0];
        let eps = [1.0f64, 3.999f64.sqrt(), 5.0f64.sqrt()];
        let out = from_sanity(&xstar, &eps, 1, 3, &v, None).unwrap();
        assert_eq!(out.features, vec![0]);
        assert_eq!(out.dropped, vec![1, 2]);
        assert!(out.means.iter().all(|x| x.is_finite()));
        assert!(out.sds.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    #[test]
    fn test_sanity_refuses_a_dataset_that_is_entirely_ill_conditioned() {
        let v = [1.0f64; 3];
        let xstar = [0.5f64; 3];
        let eps = [1.5f64; 3];
        assert!(matches!(
            from_sanity(&xstar, &eps, 1, 3, &v, None),
            Err(BonsaiErrors::IllConditionedConversion { .. })
        ));
    }

    #[test]
    fn test_sanity_output_feeds_prepare() {
        let (n, p) = (48, 12);
        let mut rng = SplitMix64::new(0xFEED_FACE);
        let v: Vec<f64> = (0..p).map(|_| 0.5 + rng.uniform()).collect();
        let mut xstar = vec![0.0f64; n * p];
        let mut eps = vec![0.0f64; n * p];
        for cell in 0..n {
            for g in 0..p {
                // A posterior tighter than the prior by a comfortable margin,
                // which is the regime the conversion is meant for.
                eps[cell * p + g] = (0.2 * v[g]).sqrt();
                xstar[cell * p + g] = v[g].sqrt() * rng.normal();
            }
        }
        let sanity = from_sanity(&xstar, &eps, n, p, &v, None).unwrap();
        assert!(sanity.dropped.is_empty());
        let prepared = prepare(
            &sanity.means,
            &sanity.sds,
            n,
            sanity.features.len(),
            Some(&sanity.variances),
            Some(keep_everything()),
        )
        .unwrap();
        assert_eq!(prepared.n_features(), p);
        assert!(prepared.transformed_means.iter().all(|x| x.is_finite()));
        assert!(
            prepared
                .transformed_precisions
                .iter()
                .all(|x| x.is_finite() && *x > 0.0)
        );
    }

    // -- rejection paths --

    #[test]
    fn test_a_non_finite_mean_is_rejected() {
        let mut mu = vec![1.0f64; 6];
        mu[3] = f64::NAN;
        let sd = vec![0.1f64; 6];
        assert!(matches!(
            prepare(&mu, &sd, 3, 2, None, None),
            Err(BonsaiErrors::NonFiniteMean { .. })
        ));
        mu[3] = f64::INFINITY;
        assert!(matches!(
            prepare(&mu, &sd, 3, 2, None, None),
            Err(BonsaiErrors::NonFiniteMean { .. })
        ));
    }

    #[test]
    fn test_a_non_positive_standard_deviation_is_rejected() {
        let mu = vec![1.0f64; 6];
        for bad in [0.0f64, -0.5, f64::NAN, f64::INFINITY] {
            let mut sd = vec![0.1f64; 6];
            sd[4] = bad;
            assert!(
                matches!(
                    prepare(&mu, &sd, 3, 2, None, None),
                    Err(BonsaiErrors::NonPositiveSd { .. })
                ),
                "sd {bad} was accepted"
            );
        }
    }

    #[test]
    fn test_a_non_positive_variance_is_rejected() {
        let mu = vec![1.0f64; 6];
        let sd = vec![0.1f64; 6];
        for bad in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            let v = vec![bad, 1.0];
            assert!(
                matches!(
                    prepare(&mu, &sd, 3, 2, Some(&v), None),
                    Err(BonsaiErrors::NonPositiveVariance { .. })
                ),
                "variance {bad} was accepted"
            );
        }
    }

    #[test]
    fn test_mismatched_shapes_are_rejected() {
        let mu = vec![1.0f64; 6];
        assert!(matches!(
            prepare(&mu, &[0.1f64; 5], 3, 2, None, None),
            Err(BonsaiErrors::ShapeMismatch { .. })
        ));
        assert!(matches!(
            prepare(&mu, &[0.1f64; 6], 4, 2, None, None),
            Err(BonsaiErrors::ShapeMismatch { .. })
        ));
        assert!(matches!(
            prepare(&mu, &[0.1f64; 6], 3, 2, Some(&[1.0, 1.0, 1.0]), None),
            Err(BonsaiErrors::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_an_empty_dataset_is_rejected() {
        assert!(matches!(
            prepare::<f64>(&[], &[], 0, 4, None, None),
            Err(BonsaiErrors::EmptyInput { .. })
        ));
        assert!(matches!(
            prepare::<f64>(&[], &[], 4, 0, None, None),
            Err(BonsaiErrors::EmptyInput { .. })
        ));
    }

    #[test]
    fn test_a_threshold_that_retains_nothing_is_rejected() {
        let (n, p) = (64, 4);
        let (mu, sd) = homoscedastic(n, &[1.0; 4], &[0.1; 4], 0x2222_3333);
        let impossible = IngestParams {
            min_signal_to_noise: f64::INFINITY,
            ..IngestParams::default()
        };
        assert!(matches!(
            prepare(&mu, &sd, n, p, None, Some(impossible)),
            Err(BonsaiErrors::NoFeaturesRetained { .. })
        ));
    }

    #[test]
    fn test_restore_scale_rejects_a_ragged_input() {
        let (n, p) = (8, 3);
        let (mu, sd) = homoscedastic(n, &[1.0; 3], &[0.1; 3], 0x4444_5555);
        let prepared = prepare(&mu, &sd, n, p, None, Some(keep_everything())).unwrap();
        assert!(matches!(
            prepared.restore_scale(&[0.0f64; 4]),
            Err(BonsaiErrors::ShapeMismatch { .. })
        ));
    }

    // -- end to end --

    #[test]
    fn test_ingest_and_the_model_agree_on_a_simulated_dataset() {
        let params = SimulationParams {
            n_leaves: 128,
            n_features: 200,
            noise_sd: 0.1,
            seed: 0x7777_8888,
            ..SimulationParams::default()
        };
        let data = simulate_binary::<f64>(Some(params)).unwrap();
        let (n, p) = (data.n_leaves, data.n_features);

        // The generator hands back transformed units, so undo the transform to
        // get something an actual caller would have.
        let scale: Vec<f64> = data.variances.iter().map(|v| v.sqrt()).collect();
        let mut mu = vec![0.0f64; n * p];
        let mut sd = vec![0.0f64; n * p];
        for cell in 0..n {
            for g in 0..p {
                mu[cell * p + g] = data.means[cell * p + g] * scale[g];
                sd[cell * p + g] = data.sds[cell * p + g] * scale[g];
            }
        }

        let prepared = prepare(&mu, &sd, n, p, None, None).unwrap();
        // Every feature of this fixture carries signal worth roughly
        // `1/noise_sd^2`, two orders of magnitude above the default threshold.
        assert_eq!(prepared.n_features(), p);
        for (slot, &g) in prepared.features.iter().enumerate() {
            assert_relative_eq!(
                prepared.variances[slot],
                data.variances[g],
                max_relative = 0.1
            );
        }

        let mut state = NodeState::new(
            data.tree.n_nodes(),
            p,
            &prepared.transformed_means,
            &prepared.transformed_precisions,
        )
        .unwrap();
        let truth = state.prune(&data.tree);
        assert!(truth.is_finite(), "loglikelihood was {truth}");

        // Same tree, leaves relabelled. The topology the data was generated on
        // has to beat it, and by a wide margin at this noise level.
        let mut rng = SplitMix64::new(0x9999_AAAA);
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut m = vec![0.0f64; n * p];
        let mut w = vec![0.0f64; n * p];
        for cell in 0..n {
            let src = perm[cell] * p;
            m[cell * p..cell * p + p].copy_from_slice(&prepared.transformed_means[src..src + p]);
            w[cell * p..cell * p + p]
                .copy_from_slice(&prepared.transformed_precisions[src..src + p]);
        }
        let mut shuffled = NodeState::new(data.tree.n_nodes(), p, &m, &w).unwrap();
        let scrambled = shuffled.prune(&data.tree);
        assert!(scrambled.is_finite());
        assert!(
            truth > scrambled,
            "the true topology scored {truth} against {scrambled} for a shuffle"
        );
    }

    #[test]
    fn test_the_per_feature_pass_is_independent_of_the_chunking() {
        // The chunked gather is the one place a feature's data could be picked
        // up from the wrong column, and a feature count that straddles a chunk
        // boundary is where that would show.
        let (n, p) = (40, INGEST_BLOCK + 7);
        let spreads: Vec<f64> = (0..p).map(|g| 0.1 + 0.01 * g as f64).collect();
        let sds: Vec<f64> = (0..p).map(|g| 0.02 + 0.001 * g as f64).collect();
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0xC0FF_EE00);
        let prepared = prepare(&mu, &sd, n, p, None, Some(keep_everything())).unwrap();
        assert_eq!(prepared.n_features(), p);
        for (slot, &g) in prepared.features.iter().enumerate() {
            let col: Vec<f64> = (0..n).map(|i| mu[i * p + g]).collect();
            let sig2: Vec<f64> = (0..n).map(|i| sd[i * p + g] * sd[i * p + g]).collect();
            let mut d2 = vec![0.0f64; n];
            let v = solve_variance(&col, &sig2, &mut d2).unwrap();
            assert_relative_eq!(prepared.variances[slot], v, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_f32_storage_selects_the_same_features_as_f64() {
        // `prepare` is the only public generic entry into the module, and it is
        // the point where an `f32` caller's data first meets the `f64`
        // accumulation policy. The selection is a decision, so it has to be
        // identical; the transformed values only have to agree to `f32`.
        let (n, p) = (96, 20);
        // Feature zero is deliberately below the threshold, so the comparison
        // is on a selection that actually made a decision.
        let spreads: Vec<f64> = (0..p).map(|g| 0.005 + 0.03 * g as f64).collect();
        let sds: Vec<f64> = (0..p).map(|g| 0.04 + 0.002 * g as f64).collect();
        let (mu, sd) = homoscedastic(n, &spreads, &sds, 0x3141_5926);
        let narrow_mu: Vec<f32> = mu.iter().map(|&x| x as f32).collect();
        let narrow_sd: Vec<f32> = sd.iter().map(|&x| x as f32).collect();

        let wide = prepare(&mu, &sd, n, p, None, None).unwrap();
        let thin = prepare(&narrow_mu, &narrow_sd, n, p, None, None).unwrap();
        assert_eq!(wide.features, thin.features);
        assert!(!wide.features.is_empty() && wide.features.len() < p);
        for k in 0..n * wide.n_features() {
            assert_relative_eq!(
                thin.transformed_means[k] as f64,
                wide.transformed_means[k],
                max_relative = 1e-5
            );
            assert_relative_eq!(
                thin.transformed_precisions[k] as f64,
                wide.transformed_precisions[k],
                max_relative = 1e-5
            );
        }
    }
}

#[cfg(all(test, feature = "sanity"))]
mod sanity_tests {
    use super::*;
    use crate::bonsai::bonsai;
    use crate::tree::distance::{MAX_PAIRS, distance_recovery};
    use crate::tree::simulate::{
        SimulatedData, SimulationParams, robinson_foulds, simulate_binary,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, LogNormal, Poisson};
    use sanity_sc_rs::input::CountMatrix;
    use sanity_sc_rs::sanity;

    /// Counts of SPEC 13.1 on a 64-leaf balanced tree: the noise-free leaf
    /// positions, back on the raw scale, as log fold changes about a per-gene
    /// mean quotient.
    ///
    /// ### Returns
    ///
    /// The simulation, the counts and the per-cell totals.
    fn simulated_counts() -> (SimulatedData<f64>, CountMatrix, Vec<f64>) {
        let (n, p) = (64, 300);
        let sim = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            seed: 11,
            ..SimulationParams::default()
        }))
        .unwrap();

        let mut rng = StdRng::seed_from_u64(3);
        let library = LogNormal::new(8.0f64, 0.3).unwrap();
        let totals: Vec<f64> = (0..n).map(|_| library.sample(&mut rng).round()).collect();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut indptr = vec![0usize];
        for g in 0..p {
            let log_q = -(p as f64).ln() + 2.0 * (g as f64 / p as f64 - 0.5);
            let scale = sim.variances[g].sqrt();
            for c in 0..n {
                let rate = totals[c] * (log_q + sim.truth[c * p + g] * scale).exp();
                let k = Poisson::new(rate).unwrap().sample(&mut rng) as u32;
                if k > 0 {
                    indices.push(c as u32);
                    values.push(k);
                }
            }
            indptr.push(indices.len());
        }
        let counts = CountMatrix::new(indices, values, indptr, n).unwrap();
        (sim, counts, totals)
    }

    /// Reconstruct from a Sanity run and score it against the generating tree.
    ///
    /// ### Params
    ///
    /// * `sim` - The simulation the counts came from
    /// * `out` - A finished Sanity run over those counts
    ///
    /// ### Returns
    ///
    /// Robinson-Foulds to the generating tree and distance recovery.
    fn score(sim: &SimulatedData<f64>, out: &sanity_sc_rs::SanityOutput<f64>) -> (usize, f64) {
        let (n, p) = (sim.n_leaves, sim.n_features);
        let lik = from_sanity_output(out, None).unwrap();
        let n_kept = lik.features.len();
        let res = bonsai(&lik.means, &lik.sds, n, n_kept, Some(&lik.variances), None).unwrap();
        (
            robinson_foulds(&res.tree, &sim.tree).unwrap(),
            distance_recovery(&res.tree, &sim.truth, p, MAX_PAIRS, 0),
        )
    }

    #[test]
    fn test_counts_through_sanity_recover_the_tree() {
        let (sim, counts, totals) = simulated_counts();
        let out = sanity::<f64>(&counts, &totals, None).unwrap();
        let (rf, recovery) = score(&sim, &out);
        // Measured 2026-09-24: RF 0 and recovery 0.950. The same run fed the
        // log transcription quotients lands at RF 100 of 122 and 0.016.
        assert!(rf <= 6);
        assert!(recovery > 0.8);
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn test_counts_through_gpu_sanity_recover_the_tree() {
        use cubecl::Runtime;
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        use sanity_sc_rs::gpu::sanity_gpu;

        // The device path is f32 throughout; the tree it hands on has to be as
        // good as the CPU run's, not bit-equal to it.
        let (sim, counts, totals) = simulated_counts();
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let out = sanity_gpu::<f64, WgpuRuntime>(&counts, &totals, None, &client).unwrap();
        let (rf, recovery) = score(&sim, &out);
        assert!(rf <= 6);
        assert!(recovery > 0.8);
    }
}
