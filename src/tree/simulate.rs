//! Ground-truth simulated datasets (SPEC.md section 13.1).
//!
//! Three generators, all producing a known tree together with data drawn on it,
//! so that structural tests elsewhere in the crate can check *correctness*
//! against a known answer rather than merely self-consistency.
//!
//! * [`simulate_binary`] - balanced binary tree, constant branch lengths
//!   (SI.E.2.1)
//! * [`simulate_binary_random_branches`] - the same topology with `log t`
//!   uniform on `[log 0.5, log 2]` per child (SI.E.2.2)
//! * [`simulate_unbalanced`] - grown by repeatedly splitting a randomly chosen
//!   leaf (SI.E.2.3)
//!
//! [`robinson_foulds`] compares a recovered topology against the ground truth.
//!
//! ### Units, which matter and are easy to get wrong
//!
//! **Everything returned by this module is in the transformed units of SPEC.md
//! section 3.1**, that is, already divided by `sqrt(v[g])`. That is what
//! [`crate::model::likelihood::NodeState`] consumes, so simulated data feeds
//! straight in with no further work. The per-feature variances `v[g]` are
//! returned alongside so a caller who wants the untransformed scale can
//! multiply by `sqrt(v[g])`; see [`SimulatedData::variances`].
//!
//! A consequence worth stating outright: the rescaling step of SI.E.2.1
//! ("rescale to variance `v[g]`") becomes "rescale to unit variance" once
//! divided by `sqrt(v[g])`, so `v[g]` cancels out of the transformed data
//! entirely, except through the per-feature target means. The generator
//! therefore draws its Brownian steps directly in transformed units, with step
//! variance `t` rather than `t * v[g]`. The two are algebraically identical.
//!
//! ### What the rescaling costs
//!
//! Centring and rescaling each feature multiplies feature `g` by its own factor
//! `1 / sd_g`. The topology is untouched, but the *effective* branch lengths
//! seen by feature `g` are the generating ones times `1 / sd_g^2`, and that
//! factor differs from feature to feature. So the returned tree's branch
//! lengths are the values the walk was generated with, and the data is exactly
//! Brownian on that tree only up to a per-feature scale. Topology recovery is
//! unaffected; branch-length recovery tests should expect a common offset.
//!
//! ### Determinism
//!
//! The random stream is deliberately fixed. A local counter-based splitmix64
//! (the same shape as `benches/prune_sweep.rs`) is used rather than `rand`, so
//! the output is byte-identical on every platform and independent of thread
//! count. Normal variates come from Box-Muller on two uniforms and exponentials
//! from the inverse transform, both exactly reproducible. Draws are consumed in
//! one fixed order: `v[g]`, then the target means, then the topology, then node
//! positions, then measurement noise. Changing that order changes every
//! fixture in the crate, so do not reorder it casually.

use crate::errors::BonsaiErrors;
use crate::tree::{NO_NODE, Tree};
use crate::utils::traits::{BonsaiFloat, narrow};
use rustc_hash::FxHashSet;

///////////////
// Constants //
///////////////

/// Golden-ratio increment of the splitmix64 stream, `floor(2^64 / phi)`.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First multiplicative mixing constant of splitmix64.
const SPLITMIX_MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second multiplicative mixing constant of splitmix64.
const SPLITMIX_MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// Mantissa bits used when turning a `u64` into a double in `[0, 1)`.
const MANTISSA_BITS: u32 = 53;

/// Mean of the exponential distribution the per-feature variances `v[g]` are
/// drawn from (SPEC.md section 13.1: "drawn from an exponential distribution
/// with mean 2, which is what they observed in real data").
const VARIANCE_MEAN: f64 = 2.0;

/// Lower end of the random branch length range (SPEC.md section 13.1,
/// SI.E.2.2: `log(t)` uniform on `[log 0.5, log 2]`).
const RANDOM_BRANCH_LO: f64 = 0.5;

/// Upper end of the random branch length range (SPEC.md section 13.1,
/// SI.E.2.2).
const RANDOM_BRANCH_HI: f64 = 2.0;

/// Smallest per-feature variance that is rescaled rather than left alone.
///
/// Below this the feature is constant across cells to within rounding and
/// dividing by its standard deviation would manufacture noise. Our value, not
/// the paper's; it only guards a degenerate case that a well-formed simulation
/// never reaches.
const MIN_FEATURE_VARIANCE: f64 = 1e-300;

//////////////
// The PRNG //
//////////////

/// A splitmix64 stream.
///
/// Three lines of state advance and mixing, no dependency, identical output
/// everywhere. Seeded directly with the caller's seed, so distinct seeds give
/// distinct streams from the first draw.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    /// Stream position; advanced by [`SPLITMIX_GAMMA`] per draw.
    state: u64,
}

impl SplitMix64 {
    /// Start a stream at a seed.
    ///
    /// ### Params
    ///
    /// * `seed` - Seed value; any `u64` is valid, including zero
    ///
    /// ### Returns
    ///
    /// The stream.
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next raw 64-bit word.
    ///
    /// ### Returns
    ///
    /// A uniformly distributed `u64`.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MIX_B);
        z ^ (z >> 31)
    }

    /// Draw a uniform on the half-open interval `[0, 1)`.
    ///
    /// ### Returns
    ///
    /// The variate.
    #[inline]
    fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> (64 - MANTISSA_BITS);
        bits as f64 / (1u64 << MANTISSA_BITS) as f64
    }

    /// Draw a uniform on the half-open interval `(0, 1]`.
    ///
    /// Needed wherever a logarithm is taken of the variate, which is both
    /// Box-Muller and the exponential inverse transform.
    ///
    /// ### Returns
    ///
    /// The variate, never zero.
    #[inline]
    fn uniform_nonzero(&mut self) -> f64 {
        let bits = self.next_u64() >> (64 - MANTISSA_BITS);
        (bits as f64 + 1.0) / (1u64 << MANTISSA_BITS) as f64
    }

    /// Draw a standard normal variate by Box-Muller.
    ///
    /// The second variate of the pair is discarded rather than cached. Caching
    /// would halve the cost but would make the stream position depend on the
    /// parity of previous calls, and this module interleaves normal and uniform
    /// draws; a fixed two-uniforms-per-normal cost keeps the stream trivially
    /// auditable.
    ///
    /// ### Returns
    ///
    /// A draw from `N(0, 1)`.
    #[inline]
    fn normal(&mut self) -> f64 {
        let radial = (-2.0 * self.uniform_nonzero().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        radial * angle.cos()
    }

    /// Draw an exponential variate with a given mean, by inverse transform.
    ///
    /// ### Params
    ///
    /// * `mean` - Mean of the distribution, strictly positive
    ///
    /// ### Returns
    ///
    /// The variate, strictly positive.
    #[inline]
    fn exponential(&mut self, mean: f64) -> f64 {
        -mean * self.uniform_nonzero().ln()
    }

    /// Draw a log-uniform variate on `[lo, hi]`.
    ///
    /// ### Params
    ///
    /// * `lo` - Lower bound, strictly positive
    /// * `hi` - Upper bound, at least `lo`
    ///
    /// ### Returns
    ///
    /// The variate.
    #[inline]
    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let (log_lo, log_hi) = (lo.ln(), hi.ln());
        (log_lo + self.uniform() * (log_hi - log_lo)).exp()
    }

    /// Draw an index uniformly from `0..n`.
    ///
    /// Uses the low bits of a 64-bit draw. The modulo bias is on the order of
    /// `n / 2^64` and is irrelevant for the leaf counts this module handles.
    ///
    /// ### Params
    ///
    /// * `n` - Exclusive upper bound, strictly positive
    ///
    /// ### Returns
    ///
    /// An index in `0..n`.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

////////////////
// Parameters //
////////////////

/// Everything the three generators can be steered by.
///
/// `n_leaves` is interpreted differently by the balanced generators, which
/// require a power of two and derive the generation count from it, and by
/// [`simulate_unbalanced`], which accepts any count of at least two.
#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    /// Number of cells, that is, leaves of the ground-truth tree. The balanced
    /// generators require a power of two of at least two, and take the number
    /// of generations to be its base-two logarithm.
    pub n_leaves: usize,
    /// Number of features.
    pub n_features: usize,
    /// Branch length `t` used by [`simulate_binary`] and
    /// [`simulate_unbalanced`]. Ignored by
    /// [`simulate_binary_random_branches`].
    pub branch_length: f64,
    /// Measurement noise level, in transformed units. Since the true positions
    /// are rescaled to unit variance per feature, this is directly the
    /// noise-to-signal ratio: `0.1` means an error bar a tenth of the spread of
    /// the data. Must be strictly positive, because a zero standard deviation
    /// is an infinite precision and the pruning recursion cannot carry it.
    pub noise_sd: f64,
    /// Spread of the per-cell per-feature error bars about `noise_sd`. Each
    /// standard deviation is multiplied by a log-uniform draw on
    /// `[1 / noise_spread, noise_spread]`, so `1.0` is homoscedastic. Real data
    /// is not homoscedastic and the precision-weighted machinery deserves to be
    /// exercised, so the default is not `1.0`.
    pub noise_spread: f64,
    /// Standard deviation of the per-feature target means `mu[g]` of SPEC.md
    /// section 13.1, in *untransformed* units. The tree loglikelihood depends
    /// only on differences of means within a feature, so this shifts the data
    /// without changing any score; it defaults to zero and exists so that the
    /// generator is faithful to the specification.
    pub feature_mean_sd: f64,
    /// Seed for the splitmix64 stream.
    pub seed: u64,
}

impl SimulationParams {
    /// Build a parameter set explicitly.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of cells
    /// * `n_features` - Number of features
    /// * `branch_length` - Constant branch length `t`
    /// * `noise_sd` - Measurement noise level in transformed units
    /// * `noise_spread` - Log-uniform spread factor on the error bars
    /// * `feature_mean_sd` - Spread of the per-feature target means
    /// * `seed` - Seed for the random stream
    ///
    /// ### Returns
    ///
    /// The parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_leaves: usize,
        n_features: usize,
        branch_length: f64,
        noise_sd: f64,
        noise_spread: f64,
        feature_mean_sd: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_leaves,
            n_features,
            branch_length,
            noise_sd,
            noise_spread,
            feature_mean_sd,
            seed,
        }
    }
}

impl Default for SimulationParams {
    /// A small, fast fixture: 64 cells by 200 features on unit branches, with
    /// error bars a tenth of the data spread and a factor-of-two spread on
    /// them.
    ///
    /// ### Returns
    ///
    /// The default parameter set.
    fn default() -> Self {
        Self {
            n_leaves: 64,
            n_features: 200,
            branch_length: 1.0,
            noise_sd: 0.1,
            noise_spread: 2.0,
            feature_mean_sd: 0.0,
            seed: 0,
        }
    }
}

////////////
// Output //
////////////

/// One simulated dataset with its ground truth.
///
/// All three matrices are row-major `[leaf][feature]`, the layout
/// [`crate::model::likelihood::NodeState::new`] expects, and all three are in
/// the transformed units of SPEC.md section 3.1. Multiply column `g` by
/// `variances[g].sqrt()` to recover the untransformed scale.
#[derive(Clone, Debug)]
pub struct SimulatedData<T> {
    /// The tree the data was generated on, with the generating branch lengths.
    pub tree: Tree,
    /// True leaf positions, noise-free, `[leaf][feature]`.
    pub truth: Vec<T>,
    /// Observed means: `truth` plus a draw from `N(0, sds^2)`,
    /// `[leaf][feature]`.
    pub means: Vec<T>,
    /// Observed standard deviations on those means, `[leaf][feature]`.
    pub sds: Vec<T>,
    /// Per-feature total variances `v[g]`, on the *untransformed* scale. These
    /// are what the transform of SPEC.md section 3.1 divided out.
    pub variances: Vec<f64>,
    /// Number of leaves.
    pub n_leaves: usize,
    /// Number of features.
    pub n_features: usize,
}

impl<T: BonsaiFloat> SimulatedData<T> {
    /// Observed precisions, `1 / sd^2`, ready for
    /// [`crate::model::likelihood::NodeState::new`].
    ///
    /// ### Returns
    ///
    /// The precisions, row-major `[leaf][feature]`.
    pub fn precisions(&self) -> Vec<T> {
        self.sds
            .iter()
            .map(|&s| {
                let s = s.to_f64().unwrap_or(f64::NAN);
                narrow(1.0 / (s * s))
            })
            .collect()
    }
}

////////////////
// Generators //
////////////////

/// Validate a parameter set and open its random stream.
///
/// Draws the per-feature variances and target means, which come first in the
/// stream for every generator so that the three agree on those quantities for a
/// given seed.
///
/// ### Params
///
/// * `params` - Resolved parameter set
///
/// ### Returns
///
/// The stream, the per-feature variances `v[g]`, and the per-feature target
/// means already divided by `sqrt(v[g])` so they can be added in transformed
/// units; or an error if the parameters are unusable.
fn open_stream(
    params: &SimulationParams,
) -> Result<(SplitMix64, Vec<f64>, Vec<f64>), BonsaiErrors> {
    if params.n_features == 0 || params.n_leaves == 0 {
        return Err(BonsaiErrors::EmptyInput {
            n_cells: params.n_leaves,
            n_features: params.n_features,
        });
    }
    if params.n_leaves < 2 {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("{} leaves is not enough to simulate a tree", params.n_leaves),
        });
    }
    // `is_finite` first, so a NaN is rejected without a negated comparison.
    if !params.noise_sd.is_finite() || params.noise_sd <= 0.0 {
        return Err(BonsaiErrors::NonPositiveSd {
            value: params.noise_sd,
            cell: 0,
            feature: 0,
        });
    }
    if !params.noise_spread.is_finite() || params.noise_spread < 1.0 {
        return Err(BonsaiErrors::NonPositiveSd {
            value: params.noise_spread,
            cell: 0,
            feature: 0,
        });
    }

    let mut rng = SplitMix64::new(params.seed);
    let variances: Vec<f64> = (0..params.n_features)
        .map(|_| rng.exponential(VARIANCE_MEAN))
        .collect();
    // Drawn unconditionally, even when the spread is zero, so that the stream
    // position after this point does not depend on `feature_mean_sd`.
    let offsets: Vec<f64> = variances
        .iter()
        .map(|&v| params.feature_mean_sd * rng.normal() / v.sqrt())
        .collect();
    Ok((rng, variances, offsets))
}

/// Parent array of a balanced binary tree over `n_leaves` leaves.
///
/// Built bottom-up by pairing, which is exactly the numbering the arena
/// invariant wants: leaves occupy `0..n_leaves` and every internal node is
/// allocated after both of its children.
///
/// ### Params
///
/// * `n_leaves` - Number of leaves, a power of two of at least two
///
/// ### Returns
///
/// The parent array, or `MalformedTree` if the leaf count is not a power of
/// two.
fn balanced_parents(n_leaves: usize) -> Result<Vec<u32>, BonsaiErrors> {
    if n_leaves < 2 || !n_leaves.is_power_of_two() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "{n_leaves} leaves: the balanced generators need a power of two of at least two, \
                 since the generation count is its base-two logarithm"
            ),
        });
    }
    let n_nodes = 2 * n_leaves - 1;
    let mut parent = vec![NO_NODE; n_nodes];
    let mut level: Vec<u32> = (0..n_leaves as u32).collect();
    let mut next_free = n_leaves as u32;
    while level.len() > 1 {
        let mut up = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            let ancestor = next_free;
            next_free += 1;
            parent[pair[0] as usize] = ancestor;
            parent[pair[1] as usize] = ancestor;
            up.push(ancestor);
        }
        level = up;
    }
    Ok(parent)
}

/// Grow an unbalanced topology by repeatedly splitting a random leaf
/// (SPEC.md section 13.1, SI.E.2.3).
///
/// The construction numbers the root zero and appends children, which is the
/// reverse of what the arena wants, so the result is relabelled: leaves take
/// `0..n_leaves` in construction order and internal nodes are sorted by height
/// above the leaves. A parent's height strictly exceeds every child's, so that
/// ordering satisfies the "parents have larger indices" requirement of
/// [`Tree::from_parents`].
///
/// ### Params
///
/// * `rng` - Random stream, advanced by one draw per split
/// * `n_leaves` - Number of leaves to grow to, at least two
///
/// ### Returns
///
/// The parent array in arena numbering.
fn unbalanced_parents(rng: &mut SplitMix64, n_leaves: usize) -> Vec<u32> {
    // Construction numbering: node 0 is the root, children are appended, so a
    // child's index always exceeds its parent's.
    let mut built = vec![NO_NODE];
    let mut open: Vec<u32> = vec![0];
    for _ in 0..n_leaves - 1 {
        let slot = rng.below(open.len());
        let chosen = open[slot];
        let left = built.len() as u32;
        built.push(chosen);
        let right = built.len() as u32;
        built.push(chosen);
        open[slot] = left;
        open.push(right);
    }

    let n_nodes = built.len();
    let mut is_leaf = vec![true; n_nodes];
    for &par in built.iter() {
        if par != NO_NODE {
            is_leaf[par as usize] = false;
        }
    }

    // Children have larger construction indices, so a descending scan settles
    // every child before its parent.
    let mut height = vec![0u32; n_nodes];
    for node in (1..n_nodes).rev() {
        let par = built[node] as usize;
        height[par] = height[par].max(height[node] + 1);
    }

    let mut relabel = vec![0u32; n_nodes];
    let mut leaf_count = 0u32;
    let mut internal: Vec<u32> = Vec::with_capacity(n_leaves - 1);
    for node in 0..n_nodes {
        if is_leaf[node] {
            relabel[node] = leaf_count;
            leaf_count += 1;
        } else {
            internal.push(node as u32);
        }
    }
    internal.sort_unstable_by_key(|&i| (height[i as usize], i));
    for (slot, &old) in internal.iter().enumerate() {
        relabel[old as usize] = (n_leaves + slot) as u32;
    }

    let mut parent = vec![NO_NODE; n_nodes];
    for old in 0..n_nodes {
        parent[relabel[old] as usize] = match built[old] {
            NO_NODE => NO_NODE,
            par => relabel[par as usize],
        };
    }
    parent
}

/// Walk the tree from the root, centre and rescale, then add measurement noise.
///
/// Shared tail of all three generators. Positions are drawn in transformed
/// units, so the Brownian step along a branch of length `t` has variance `t`
/// rather than `t * v[g]`; see the module documentation for why those are the
/// same thing once the rescaling step is applied.
///
/// ### Params
///
/// * `rng` - Random stream, positioned after the variance and mean draws
/// * `parent` - Parent array in arena numbering
/// * `branch` - Branch length above each node, root entry ignored
/// * `n_leaves` - Number of leaves
/// * `variances` - Per-feature variances `v[g]`
/// * `offsets` - Per-feature target means, already in transformed units
/// * `params` - Resolved parameter set
///
/// ### Returns
///
/// The finished dataset, or an error from [`Tree::from_parents`].
fn assemble<T: BonsaiFloat>(
    rng: &mut SplitMix64,
    parent: Vec<u32>,
    branch: Vec<f64>,
    n_leaves: usize,
    variances: Vec<f64>,
    offsets: &[f64],
    params: &SimulationParams,
) -> Result<SimulatedData<T>, BonsaiErrors> {
    let p = params.n_features;
    let n_nodes = parent.len();

    // Descending index order is a valid pre-order: the arena invariant puts
    // every parent above its children, and the root is the last node.
    let mut pos = vec![0.0f64; n_nodes * p];
    for node in (0..n_nodes - 1).rev() {
        let par = parent[node] as usize;
        let step_sd = branch[node].sqrt();
        for g in 0..p {
            let from = pos[par * p + g];
            pos[node * p + g] = from + step_sd * rng.normal();
        }
    }

    // Per feature: centre across cells, then rescale to variance `v[g]`. In
    // transformed units that target variance is one, since the whole feature
    // has already been divided by `sqrt(v[g])`.
    let n = n_leaves as f64;
    let mut truth = vec![0.0f64; n_leaves * p];
    for g in 0..p {
        let mut mean = 0.0f64;
        for i in 0..n_leaves {
            mean += pos[i * p + g];
        }
        mean /= n;
        let mut var = 0.0f64;
        for i in 0..n_leaves {
            let d = pos[i * p + g] - mean;
            var += d * d;
        }
        var /= n;
        let scale = if var > MIN_FEATURE_VARIANCE {
            var.sqrt().recip()
        } else {
            0.0
        };
        for i in 0..n_leaves {
            truth[i * p + g] = (pos[i * p + g] - mean) * scale + offsets[g];
        }
    }
    drop(pos);

    // Measurement noise: the Gaussian likelihood of SPEC.md section 2 (S2).
    let log_spread = params.noise_spread.ln();
    let mut sds = vec![0.0f64; n_leaves * p];
    let mut means = vec![0.0f64; n_leaves * p];
    for k in 0..n_leaves * p {
        let sd = params.noise_sd * (log_spread * (2.0 * rng.uniform() - 1.0)).exp();
        sds[k] = sd;
        means[k] = truth[k] + sd * rng.normal();
    }

    let tree = Tree::from_parents(parent, branch, n_leaves)?;
    Ok(SimulatedData {
        tree,
        truth: truth.into_iter().map(narrow).collect(),
        means: means.into_iter().map(narrow).collect(),
        sds: sds.into_iter().map(narrow).collect(),
        variances,
        n_leaves,
        n_features: p,
    })
}

/// Simulate on a balanced binary tree with constant branch lengths
/// (SPEC.md section 13.1, SI.E.2.1).
///
/// The root sits at the origin in `p` dimensions and each child adds a
/// per-feature Gaussian step of variance `t * v[g]`, which is variance `t` in
/// the transformed units this returns. `n_leaves` must be a power of two; its
/// base-two logarithm is the generation count, and only the last generation is
/// kept as data.
///
/// ### Params
///
/// * `params` - Parameter set, or `None` for [`SimulationParams::default`]
///
/// ### Returns
///
/// The dataset, in transformed units, or an error if the parameters are
/// unusable.
pub fn simulate_binary<T: BonsaiFloat>(
    params: Option<SimulationParams>,
) -> Result<SimulatedData<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let (mut rng, variances, offsets) = open_stream(&params)?;
    let parent = balanced_parents(params.n_leaves)?;
    let branch = vec![params.branch_length; parent.len()];
    assemble(
        &mut rng,
        parent,
        branch,
        params.n_leaves,
        variances,
        &offsets,
        &params,
    )
}

/// Simulate on a balanced binary tree with random branch lengths
/// (SPEC.md section 13.1, SI.E.2.2).
///
/// Identical to [`simulate_binary`] except that `log(t)` is uniform on
/// `[log 0.5, log 2]`, drawn independently for every non-root node.
/// `params.branch_length` is ignored.
///
/// ### Params
///
/// * `params` - Parameter set, or `None` for [`SimulationParams::default`]
///
/// ### Returns
///
/// The dataset, in transformed units, or an error if the parameters are
/// unusable.
pub fn simulate_binary_random_branches<T: BonsaiFloat>(
    params: Option<SimulationParams>,
) -> Result<SimulatedData<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let (mut rng, variances, offsets) = open_stream(&params)?;
    let parent = balanced_parents(params.n_leaves)?;
    let n_nodes = parent.len();
    // Ascending node order, root last and left at zero: its branch is unused.
    let mut branch: Vec<f64> = (0..n_nodes - 1)
        .map(|_| rng.log_uniform(RANDOM_BRANCH_LO, RANDOM_BRANCH_HI))
        .collect();
    branch.push(0.0);
    assemble(
        &mut rng,
        parent,
        branch,
        params.n_leaves,
        variances,
        &offsets,
        &params,
    )
}

/// Simulate on an unbalanced tree grown by splitting random leaves
/// (SPEC.md section 13.1, SI.E.2.3).
///
/// Starts from a single node and repeats `n_leaves - 1` times: pick a leaf
/// uniformly at random, give it two children, and replace it in the leaf list
/// with them. Any `n_leaves` of at least two works. The specification says
/// nothing about branch lengths here, so `params.branch_length` is used
/// throughout, as in [`simulate_binary`].
///
/// ### Params
///
/// * `params` - Parameter set, or `None` for [`SimulationParams::default`]
///
/// ### Returns
///
/// The dataset, in transformed units, or an error if the parameters are
/// unusable.
pub fn simulate_unbalanced<T: BonsaiFloat>(
    params: Option<SimulationParams>,
) -> Result<SimulatedData<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let (mut rng, variances, offsets) = open_stream(&params)?;
    let parent = unbalanced_parents(&mut rng, params.n_leaves);
    let branch = vec![params.branch_length; parent.len()];
    assemble(
        &mut rng,
        parent,
        branch,
        params.n_leaves,
        variances,
        &offsets,
        &params,
    )
}

//////////////////////
// Robinson-Foulds //
//////////////////////

/// The set of non-trivial splits of a tree, each as a sorted leaf-index list.
///
/// The arena holds an unrooted tree in a rooted representation, and the
/// loglikelihood does not depend on where the root sits (SPEC.md section 2,
/// S14), so the comparison must be over unrooted splits. Each internal node
/// below the root induces a bipartition of the leaves; it is canonicalised to
/// whichever side does not contain leaf zero, so that a tree and any rerooting
/// of it produce the same set. Splits with fewer than two leaves on either side
/// are trivial, present in every tree over the leaf set, and dropped.
///
/// ### Params
///
/// * `tree` - Tree to describe
///
/// ### Returns
///
/// The canonical non-trivial splits.
fn splits(tree: &Tree) -> FxHashSet<Vec<u32>> {
    let n_leaves = tree.n_leaves();
    let n_nodes = tree.n_nodes();

    // Ascending index order is a post-order, so a node's children are settled
    // by the time it is reached.
    let mut clade: Vec<Vec<u32>> = Vec::with_capacity(n_nodes);
    for node in 0..n_nodes {
        if node < n_leaves {
            clade.push(vec![node as u32]);
        } else {
            let mut merged: Vec<u32> = Vec::new();
            for &child in tree.children(node as u32) {
                merged.extend_from_slice(&clade[child as usize]);
            }
            merged.sort_unstable();
            clade.push(merged);
        }
    }

    let mut out = FxHashSet::default();
    let mut present = vec![false; n_leaves];
    for node in n_leaves..n_nodes {
        let below = &clade[node];
        // Take the side without leaf zero, so a rerooting maps to the same key.
        let side: Vec<u32> = if below.first() == Some(&0) {
            for f in present.iter_mut() {
                *f = false;
            }
            for &leaf in below.iter() {
                present[leaf as usize] = true;
            }
            (0..n_leaves as u32)
                .filter(|&leaf| !present[leaf as usize])
                .collect()
        } else {
            below.clone()
        };
        if side.len() >= 2 && side.len() <= n_leaves - 2 {
            out.insert(side);
        }
    }
    out
}

/// Robinson-Foulds distance between two trees over the same leaf set.
///
/// The size of the symmetric difference of the two sets of non-trivial splits.
/// Zero means the two topologies are identical as unrooted trees; the maximum
/// for two binary trees over `n` leaves is `2 * (n - 3)`. Branch lengths are
/// ignored.
///
/// Leaves are matched by index, which is what makes this usable against the
/// simulator: [`Tree::from_parents`] relabels internal nodes but never leaves,
/// so a recovered tree indexes cells the same way the ground truth does.
///
/// ### Params
///
/// * `left` - First tree
/// * `right` - Second tree, over the same leaves
///
/// ### Returns
///
/// The distance, or `MalformedTree` if the two leaf sets differ in size.
pub fn robinson_foulds(left: &Tree, right: &Tree) -> Result<usize, BonsaiErrors> {
    if left.n_leaves() != right.n_leaves() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "Robinson-Foulds needs a shared leaf set: {} leaves against {}",
                left.n_leaves(),
                right.n_leaves()
            ),
        });
    }
    let a = splits(left);
    let b = splits(right);
    Ok(a.symmetric_difference(&b).count())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::likelihood::NodeState;
    use approx::assert_relative_eq;

    /// Bit-exact fingerprint of a dataset, tree included.
    ///
    /// ### Params
    ///
    /// * `data` - Dataset to fingerprint
    ///
    /// ### Returns
    ///
    /// The raw bits of every returned quantity, in a fixed order.
    fn fingerprint(data: &SimulatedData<f64>) -> Vec<u64> {
        let mut out = Vec::new();
        out.push(data.tree.n_nodes() as u64);
        out.push(data.tree.n_leaves() as u64);
        for node in 0..data.tree.n_nodes() as u32 {
            out.push(data.tree.parent(node).unwrap_or(NO_NODE) as u64);
            out.push(data.tree.branch(node).to_bits());
        }
        for v in data
            .truth
            .iter()
            .chain(&data.means)
            .chain(&data.sds)
            .chain(&data.variances)
        {
            out.push(v.to_bits());
        }
        out
    }

    /// A balanced binary tree whose leaf labels have been permuted, which makes
    /// it a genuinely different topology over the same cells.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, a power of two
    /// * `branch` - Branch length for every edge
    /// * `seed` - Seed for the permutation
    ///
    /// ### Returns
    ///
    /// The relabelled tree.
    fn shuffled_binary(n_leaves: usize, branch: f64, seed: u64) -> Tree {
        let old = balanced_parents(n_leaves).unwrap();
        let mut rng = SplitMix64::new(seed);
        let mut perm: Vec<u32> = (0..n_leaves as u32).collect();
        for i in (1..n_leaves).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut parent = old.clone();
        for slot in 0..n_leaves {
            parent[perm[slot] as usize] = old[slot];
        }
        Tree::from_parents(parent, vec![branch; old.len()], n_leaves).unwrap()
    }

    /// Depth of every leaf, counted in edges from the root.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to measure
    ///
    /// ### Returns
    ///
    /// One depth per leaf.
    fn leaf_depths(tree: &Tree) -> Vec<usize> {
        (0..tree.n_leaves() as u32)
            .map(|leaf| {
                let mut node = leaf;
                let mut depth = 0;
                while let Some(par) = tree.parent(node) {
                    node = par;
                    depth += 1;
                }
                depth
            })
            .collect()
    }

    #[test]
    fn test_same_seed_gives_byte_identical_output() {
        let params = SimulationParams {
            n_leaves: 16,
            n_features: 40,
            seed: 12345,
            ..Default::default()
        };
        for _ in 0..2 {
            let a = simulate_binary::<f64>(Some(params)).unwrap();
            let b = simulate_binary::<f64>(Some(params)).unwrap();
            assert_eq!(fingerprint(&a), fingerprint(&b));
        }

        let unbalanced = SimulationParams {
            n_leaves: 23,
            ..params
        };
        let a = simulate_unbalanced::<f64>(Some(unbalanced)).unwrap();
        let b = simulate_unbalanced::<f64>(Some(unbalanced)).unwrap();
        assert_eq!(fingerprint(&a), fingerprint(&b));

        let a = simulate_binary_random_branches::<f64>(Some(params)).unwrap();
        let b = simulate_binary_random_branches::<f64>(Some(params)).unwrap();
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn test_different_seeds_give_different_data() {
        let base = SimulationParams {
            n_leaves: 16,
            n_features: 40,
            seed: 1,
            ..Default::default()
        };
        let other = SimulationParams { seed: 2, ..base };

        let a = simulate_binary::<f64>(Some(base)).unwrap();
        let b = simulate_binary::<f64>(Some(other)).unwrap();
        assert_ne!(fingerprint(&a), fingerprint(&b));
        assert_ne!(a.variances, b.variances);

        // And the seed must reach the topology, not just the data.
        let unbalanced_a = simulate_unbalanced::<f64>(Some(SimulationParams {
            n_leaves: 64,
            ..base
        }))
        .unwrap();
        let unbalanced_b = simulate_unbalanced::<f64>(Some(SimulationParams {
            n_leaves: 64,
            ..other
        }))
        .unwrap();
        assert!(robinson_foulds(&unbalanced_a.tree, &unbalanced_b.tree).unwrap() > 0);
    }

    #[test]
    fn test_robinson_foulds_of_a_hand_checked_quartet() {
        // Four leaves, so exactly one non-trivial split per binary tree.
        // ((0,1),(2,3)) against ((0,2),(1,3)): the split sets are {2,3} and
        // {1,3}, disjoint, so the symmetric difference has two elements.
        let ab_cd =
            Tree::from_parents(vec![4, 4, 5, 5, 6, 6, NO_NODE], vec![1.0; 7], 4).unwrap();
        let ac_bd =
            Tree::from_parents(vec![4, 5, 4, 5, 6, 6, NO_NODE], vec![1.0; 7], 4).unwrap();
        assert_eq!(robinson_foulds(&ab_cd, &ac_bd).unwrap(), 2);

        // Six leaves as ((0,1),(2,3),(4,5)) fully resolved one way, then with
        // the (2,3) and (4,5) cherries recombined. Splits differ in exactly one
        // element on each side.
        let left = Tree::from_parents(
            vec![6, 6, 7, 7, 8, 8, 9, 9, 10, 10, NO_NODE],
            vec![1.0; 11],
            6,
        )
        .unwrap();
        assert_eq!(splits(&left).len(), 3);
    }

    #[test]
    fn test_robinson_foulds_is_a_metric() {
        let n = 32usize;
        let a = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: 4,
            seed: 7,
            ..Default::default()
        }))
        .unwrap()
        .tree;
        let b = shuffled_binary(n, 1.0, 99);
        let c = shuffled_binary(n, 1.0, 4242);

        assert_eq!(robinson_foulds(&a, &a).unwrap(), 0);
        assert_eq!(robinson_foulds(&b, &b).unwrap(), 0);
        assert_eq!(
            robinson_foulds(&a, &b).unwrap(),
            robinson_foulds(&b, &a).unwrap()
        );
        assert!(robinson_foulds(&a, &b).unwrap() > 0);
        assert!(robinson_foulds(&a, &c).unwrap() > 0);
        // Symmetric difference of set differences: the triangle inequality
        // holds for any symmetric difference, so this is a real check on the
        // split extraction rather than on the metric itself.
        assert!(
            robinson_foulds(&a, &c).unwrap()
                <= robinson_foulds(&a, &b).unwrap() + robinson_foulds(&b, &c).unwrap()
        );
        // Two binary trees over n leaves can differ by at most 2 * (n - 3).
        assert!(robinson_foulds(&a, &b).unwrap() <= 2 * (n - 3));
    }

    #[test]
    fn test_robinson_foulds_ignores_internal_relabelling_and_rerooting() {
        // The same topology with the two cherries allocated to swapped internal
        // indices; `from_parents` relabels either way.
        let straight =
            Tree::from_parents(vec![4, 4, 5, 5, 6, 6, NO_NODE], vec![1.0; 7], 4).unwrap();
        let swapped =
            Tree::from_parents(vec![5, 5, 4, 4, 6, 6, NO_NODE], vec![1.0; 7], 4).unwrap();
        assert_eq!(robinson_foulds(&straight, &swapped).unwrap(), 0);

        // Same unrooted tree, rooted on the branch leading to leaf 0 instead:
        // 0 hangs off the root, then 1, then the cherry (2,3).
        let rerooted =
            Tree::from_parents(vec![6, 5, 4, 4, 5, 6, NO_NODE], vec![1.0; 7], 4).unwrap();
        assert_eq!(robinson_foulds(&straight, &rerooted).unwrap(), 0);

        // A larger case: reverse the internal numbering within each level,
        // which is still a legal arena numbering because a parent stays in a
        // strictly higher level than its children. `from_parents` keeps that
        // labelling, so the two trees genuinely carry different internal
        // indices for the same topology.
        let big = simulate_unbalanced::<f64>(Some(SimulationParams {
            n_leaves: 40,
            n_features: 2,
            seed: 3,
            ..Default::default()
        }))
        .unwrap()
        .tree;
        let mut relabel: Vec<u32> = (0..big.n_nodes() as u32).collect();
        for level in 0..big.n_levels() {
            let (start, end) = big.level(level);
            for node in start..end {
                relabel[node] = (start + end - 1 - node) as u32;
            }
        }
        let mut parent = vec![NO_NODE; big.n_nodes()];
        for node in 0..big.n_nodes() as u32 {
            parent[relabel[node as usize] as usize] = match big.parent(node) {
                None => NO_NODE,
                Some(par) => relabel[par as usize],
            };
        }
        let original: Vec<u32> = (0..big.n_nodes() as u32)
            .map(|node| big.parent(node).unwrap_or(NO_NODE))
            .collect();
        assert_ne!(parent, original, "the relabelling did not change anything");
        let reversed = Tree::from_parents(parent, vec![1.0; big.n_nodes()], big.n_leaves()).unwrap();
        assert_eq!(robinson_foulds(&big, &reversed).unwrap(), 0);
    }

    #[test]
    fn test_true_tree_beats_a_shuffled_one_under_the_likelihood() {
        // The load-bearing test: the simulator and `NodeState::prune` must
        // agree about what a tree means. With small error bars the true
        // topology has to score higher than a permuted one on the same cells.
        let params = SimulationParams {
            n_leaves: 32,
            n_features: 300,
            branch_length: 1.0,
            noise_sd: 1e-3,
            noise_spread: 1.0,
            feature_mean_sd: 0.0,
            seed: 20260827,
        };
        let data = simulate_binary::<f64>(Some(params)).unwrap();
        let precisions = data.precisions();

        let mut truth_state = NodeState::new(
            data.tree.n_nodes(),
            data.n_features,
            &data.means,
            &precisions,
        )
        .unwrap();
        let l_true = truth_state.prune(&data.tree);

        let wrong = shuffled_binary(params.n_leaves, params.branch_length, 555);
        assert!(robinson_foulds(&data.tree, &wrong).unwrap() > 0);
        let mut wrong_state =
            NodeState::new(wrong.n_nodes(), data.n_features, &data.means, &precisions).unwrap();
        let l_wrong = wrong_state.prune(&wrong);

        assert!(
            l_true > l_wrong,
            "true topology scored {l_true}, shuffled scored {l_wrong}"
        );

        // The same must hold for the unbalanced generator, where the wrong tree
        // has the wrong shape as well as the wrong labels.
        let data = simulate_unbalanced::<f64>(Some(params)).unwrap();
        let precisions = data.precisions();
        let mut truth_state = NodeState::new(
            data.tree.n_nodes(),
            data.n_features,
            &data.means,
            &precisions,
        )
        .unwrap();
        let l_true = truth_state.prune(&data.tree);
        let flat = Tree::balanced_binary(params.n_leaves, params.branch_length).unwrap();
        let mut flat_state =
            NodeState::new(flat.n_nodes(), data.n_features, &data.means, &precisions).unwrap();
        let l_flat = flat_state.prune(&flat);
        assert!(
            l_true > l_flat,
            "true topology scored {l_true}, balanced scored {l_flat}"
        );
    }

    #[test]
    fn test_per_feature_variance_is_v_after_rescaling() {
        // SPEC.md section 13.1 rescales each feature to variance `v[g]` in
        // untransformed units. The rescale is exact rather than sampled, so the
        // tolerance is a rounding tolerance, not a sampling one.
        let params = SimulationParams {
            n_leaves: 64,
            n_features: 128,
            seed: 11,
            ..Default::default()
        };
        let data = simulate_binary::<f64>(Some(params)).unwrap();
        let (n, p) = (data.n_leaves, data.n_features);
        for g in 0..p {
            let scale = data.variances[g].sqrt();
            let untransformed: Vec<f64> =
                (0..n).map(|i| data.truth[i * p + g] * scale).collect();
            let mean = untransformed.iter().sum::<f64>() / n as f64;
            let var = untransformed
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f64>()
                / n as f64;
            assert_relative_eq!(var, data.variances[g], max_relative = 1e-10);
        }
        // Feature variances are drawn exponential with mean 2, so the sample
        // mean over 128 features should be in the right neighbourhood.
        let mean_v = data.variances.iter().sum::<f64>() / p as f64;
        assert!(
            (mean_v - VARIANCE_MEAN).abs() < 0.6,
            "mean v[g] was {mean_v}"
        );
    }

    #[test]
    fn test_shapes_and_tree_validity() {
        let params = SimulationParams {
            n_leaves: 16,
            n_features: 25,
            seed: 5,
            ..Default::default()
        };
        for data in [
            simulate_binary::<f32>(Some(params)).unwrap(),
            simulate_binary_random_branches::<f32>(Some(params)).unwrap(),
            simulate_unbalanced::<f32>(Some(params)).unwrap(),
        ] {
            assert_eq!(data.n_leaves, 16);
            assert_eq!(data.n_features, 25);
            assert_eq!(data.tree.n_leaves(), 16);
            assert_eq!(data.tree.n_nodes(), 31);
            assert_eq!(data.truth.len(), 16 * 25);
            assert_eq!(data.means.len(), 16 * 25);
            assert_eq!(data.sds.len(), 16 * 25);
            assert_eq!(data.variances.len(), 25);
            for node in data.tree.internal_postorder() {
                assert_eq!(data.tree.children(node).len(), 2);
            }
            assert!(data.sds.iter().all(|&s| s > 0.0));
            assert!(data.precisions().iter().all(|&w| w.is_finite() && w > 0.0));
        }
    }

    #[test]
    fn test_random_branch_lengths_lie_in_the_specified_range() {
        let data = simulate_binary_random_branches::<f64>(Some(SimulationParams {
            n_leaves: 64,
            n_features: 4,
            seed: 8,
            ..Default::default()
        }))
        .unwrap();
        let root = data.tree.root();
        let mut seen_short = false;
        let mut seen_long = false;
        for node in 0..data.tree.n_nodes() as u32 {
            if node == root {
                continue;
            }
            let t = data.tree.branch(node);
            assert!((RANDOM_BRANCH_LO..=RANDOM_BRANCH_HI).contains(&t), "t was {t}");
            seen_short |= t < 1.0;
            seen_long |= t > 1.0;
        }
        assert!(seen_short && seen_long, "branch lengths were not varied");
    }

    #[test]
    fn test_unbalanced_generator_is_actually_unbalanced() {
        let data = simulate_unbalanced::<f64>(Some(SimulationParams {
            n_leaves: 64,
            n_features: 4,
            seed: 13,
            ..Default::default()
        }))
        .unwrap();
        let depths = leaf_depths(&data.tree);
        assert_eq!(depths.len(), 64);
        let lo = depths.iter().copied().min().unwrap_or(0);
        let hi = depths.iter().copied().max().unwrap_or(0);
        // The balanced tree over 64 leaves has every leaf at depth six.
        assert!(hi > lo, "depths were all {lo}, which is the balanced case");
        assert!(hi > 6, "deepest leaf at {hi}, no deeper than balanced");

        let balanced = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: 64,
            n_features: 4,
            seed: 13,
            ..Default::default()
        }))
        .unwrap();
        assert!(leaf_depths(&balanced.tree).iter().all(|&d| d == 6));
        assert!(robinson_foulds(&data.tree, &balanced.tree).unwrap() > 0);
    }

    #[test]
    fn test_rejects_unusable_parameters() {
        let bad_power = SimulationParams {
            n_leaves: 17,
            ..Default::default()
        };
        assert!(matches!(
            simulate_binary::<f64>(Some(bad_power)),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        // The unbalanced generator has no power-of-two requirement.
        assert!(simulate_unbalanced::<f64>(Some(bad_power)).is_ok());

        let no_features = SimulationParams {
            n_features: 0,
            ..Default::default()
        };
        assert!(matches!(
            simulate_binary::<f64>(Some(no_features)),
            Err(BonsaiErrors::EmptyInput { .. })
        ));

        let no_noise = SimulationParams {
            noise_sd: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            simulate_binary::<f64>(Some(no_noise)),
            Err(BonsaiErrors::NonPositiveSd { .. })
        ));
    }

    #[test]
    fn test_noise_level_is_honoured() {
        // The observed means must sit around the truth at the requested scale.
        let params = SimulationParams {
            n_leaves: 64,
            n_features: 200,
            noise_sd: 0.05,
            noise_spread: 1.0,
            ..Default::default()
        };
        let data = simulate_binary::<f64>(Some(params)).unwrap();
        let residual: f64 = data
            .means
            .iter()
            .zip(&data.truth)
            .map(|(m, t)| (m - t) * (m - t))
            .sum::<f64>()
            / data.means.len() as f64;
        assert_relative_eq!(residual.sqrt(), params.noise_sd, max_relative = 0.05);
        assert!(data.sds.iter().all(|&s| (s - params.noise_sd).abs() < 1e-12));
    }
}
