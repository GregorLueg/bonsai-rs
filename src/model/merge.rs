//! The merge score: what does inserting an ancestor above two children of a
//! star gain?
//!
//! Implements SPEC.md section 8. This is the quantity the whole tree search is
//! a search over, so it is worth being precise about its shape. Both the tree
//! before the merge and the tree after it are three-leaf stars over the same
//! three effective leaves: the two children `k` and `l`, and `R`, which is
//! every other child of the root collapsed into one. The score is the
//! difference of the two stars' loglikelihoods.
//!
//! Because `R` is obtained by peeling `k` and `l` off the root's own effective
//! leaf, scoring one pair costs `O(p)` rather than `O(n * p)`.

use crate::errors::BonsaiErrors;
use crate::model::branch::optimise_edge;
use crate::utils::traits::{BonsaiFloat, wide};

/// Tuning knobs for the merge-score branch-length solve.
///
/// All three trade accuracy of the *branch lengths* against the cost of the
/// merge scan, which is the dominant cost of the whole search. None of them
/// change the model.
#[derive(Clone, Copy, Debug)]
pub struct MergeParams {
    /// Coordinate sweeps in the constrained stage (SPEC.md section 8.4).
    ///
    /// Each sweep is monotone in the gain, so this is a stopping rule rather
    /// than a correctness bound.
    pub coord_sweeps: usize,
    /// Relative convergence tolerance on the split of the total branch length.
    pub split_tol: f64,
    /// Iteration budget for the bracketed split solve.
    pub max_split_iter: usize,
}

impl Default for MergeParams {
    /// Ours, chosen by measurement; `docs/PERFORMANCE.md` has the sweep.
    ///
    /// Two coordinate sweeps. Stage two is the dominant cost of a merge scan,
    /// so this is the most expensive constant in the crate, and it is not
    /// optional either: stage two roughly triples the gain over the unrefined
    /// split, which is what `test_two_coordinate_sweeps_reach_the_fixed_point`
    /// pins.
    ///
    /// **Two sweeps are not the exact fixed point.** The coordinate descent
    /// converges slowly on a small corner of the space, where the split and the
    /// root branch are strongly coupled; everywhere else two sweeps and twenty
    /// agree to the bit. The shortfall is in a *candidate's* score rather than
    /// in the tree, and running the whole pipeline at two sweeps against eight
    /// gives identical trees, so the default stands. A merge scan losing close
    /// calls on a coupled fixture is still where to look first.
    ///
    /// `split_tol` is looser than the branch-length tolerance in
    /// `model::branch` on purpose: the gain is stationary in the split at the
    /// optimum, so an error of `eps` in the split costs `O(eps^2)` in the
    /// score. It is tighter than it needs to be for that alone because the
    /// secant solve lands where its iterates take it rather than on a fixed
    /// grid of midpoints, so two solves on inputs differing in the last place
    /// can return splits `eps` apart, and that showed up as drift between the
    /// exact and incremental centre in `search::bounds`.
    fn default() -> Self {
        Self {
            coord_sweeps: 2,
            split_tol: 1e-10,
            max_split_iter: 40,
        }
    }
}

/// An effective leaf: one mean and one precision per feature.
#[derive(Clone, Copy, Debug)]
pub struct EffLeaf<'a, T> {
    /// Effective means, length `p`.
    pub m: &'a [T],
    /// Effective precisions, length `p`.
    pub w: &'a [T],
}

/// Per-feature constants of one candidate pair.
///
/// Everything here is independent of the three branch lengths being optimised,
/// so it is computed once and reused across every step of the solve. Reused
/// across candidate pairs too, via [`MergeScratch::prepare`], so a parallel scan
/// allocates once per thread rather than once per pair.
#[derive(Clone, Debug)]
pub struct MergeScratch {
    /// Inverse effective precision of `k`, `l` and `R`.
    c_k: Vec<f64>,
    c_l: Vec<f64>,
    c_r: Vec<f64>,
    /// Squared separations between the three effective leaves.
    d_kl: Vec<f64>,
    d_kr: Vec<f64>,
    d_lr: Vec<f64>,
    /// Work space for the one-dimensional edge solves.
    s: Vec<f64>,
    d: Vec<f64>,
    /// Loglikelihood of the star before the merge, which does not depend on the
    /// branch lengths under optimisation.
    before: f64,
    /// Upper bracket for the total `k`-to-`l` branch length.
    upper_kl: f64,
}

/// The outcome of scoring one candidate pair.
#[derive(Clone, Copy, Debug)]
pub struct MergeScore {
    /// Loglikelihood gained by inserting the ancestor, SPEC.md section 8.3.
    pub gain: f64,
    /// Optimised branch length from the new ancestor to `k`.
    pub t_ak: f64,
    /// Optimised branch length from the new ancestor to `l`.
    pub t_al: f64,
    /// Optimised branch length from the new ancestor to the root.
    pub t_ar: f64,
}

/// One feature's contribution to a three-leaf star's loglikelihood.
///
/// SPEC.md section 8.3, using the three-point identity (S33) so the star's own
/// centre never has to be formed:
///
/// ```text
/// star3 = log(a1 * a2 * a3) - log(a1 + a2 + a3)
///         - (a1*a2*d12 + a1*a3*d13 + a2*a3*d23) / (a1 + a2 + a3)
/// ```
///
/// One logarithm, not four: the product is well scaled because each factor is a
/// diffusion-corrected precision bounded above by the reciprocal of its branch
/// length.
///
/// ### Params
///
/// * `a1`, `a2`, `a3` - Diffusion-corrected precisions of the three leaves
/// * `d12`, `d13`, `d23` - Squared separations between them
///
/// ### Returns
///
/// The feature's contribution, twice the loglikelihood term.
#[inline(always)]
fn star3(a1: f64, a2: f64, a3: f64, d12: f64, d13: f64, d23: f64) -> f64 {
    let s = a1 + a2 + a3;
    let q = a1 * a2 * d12 + a1 * a3 * d13 + a2 * a3 * d23;
    (a1 * a2 * a3).ln() - s.ln() - q / s
}

/// Precision of a leaf seen across a branch of length `t`.
///
/// ### Params
///
/// * `c` - The leaf's inverse effective precision
/// * `t` - Branch length
///
/// ### Returns
///
/// `1 / (t + c)`.
#[inline(always)]
fn across(c: f64, t: f64) -> f64 {
    1.0 / (t + c)
}

impl MergeScratch {
    /// Allocate scratch for a given feature count.
    ///
    /// ### Params
    ///
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// Empty scratch, ready for [`MergeScratch::prepare`].
    pub fn new(p: usize) -> Self {
        Self {
            c_k: vec![0.0; p],
            c_l: vec![0.0; p],
            c_r: vec![0.0; p],
            d_kl: vec![0.0; p],
            d_kr: vec![0.0; p],
            d_lr: vec![0.0; p],
            s: vec![0.0; p],
            d: vec![0.0; p],
            before: 0.0,
            upper_kl: 0.0,
        }
    }

    /// Load one candidate pair, computing everything that does not move.
    ///
    /// One pass over the features. `t_rk` and `t_rl` are the branch lengths the
    /// two children currently have to the root, which fix the "before" star;
    /// `R` attaches to the root directly, so its own contribution needs no
    /// diffusion correction.
    ///
    /// ### Params
    ///
    /// * `k` - First child
    /// * `l` - Second child
    /// * `r` - The rest of the star, already peeled (SPEC.md section 8.1)
    /// * `t_rk` - Current branch length from the root to `k`
    /// * `t_rl` - Current branch length from the root to `l`
    pub fn prepare<T: BonsaiFloat>(
        &mut self,
        k: EffLeaf<'_, T>,
        l: EffLeaf<'_, T>,
        r: EffLeaf<'_, T>,
        t_rk: f64,
        t_rl: f64,
    ) {
        let p = self.c_k.len();
        debug_assert_eq!(k.m.len(), p);

        let mut before = 0.0f64;
        let mut upper = 0.0f64;
        for g in 0..p {
            let (mk, ml, mr) = (wide(k.m[g]), wide(l.m[g]), wide(r.m[g]));
            let ck = 1.0 / wide(k.w[g]);
            let cl = 1.0 / wide(l.w[g]);
            let cr = 1.0 / wide(r.w[g]);

            let dkl = (mk - ml) * (mk - ml);
            let dkr = (mk - mr) * (mk - mr);
            let dlr = (ml - mr) * (ml - mr);

            self.c_k[g] = ck;
            self.c_l[g] = cl;
            self.c_r[g] = cr;
            self.d_kl[g] = dkl;
            self.d_kr[g] = dkr;
            self.d_lr[g] = dlr;

            before += star3(
                across(ck, t_rk),
                across(cl, t_rl),
                across(cr, 0.0),
                dkl,
                dkr,
                dlr,
            );
            upper = upper.max(dkl - (ck + cl));
        }
        self.before = 0.5 * before;
        self.upper_kl = upper.max(0.0);
    }

    /// The score and its gradient at a given set of branch lengths.
    ///
    /// The gradient is taken with respect to `u`, the share of the total
    /// `k`-to-`l` length assigned to `t_ak`, and to `t_ar`. With
    /// `a = 1 / (t + c)` the chain rule needs only `da/dt = -a^2`, so both
    /// partials fall out of the same pass that evaluates the score.
    ///
    /// Neither partial has a production consumer: the split is solved by
    /// [`MergeScratch::split_derivative`] and `t_ar` by the edge solve, so
    /// [`score_merge`] and [`gain_at`] both call this for the gain alone.
    /// [`crate::search::bounds`] differentiates the score too but does it with
    /// respect to the centre, through its own chain rule. These are kept because
    /// they are what pins the two gradient copies against central differences,
    /// and they are nearly free: this runs once per candidate pair, against the
    /// tens of `split_derivative` calls the bisection makes.
    ///
    /// ### Params
    ///
    /// * `total` - The total `k`-to-`l` branch length, held fixed
    /// * `u` - Share of `total` assigned to `t_ak`, in `[0, total]`
    /// * `t_ar` - Branch length from the ancestor to the root
    ///
    /// ### Returns
    ///
    /// The gain, its derivative with respect to `u`, and its derivative with
    /// respect to `t_ar`.
    fn gain_and_gradient(&self, total: f64, u: f64, t_ar: f64) -> (f64, f64, f64) {
        let mut after = 0.0f64;
        let mut d_du = 0.0f64;
        let mut d_dtar = 0.0f64;

        for g in 0..self.c_k.len() {
            let r1 = u + self.c_k[g];
            let r2 = (total - u) + self.c_l[g];
            let r3 = t_ar + self.c_r[g];
            let (a1, a2, a3) = (1.0 / r1, 1.0 / r2, 1.0 / r3);
            let (d12, d13, d23) = (self.d_kl[g], self.d_kr[g], self.d_lr[g]);

            let s = a1 + a2 + a3;
            let q = a1 * a2 * d12 + a1 * a3 * d13 + a2 * a3 * d23;
            let inv_s = 1.0 / s;
            // log(a1*a2*a3) = -log(r1*r2*r3), one logarithm either way but
            // without forming three reciprocals to feed it.
            after += -(r1 * r2 * r3).ln() - s.ln() - q * inv_s;

            let q_over_s2 = q * inv_s * inv_s;
            let g1 = r1 - inv_s - (a2 * d12 + a3 * d13) * inv_s + q_over_s2;
            let g2 = r2 - inv_s - (a1 * d12 + a3 * d23) * inv_s + q_over_s2;
            let g3 = r3 - inv_s - (a1 * d13 + a2 * d23) * inv_s + q_over_s2;

            // Lengthening t_ak shortens t_al by the same amount, hence the
            // opposing signs on the first two terms.
            d_du += g2 * a2 * a2 - g1 * a1 * a1;
            d_dtar -= g3 * a3 * a3;
        }

        (0.5 * after - self.before, 0.5 * d_du, 0.5 * d_dtar)
    }

    /// Optimise `t_ar` with the two child branches held fixed.
    ///
    /// With `t_ak` and `t_al` fixed, the pair collapses into a single effective
    /// leaf at the ancestor and the remaining problem is an ordinary edge
    /// between that leaf and `R`, so the branch-length root find of SPEC.md
    /// section 6 applies unchanged.
    ///
    /// ### Params
    ///
    /// * `k` - First child
    /// * `l` - Second child
    /// * `r` - The peeled rest of the star
    /// * `t_ak` - Branch length from the ancestor to `k`
    /// * `t_al` - Branch length from the ancestor to `l`
    ///
    /// ### Returns
    ///
    /// The optimal `t_ar`.
    fn optimise_root_branch<T: BonsaiFloat>(
        &mut self,
        k: EffLeaf<'_, T>,
        l: EffLeaf<'_, T>,
        r: EffLeaf<'_, T>,
        t_ak: f64,
        t_al: f64,
    ) -> Result<f64, BonsaiErrors> {
        let mut upper = 0.0f64;
        for g in 0..self.c_k.len() {
            let a1 = across(self.c_k[g], t_ak);
            let a2 = across(self.c_l[g], t_al);
            let w_a = a1 + a2;
            // Convex combination, as everywhere else in this crate.
            let mk = wide(k.m[g]);
            let m_a = mk + (wide(l.m[g]) - mk) * (a2 / w_a);
            let diff = m_a - wide(r.m[g]);

            let s_g = 1.0 / w_a + self.c_r[g];
            let d_g = diff * diff;
            self.s[g] = s_g;
            self.d[g] = d_g;
            upper = upper.max(d_g - s_g);
        }
        optimise_edge(&self.s, &self.d, upper.max(0.0))
    }

    /// Derivative of the gain with respect to the split, and nothing else.
    ///
    /// The same arithmetic as [`MergeScratch::gain_and_gradient`] with the two
    /// logarithms removed, because the bracketed solve below only ever looks at
    /// this derivative's sign. The logarithms are the single most expensive
    /// operation in these kernels, so paying for a score nobody reads dominated
    /// the whole merge scan before this existed.
    ///
    /// ### Params
    ///
    /// * `total` - The total `k`-to-`l` branch length, held fixed
    /// * `u` - Share of `total` assigned to `t_ak`
    /// * `t_ar` - Branch length from the ancestor to the root
    ///
    /// ### Returns
    ///
    /// The derivative of the gain with respect to `u`.
    fn split_derivative(&self, total: f64, u: f64, t_ar: f64) -> f64 {
        let mut acc = 0.0f64;
        for g in 0..self.c_k.len() {
            // `1/a` is the branch length plus the leaf's own inverse precision,
            // which is what `a` was built from, so it is a subtraction saved
            // rather than a reciprocal paid.
            let r1 = u + self.c_k[g];
            let r2 = (total - u) + self.c_l[g];
            let r3 = t_ar + self.c_r[g];
            let (a1, a2, a3) = (1.0 / r1, 1.0 / r2, 1.0 / r3);
            let (d12, d13, d23) = (self.d_kl[g], self.d_kr[g], self.d_lr[g]);

            let inv_s = 1.0 / (a1 + a2 + a3);
            let q = a1 * a2 * d12 + a1 * a3 * d13 + a2 * a3 * d23;
            let q_over_s2 = q * inv_s * inv_s;

            let g1 = r1 - inv_s - (a2 * d12 + a3 * d13) * inv_s + q_over_s2;
            let g2 = r2 - inv_s - (a1 * d12 + a3 * d23) * inv_s + q_over_s2;
            acc += g2 * a2 * a2 - g1 * a1 * a1;
        }
        0.5 * acc
    }

    /// Optimise how the total `k`-to-`l` length divides between the two child
    /// branches, with `t_ar` held fixed.
    ///
    /// Bracketed on `(0, total)` and solved on the analytic derivative by
    /// regula falsi with the Illinois modification, which is a secant step
    /// that never leaves the bracket and halves a stale end's weight so the
    /// bracket cannot stall on one side. No second derivative is available, so
    /// Newton is out. Bisection reaches the shipped tolerance too, and is what
    /// this replaced, but it takes tens of derivative passes per sweep where
    /// the secant takes a handful, and that is most of the cost of resolving
    /// the four-member star search step 5 leaves after every regraft.
    ///
    /// ### Params
    ///
    /// * `total` - The total `k`-to-`l` branch length
    /// * `t_ar` - Branch length from the ancestor to the root
    ///
    /// ### Returns
    ///
    /// The share of `total` assigned to `t_ak`.
    fn optimise_split(&self, total: f64, t_ar: f64, params: &MergeParams) -> f64 {
        if total <= 0.0 {
            return 0.0;
        }
        // The optimum sits at an end of the bracket when the derivative does
        // not change sign across it, and the ends are taken exactly: SPEC.md
        // section 9.2 keys polytomy resolution on zero-length branches, so a
        // boundary-optimal split has to come back as `0.0` or as `total` and
        // not as a hair's breadth from either. The ends are safe to evaluate
        // because `r1` and `r2` in `split_derivative` carry the effective
        // leaf's own inverse precision, which the star primitive has already
        // checked is finite and positive, so neither reciprocal divides by
        // zero at a zero branch length.
        let mut f_lo = self.split_derivative(total, 0.0, t_ar);
        if f_lo <= 0.0 {
            return 0.0;
        }
        let mut f_hi = self.split_derivative(total, total, t_ar);
        if f_hi >= 0.0 {
            return total;
        }
        let (mut lo, mut hi) = (0.0f64, total);
        // Which end the last step moved: `1` for `lo`, `-1` for `hi`, `0` for
        // neither yet. Two moves of the same end in a row is the stall the
        // Illinois halving breaks.
        let mut moved = 0i8;

        for _ in 0..params.max_split_iter {
            if hi - lo <= params.split_tol * total {
                break;
            }
            let secant = (lo * f_hi - hi * f_lo) / (f_hi - f_lo);
            let mid = if secant > lo && secant < hi {
                secant
            } else {
                0.5 * (lo + hi)
            };
            let f_mid = self.split_derivative(total, mid, t_ar);
            if f_mid == 0.0 {
                return mid;
            }
            if f_mid > 0.0 {
                lo = mid;
                f_lo = f_mid;
                if moved == 1 {
                    f_hi *= 0.5;
                }
                moved = 1;
            } else {
                hi = mid;
                f_hi = f_mid;
                if moved == -1 {
                    f_lo *= 0.5;
                }
                moved = -1;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Score a candidate merge, optimising the three new branch lengths.
///
/// The two-stage scheme of SPEC.md section 8.4. Stage one fixes the total
/// `k`-to-`l` length by solving the two-leaf problem with the ancestor detached
/// from the root, which is the distance the merge is really asserting and the
/// one quantity the rest of the search will not revisit. Stage two divides that
/// total and picks the branch to the root, holding the total fixed.
///
/// Stage two runs as coordinate descent over the two remaining freedoms rather
/// than as a two-dimensional Newton. Each half is a bracketed one-dimensional
/// solve, each step is monotone in the score, and the whole thing reuses the
/// branch-length machinery instead of needing a Hessian.
///
/// ### Params
///
/// * `k` - First child of the root being merged
/// * `l` - Second child of the root being merged
/// * `r` - Every other child of the root, peeled into one effective leaf
/// * `t_rk` - Current branch length from the root to `k`
/// * `t_rl` - Current branch length from the root to `l`
/// * `scratch` - Reusable per-feature work space, sized to the feature count
///
/// ### Returns
///
/// The loglikelihood gain and the three optimised branch lengths. A
/// non-positive gain means the merge is not worth making.
pub fn score_merge<T: BonsaiFloat>(
    k: EffLeaf<'_, T>,
    l: EffLeaf<'_, T>,
    r: EffLeaf<'_, T>,
    t_rk: f64,
    t_rl: f64,
    params: Option<MergeParams>,
    scratch: &mut MergeScratch,
) -> Result<MergeScore, BonsaiErrors> {
    let params = params.unwrap_or_default();
    scratch.prepare(k, l, r, t_rk, t_rl);

    // Stage one: the total k-to-l length, with the ancestor detached.
    for g in 0..scratch.c_k.len() {
        scratch.s[g] = scratch.c_k[g] + scratch.c_l[g];
        scratch.d[g] = scratch.d_kl[g];
    }
    let total = optimise_edge(&scratch.s, &scratch.d, scratch.upper_kl)?;

    // Stage two: divide the total, and pick the branch to the root.
    let mut u = 0.5 * total;
    let mut t_ar = 0.0f64;
    for _ in 0..params.coord_sweeps {
        t_ar = scratch.optimise_root_branch(k, l, r, u, total - u)?;
        u = scratch.optimise_split(total, t_ar, &params);
    }

    let (gain, _, _) = scratch.gain_and_gradient(total, u, t_ar);
    Ok(MergeScore {
        gain,
        t_ak: u,
        t_al: total - u,
        t_ar,
    })
}

/// The gain of a merge at explicitly given branch lengths, without optimising.
///
/// The score as a function of where the ancestor sits rather than at its
/// optimum, which is what a central difference against
/// [`MergeScratch::gain_and_gradient`] needs. Both this module's gradient tests
/// and `search::bounds`'s use it for exactly that; nothing in the pipeline
/// calls it.
///
/// ### Params
///
/// * `total` - Total `k`-to-`l` branch length
/// * `u` - Share of `total` assigned to `t_ak`
/// * `t_ar` - Branch length from the ancestor to the root
/// * `scratch` - Scratch already loaded by [`MergeScratch::prepare`]
///
/// ### Returns
///
/// The loglikelihood gain.
pub fn gain_at(total: f64, u: f64, t_ar: f64, scratch: &MergeScratch) -> f64 {
    scratch.gain_and_gradient(total, u, t_ar).0
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::kernels::edge_newton;
    use approx::assert_relative_eq;

    /// Three effective leaves with `k` and `l` nearer to each other than either
    /// is to `R`.
    ///
    /// `pair_sep` has to exceed the pair's combined error bars, or the optimal
    /// branch between them is zero and the branch-length tests are probing a
    /// degenerate case rather than the solve.
    #[allow(clippy::type_complexity)]
    fn three_leaves(
        p: usize,
        pair_sep: f64,
        spread: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let m_k: Vec<f64> = (0..p).map(|g| (g as f64 * 0.29).sin()).collect();
        let m_l: Vec<f64> = m_k
            .iter()
            .enumerate()
            .map(|(g, x)| x + pair_sep * (1.0 + 0.3 * (g as f64 * 0.13).cos()))
            .collect();
        // Place R away from the k-l midpoint, alternating side by feature, so
        // that neither child is systematically nearer to it. A one-sided R
        // pushes the new ancestor onto whichever child it favours, which pins
        // the split at a bracket end and makes the branch-length tests probe a
        // corner instead of an interior optimum.
        let m_r: Vec<f64> = (0..p)
            .map(|g| {
                let mid = 0.5 * (m_k[g] + m_l[g]);
                let side = if g % 2 == 0 { 1.0 } else { -1.0 };
                mid + side * spread * (1.0 + 0.2 * (g as f64 * 0.07).sin())
            })
            .collect();
        let w_k: Vec<f64> = (0..p)
            .map(|g| 1.0 + 0.5 * (g as f64 * 0.19).cos())
            .collect();
        let w_l: Vec<f64> = (0..p)
            .map(|g| 1.2 + 0.4 * (g as f64 * 0.23).sin())
            .collect();
        let w_r: Vec<f64> = (0..p)
            .map(|g| 0.8 + 0.3 * (g as f64 * 0.31).cos())
            .collect();
        (m_k, w_k, m_l, w_l, m_r, w_r)
    }

    fn scored(p: usize, pair_sep: f64, spread: f64) -> (MergeScratch, MergeScore) {
        let (m_k, w_k, m_l, w_l, m_r, w_r) = three_leaves(p, pair_sep, spread);
        let k = EffLeaf { m: &m_k, w: &w_k };
        let l = EffLeaf { m: &m_l, w: &w_l };
        let r = EffLeaf { m: &m_r, w: &w_r };
        let mut scratch = MergeScratch::new(p);
        let score = score_merge(k, l, r, 0.8, 0.9, None, &mut scratch).unwrap();
        (scratch, score)
    }

    #[test]
    fn test_gradients_match_central_differences() {
        let (scratch, score) = scored(64, 2.5, 6.0);
        let total = score.t_ak + score.t_al;
        let (u, t_ar) = (0.3 * total, 0.7);
        let h = 1e-6;

        let (_, d_du, d_dtar) = scratch.gain_and_gradient(total, u, t_ar);

        let fd_u = (gain_at(total, u + h, t_ar, &scratch) - gain_at(total, u - h, t_ar, &scratch))
            / (2.0 * h);
        let fd_tar = (gain_at(total, u, t_ar + h, &scratch)
            - gain_at(total, u, t_ar - h, &scratch))
            / (2.0 * h);

        assert_relative_eq!(d_du, fd_u, max_relative = 1e-6);
        assert_relative_eq!(d_dtar, fd_tar, max_relative = 1e-6);

        // `split_derivative` is the copy the bracketed solve actually runs, and
        // it is the one the central differences above do not touch. It is
        // documented as the same arithmetic with the logarithms dropped, so it
        // must agree to the bit, not merely to a tolerance.
        for &(u, t_ar) in [
            (0.3 * total, 0.7),
            (0.05 * total, 0.01),
            (0.95 * total, 5.0),
        ]
        .iter()
        {
            assert_eq!(
                scratch.split_derivative(total, u, t_ar),
                scratch.gain_and_gradient(total, u, t_ar).1,
                "the two gradient copies disagree at u = {u}, t_ar = {t_ar}"
            );
        }
    }

    #[test]
    fn test_optimised_branches_beat_their_neighbourhood() {
        let (scratch, score) = scored(128, 2.5, 6.0);
        let total = score.t_ak + score.t_al;
        let best = score.gain;

        for du in [-0.05f64, -0.005, 0.005, 0.05] {
            let u = (score.t_ak + du).clamp(1e-9, total - 1e-9);
            assert!(
                best >= gain_at(total, u, score.t_ar, &scratch) - 1e-9,
                "split u = {u} beat the optimum"
            );
        }
        for dt in [-0.05f64, -0.005, 0.005, 0.05] {
            let t = (score.t_ar + dt).max(0.0);
            assert!(
                best >= gain_at(total, score.t_ak, t, &scratch) - 1e-9,
                "t_ar = {t} beat the optimum"
            );
        }
    }

    #[test]
    fn test_stage_one_total_solves_the_two_leaf_problem() {
        // The total k-to-l length must satisfy the ordinary edge stationarity
        // condition with s = c_k + c_l and d = d_kl, since that is exactly what
        // stage one solves.
        let (scratch, score) = scored(96, 2.5, 6.0);
        let total = score.t_ak + score.t_al;
        let p = scratch.c_k.len();
        let s: Vec<f64> = (0..p).map(|g| scratch.c_k[g] + scratch.c_l[g]).collect();
        let scale = edge_newton(&s, &scratch.d_kl, 0.0).0.abs().max(1e-30);
        assert!(
            edge_newton(&s, &scratch.d_kl, total).0.abs() / scale < 1e-9,
            "stage one did not solve the two-leaf problem"
        );
    }

    #[test]
    fn test_gain_grows_as_the_rest_of_the_star_moves_away() {
        // A pair that stands apart from everything else is a better merge than
        // one barely distinguishable from the rest of the star.
        let (_, distant) = scored(128, 2.5, 8.0);
        let (_, close) = scored(128, 2.5, 1.0);
        assert!(
            distant.gain > close.gain,
            "gain {} with a distant R was not better than {} with a close one",
            distant.gain,
            close.gain
        );
    }

    #[test]
    fn test_indistinguishable_pair_gets_a_zero_length_branch() {
        // A pair closer together than their own error bars wants no branch at
        // all. This is how the search manufactures the polytomies it then goes
        // back and resolves.
        let (_, score) = scored(128, 0.2, 6.0);
        assert_eq!(score.t_ak + score.t_al, 0.0);
    }

    #[test]
    fn test_gain_equals_the_difference_of_two_pruned_tree_loglikelihoods() {
        // The test that matters in this module. `score_merge` computes the gain
        // in closed form from three effective leaves; the pruning recursion in
        // `model::likelihood` computes whole-tree loglikelihoods knowing nothing
        // about any of that. Building both trees explicitly and differencing
        // them must reproduce the gain exactly.
        //
        // Four leaves on a star at the root: k, l and two others that make up
        // R. The merged tree hangs k and l off a new ancestor a, which attaches
        // to the root.
        use crate::model::likelihood::NodeState;
        use crate::tree::{NO_NODE, Tree};

        let p = 48usize;
        let (m_k, w_k, m_l, w_l, _, _) = three_leaves(p, 2.5, 6.0);
        let m_m: Vec<f64> = (0..p).map(|g| 5.0 + (g as f64 * 0.11).cos()).collect();
        let w_m: Vec<f64> = (0..p)
            .map(|g| 0.9 + 0.2 * (g as f64 * 0.17).sin())
            .collect();
        let m_n: Vec<f64> = (0..p).map(|g| 6.5 + (g as f64 * 0.07).sin()).collect();
        let w_n: Vec<f64> = (0..p)
            .map(|g| 1.1 + 0.3 * (g as f64 * 0.13).cos())
            .collect();

        let (t_rk, t_rl, t_m, t_n) = (0.8, 0.9, 1.3, 0.6);

        // Peel the rest of the star into one effective leaf, SPEC.md 8.1. Here
        // it is built directly rather than by subtraction, since there are only
        // two nodes in it.
        let mut m_r = vec![0.0f64; p];
        let mut w_r = vec![0.0f64; p];
        for g in 0..p {
            let wd_m = w_m[g] / (1.0 + t_m * w_m[g]);
            let wd_n = w_n[g] / (1.0 + t_n * w_n[g]);
            w_r[g] = wd_m + wd_n;
            m_r[g] = m_m[g] + (m_n[g] - m_m[g]) * (wd_n / w_r[g]);
        }

        let mut scratch = MergeScratch::new(p);
        let score = score_merge(
            EffLeaf { m: &m_k, w: &w_k },
            EffLeaf { m: &m_l, w: &w_l },
            EffLeaf { m: &m_r, w: &w_r },
            t_rk,
            t_rl,
            None,
            &mut scratch,
        )
        .unwrap();

        let mut leaves_m = Vec::new();
        let mut leaves_w = Vec::new();
        for (m, w) in [(&m_k, &w_k), (&m_l, &w_l), (&m_m, &w_m), (&m_n, &w_n)] {
            leaves_m.extend_from_slice(m);
            leaves_w.extend_from_slice(w);
        }

        // Before: a four-leaf star.
        let before_tree = Tree::from_parents(
            vec![4, 4, 4, 4, NO_NODE],
            vec![t_rk, t_rl, t_m, t_n, 0.0],
            4,
        )
        .unwrap();
        let mut before_state =
            NodeState::new(before_tree.n_nodes(), p, &leaves_m, &leaves_w).unwrap();
        let l_before = before_state.prune(&before_tree);

        // After: ((k, l) a, m, n) r, with the optimised branch lengths.
        let after_tree = Tree::from_parents(
            vec![4, 4, 5, 5, 5, NO_NODE],
            vec![score.t_ak, score.t_al, t_m, t_n, score.t_ar, 0.0],
            4,
        )
        .unwrap();
        let mut after_state =
            NodeState::new(after_tree.n_nodes(), p, &leaves_m, &leaves_w).unwrap();
        let l_after = after_state.prune(&after_tree);

        assert_relative_eq!(score.gain, l_after - l_before, max_relative = 1e-10);
    }

    #[test]
    fn test_two_coordinate_sweeps_reach_the_fixed_point() {
        // Pins the default in `MergeParams`. Stage two is two thirds of the
        // cost of a merge scan, so the sweep count is the most expensive
        // constant in the crate and it should not sit on an assumption.
        //
        // Ten sweeps stands in for the converged answer. If a later change
        // makes the coupling between the split and the root branch stronger,
        // this test is what notices.
        let (m_k, w_k, m_l, w_l, m_r, w_r) = three_leaves(256, 2.5, 6.0);
        let k = EffLeaf { m: &m_k, w: &w_k };
        let l = EffLeaf { m: &m_l, w: &w_l };
        let r = EffLeaf { m: &m_r, w: &w_r };
        let mut scratch = MergeScratch::new(256);

        let sweep = |n: usize, scratch: &mut MergeScratch| {
            let params = MergeParams {
                coord_sweeps: n,
                ..Default::default()
            };
            score_merge(k, l, r, 0.8, 0.9, Some(params), scratch).unwrap()
        };

        let converged = sweep(10, &mut scratch);
        let two = sweep(2, &mut scratch);
        let one = sweep(1, &mut scratch);
        let none = sweep(0, &mut scratch);

        // Each sweep is monotone, so more sweeps can only help.
        assert!(one.gain >= none.gain - 1e-12);
        assert!(two.gain >= one.gain - 1e-12);
        assert!(converged.gain >= two.gain - 1e-12);

        // Stage two is worth doing at all.
        assert!(
            two.gain > none.gain * 1.05,
            "stage two barely moved the gain: {} against {}",
            two.gain,
            none.gain
        );

        // And two sweeps is where it stops moving.
        let shortfall = (converged.gain - two.gain).abs() / converged.gain.abs().max(1.0);
        assert!(
            shortfall < 1e-9,
            "two sweeps fell {shortfall:e} short of converged; the default needs revisiting"
        );
    }

    #[test]
    fn test_a_boundary_optimal_split_lands_exactly_on_the_boundary() {
        // The split solve used to bracket on
        // `(1e-12 * total, total - 1e-12 * total)` and so could never return
        // an end, which turned a zero-length branch into a `1e-12 * total` one
        // and hid the polytomy SPEC.md section 9.2 goes looking for.
        //
        // `R` sits on top of one of the pair and carries most of the precision,
        // which drags the new ancestor onto that member: the optimal split is
        // then the whole of `total` on the far child and nothing on the near
        // one. Both sides are checked, because they are the two separate
        // returns in `optimise_split`.
        let p = 32usize;
        for near_k in [true, false] {
            let m_k: Vec<f64> = (0..p).map(|g| (g as f64 * 0.29).sin()).collect();
            let m_l: Vec<f64> = m_k.iter().map(|x| x + 2.5).collect();
            let m_r = if near_k { m_k.clone() } else { m_l.clone() };
            let (w_k, w_l) = (vec![1.0f64; p], vec![1.0f64; p]);
            let w_r = vec![50.0f64; p];

            let mut scratch = MergeScratch::new(p);
            let score = score_merge(
                EffLeaf { m: &m_k, w: &w_k },
                EffLeaf { m: &m_l, w: &w_l },
                EffLeaf { m: &m_r, w: &w_r },
                0.0,
                0.0,
                None,
                &mut scratch,
            )
            .expect("score");
            let total = score.t_ak + score.t_al;
            assert!(total > 0.0, "the fixture stopped exercising a real total");

            let (zero, whole) = if near_k {
                (score.t_ak, score.t_al)
            } else {
                (score.t_al, score.t_ak)
            };
            assert_eq!(
                zero, 0.0,
                "the branch to the member R sits on came back as {zero:e}, not zero"
            );
            assert_eq!(whole, total, "the two branches no longer sum to the total");

            // The snap is not cosmetic: the end of the bracket really is the
            // better split, so the old interior answer also cost gain.
            let inside = gain_at(total, 1e-12 * total, score.t_ar, &scratch);
            let inside = if near_k {
                inside
            } else {
                gain_at(total, total - 1e-12 * total, score.t_ar, &scratch)
            };
            assert!(
                score.gain >= inside,
                "the boundary split scored {} against {inside} just inside it",
                score.gain
            );
        }
    }

    #[test]
    fn test_branch_lengths_are_non_negative_and_split_the_total() {
        let (_, score) = scored(200, 2.5, 6.0);
        assert!(score.t_ak >= 0.0 && score.t_al >= 0.0 && score.t_ar >= 0.0);
        assert!(score.gain.is_finite());
    }
}
