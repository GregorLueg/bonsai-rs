//! SIMD tiers for the feature-axis kernels.
//!
//! Portable SIMD via the `wide` crate. This is the only file in the crate that
//! names a `wide` type; algorithm code stays generic over [`BonsaiSimd`].
//!
//! ### What is vectorised, and why those two
//!
//! Two kernels: [`edge_newton_simd`], which the bracketed branch-length solve
//! calls 35 to 45 times per candidate pair, and the `f32` binary prune. Nothing
//! else, because nothing else runs often enough to matter.
//!
//! Picking those by call count rather than by how vectorisable they looked is
//! the whole trick. `model::merge::MergeScratch::split_derivative` reads like
//! the hot loop and is not: instrumenting `benches/merge_scan.rs` puts it at
//! **0.07 calls per pair**, because a pair whose total branch length is zero,
//! or whose split is optimal at a bracket end, never reaches the bisection. A
//! `wide::f64x4` tier written for it measured flat, as did batching its
//! reciprocals. The same `f64x4` treatment applied to `edge_newton` took 14 per
//! cent off the merge scan.
//!
//! The compiler will not do this for you. `edge_newton` and `split_derivative`
//! both accumulate into floating-point reductions that LLVM may not reorder, so
//! neither is auto-vectorised: scalar divides on aarch64, and on x86-64 the same
//! at baseline, at `x86-64-v3` and at `x86-64-v4`.
//!
//! ### Lane width is set by the target, not by the source
//!
//! `wide` compiles to the baseline instruction set. On aarch64 that is NEON, so
//! an `f64x4` is two registers and an `f32x8` is two. On x86-64 with no
//! `target-cpu` it is SSE2, and `benches/prune_sweep.rs` built for x86-64 shows
//! it: 104 128-bit operations at the baseline against 22 256-bit ones at
//! `x86-64-v3`. `x86-64-v4` is identical to v3, because an `f32x8` already fills
//! a 256-bit register.
//!
//! So AVX2 would widen both kernels on x86 and AVX-512 would need an `f32x16`
//! and an `f64x8` to reach. Neither is measured; there is no x86 machine here.

use wide::{f32x8, f64x4};

use crate::utils::kernels::prune_binary_scalar;
use crate::utils::traits::BonsaiFloat;

/// Lanes in the `f32` vector type.
const LANES_F32: usize = 8;

/// Vectors accumulated in `f32` lanes before flushing into the `f64` total.
///
/// Sets the trade between logarithm throughput and accumulation error. At 32
/// vectors the block holds 256 features, so the `f32` rounding error inside a
/// block grows as `sqrt(256) * 6e-8`, around `1e-6` relative, which is
/// comfortably below the error already present in `f32` input data. Chosen on
/// that argument, 2026-08-27; no measurement says the exact value matters.
const FLUSH_BLOCKS: usize = 32;

/// Vectorised feature-axis kernels, one implementation per storage type.
///
/// Algorithm code calls these through the [`BonsaiFloat`] bound rather than
/// naming a lane width, so a new tier or a new storage type is a change here
/// and nowhere else.
pub trait BonsaiSimd: Sized + Copy {
    /// Fused prune of a node with exactly two children.
    ///
    /// Semantics are identical to
    /// [`crate::utils::kernels::prune_binary_scalar`]; see that function for
    /// the equations and the argument meanings.
    ///
    /// ### Params
    ///
    /// * `m_k` - Effective means of the first child, length `p`
    /// * `w_k` - Effective precisions of the first child, length `p`
    /// * `t_k` - Branch length above the first child
    /// * `m_l` - Effective means of the second child, length `p`
    /// * `w_l` - Effective precisions of the second child, length `p`
    /// * `t_l` - Branch length above the second child
    /// * `m_out` - Destination for the parent's effective means, length `p`
    /// * `w_out` - Destination for the parent's effective precisions, length `p`
    ///
    /// ### Returns
    ///
    /// The node's loglikelihood contribution.
    #[allow(clippy::too_many_arguments)]
    fn prune_binary_simd(
        m_k: &[Self],
        w_k: &[Self],
        t_k: f64,
        m_l: &[Self],
        w_l: &[Self],
        t_l: f64,
        m_out: &mut [Self],
        w_out: &mut [Self],
    ) -> f64;
}

/// Load eight consecutive `f32` into a vector register.
///
/// ### Params
///
/// * `s` - Slice of at least eight elements
///
/// ### Returns
///
/// The first eight elements as a vector.
#[inline(always)]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Store a vector register into eight consecutive `f32`.
///
/// ### Params
///
/// * `v` - Vector to store
/// * `s` - Destination of at least eight elements
#[inline(always)]
fn store8(v: f32x8, s: &mut [f32]) {
    s[..LANES_F32].copy_from_slice(&v.to_array());
}

impl BonsaiSimd for f64 {
    /// `f64` storage has no vector tier and runs the scalar path.
    ///
    /// No measurement says it should not have one; see the module docs.
    #[inline]
    fn prune_binary_simd(
        m_k: &[f64],
        w_k: &[f64],
        t_k: f64,
        m_l: &[f64],
        w_l: &[f64],
        t_l: f64,
        m_out: &mut [f64],
        w_out: &mut [f64],
    ) -> f64 {
        prune_binary_scalar(m_k, w_k, t_k, m_l, w_l, t_l, m_out, w_out)
    }
}

impl BonsaiSimd for f32 {
    #[inline]
    fn prune_binary_simd(
        m_k: &[f32],
        w_k: &[f32],
        t_k: f64,
        m_l: &[f32],
        w_l: &[f32],
        t_l: f64,
        m_out: &mut [f32],
        w_out: &mut [f32],
    ) -> f64 {
        let n = m_k.len();
        let n_vec = n - n % LANES_F32;

        let one = f32x8::splat(1.0);
        let tk = f32x8::splat(t_k as f32);
        let tl = f32x8::splat(t_l as f32);

        // Lane accumulator flushed into `f64` every FLUSH_BLOCKS vectors, so
        // the `f32` rounding error grows as the square root of the block length
        // rather than of the whole feature axis, while the logarithm still gets
        // to run eight lanes wide.
        let mut acc = 0.0f64;
        let mut lanes = f32x8::splat(0.0);
        let mut since_flush = 0usize;

        let mut i = 0;
        while i < n_vec {
            let mk = load8(&m_k[i..]);
            let wk = load8(&w_k[i..]);
            let ml = load8(&m_l[i..]);
            let wl = load8(&w_l[i..]);

            let wdk = wk / tk.mul_add(wk, one);
            let wdl = wl / tl.mul_add(wl, one);
            let wa = wdk + wdl;
            let inv = one / wa;

            // Associate as `(wdk * inv) * wdl`, never `wdk * wdl * inv`. The
            // first factor is a weight in [0, 1], so nothing can leave range.
            // Forming the bare product first overflows f32 above 1.8e19 per
            // factor and goes subnormal below 1.1e-19, and `ln` of the result
            // is then a signed infinity. The scalar tier is saved from this
            // only by widening to f64 first, which is not a guarantee, so it
            // associates the same way.
            let reduced = (wdk * inv) * wdl;
            let diff = ml - mk;
            lanes += reduced.ln() - reduced * diff * diff;

            // Convex combination, not a ratio of sums; see the scalar kernel.
            store8(diff.mul_add(wdl * inv, mk), &mut m_out[i..]);
            store8(wa, &mut w_out[i..]);

            i += LANES_F32;
            since_flush += 1;
            if since_flush == FLUSH_BLOCKS {
                acc += lanes.reduce_add() as f64;
                lanes = f32x8::splat(0.0);
                since_flush = 0;
            }
        }
        acc += lanes.reduce_add() as f64;

        if n_vec < n {
            acc += 2.0
                * prune_binary_scalar(
                    &m_k[n_vec..],
                    &w_k[n_vec..],
                    t_k,
                    &m_l[n_vec..],
                    &w_l[n_vec..],
                    t_l,
                    &mut m_out[n_vec..],
                    &mut w_out[n_vec..],
                );
        }
        0.5 * acc
    }
}

/// Lanes in the `f64` vector type.
const LANES_F64: usize = 4;

/// Load four consecutive `f64` into a vector register.
///
/// ### Params
///
/// * `s` - Slice of at least four elements
///
/// ### Returns
///
/// The first four elements as a vector.
#[inline(always)]
fn load4(s: &[f64]) -> f64x4 {
    f64x4::from([s[0], s[1], s[2], s[3]])
}

/// One Newton evaluation of the branch-length stationarity condition,
/// vectorised.
///
/// Semantics are identical to [`crate::utils::kernels::edge_newton`]; see that
/// function for the equations.
///
/// This is the kernel the search spends its time in. Instrumenting
/// `benches/merge_scan.rs` gives **35 to 45 calls per candidate pair**, against
/// 0.07 for `model::merge::MergeScratch::split_derivative` and one each for
/// `prep_edge` and the peel: `model::branch::optimise_edge` is a bracketed
/// Newton and every iteration is one pass over the feature axis.
///
/// Four lanes, two lane accumulators, reduced in a fixed order so the result
/// does not depend on how the work was scheduled. Accumulation stays in `f64`.
///
/// Measured 2026-09-12 on an M1 Max against the scalar tier:
///
/// | bench | scalar | here |
/// |---|---|---|
/// | `kernels`, ns per feature | 0.95 | 0.71 |
/// | `merge_scan`, 8192 by 2000, ms | 1365 | 1168 |
/// | `pipeline`, 2048 by 2000, s | 71.4 | 65.7 |
///
/// So a third off the kernel, 14 per cent off the merge scan and 8 per cent off
/// the whole run. Trees and loglikelihoods are unchanged across all ten
/// `pipeline` configurations.
///
/// ### Params
///
/// * `s` - Summed inverse precisions from `prep_edge`, length `p`
/// * `d` - Squared separations from `prep_edge`, length `p`
/// * `t` - Branch length at which to evaluate
///
/// ### Returns
///
/// The pair `(f(t), f'(t))`.
pub fn edge_newton_simd(s: &[f64], d: &[f64], t: f64) -> (f64, f64) {
    let n = s.len();
    let n_vec = n - n % LANES_F64;

    let one = f64x4::splat(1.0);
    let two = f64x4::splat(2.0);
    let vt = f64x4::splat(t);

    let mut vf = f64x4::splat(0.0);
    let mut vfp = f64x4::splat(0.0);

    let mut i = 0;
    while i < n_vec {
        let r = one / (load4(&s[i..]) + vt);
        let dr = load4(&d[i..]) * r;
        vf += r * (one - dr);
        vfp += r * r * (two * dr - one);
        i += LANES_F64;
    }

    let mut f = vf.reduce_add();
    let mut fp = vfp.reduce_add();
    for g in n_vec..n {
        let r = 1.0 / (s[g] + t);
        let dr = d[g] * r;
        f += r * (1.0 - dr);
        fp += r * r * (2.0 * dr - 1.0);
    }
    (f, fp)
}

/// Fused prune of a node with exactly two children, dispatched by storage type.
///
/// ### Params
///
/// See [`BonsaiSimd::prune_binary_simd`].
///
/// ### Returns
///
/// The node's loglikelihood contribution.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn prune_binary<T: BonsaiFloat>(
    m_k: &[T],
    w_k: &[T],
    t_k: f64,
    m_l: &[T],
    w_l: &[T],
    t_l: f64,
    m_out: &mut [T],
    w_out: &mut [T],
) -> f64 {
    T::prune_binary_simd(m_k, w_k, t_k, m_l, w_l, t_l, m_out, w_out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two effective leaves of `p` features, spanning several orders of
    /// magnitude in precision so the logarithm is exercised over a realistic
    /// range rather than a comfortable one.
    fn toy_pair(p: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut m_k = Vec::with_capacity(p);
        let mut w_k = Vec::with_capacity(p);
        let mut m_l = Vec::with_capacity(p);
        let mut w_l = Vec::with_capacity(p);
        for g in 0..p {
            let f = g as f64;
            m_k.push((f * 0.37).sin() * 3.0);
            w_k.push(0.05 + (f * 0.11).cos().abs() * 8.0);
            m_l.push((f * 0.53).cos() * 3.0);
            w_l.push(0.05 + (f * 0.19).sin().abs() * 8.0);
        }
        (m_k, w_k, m_l, w_l)
    }

    #[test]
    fn test_vector_edge_newton_tracks_the_scalar_one() {
        use crate::utils::kernels::edge_newton;

        // A length that is not a multiple of the lane count, so the tail runs,
        // and precisions spanning several orders of magnitude so the reciprocal
        // is exercised over the range a deep tree produces.
        let p = 2053usize;
        let s: Vec<f64> = (0..p)
            .map(|g| 1e-3 * (1.0 + (g as f64 * 0.37).sin().abs() * 1e4))
            .collect();
        let d: Vec<f64> = (0..p)
            .map(|g| (g as f64 * 0.53).cos().abs() * 9.0)
            .collect();

        for t in [0.0f64, 1e-9, 0.83, 17.5, 1e6] {
            let (f_s, fp_s) = edge_newton(&s, &d, t);
            let (f_v, fp_v) = edge_newton_simd(&s, &d, t);
            assert_relative_eq!(f_v, f_s, max_relative = 1e-12);
            assert_relative_eq!(fp_v, fp_s, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_vector_edge_newton_is_deterministic_across_lengths() {
        // Lane count decides which features take the vector path and which fall
        // to the tail, so a per-feature-constant input must give a per-feature
        // constant answer whatever the length. This is the shape of bug
        // `test_extreme_precisions_do_not_leave_f32_range` pins for the prune.
        let mut previous: Option<(f64, f64)> = None;
        for p in [3usize, 4, 7, 8, 64, 1000] {
            let s = vec![0.25f64; p];
            let d = vec![1.5f64; p];
            let (f, fp) = edge_newton_simd(&s, &d, 0.4);
            let per = (f / p as f64, fp / p as f64);
            if let Some((wf, wfp)) = previous {
                assert_relative_eq!(per.0, wf, max_relative = 1e-12);
                assert_relative_eq!(per.1, wfp, max_relative = 1e-12);
            }
            previous = Some(per);
        }
    }

    #[test]
    fn test_wide_ln_is_accurate_over_the_range_the_kernels_see() {
        // Effective precisions span many orders of magnitude in a deep tree, so
        // check the whole plausible range rather than a comfortable slice of it.
        let mut worst = 0.0f64;
        let mut x = 1e-12f32;
        while x < 1e12 {
            let v = f32x8::splat(x).ln().to_array()[0] as f64;
            let want = (x as f64).ln();
            worst = worst.max(((v - want) / want.abs().max(1.0)).abs());
            x *= 1.7;
        }
        assert!(worst < 1e-6, "worst relative ln error {worst:e}");
    }

    #[test]
    fn test_f32_simd_prune_tracks_the_f64_scalar_prune() {
        // A length that is not a multiple of the lane count, so the tail path
        // runs too.
        let p = 4099usize;
        let (m_k, w_k, m_l, w_l) = toy_pair(p);
        let (t_k, t_l) = (0.42, 1.73);

        let mut m_s = vec![0.0f64; p];
        let mut w_s = vec![0.0f64; p];
        let l_s = prune_binary_scalar(&m_k, &w_k, t_k, &m_l, &w_l, t_l, &mut m_s, &mut w_s);

        let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let (mk32, wk32, ml32, wl32) = (f32v(&m_k), f32v(&w_k), f32v(&m_l), f32v(&w_l));
        let mut m_v = vec![0.0f32; p];
        let mut w_v = vec![0.0f32; p];
        let l_v = f32::prune_binary_simd(&mk32, &wk32, t_k, &ml32, &wl32, t_l, &mut m_v, &mut w_v);

        assert_relative_eq!(l_v, l_s, max_relative = 1e-5);
        for g in 0..p {
            assert_relative_eq!(m_v[g] as f64, m_s[g], max_relative = 1e-4);
            assert_relative_eq!(w_v[g] as f64, w_s[g], max_relative = 1e-5);
        }
    }

    #[test]
    fn test_extreme_precisions_do_not_leave_f32_range() {
        // Regression, adversarial review 2026-08-27. Forming `wdk * wdl` before
        // dividing overflowed f32 above 1.8e19 per factor and went subnormal
        // below 1.1e-19, so `reduced.ln()` came back as a signed infinity while
        // the scalar tier, which widens to f64 first, was fine.
        //
        // The lane count is part of the bug, not incidental to it: `p % 8`
        // decides which features take the vector path and which fall to the
        // scalar tail, so the same per-element data gave different answers at
        // different feature counts. Hence the p = 7 against p = 8 comparison.
        for (precision, branch) in [(1e-25f32, 1.0f64), (1e30f32, 1e-30f64)] {
            let mut previous: Option<f64> = None;
            for p in [7usize, 8, 15, 16, 64] {
                let m_k = vec![0.5f32; p];
                let m_l = vec![-0.25f32; p];
                let w = vec![precision; p];
                let mut m_out = vec![0.0f32; p];
                let mut w_out = vec![0.0f32; p];

                let vector = f32::prune_binary_simd(
                    &m_k, &w, branch, &m_l, &w, branch, &mut m_out, &mut w_out,
                );
                assert!(
                    vector.is_finite(),
                    "precision {precision:e}, p = {p}: vector tier gave {vector}"
                );

                let scalar =
                    prune_binary_scalar(&m_k, &w, branch, &m_l, &w, branch, &mut m_out, &mut w_out);
                assert_relative_eq!(vector, scalar, max_relative = 1e-4);

                // Per-feature, so feature counts are comparable to each other.
                let per_feature = vector / p as f64;
                if let Some(want) = previous {
                    assert_relative_eq!(per_feature, want, max_relative = 1e-4);
                }
                previous = Some(per_feature);
            }
        }
    }

    #[test]
    fn test_f32_simd_prune_matches_the_f32_scalar_prune() {
        // Same storage type on both sides, so this isolates the lane arithmetic
        // and the block flushing from the cost of storing in f32. The scalar
        // tier still widens to f64 internally, so what is being pinned is the
        // extra error from keeping the intermediates in f32 lanes.
        //
        // The means are compared on an absolute tolerance. An effective mean
        // that lands near zero between two child means of order one carries
        // absolute error around `3 * f32::EPSILON`, which reads as a large
        // relative error while being irrelevant: everything downstream consumes
        // squared *differences* of means, so absolute accuracy is what the
        // likelihood is sensitive to.
        let p = 2053usize;
        let (m_k, w_k, m_l, w_l) = toy_pair(p);
        let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let (mk32, wk32, ml32, wl32) = (f32v(&m_k), f32v(&w_k), f32v(&m_l), f32v(&w_l));
        let (t_k, t_l) = (0.91, 0.08);

        let mut m_s = vec![0.0f32; p];
        let mut w_s = vec![0.0f32; p];
        let l_s = prune_binary_scalar(&mk32, &wk32, t_k, &ml32, &wl32, t_l, &mut m_s, &mut w_s);

        let mut m_v = vec![0.0f32; p];
        let mut w_v = vec![0.0f32; p];
        let l_v = f32::prune_binary_simd(&mk32, &wk32, t_k, &ml32, &wl32, t_l, &mut m_v, &mut w_v);

        assert_relative_eq!(l_v, l_s, max_relative = 1e-5);
        for g in 0..p {
            assert_relative_eq!(m_v[g], m_s[g], epsilon = 1e-6);
            assert_relative_eq!(w_v[g], w_s[g], max_relative = 1e-6);
        }
    }
}
