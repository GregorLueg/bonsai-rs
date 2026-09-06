//! SIMD tiers for the feature-axis kernels.
//!
//! Portable SIMD via the `wide` crate. This is the only file in the crate that
//! names a `wide` type; algorithm code stays generic over [`BonsaiSimd`].
//!
//! The kernels here are logarithm bound, not bandwidth bound: `ln` costs about
//! 3.3 times everything else in `prune_binary` put together. That single fact
//! decides which tiers exist.
//!
//! ### Why `f64` is scalar here
//!
//! Measured on Apple M-series, 2026-08-27, 4.2M elements, best of five:
//!
//! | path | throughput |
//! |---|---|
//! | scalar `f64::ln` | 365 Melem/s |
//! | `wide::f64x4::ln` | 367 Melem/s |
//! | `wide::f32x8::ln` | 759 Melem/s |
//! | same loop with no `ln` at all | 1211 Melem/s |
//!
//! macOS libm's `log` is already about as fast as a four-lane polynomial, so a
//! `f64x4` tier bought nothing: wiring one into the pruning sweep made it
//! *slower*, 78.6 ms against 73.7 ms at 8192 leaves by 2000 features, because
//! the loads and stores cost more than the logarithm saved. It was removed. The
//! `f32x8` tier is a genuine 2.07x on the dominant term and stays.
//!
//! ### What users on x86-64 actually get
//!
//! Those numbers are from Apple Silicon, where the NEON baseline is 128 bits
//! and is always available. **On x86-64 the baseline is SSE2**, also 128 bits,
//! because a crate shipped to crates.io cannot be built with
//! `-C target-cpu=native` without handing an illegal-instruction crash to
//! anyone whose machine is older than the build machine's. So `f32x8` there
//! compiles to a pair of SSE2 registers rather than one AVX2 register, and the
//! 2.07x above is an aarch64 measurement that has not been reproduced on x86.
//!
//! Do not quote it as an x86 number. When someone measures on x86-64, the
//! options are, in order: runtime dispatch on `is_x86_feature_detected!` around
//! an AVX2 arm; raising the baseline to `x86-64-v3` and documenting the
//! requirement; or accepting SSE2 and saying so.
//!
//! Two other x86-only traps that do not reproduce on a Mac. Denormal floats can
//! be an order of magnitude slower there, and log-transformed near-zero
//! expression values drift straight into that range, so a kernel that is fine
//! on synthetic data and inexplicably slow on real data wants flush-to-zero
//! checked first. And glibc's scalar `log` is weaker than macOS libm's, so the
//! `f64x4` experiment that failed here may well pay there; the numbers to beat
//! are in the table.
//!
//! ### Accuracy
//!
//! `wide`'s `ln` is a polynomial approximation, not correctly rounded. The tests
//! below pin it against `f64::ln` over the range these kernels actually see, and
//! the pruning tests pin the assembled result against the scalar path.

use wide::f32x8;

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
    /// `f64` storage runs the scalar path, on measurement rather than by
    /// oversight. See the module docs.
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
