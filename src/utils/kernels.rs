//! The inner kernels: every loop in this crate that runs over features.
//!
//! Each function here corresponds to one equation in `docs/SPEC.md` and is
//! written as a single fused pass over the feature axis. They are deliberately
//! sequential: parallelism in this crate lives one level up, over candidate
//! pairs in a merge round, so a nested rayon fan-out here would only
//! oversubscribe.
//!
//! Storage is `T`, accumulation is always `f64`. See `BonsaiFloat`.
//!
//! These are the scalar tiers and the reference the vector tiers in
//! `utils::simd` are pinned against.

use crate::utils::traits::{BonsaiFloat, narrow, wide};

////////////////////////
// Pruning recursion  //
////////////////////////

/// Fused prune of a node with exactly two children, scalar tier.
///
/// The dominant case: Bonsai trees are binary apart from polytomies that the
/// search is actively removing. Computes the diffusion-corrected precisions,
/// the parent's effective mean and precision, and the node's loglikelihood
/// contribution in one pass over the features.
///
/// Implements the two-child case of SPEC.md sections 4 and 5, using the
/// pairwise form of the quadratic term (identity S33) so that no large
/// cancelling difference is ever formed:
///
/// ```text
/// wd_k    = w_k / (1 + t_k * w_k)
/// w_out   = wd_k + wd_l
/// reduced = wd_k * wd_l / w_out
/// m_out   = m_k + (m_l - m_k) * wd_l / w_out
/// contrib = 1/2 * sum_g [ log reduced - reduced * (m_l - m_k)^2 ]
/// ```
///
/// Two rewrites of the SI form are load-bearing and are explained at their use
/// sites: the three logarithms collapse into one, and the effective mean is a
/// convex combination rather than a ratio of sums.
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
/// The loglikelihood contribution of this node, up to the additive constants
/// dropped in SPEC.md section 3.
#[allow(clippy::too_many_arguments)]
pub fn prune_binary_scalar<T: BonsaiFloat>(
    m_k: &[T],
    w_k: &[T],
    t_k: f64,
    m_l: &[T],
    w_l: &[T],
    t_l: f64,
    m_out: &mut [T],
    w_out: &mut [T],
) -> f64 {
    debug_assert_eq!(m_k.len(), w_k.len());
    debug_assert_eq!(m_k.len(), m_l.len());
    debug_assert_eq!(m_k.len(), m_out.len());
    debug_assert_eq!(m_k.len(), w_out.len());

    let mut acc = 0.0f64;
    for g in 0..m_k.len() {
        let (mk, wk) = (wide(m_k[g]), wide(w_k[g]));
        let (ml, wl) = (wide(m_l[g]), wide(w_l[g]));

        let wdk = wk / (1.0 + t_k * wk);
        let wdl = wl / (1.0 + t_l * wl);
        let wa = wdk + wdl;
        let inv = 1.0 / wa;

        // One log, not three: log(a) + log(b) - log(a+b) = log(a*b/(a+b)).
        // The argument is the pair's reduced precision, bounded above by
        // min(wdk, wdl), so it is as well scaled as the inputs are and cannot
        // manufacture an overflow the separate logs would have avoided.
        let reduced = (wdk * inv) * wdl;
        let diff = ml - mk;
        acc += reduced.ln() - reduced * diff * diff;

        // The effective mean as a convex combination rather than
        // `(wdk*mk + wdl*ml)/wa`. Algebraically the same, but `wdl*inv` lies in
        // `[0, 1]` so the result is pinned between the two child means and
        // cannot cancel when they differ in sign. That matters: the mean feeds
        // straight back into the recursion, so an error here compounds up the
        // tree.
        m_out[g] = narrow(diff.mul_add(wdl * inv, mk));
        w_out[g] = narrow(wa);
    }
    0.5 * acc
}

/// Prune of a node with an arbitrary number of children.
///
/// The polytomy case. Two passes over the features: one to accumulate the
/// effective precision and mean, one for the quadratic term. The second pass
/// uses the direct form `sum_k wd_k * (m_a - m_k)^2` rather than the algebraic
/// shortcut `sum_k wd_k * m_k^2 - w_a * m_a^2`, because the shortcut differences
/// two large similar numbers.
///
/// ### Params
///
/// * `children` - Effective means, effective precisions and upstream branch
///   length of each child
/// * `m_out` - Destination for the parent's effective means, length `p`
/// * `w_out` - Destination for the parent's effective precisions, length `p`
/// * `scratch` - Scratch space of length `p * children.len()` holding the
///   diffusion-corrected precisions between the two passes
///
/// ### Returns
///
/// The loglikelihood contribution of this node.
pub fn prune_general<T: BonsaiFloat>(
    children: &[(&[T], &[T], f64)],
    m_out: &mut [T],
    w_out: &mut [T],
    scratch: &mut [f64],
) -> f64 {
    let p = m_out.len();
    let n_child = children.len();
    debug_assert!(n_child >= 2);
    debug_assert_eq!(scratch.len(), p * n_child);

    let mut acc = 0.0f64;

    // Pass one: diffusion-corrected precisions, effective precision and mean.
    // The mean accumulates as a running weighted average rather than a ratio of
    // sums, for the same conditioning reason as in `prune_binary_scalar`: every
    // partial value stays inside the convex hull of the child means.
    for g in 0..p {
        let mut wa = 0.0f64;
        let mut ma = 0.0f64;
        for (c, &(mc, wc, tc)) in children.iter().enumerate() {
            let w = wide(wc[g]);
            let wd = w / (1.0 + tc * w);
            scratch[c * p + g] = wd;
            wa += wd;
            ma += (wide(mc[g]) - ma) * (wd / wa);
            acc += wd.ln();
        }
        acc -= wa.ln();
        m_out[g] = narrow(ma);
        w_out[g] = narrow(wa);
    }

    // Pass two: the quadratic term against the settled parent mean.
    for g in 0..p {
        let ma = wide(m_out[g]);
        for (c, &(mc, _, _)) in children.iter().enumerate() {
            let diff = ma - wide(mc[g]);
            acc -= scratch[c * p + g] * diff * diff;
        }
    }

    0.5 * acc
}

///////////////////////////
// Branch length kernels //
///////////////////////////

/// Prepare the per-feature constants of an edge between two effective leaves.
///
/// Implements the definitions above SPEC.md section 6's `L(t)`:
///
/// ```text
/// s[g] = 1/w_k[g] + 1/w_l[g]
/// d[g] = (m_k[g] - m_l[g])^2
/// ```
///
/// Both are independent of the branch length, so they are computed once and
/// reused across every step of the root find.
///
/// ### Params
///
/// * `m_k` - Effective means on one side of the edge, length `p`
/// * `w_k` - Effective precisions on one side, length `p`
/// * `m_l` - Effective means on the other side, length `p`
/// * `w_l` - Effective precisions on the other side, length `p`
/// * `s` - Destination for the summed inverse precisions, length `p`
/// * `d` - Destination for the squared separations, length `p`
///
/// ### Returns
///
/// `max(0, max_g (d[g] - s[g]))`, which brackets the optimal branch length from
/// above. Beyond it every term of the stationarity condition is positive, so
/// the root cannot lie further out. Computed here because it is free on a pass
/// that already touches both arrays.
pub fn prep_edge<T: BonsaiFloat>(
    m_k: &[T],
    w_k: &[T],
    m_l: &[T],
    w_l: &[T],
    s: &mut [f64],
    d: &mut [f64],
) -> f64 {
    let mut upper = 0.0f64;
    for g in 0..m_k.len() {
        let s_g = 1.0 / wide(w_k[g]) + 1.0 / wide(w_l[g]);
        let diff = wide(m_k[g]) - wide(m_l[g]);
        let d_g = diff * diff;
        s[g] = s_g;
        d[g] = d_g;
        upper = upper.max(d_g - s_g);
    }
    upper
}

/// One Newton evaluation of the branch-length stationarity condition.
///
/// The crate's hottest kernel. With `r = 1/(s[g] + t)`, SPEC.md section 6 gives
/// `dL/dt = -1/2 * f(t)` where
///
/// ```text
/// f(t)  = sum_g r * (1 - d[g] * r)
/// f'(t) = sum_g r^2 * (2 * d[g] * r - 1)
/// ```
///
/// so the branch length that maximises the loglikelihood is the root of `f`.
/// No logarithm appears: one reciprocal, one fused multiply-add and a handful of
/// multiplies per feature. `edge_loglik` is the only place a log is paid, and it
/// is called once per solve rather than once per iteration.
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
#[inline]
pub fn edge_newton(s: &[f64], d: &[f64], t: f64) -> (f64, f64) {
    let mut f = 0.0f64;
    let mut fp = 0.0f64;
    for g in 0..s.len() {
        let r = 1.0 / (s[g] + t);
        let dr = d[g] * r;
        f += r * (1.0 - dr);
        fp += r * r * (2.0 * dr - 1.0);
    }
    (f, fp)
}

/// The loglikelihood contribution of a single edge at a given branch length.
///
/// SPEC.md section 6:
///
/// ```text
/// L(t) = -1/2 * sum_g [ log(s[g] + t) + d[g] / (s[g] + t) ]
/// ```
///
/// ### Params
///
/// * `s` - Summed inverse precisions from `prep_edge`, length `p`
/// * `d` - Squared separations from `prep_edge`, length `p`
/// * `t` - Branch length at which to evaluate
///
/// ### Returns
///
/// The edge's loglikelihood contribution, up to the dropped additive constants.
pub fn edge_loglik(s: &[f64], d: &[f64], t: f64) -> f64 {
    let mut acc = 0.0f64;
    for g in 0..s.len() {
        let u = s[g] + t;
        acc += u.ln() + d[g] / u;
    }
    -0.5 * acc
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reference implementation of the two-child prune, written the obvious way
    /// with the direct quadratic form, to pin the fused pairwise version.
    fn prune_binary_naive(
        m_k: &[f64],
        w_k: &[f64],
        t_k: f64,
        m_l: &[f64],
        w_l: &[f64],
        t_l: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let p = m_k.len();
        let mut m_out = vec![0.0; p];
        let mut w_out = vec![0.0; p];
        let mut acc = 0.0;
        for g in 0..p {
            let wdk = w_k[g] / (1.0 + t_k * w_k[g]);
            let wdl = w_l[g] / (1.0 + t_l * w_l[g]);
            let wa = wdk + wdl;
            let ma = (wdk * m_k[g] + wdl * m_l[g]) / wa;
            acc += wdk.ln() + wdl.ln()
                - wa.ln()
                - wdk * (ma - m_k[g]).powi(2)
                - wdl * (ma - m_l[g]).powi(2);
            m_out[g] = ma;
            w_out[g] = wa;
        }
        (m_out, w_out, 0.5 * acc)
    }

    fn toy_edge() -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>, f64) {
        let m_k = vec![0.3, -1.2, 2.0, 0.0, 5.5];
        let w_k = vec![1.0, 0.5, 4.0, 2.5, 0.2];
        let m_l = vec![-0.7, 0.4, 2.4, 1.0, 5.0];
        let w_l = vec![3.0, 1.5, 0.8, 1.0, 7.0];
        (m_k, w_k, 0.37, m_l, w_l, 1.9)
    }

    #[test]
    fn test_prune_binary_matches_direct_quadratic_form() {
        let (m_k, w_k, t_k, m_l, w_l, t_l) = toy_edge();
        let (m_ref, w_ref, l_ref) = prune_binary_naive(&m_k, &w_k, t_k, &m_l, &w_l, t_l);

        let mut m_out = vec![0.0f64; m_k.len()];
        let mut w_out = vec![0.0f64; m_k.len()];
        let l = prune_binary_scalar(&m_k, &w_k, t_k, &m_l, &w_l, t_l, &mut m_out, &mut w_out);

        assert_relative_eq!(l, l_ref, epsilon = 1e-12);
        for g in 0..m_k.len() {
            assert_relative_eq!(m_out[g], m_ref[g], epsilon = 1e-12);
            assert_relative_eq!(w_out[g], w_ref[g], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_prune_general_agrees_with_prune_binary_on_two_children() {
        let (m_k, w_k, t_k, m_l, w_l, t_l) = toy_edge();
        let p = m_k.len();

        let mut m_bin = vec![0.0f64; p];
        let mut w_bin = vec![0.0f64; p];
        let l_bin = prune_binary_scalar(&m_k, &w_k, t_k, &m_l, &w_l, t_l, &mut m_bin, &mut w_bin);

        let children: Vec<(&[f64], &[f64], f64)> = vec![
            (m_k.as_slice(), w_k.as_slice(), t_k),
            (m_l.as_slice(), w_l.as_slice(), t_l),
        ];
        let mut m_gen = vec![0.0f64; p];
        let mut w_gen = vec![0.0f64; p];
        let mut scratch = vec![0.0f64; p * 2];
        let l_gen = prune_general(&children, &mut m_gen, &mut w_gen, &mut scratch);

        assert_relative_eq!(l_bin, l_gen, epsilon = 1e-12);
        for g in 0..p {
            assert_relative_eq!(m_bin[g], m_gen[g], epsilon = 1e-12);
            assert_relative_eq!(w_bin[g], w_gen[g], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_edge_newton_matches_central_differences() {
        let (m_k, w_k, _, m_l, w_l, _) = toy_edge();
        let p = m_k.len();
        let mut s = vec![0.0; p];
        let mut d = vec![0.0; p];
        prep_edge(&m_k, &w_k, &m_l, &w_l, &mut s, &mut d);

        // f is -2 dL/dt, so check it against a central difference of the
        // loglikelihood, and f' against a central difference of f.
        let t = 0.83;
        let h = 1e-6;
        let (f, fp) = edge_newton(&s, &d, t);

        let dl = (edge_loglik(&s, &d, t + h) - edge_loglik(&s, &d, t - h)) / (2.0 * h);
        assert_relative_eq!(-0.5 * f, dl, epsilon = 1e-7);

        let f_hi = edge_newton(&s, &d, t + h).0;
        let f_lo = edge_newton(&s, &d, t - h).0;
        assert_relative_eq!(fp, (f_hi - f_lo) / (2.0 * h), epsilon = 1e-6);
    }

    #[test]
    fn test_edge_loglik_single_feature_peaks_where_predicted() {
        // For one feature, L(t) = -1/2 [log(s+t) + d/(s+t)] is maximised at
        // s + t = d, so t* = d - s when that is positive.
        let s = vec![0.4];
        let d = vec![2.9];
        let t_star = d[0] - s[0];

        let peak = edge_loglik(&s, &d, t_star);
        assert!(peak > edge_loglik(&s, &d, t_star - 0.1));
        assert!(peak > edge_loglik(&s, &d, t_star + 0.1));
        assert_relative_eq!(edge_newton(&s, &d, t_star).0, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_f32_storage_tracks_f64_storage() {
        let (m_k, w_k, t_k, m_l, w_l, t_l) = toy_edge();
        let p = m_k.len();

        let mut m64 = vec![0.0f64; p];
        let mut w64 = vec![0.0f64; p];
        let l64 = prune_binary_scalar(&m_k, &w_k, t_k, &m_l, &w_l, t_l, &mut m64, &mut w64);

        let m_k32: Vec<f32> = m_k.iter().map(|&x| x as f32).collect();
        let w_k32: Vec<f32> = w_k.iter().map(|&x| x as f32).collect();
        let m_l32: Vec<f32> = m_l.iter().map(|&x| x as f32).collect();
        let w_l32: Vec<f32> = w_l.iter().map(|&x| x as f32).collect();
        let mut m32 = vec![0.0f32; p];
        let mut w32 = vec![0.0f32; p];
        let l32 = prune_binary_scalar(&m_k32, &w_k32, t_k, &m_l32, &w_l32, t_l, &mut m32, &mut w32);

        assert_relative_eq!(l32, l64, epsilon = 1e-5);
    }
}
