//! Branch-length optimisation.
//!
//! Implements SPEC.md section 6. Collapsing everything but one edge into an
//! effective leaf on each side reduces the branch-length problem to a
//! one-dimensional root find whose derivative is available in closed form, so
//! there is no call to a general-purpose optimiser anywhere in this crate.

use crate::errors::BonsaiErrors;
use crate::utils::kernels::edge_loglik;
use crate::utils::simd::edge_newton_simd as edge_newton;

////////////
// Consts //
////////////

/// Iteration budget for the safeguarded Newton solve.
///
/// The iteration is quadratically convergent inside the bracket and the bracket
/// halves on every safeguarded step, so this is a runaway guard rather than a
/// working limit. A bracket of any plausible width is exhausted by bisection
/// alone in fewer than 60 halvings of `f64` precision.
const MAX_NEWTON_ITER: usize = 100;

/// Relative convergence tolerance on the branch length.
///
/// Tightened well past what the search needs, because the stopping point of the
/// inner solve should never be what limits agreement between two runs.
const BRANCH_TOL: f64 = 1e-12;

///////////////
// Functions //
///////////////

/// Optimise one branch length given the edge's precomputed constants.
///
/// The loglikelihood of an edge is unimodal in its length: SPEC.md section 6
/// gives `dL/dt = -f(t)/2`, and for a single feature the stationary point sits
/// at `s + t = d`, so each term of `f` changes sign exactly once. The solve is
/// therefore a bracketed root find, with the bracket coming from `prep_edge`.
///
/// A safeguarded Newton is used rather than a plain one: the Newton step is
/// taken when it lands inside the current bracket and makes progress, and a
/// bisection step otherwise. That keeps the quadratic convergence where it
/// applies without ever leaving the bracket, which matters because `f` is not
/// globally monotone. Working directly in `t` rather than in `log t` keeps
/// transcendentals out of the iteration entirely; positivity comes from the
/// bracket rather than from a change of variable.
///
/// Two details decide the iteration count, which is the cost of every merge,
/// placement and branch solve in the crate, and both are measured over the
/// solves the SPR beam makes on a real search:
///
/// - **Where it starts.** `upper` is a maximum over features and sits an order
///   of magnitude above the root, so from `upper / 2` the first three or four
///   steps are bisections. The mean of `d - s` over features is the exact
///   optimum when the features agree and lands inside Newton's basin when they
///   do not. The neighbour's optimum, which the beam could hand down, was
///   tried and is worse, by around half again as many iterations per solve.
/// - **When it stops.** Convergence is tested on the Newton correction before
///   the safeguard sees it. Near the root the correction points exactly at a
///   bracket end, the safeguard refuses it as not strictly inside, and the loop
///   would otherwise bisect a bracket already narrower than the tolerance, one
///   full pass over the features per halving: seven wasted passes out of
///   fifteen on a typical solve. 22.4 iterations per solve before, 10.9 after,
///   5.8 with the mean start as well.
///
/// ### Params
///
/// * `s` - Summed inverse precisions from `prep_edge`, length `p`
/// * `d` - Squared separations from `prep_edge`, length `p`, same length as `s`
/// * `upper` - Upper bracket from `prep_edge`
///
/// ### Returns
///
/// The branch length maximising the edge's loglikelihood, which is zero when
/// the two effective leaves are closer together than their own uncertainty
/// already allows. `RootFindDiverged` if the iteration budget is exhausted,
/// which would mean the bracket was wrong, or if the bracket or the derivative
/// at the origin is not finite, which is reported with `max_iter: 0` because
/// the iteration never started.
pub fn optimise_edge(s: &[f64], d: &[f64], upper: f64) -> Result<f64, BonsaiErrors> {
    debug_assert_eq!(s.len(), d.len(), "prep_edge sizes s and d together");

    if !upper.is_finite() {
        return Err(BonsaiErrors::RootFindDiverged {
            max_iter: 0,
            last_step: upper,
        });
    }

    if upper <= 0.0 {
        return Ok(0.0);
    }
    let f0 = edge_newton(s, d, 0.0).0;
    if !f0.is_finite() {
        return Err(BonsaiErrors::RootFindDiverged {
            max_iter: 0,
            last_step: f0,
        });
    }
    if f0 >= 0.0 {
        return Ok(0.0);
    }

    let (mut lo, mut hi) = (0.0f64, upper);
    let mean: f64 = s.iter().zip(d).map(|(&s_g, &d_g)| d_g - s_g).sum::<f64>() / s.len() as f64;
    let mut t = if mean > 0.0 && mean < upper {
        mean
    } else {
        0.5 * upper
    };
    let mut step = upper;

    for _ in 0..MAX_NEWTON_ITER {
        let (f, fp) = edge_newton(s, d, t);
        if f < 0.0 {
            lo = t;
        } else {
            hi = t;
        }

        let newton = t - f / fp;
        if (newton - t).abs() <= BRANCH_TOL * t.max(BRANCH_TOL) {
            return Ok(t);
        }
        let prev = t;
        if newton > lo && newton < hi && (newton - t).abs() < 0.5 * step {
            step = (newton - t).abs();
            t = newton;
        } else {
            step = 0.5 * (hi - lo);
            t = 0.5 * (lo + hi);
        }

        if (t - prev).abs() <= BRANCH_TOL * t.max(BRANCH_TOL) {
            return Ok(t);
        }
    }

    Err(BonsaiErrors::RootFindDiverged {
        max_iter: MAX_NEWTON_ITER,
        last_step: step,
    })
}

/// The loglikelihood of an edge at its optimal length.
///
/// Convenience for the merge and placement scores, which want the optimised
/// value rather than the length itself.
///
/// ### Params
///
/// * `s` - Summed inverse precisions from `prep_edge`, length `p`
/// * `d` - Squared separations from `prep_edge`, length `p`
/// * `upper` - Upper bracket from `prep_edge`
///
/// ### Returns
///
/// The optimal branch length and the edge loglikelihood there.
pub fn optimise_edge_loglik(s: &[f64], d: &[f64], upper: f64) -> Result<(f64, f64), BonsaiErrors> {
    let t = optimise_edge(s, d, upper)?;
    Ok((t, edge_loglik(s, d, t)))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::kernels::prep_edge;
    use approx::assert_relative_eq;

    /// Effective leaves separated by a controllable amount.
    ///
    /// Both the separation and the precision vary across features, so the
    /// optimum is a genuine compromise between features that disagree rather
    /// than the degenerate case where every feature wants the same answer.
    fn edge(p: usize, separation: f64, precision: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let m_k: Vec<f64> = (0..p).map(|g| (g as f64 * 0.31).sin()).collect();
        let m_l: Vec<f64> = m_k
            .iter()
            .enumerate()
            .map(|(g, x)| x + separation * (1.0 + 0.4 * (g as f64 * 0.17).sin()))
            .collect();
        let w_k: Vec<f64> = (0..p)
            .map(|g| precision * (1.0 + 0.5 * (g as f64 * 0.23).cos()))
            .collect();
        let w_l: Vec<f64> = (0..p)
            .map(|g| precision * (1.0 + 0.5 * (g as f64 * 0.41).sin()))
            .collect();
        let (mut s, mut d) = (vec![0.0; p], vec![0.0; p]);
        let upper = prep_edge(&m_k, &w_k, &m_l, &w_l, &mut s, &mut d);
        (s, d, upper)
    }

    #[test]
    fn test_single_feature_optimum_is_d_minus_s() {
        // With one feature the stationary point is at s + t = d exactly.
        let s = vec![0.4];
        let d = vec![2.9];
        let upper = d[0] - s[0];
        let t = optimise_edge(&s, &d, upper).unwrap();
        assert_relative_eq!(t, d[0] - s[0], max_relative = 1e-10);
    }

    #[test]
    fn test_identical_features_optimum_is_the_common_value() {
        // When every feature agrees on d - s, so does the sum.
        let p = 64;
        let s = vec![0.7; p];
        let d = vec![3.2; p];
        let t = optimise_edge(&s, &d, d[0] - s[0]).unwrap();
        assert_relative_eq!(t, 2.5, max_relative = 1e-10);
    }

    #[test]
    fn test_close_leaves_collapse_the_branch_to_zero() {
        // Leaves nearer to each other than their own error bars want no branch
        // at all, which is exactly how polytomies arise in the search. A
        // separation of 0.3 against precisions around 0.5 puts every feature's
        // squared separation well inside its combined variance.
        let (s, d, upper) = edge(128, 0.3, 0.5);
        assert_eq!(optimise_edge(&s, &d, upper).unwrap(), 0.0);
    }

    #[test]
    fn test_optimum_beats_its_neighbourhood() {
        for separation in [2.0f64, 4.0, 8.0] {
            let (s, d, upper) = edge(200, separation, 1.5);
            let (t, best) = optimise_edge_loglik(&s, &d, upper).unwrap();
            assert!(
                t > 0.0,
                "expected a positive branch at separation {separation}"
            );
            for delta in [-0.2f64, -0.01, 0.01, 0.2] {
                let other = (t + delta).max(0.0);
                assert!(
                    best >= edge_loglik(&s, &d, other),
                    "t={t} was beaten by {other} at separation {separation}"
                );
            }
        }
    }

    #[test]
    fn test_derivative_vanishes_at_the_optimum() {
        let (s, d, upper) = edge(300, 3.0, 0.8);
        let t = optimise_edge(&s, &d, upper).unwrap();
        let (f, _) = edge_newton(&s, &d, t);
        // Scale against the derivative's magnitude at the bracket ends, so the
        // assertion means "indistinguishable from zero" rather than "small".
        let scale = edge_newton(&s, &d, 0.0).0.abs().max(1e-30);
        assert!(f.abs() / scale < 1e-10, "f = {f:e} at t = {t}");
    }

    #[test]
    fn test_more_separation_gives_a_longer_branch() {
        let mut previous = 0.0;
        for separation in [2.0f64, 3.0, 4.0, 6.0, 8.0] {
            let (s, d, upper) = edge(128, separation, 1.0);
            let t = optimise_edge(&s, &d, upper).unwrap();
            assert!(
                t > previous,
                "branch shrank going to separation {separation}"
            );
            previous = t;
        }
    }

    #[test]
    fn test_a_non_finite_edge_is_an_error_and_not_a_branch_length() {
        // Every comparison against a `NaN` is false, so a `NaN` used to skip
        // the early return, then send `hi = t`
        // on every iteration, and come back as `Ok(6.2e-25)`: a garbage branch
        // length reported as a success.
        for (s, d, upper) in [
            (vec![1.0, 1.0], vec![f64::NAN, 4.0], 3.0),
            (vec![f64::NAN, 1.0], vec![9.0, 4.0], 3.0),
            (vec![1.0, 1.0], vec![f64::INFINITY, 4.0], 3.0),
            (vec![1.0, 1.0], vec![9.0, 4.0], f64::NAN),
            (vec![1.0, 1.0], vec![9.0, 4.0], f64::INFINITY),
        ] {
            let out = optimise_edge(&s, &d, upper);
            assert!(
                matches!(out, Err(BonsaiErrors::RootFindDiverged { .. })),
                "s = {s:?}, d = {d:?}, upper = {upper} returned {out:?}"
            );
        }

        // The guards are on the input and not on the answer: a well-posed edge
        // with a large but finite bracket still solves.
        let (s, d, upper) = edge(64, 2.5, 1.0);
        assert!(optimise_edge(&s, &d, upper).is_ok());
        assert_eq!(optimise_edge(&s, &d, 0.0).expect("zero bracket"), 0.0);
    }

    #[test]
    fn test_upper_bracket_from_prep_edge_contains_the_root() {
        let (s, d, upper) = edge(256, 2.5, 1.2);
        let t = optimise_edge(&s, &d, upper).unwrap();
        assert!(t <= upper, "root {t} escaped its bracket {upper}");
        assert!(
            edge_newton(&s, &d, upper).0 >= 0.0,
            "bracket does not bracket"
        );
    }
}
