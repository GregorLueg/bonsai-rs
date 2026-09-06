//! Timing harness for the pruning sweep, and the numerical check against the
//! numpy reference in `reference/bonsai_ref.py`.
//!
//! Both sides generate leaf data from the same counter-based splitmix64 stream,
//! so the loglikelihoods are comparable to the last bit that the arithmetic
//! allows. Run the two and compare:
//!
//! ```sh
//! cargo bench --bench prune_sweep
//! uv run --with numpy reference/bonsai_ref.py --leaves 8192 --features 2000
//! ```
//!
//! Two tree shapes are swept. Balanced binary is the friendly case: sibling
//! rows sit next to each other and the sweep streams. The ladder is the
//! pathological one, one node per level and maximally deep, which is what bounds
//! the damage when a real dataset produces a deep laddery tree.
//!
//! Both storage widths are timed because `f32` is the fastest path and the one
//! most likely to be used at scale, so it needs a measurement at scale of its
//! own rather than an extrapolation from the `f64` column. This bench used to
//! carry two `BlockedState` columns as well; that type was deleted on
//! 2026-09-06 once measurement showed the whole prune is 2.4 per cent of the
//! pipeline, so parallelising it could not matter.
//!
//! Plain `main`, no criterion. The sweep is deterministic and long enough that
//! best-of-N over a handful of repeats is stable.

use bonsai_rs::model::likelihood::NodeState;
use bonsai_rs::tree::Tree;
use bonsai_rs::utils::rng::splitmix64_at;
use std::time::Instant;

/// Leaf counts swept, all powers of two so the balanced binary builder applies.
const LEAF_COUNTS: [usize; 3] = [1024, 4096, 8192];

/// Feature counts swept. 2000 is the scale the paper works at after the
/// signal-to-noise filter.
const FEATURE_COUNTS: [usize; 2] = [500, 2000];

/// Repeats per configuration; the reported time is the best of these.
const REPEATS: usize = 5;

/// Branch length assigned to every edge in the fixture tree.
const FIXTURE_BRANCH: f64 = 0.6;

/// Leaf means and precisions in transformed units, row-major `[leaf][feature]`.
///
/// ### Params
///
/// * `n_leaves` - Number of cells
/// * `n_features` - Number of genes
///
/// ### Returns
///
/// The means and precisions blocks, each `n_leaves * n_features` long.
fn make_fixture(n_leaves: usize, n_features: usize) -> (Vec<f64>, Vec<f64>) {
    let n = (n_leaves * n_features) as u64;
    let means = (0..n).map(|i| splitmix64_at(i) * 4.0 - 2.0).collect();
    let precisions = (n..2 * n).map(|i| 0.25 + splitmix64_at(i) * 3.0).collect();
    (means, precisions)
}

/// Best-of-N wall time for a closure, with a finiteness check on its result.
///
/// The check is not decoration: a sweep that silently did no work would
/// otherwise report as an excellent timing.
///
/// ### Params
///
/// * `f` - Closure returning the loglikelihood
///
/// ### Returns
///
/// The best time in seconds and the loglikelihood.
fn time<F: FnMut() -> f64>(mut f: F) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut out = f64::NAN;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        out = f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    assert!(out.is_finite(), "non-finite loglikelihood");
    (best, out)
}

fn main() {
    println!("threads {}", rayon::current_num_threads());
    println!(
        "{:>9} {:>7} {:>6} {:>9} {:>9} {:>7} {:>22}",
        "shape", "leaves", "feat", "f64_ms", "f32_ms", "f32_x", "loglik_f64"
    );

    for &n_leaves in LEAF_COUNTS.iter() {
        for &n_features in FEATURE_COUNTS.iter() {
            let (means, precisions) = make_fixture(n_leaves, n_features);
            let means32: Vec<f32> = means.iter().map(|&x| x as f32).collect();
            let precisions32: Vec<f32> = precisions.iter().map(|&x| x as f32).collect();

            for (shape, tree) in [
                (
                    "balanced",
                    Tree::balanced_binary(n_leaves, FIXTURE_BRANCH).unwrap(),
                ),
                ("ladder", Tree::ladder(n_leaves, FIXTURE_BRANCH).unwrap()),
            ] {
                let n = tree.n_nodes();
                let mut flat = NodeState::new(n, n_features, &means, &precisions).unwrap();
                let (seq, loglik) = time(|| flat.prune(&tree));

                let mut flat32 = NodeState::new(n, n_features, &means32, &precisions32).unwrap();
                let (seq32, loglik32) = time(|| flat32.prune(&tree));
                // `f32` storage still accumulates in `f64`, so the two agree far
                // better than `f32` epsilon; loose enough to pass, tight enough
                // that a storage path gone wrong fails.
                assert!((loglik32 - loglik).abs() <= 1e-4 * loglik.abs());

                println!(
                    "{:>9} {:>7} {:>6} {:>9.3} {:>9.3} {:>7.2} {:>22.12e}",
                    shape,
                    n_leaves,
                    n_features,
                    seq * 1e3,
                    seq32 * 1e3,
                    seq / seq32,
                    loglik
                );
            }
        }
    }
}
