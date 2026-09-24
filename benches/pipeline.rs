//! End-to-end scaling of the whole seven-step search.
//!
//! This is the number the project was justified on. The original criterion was
//! thirty thousand cells by two thousand features in under an hour on one
//! machine; everything measured before this point was the likelihood kernel
//! alone, with no search around it.
//!
//! Reports wall time and Robinson-Foulds to the generating tree, so a
//! configuration that got fast by getting worse is visible rather than
//! flattering.
//!
//! **Run on a quiet machine.** Check `uptime` first.
//!
//! ```sh
//! cargo bench --bench pipeline
//! ```

use bonsai_rs::bonsai::bonsai_prepared;
use bonsai_rs::ingest::PreparedData;
use bonsai_rs::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};
use std::time::Instant;

/// Leaf counts swept, powers of two for the binary generator.
const LEAVES: [usize; 5] = [128, 256, 512, 1024, 2048];

/// Feature counts swept.
const FEATURES: [usize; 2] = [200, 2000];

/// Measurement noise, in transformed units. 0.3 sits inside the range where the
/// greedy step recovers the topology well, so the timing is of a search doing
/// real work rather than one thrashing on noise.
const NOISE: f64 = 0.3;

/// Wall-clock ceiling per configuration. Past this the sweep stops growing,
/// so an unattended run cannot turn into an overnight one.
const BUDGET_SECONDS: f64 = 240.0;

fn main() {
    println!(
        "{:>7} {:>6} {:>10} {:>12} {:>9} {:>8} {:>14}",
        "leaves", "feat", "seconds", "per cell ms", "RF", "of max", "loglik"
    );

    for &p in FEATURES.iter() {
        for &n in LEAVES.iter() {
            let data = simulate_binary::<f64>(Some(SimulationParams {
                n_leaves: n,
                n_features: p,
                noise_sd: NOISE,
                seed: 31,
                ..Default::default()
            }))
            .expect("simulation");

            // `simulate` already returns transformed units, which is what the
            // pipeline works in, so ingest is bypassed here on purpose: this
            // measures the search, not the feature selection.
            let prepared = PreparedData {
                transformed_means: data.means.clone(),
                transformed_precisions: data.precisions(),
                features: (0..p).collect(),
                variances: vec![1.0; p],
                signal_to_noise: vec![f64::INFINITY; p],
                n_cells: n,
                n_features_in: p,
            };

            let t0 = Instant::now();
            let out = bonsai_prepared(&prepared, None).expect("pipeline");
            let secs = t0.elapsed().as_secs_f64();

            let rf = robinson_foulds(&out.tree, &data.tree).expect("rf");
            assert!(out.loglik.is_finite(), "non-finite loglikelihood");

            println!(
                "{n:>7} {p:>6} {secs:>10.2} {:>12.2} {rf:>9} {:>8} {:>14.1}",
                secs * 1e3 / n as f64,
                2 * (n - 3),
                out.loglik
            );

            if secs > BUDGET_SECONDS {
                println!("         (budget reached, stopping this feature count)");
                break;
            }
        }
    }
}
