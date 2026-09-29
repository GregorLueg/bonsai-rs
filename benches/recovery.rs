//! Distance recovery of our reconstructions (paper Figs. S8 and S9): how well
//! tree path distances track squared Euclidean distances in the data.
//!
//! The reconstruction can score above the generating tree, whose branch lengths
//! are expected displacements rather than the realised ones.
//!
//! ```sh
//! cargo bench --bench recovery
//! ```

use bonsai_rs::prelude::*;
use bonsai_rs::tree::distance::{MAX_PAIRS, distance_recovery};
use bonsai_rs::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};

fn main() {
    println!(
        "{:>7} {:>6} {:>7} {:>10} {:>10} {:>10} {:>6}",
        "leaves", "feat", "noise", "ceiling", "ours", "vs noisy", "RF"
    );

    for (n, p) in [(64usize, 500usize), (256, 500), (256, 2000)] {
        for noise in [0.1f64, 0.3, 0.5] {
            let d = simulate_binary::<f64>(Some(SimulationParams {
                n_leaves: n,
                n_features: p,
                noise_sd: noise,
                seed: 41,
                ..Default::default()
            }))
            .expect("simulation");

            let data = PreparedData {
                transformed_means: d.means.clone(),
                transformed_precisions: d.precisions(),
                features: (0..p).collect(),
                variances: vec![1.0; p],
                signal_to_noise: vec![f64::INFINITY; p],
                n_cells: n,
                n_features_in: p,
            };

            let out = bonsai_prepared(&data, None, Verbosity::Quiet).expect("bonsai");

            // Against the true positions: the measured ones reward fitting the noise.
            let ceiling = distance_recovery(&d.tree, &d.truth, p, MAX_PAIRS, 0);
            let ours = distance_recovery(&out.tree, &d.truth, p, MAX_PAIRS, 0);
            let fitted = distance_recovery(&out.tree, &d.means, p, MAX_PAIRS, 0);
            let rf = robinson_foulds(&out.tree, &d.tree).expect("rf");

            println!(
                "{n:>7} {p:>6} {noise:>7.1} {ceiling:>10.4} {ours:>10.4} {fitted:>10.4} {rf:>6}"
            );
        }
    }
}
