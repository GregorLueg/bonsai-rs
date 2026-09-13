//! Distance recovery of our reconstructions.
//!
//! The paper's Fig. S8 and S9 metric: how well tree path distances track
//! squared Euclidean distances in the data. This is the property Bonsai claims
//! over UMAP and tSNE, and the one Robinson-Foulds cannot see, since a tree with
//! the right topology and useless branch lengths scores RF 0.
//!
//! Reference figures, `simulate_binary`, seed 41, correlation against the
//! **true** positions:
//!
//! | cells | features | noise | generating tree | ours | RF |
//! |---|---|---|---|---|---|
//! | 64 | 500 | 0.1 | 0.9650 | 0.9804 | 0 |
//! | 64 | 500 | 0.5 | 0.9650 | 0.9716 | 0 |
//! | 256 | 500 | 0.1 | 0.9418 | 0.9628 | 0 |
//! | 256 | 500 | 0.5 | 0.9418 | 0.9527 | 0 |
//! | 256 | 2000 | 0.1 | 0.9817 | 0.9901 | 0 |
//! | 256 | 2000 | 0.5 | 0.9817 | 0.9875 | 0 |
//!
//! ### The generating tree is not a ceiling, and that is not a bug
//!
//! Our reconstruction scores above it. The generating tree carries the branch
//! lengths that *produced* the data, and those are diffusion times: the
//! **expected** squared displacement. What was realised differs from its
//! expectation by chance. Our branch lengths are fitted to what actually
//! happened, so they describe the realised structure better than the parameters
//! that generated it do. Recovering more than ground truth on this metric is
//! therefore expected, not a red flag, and the topology is exact anyway.
//!
//! Note also the third column of the original run: correlating against the
//! *measured* positions instead flatters our tree further, because its branch
//! lengths were optimised on exactly that noise. True positions are the honest
//! comparison and the one the paper makes.
//!
//! ### Blessing of dimensionality
//!
//! At 256 cells, recovery goes 0.9418 to 0.9817 as features go 500 to 2000,
//! which is the effect the paper reports in Figs. S12 and S13 and argues is why
//! trees do better in high dimensions rather than worse.
//!
//! ```sh
//! cargo bench --bench recovery
//! ```

use bonsai_rs::bonsai::bonsai_prepared;
use bonsai_rs::ingest::PreparedData;
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

            let out = bonsai_prepared(&data, None).expect("bonsai");

            // Against the TRUE positions, not the measured ones. Correlating
            // against the measurements rewards a tree for fitting their noise,
            // and our branch lengths were optimised on exactly those, so it
            // beats the generating tree there. The question worth asking is how
            // well the structure underneath was recovered.
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
