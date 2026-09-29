//! Timing harness for one full round of candidate-pair scoring: the wall time
//! of one `n * k` scan (SPEC.md section 10), paid once up front and once per
//! ellipsoid redraw. Phase totals are in `benches/steps.rs`.
//!
//! ```sh
//! cargo bench --bench merge_scan
//! ```

use bonsai_rs::model::merge::{EffLeaf, MergeScratch, score_merge};
use bonsai_rs::utils::rng::splitmix64_at;
use rayon::prelude::*;
use std::time::Instant;

/// Star sizes swept, standing in for the number of root children in a round.
const STAR_SIZES: [usize; 3] = [1024, 4096, 8192];

/// Feature counts swept.
const FEATURE_COUNTS: [usize; 2] = [500, 2000];

/// Neighbours per node, which fixes the `n * k` pair count. Cost scales linearly in `k`.
const NEIGHBOURS: usize = 10;

/// Repeats per configuration; the reported time is the best of these.
const REPEATS: usize = 3;

fn main() {
    println!("threads {}", rayon::current_num_threads());
    println!(
        "{:>7} {:>6} {:>9} {:>10} {:>12} {:>12}",
        "star", "feat", "pairs", "scan_ms", "pairs/s", "us_per_pair"
    );

    for &n in STAR_SIZES.iter() {
        for &p in FEATURE_COUNTS.iter() {
            // Clustered so neighbouring indices are close; random pairs would hit the zero-branch early exit.
            let total = (n * p) as u64;
            let m: Vec<f64> = (0..total)
                .map(|i| {
                    let cell = i / p as u64;
                    let feat = i % p as u64;
                    (cell / 8) as f64 * 0.6 + splitmix64_at(i) * 2.0 + (feat as f64 * 0.01).sin()
                })
                .collect();
            let w: Vec<f64> = (0..total)
                .map(|i| 0.4 + splitmix64_at(total + i) * 2.0)
                .collect();

            // One representative for the peeled rest of the star.
            let m_r: Vec<f64> = (0..p).map(|g| 1.0 + (g as f64 * 0.03).cos()).collect();
            let w_r: Vec<f64> = (0..p)
                .map(|g| 0.7 + 0.2 * (g as f64 * 0.05).sin())
                .collect();

            // Each node against the next NEIGHBOURS nodes.
            let pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|i| ((i + 1)..(i + 1 + NEIGHBOURS).min(n)).map(move |j| (i, j)))
                .collect();

            let mut best = f64::INFINITY;
            let mut checksum = 0.0f64;
            for _ in 0..REPEATS {
                let t0 = Instant::now();
                let sum: f64 = pairs
                    .par_iter()
                    .fold(
                        || (MergeScratch::new(p), 0.0f64),
                        |(mut scratch, acc), &(i, j)| {
                            let k = EffLeaf {
                                m: &m[i * p..i * p + p],
                                w: &w[i * p..i * p + p],
                            };
                            let l = EffLeaf {
                                m: &m[j * p..j * p + p],
                                w: &w[j * p..j * p + p],
                            };
                            let r = EffLeaf { m: &m_r, w: &w_r };
                            let gain = score_merge(k, l, r, 0.5, 0.5, None, &mut scratch)
                                .expect("merge score diverged")
                                .gain;
                            (scratch, acc + gain)
                        },
                    )
                    .map(|(_, acc)| acc)
                    .sum();
                best = best.min(t0.elapsed().as_secs_f64());
                checksum = sum;
            }

            assert!(checksum.is_finite(), "non-finite gain sum");

            println!(
                "{:>7} {:>6} {:>9} {:>10.1} {:>12.0} {:>12.2}",
                n,
                p,
                pairs.len(),
                best * 1e3,
                pairs.len() as f64 / best,
                best * 1e6 / pairs.len() as f64
            );
        }
    }
}
