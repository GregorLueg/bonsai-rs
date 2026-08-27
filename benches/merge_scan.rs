//! Timing harness for one full round of candidate-pair scoring.
//!
//! This is the cost that decides whether the tree search is viable. The naive
//! search scores every pair of the root's children every round, which SPEC.md
//! section 10 gives as `O(n^3 p)`. Restricting candidates to a `k`-nearest
//! neighbour graph (SPEC.md section 11) cuts a round to `n * k` pairs; the
//! upper-bound machinery (section 10) then means only a handful of those get
//! rescored in a typical later round, with a full rescan at each ellipsoid
//! epoch boundary.
//!
//! So the number to know is the wall time of one full `n * k` scan. A run pays
//! it once up front and once per epoch.
//!
//! ```sh
//! cargo bench --bench merge_scan
//! ```
//!
//! Plain `main`, no harness.

use bonsai_rs::model::merge::{EffLeaf, MergeScratch, score_merge};
use rayon::prelude::*;
use std::time::Instant;

/// Star sizes swept, standing in for the number of root children in a round.
const STAR_SIZES: [usize; 3] = [1024, 4096, 8192];

/// Feature counts swept.
const FEATURE_COUNTS: [usize; 2] = [500, 2000];

/// Neighbours per node. The paper reports 5 to 10 working well; the value this
/// crate ships is ours to pick once the search exists, so the scan is measured
/// at the top of that range.
const NEIGHBOURS: usize = 10;

/// Repeats per configuration; the reported time is the best of these.
const REPEATS: usize = 3;

/// One draw from the counter-based splitmix64 stream.
///
/// ### Params
///
/// * `index` - Position in the stream
///
/// ### Returns
///
/// A uniform in `[0, 1)`.
#[inline]
fn splitmix64(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    println!("threads {}", rayon::current_num_threads());
    println!(
        "{:>7} {:>6} {:>9} {:>10} {:>12} {:>12}",
        "star", "feat", "pairs", "scan_ms", "pairs/s", "us_per_pair"
    );

    for &n in STAR_SIZES.iter() {
        for &p in FEATURE_COUNTS.iter() {
            // Children of the star, clustered so that neighbouring indices are
            // genuinely close: a scan over random pairs would spend all its time
            // in the zero-branch early exit and flatter the timing.
            let total = (n * p) as u64;
            let m: Vec<f64> = (0..total)
                .map(|i| {
                    let cell = i / p as u64;
                    let feat = i % p as u64;
                    (cell / 8) as f64 * 0.6 + splitmix64(i) * 2.0 + (feat as f64 * 0.01).sin()
                })
                .collect();
            let w: Vec<f64> = (0..total).map(|i| 0.4 + splitmix64(total + i) * 2.0).collect();

            // The peeled rest of the star. In the real search this is recomputed
            // per pair by subtraction in O(p); here one representative is enough
            // to make the arithmetic honest.
            let m_r: Vec<f64> = (0..p).map(|g| 1.0 + (g as f64 * 0.03).cos()).collect();
            let w_r: Vec<f64> = (0..p).map(|g| 0.7 + 0.2 * (g as f64 * 0.05).sin()).collect();

            // Candidate pairs: each node against the next NEIGHBOURS nodes,
            // which the clustering above makes a plausible neighbour list.
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

            // Checksum before reporting: a scan that silently did nothing would
            // otherwise look like an excellent timing.
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
