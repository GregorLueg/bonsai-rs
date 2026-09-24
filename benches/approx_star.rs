//! Two properties of the merge score that decide how the star can be
//! accelerated.
//!
//! Both need the same thing, an exhaustive star primitive that can be
//! interrogated round by round, so they share one scaffold here. It reproduces
//! SPEC.md sections 4 and 8.1 over the public `score_merge`, which means it is
//! a second transcription of the recursion and not a call into
//! `search::star`; the `check_scaffold` assertion at the top of the run is what
//! keeps the two honest.
//!
//! ### Reducibility
//!
//! An objective is reducible when merging a mutually nearest pair never lifts
//! another pair above the score the merged pair had. That is the condition
//! under which nearest-neighbour chaining and Boruvka-style batch merging give
//! the same dendrogram as the round-by-round scan. If it holds here, a batch
//! of mutually best pairs can be merged in one round without a certificate.
//! If it does not, the count and size of the inversions say how much
//! verification a batch needs.
//!
//! ### Feature subsampling
//!
//! The gain is a sum over features, so it can be estimated on `p'` of them and
//! rescaled. What matters is not the error in the gain but whether the argmax
//! over a round's candidates survives, and what a wrong pick costs in nats.
//!
//! **Run on a quiet machine.** Check `uptime` first.
//!
//! ```sh
//! cargo bench --bench approx_star
//! ```
//!
//! Plain `main`, no harness.

use bonsai_rs::model::merge::{EffLeaf, MergeScratch, score_merge};
use bonsai_rs::search::star::{Star, resolve_star};
use bonsai_rs::tree::simulate::{SimulationParams, simulate_binary};
use bonsai_rs::utils::rng::splitmix64_at;
use rayon::prelude::*;
use std::time::Instant;

////////////////
// Parameters //
////////////////

/// Leaf counts for the reducibility scan.
///
/// Small on purpose: this is an exhaustive scan re-run after every merge, so
/// it is `O(n^3 p)` twice over. The property it measures is structural and
/// does not need the sizes a timing would.
const REDUCIBILITY_LEAVES: [usize; 2] = [64, 128];

/// Features for the reducibility scan. Inversions are a property of the score's
/// shape, not of how many features it sums over.
const REDUCIBILITY_FEATURES: usize = 200;

/// Leaves and features for the subsampling scan.
///
/// Two thousand features because that is where the crate expects to work and
/// the whole question is what fraction of them a ranking pass needs.
const SUBSAMPLE_LEAVES: usize = 128;

/// Features for the subsampling scan.
const SUBSAMPLE_FEATURES: usize = 2000;

/// Subsample sizes swept, as a count of features.
const SUBSAMPLE_SIZES: [usize; 5] = [32, 64, 128, 256, 512];

/// Seeds per configuration.
const SEEDS: [u64; 3] = [31, 32, 33];

/// Measurement noise in transformed units, matching the other benches.
const NOISE: f64 = 0.3;

/// Members at which the star primitive stops, matching `search::star`.
const MIN_CENTRE_MEMBERS: usize = 3;

//////////////
// Scaffold //
//////////////

/// One member of a star: an effective leaf and its branch to the centre.
#[derive(Clone, Debug)]
struct Member {
    /// Effective means, length `p`.
    m: Vec<f64>,
    /// Effective precisions, length `p`.
    w: Vec<f64>,
    /// Branch length from this member to the centre.
    branch: f64,
}

/// The centre's effective leaf, summarising every member (SPEC.md section 4).
///
/// ### Params
///
/// * `members` - The star's members
/// * `p` - Number of features
///
/// ### Returns
///
/// The centre's means and precisions, in that order.
fn centre_leaf(members: &[Member], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut wc = vec![0.0f64; p];
    let mut num = vec![0.0f64; p];
    for member in members {
        for g in 0..p {
            let wd = member.w[g] / (1.0 + member.branch * member.w[g]);
            wc[g] += wd;
            num[g] += wd * member.m[g];
        }
    }
    let mc: Vec<f64> = (0..p).map(|g| num[g] / wc[g]).collect();
    (mc, wc)
}

/// Peel two members off the centre, leaving the rest as one effective leaf.
///
/// SPEC.md section 8.1. `O(p)`, which is why scoring a pair does not cost
/// `O(n p)`.
///
/// ### Params
///
/// * `mc`, `wc` - The centre's effective leaf
/// * `k`, `l` - The two members being peeled off
/// * `p` - Number of features
///
/// ### Returns
///
/// The remainder's means and precisions, in that order.
fn peel(mc: &[f64], wc: &[f64], k: &Member, l: &Member, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mr = vec![0.0f64; p];
    let mut wr = vec![0.0f64; p];
    for g in 0..p {
        let wdk = k.w[g] / (1.0 + k.branch * k.w[g]);
        let wdl = l.w[g] / (1.0 + l.branch * l.w[g]);
        wr[g] = wc[g] - wdk - wdl;
        mr[g] = (mc[g] * wc[g] - wdk * k.m[g] - wdl * l.m[g]) / wr[g];
    }
    (mr, wr)
}

/// Every pair's gain at the current centre, over the given feature columns.
///
/// Passing a subset of columns is what the subsampling scan needs: the branch
/// lengths are then optimised against the subset too, which is what a
/// subsampled implementation would actually do, and the gain is rescaled by
/// `p / p'` on the way out.
///
/// ### Params
///
/// * `members` - The star's members
/// * `columns` - Feature indices to score over; `None` scores every feature
/// * `p` - Number of features in `members`
///
/// ### Returns
///
/// One gain per unordered pair, in `(i, j)` ascending order.
fn scan(members: &[Member], columns: Option<&[usize]>, p: usize) -> Vec<((usize, usize), f64)> {
    // Gather once per member rather than once per pair.
    let (view, q) = match columns {
        None => (members.to_vec(), p),
        Some(cols) => {
            let gathered = members
                .iter()
                .map(|member| Member {
                    m: cols.iter().map(|&g| member.m[g]).collect(),
                    w: cols.iter().map(|&g| member.w[g]).collect(),
                    branch: member.branch,
                })
                .collect();
            (gathered, cols.len())
        }
    };
    let scale = p as f64 / q as f64;

    let (mc, wc) = centre_leaf(&view, q);
    let n = view.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    pairs
        .par_iter()
        .map_init(
            || MergeScratch::new(q),
            |scratch, &(i, j)| {
                let (k, l) = (&view[i], &view[j]);
                let (mr, wr) = peel(&mc, &wc, k, l, q);
                let score = score_merge(
                    EffLeaf { m: &k.m, w: &k.w },
                    EffLeaf { m: &l.m, w: &l.w },
                    EffLeaf { m: &mr, w: &wr },
                    k.branch,
                    l.branch,
                    None,
                    scratch,
                )
                .expect("merge score");
                ((i, j), score.gain * scale)
            },
        )
        .collect()
}

/// Replace two members by the ancestor the merge inserts (SPEC.md section 4).
///
/// The effective mean is formed as a convex combination, not as a ratio of
/// weighted sums, for the reason `CLAUDE.md` gives: the result is pinned
/// between the two child means and cannot cancel.
///
/// ### Params
///
/// * `members` - The star, edited in place
/// * `i`, `j` - Indices of the pair to merge, `i < j`
/// * `t_ak`, `t_al`, `t_ar` - The three optimised branch lengths
/// * `p` - Number of features
fn apply_merge(
    members: &mut Vec<Member>,
    i: usize,
    j: usize,
    t_ak: f64,
    t_al: f64,
    t_ar: f64,
    p: usize,
) {
    let (k, l) = (members[i].clone(), members[j].clone());
    let mut m = vec![0.0f64; p];
    let mut w = vec![0.0f64; p];
    for g in 0..p {
        let wdk = k.w[g] / (1.0 + t_ak * k.w[g]);
        let wdl = l.w[g] / (1.0 + t_al * l.w[g]);
        w[g] = wdk + wdl;
        m[g] = k.m[g] + (l.m[g] - k.m[g]) * wdl / w[g];
    }
    // Remove the higher index first so the lower one is still valid.
    members.remove(j);
    members.remove(i);
    members.push(Member { m, w, branch: t_ar });
}

/// The star's members at round zero: one per cell, all on unit branches.
///
/// ### Params
///
/// * `means`, `precisions` - Transformed leaf data, row-major
/// * `n` - Number of cells
/// * `p` - Number of features
///
/// ### Returns
///
/// The initial members.
fn initial_members(means: &[f64], precisions: &[f64], n: usize, p: usize) -> Vec<Member> {
    (0..n)
        .map(|i| Member {
            m: means[i * p..(i + 1) * p].to_vec(),
            w: precisions[i * p..(i + 1) * p].to_vec(),
            branch: 1.0,
        })
        .collect()
}

/// Simulate one fixture and return its transformed leaf data.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `p` - Number of features
/// * `seed` - Simulation seed
///
/// ### Returns
///
/// Means and precisions, row-major, in that order.
fn fixture(n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let sim = simulate_binary::<f64>(Some(SimulationParams {
        n_leaves: n,
        n_features: p,
        noise_sd: NOISE,
        seed,
        ..Default::default()
    }))
    .expect("simulation");
    let precisions = sim.precisions();
    (sim.means, precisions)
}

/////////////////////////
// Scan 1: reducibility //
/////////////////////////

/// Merge greedily and count how often a merge lifts another pair above the
/// score the merged pair had.
///
/// An inversion is what makes batch merging unsafe, so both the count and the
/// worst excess matter: a handful of inversions worth a millinat is a different
/// answer from a handful worth ten nats.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `p` - Number of features
/// * `seed` - Simulation seed
///
/// ### Returns
///
/// Rounds run, inversions seen, the worst absolute excess in nats, and the
/// worst excess relative to the merged pair's gain.
fn reducibility(n: usize, p: usize, seed: u64) -> (usize, usize, f64, f64) {
    let (means, precisions) = fixture(n, p, seed);
    let mut members = initial_members(&means, &precisions, n, p);

    let mut rounds = 0usize;
    let mut inversions = 0usize;
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;

    let mut before = scan(&members, None, p);
    while members.len() > MIN_CENTRE_MEMBERS {
        let Some(&(pair, best)) = before
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite gains"))
        else {
            break;
        };
        if best <= 0.0 {
            break;
        }
        rounds += 1;

        // Re-solve the winner for its branch lengths, which the scan discards.
        let (mc, wc) = centre_leaf(&members, p);
        let (k, l) = (&members[pair.0], &members[pair.1]);
        let (mr, wr) = peel(&mc, &wc, k, l, p);
        let mut scratch = MergeScratch::new(p);
        let score = score_merge(
            EffLeaf { m: &k.m, w: &k.w },
            EffLeaf { m: &l.m, w: &l.w },
            EffLeaf { m: &mr, w: &wr },
            k.branch,
            l.branch,
            None,
            &mut scratch,
        )
        .expect("merge score");

        apply_merge(
            &mut members,
            pair.0,
            pair.1,
            score.t_ak,
            score.t_al,
            score.t_ar,
            p,
        );

        let after = scan(&members, None, p);
        for &(_, gain) in after.iter() {
            if gain > best {
                inversions += 1;
                let excess = gain - best;
                worst_abs = worst_abs.max(excess);
                worst_rel = worst_rel.max(excess / best.abs().max(1e-12));
            }
        }
        before = after;
    }

    (rounds, inversions, worst_abs, worst_rel)
}

/////////////////////////
// Scan 2: subsampling //
/////////////////////////

/// A deterministic subset of `q` feature indices out of `p`.
///
/// ### Params
///
/// * `p` - Total features
/// * `q` - Features to keep
/// * `seed` - Stream offset
///
/// ### Returns
///
/// `q` distinct indices, ascending.
fn columns(p: usize, q: usize, seed: u64) -> Vec<usize> {
    let mut taken = vec![false; p];
    let mut out = Vec::with_capacity(q);
    let mut draw = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    while out.len() < q {
        draw = draw.wrapping_add(1);
        let g = ((splitmix64_at(draw) * p as f64) as usize).min(p - 1);
        if !taken[g] {
            taken[g] = true;
            out.push(g);
        }
    }
    out.sort_unstable();
    out
}

/// Running tally of what a subsample size costs.
#[derive(Clone, Copy, Debug, Default)]
struct SubsampleTally {
    /// Checkpoints scored.
    checkpoints: usize,
    /// Checkpoints where the subsampled argmax was the full-feature argmax.
    agreed: usize,
    /// Total nats given up by taking the subsampled pick instead.
    nats_lost: f64,
    /// Worst nats given up at any one checkpoint.
    worst_nats: f64,
    /// Mean absolute relative error in the winning pair's gain.
    rel_error: f64,
}

/// Compare subsampled and full-feature scans at checkpoints through one star.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `p` - Number of features
/// * `seed` - Simulation seed
/// * `tally` - One entry per subsample size, edited in place
fn subsampling(n: usize, p: usize, seed: u64, tally: &mut [SubsampleTally]) {
    let (means, precisions) = fixture(n, p, seed);
    let mut members = initial_members(&means, &precisions, n, p);

    // Five checkpoints spread over the star, because the first round is the
    // easiest one and a method judged on it alone flatters itself.
    let total_rounds = n - MIN_CENTRE_MEMBERS;
    let checkpoints: Vec<usize> = (1..=5).map(|i| i * total_rounds / 6).collect();

    for round in 0..total_rounds {
        let full = scan(&members, None, p);
        let Some(&(best_pair, best_gain)) = full
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite gains"))
        else {
            break;
        };
        if best_gain <= 0.0 {
            break;
        }

        if checkpoints.contains(&round) {
            for (slot, &q) in SUBSAMPLE_SIZES.iter().enumerate() {
                let cols = columns(p, q, seed.wrapping_add(round as u64));
                let sub = scan(&members, Some(&cols), p);
                let &(pick, sub_gain) = sub
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite gains"))
                    .expect("a pair");

                let t = &mut tally[slot];
                t.checkpoints += 1;
                t.rel_error += ((sub_gain - best_gain) / best_gain).abs();
                if pick == best_pair {
                    t.agreed += 1;
                } else {
                    let taken = full
                        .iter()
                        .find(|&&(pair, _)| pair == pick)
                        .map(|&(_, gain)| gain)
                        .expect("the pick is a pair");
                    let lost = best_gain - taken;
                    t.nats_lost += lost;
                    t.worst_nats = t.worst_nats.max(lost);
                }
            }
        }

        let (mc, wc) = centre_leaf(&members, p);
        let (k, l) = (&members[best_pair.0], &members[best_pair.1]);
        let (mr, wr) = peel(&mc, &wc, k, l, p);
        let mut scratch = MergeScratch::new(p);
        let score = score_merge(
            EffLeaf { m: &k.m, w: &k.w },
            EffLeaf { m: &l.m, w: &l.w },
            EffLeaf { m: &mr, w: &wr },
            k.branch,
            l.branch,
            None,
            &mut scratch,
        )
        .expect("merge score");
        apply_merge(
            &mut members,
            best_pair.0,
            best_pair.1,
            score.t_ak,
            score.t_al,
            score.t_ar,
            p,
        );

        if members.len() <= MIN_CENTRE_MEMBERS {
            break;
        }
    }
}

///////////
// Check //
///////////

/// Assert the scaffold reproduces the shipped primitive's first merge.
///
/// This file transcribes SPEC.md sections 4 and 8.1 a second time so it can
/// look inside a round, and a second transcription is a second chance to get
/// it wrong. Both the pair chosen and its gain have to match
/// [`resolve_star`]'s, or nothing below means anything.
fn check_scaffold() {
    let (n, p) = (32usize, 64usize);
    let (means, precisions) = fixture(n, p, 7);
    let branch = vec![1.0f64; n];

    let shipped = resolve_star(
        Star {
            means: &means,
            precisions: &precisions,
            branch: &branch,
            n_features: p,
        },
        None,
    )
    .expect("shipped primitive");
    let first = shipped.merges.first().expect("at least one merge");

    let members = initial_members(&means, &precisions, n, p);
    let mine = scan(&members, None, p);
    let &(pair, gain) = mine
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite gains"))
        .expect("a pair");

    let shipped_pair = (
        (first.left.min(first.right)) as usize,
        (first.left.max(first.right)) as usize,
    );
    assert_eq!(pair, shipped_pair, "scaffold picked a different pair");
    assert!(
        (gain - first.gain).abs() <= 1e-9 * first.gain.abs(),
        "scaffold gain {gain} against the primitive's {}",
        first.gain
    );
    println!("scaffold agrees with search::star on {n} members: gain {gain:.6}");
}

//////////
// Main //
//////////

fn main() {
    println!("threads {}", rayon::current_num_threads());
    check_scaffold();

    println!("\n=== reducibility: does a merge ever lift another pair above it? ===");
    println!(
        "{:>7} {:>6} {:>6} {:>8} {:>11} {:>13} {:>13}",
        "leaves", "feat", "seed", "rounds", "inversions", "worst nats", "worst rel"
    );
    for &n in REDUCIBILITY_LEAVES.iter() {
        for &seed in SEEDS.iter() {
            let t0 = Instant::now();
            let (rounds, inversions, worst_abs, worst_rel) =
                reducibility(n, REDUCIBILITY_FEATURES, seed);
            println!(
                "{n:>7} {:>6} {seed:>6} {rounds:>8} {inversions:>11} {worst_abs:>13.4} {worst_rel:>13.2e}   ({:.1} s)",
                REDUCIBILITY_FEATURES,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    println!("\n=== feature subsampling: does the argmax survive? ===");
    println!(
        "{:>7} {:>6} {:>6} {:>10} {:>11} {:>12} {:>12}",
        "feat", "sub", "of", "agreed", "mean rel err", "nats lost", "worst nats"
    );
    let mut tally = vec![SubsampleTally::default(); SUBSAMPLE_SIZES.len()];
    for &seed in SEEDS.iter() {
        subsampling(SUBSAMPLE_LEAVES, SUBSAMPLE_FEATURES, seed, &mut tally);
    }
    for (slot, &q) in SUBSAMPLE_SIZES.iter().enumerate() {
        let t = tally[slot];
        let reps = t.checkpoints.max(1) as f64;
        println!(
            "{:>7} {q:>6} {:>6} {:>10} {:>11.2e} {:>12.3} {:>12.3}",
            SUBSAMPLE_FEATURES,
            t.checkpoints,
            t.agreed,
            t.rel_error / reps,
            t.nats_lost / reps,
            t.worst_nats
        );
    }
}
