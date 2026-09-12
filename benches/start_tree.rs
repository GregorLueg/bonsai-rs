//! Does search step 2 have to exist?
//!
//! Steps 5, 6 and 7 are a local search and step 2 only supplies the tree they
//! start from. Step 2 is 31 to 49 per cent of a run and scales `n^1.65`, so if
//! a cheap starting tree lands in the same place after refinement then the
//! greedy agglomeration is replaceable by something linear.
//!
//! Four starts, all refined by the identical steps 3 to 7:
//!
//! - **greedy**, the current steps 1 and 2, as the baseline;
//! - **average**, UPGMA on Euclidean distance between the transformed means;
//! - **ward**, the same chain on squared Euclidean with Ward's update;
//! - **random**, a uniformly random binary topology, as the null.
//!
//! The linkages are built here rather than in `src/` on purpose. This is an
//! experiment, and nothing about it should ship until the table says it should.
//! Both are exact under nearest-neighbour chaining because both objectives are
//! reducible (Bruynooghe 1977, Murtagh 1983), so the `O(n^2)` chain gives the
//! same dendrogram as the `O(n^3)` naive scan.
//!
//! What to read: `RF` against the generating tree is the answer, `loglik` is
//! the tiebreak, and `build` against `refine` says what would be saved.
//!
//! **Run on a quiet machine.** Check `uptime` first.
//!
//! ```sh
//! cargo bench --bench start_tree
//! ```
//!
//! Plain `main`, no harness.

use bonsai_rs::bonsai::refine;
use bonsai_rs::ingest::PreparedData;
use bonsai_rs::model::global::{collapse_onto_every_node, optimise_branch_lengths};
use bonsai_rs::model::likelihood::NodeState;
use bonsai_rs::model::merge::EffLeaf;
use bonsai_rs::model::place::place;
use bonsai_rs::search::Leaves;
use bonsai_rs::search::bounds::EllipsoidBounds;
use bonsai_rs::search::candidates::KnnCandidates;
use bonsai_rs::search::nni::nni;
use bonsai_rs::search::polytomy::resolve_polytomies;
use bonsai_rs::search::spr::spr;
use bonsai_rs::search::star::{Star, star_tree_with};
use bonsai_rs::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};
use bonsai_rs::tree::{NO_NODE, Tree};
use bonsai_rs::utils::rng::splitmix64_at;
use rayon::prelude::*;
use std::time::Instant;

////////////////
// Parameters //
////////////////

/// Leaf counts swept. The top end is where `benches/pipeline.rs` stops, so the
/// baseline column is comparable to the published table.
const LEAVES: [usize; 4] = [256, 512, 1024, 2048];

/// Feature counts swept. Recovery improves with features, so a start that only
/// works at 2000 is not a start.
const FEATURES: [usize; 2] = [200, 2000];

/// Simulation seeds per configuration. Robinson-Foulds moves by a few splits
/// between seeds, so a single one decides nothing.
const SEEDS: [u64; 5] = [31, 32, 33, 34, 35];

/// Measurement noise in transformed units, matching `benches/pipeline.rs`.
const NOISE: f64 = 0.3;

/// Leaf count for the noise sweep.
///
/// The size sweep runs at one noise level, where every start recovers nearly
/// the whole topology. That is a regime where the starts cannot be told apart,
/// so it cannot be the only evidence: this block holds the size fixed and turns
/// the noise up until the search stops being easy.
const NOISE_SWEEP_LEAVES: usize = 512;

/// Features for the noise sweep.
const NOISE_SWEEP_FEATURES: usize = 2000;

/// Noise levels swept, in transformed units.
const NOISES: [f64; 4] = [0.3, 0.6, 1.0, 1.6];

/// Leaf counts for the per-step scaling block.
///
/// The published per-step attribution is at 200 features and stops at 2048,
/// where polytomy resolution reads `n^2.83` and the interchanges `n^2.60`. Both
/// are round counts, and the round counts fall as the feature axis grows, so
/// those exponents say nothing about the regime the crate is meant for. This
/// block is at 2000 features from a Ward start and goes far enough to fit one.
const STEP_LEAVES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];

/// Features for the per-step scaling block.
const STEP_FEATURES: usize = 2000;

/// Seeds for the per-step scaling block. Two, because the top size is an
/// `O(n^2 p)` linkage over a 2 GB distance matrix and the exponent is what is
/// wanted, not a tight mean.
const STEP_SEEDS: [u64; 2] = [31, 32];

/// Wall-clock ceiling for one configuration of the per-step block.
///
/// Past this the sweep stops growing, so an unattended run cannot turn into an
/// overnight one. Same guard `benches/pipeline.rs` uses.
const STEP_BUDGET_SECONDS: f64 = 1200.0;

/// Placement queries per size in the beam block.
///
/// SPR is the only superlinear step left and its cost is `rounds` x `O(n)`
/// candidates x the nodes the placement beam scores. Rounds are flat at two to
/// four, so the exponent lives in the beam, and this measures it directly:
/// a cell already in the tree, placed back onto it.
const BEAM_QUERIES: usize = 32;

/// Leaf count for the SPR ablation at raised noise.
///
/// Small enough that four noise levels are minutes rather than an hour, and
/// large enough that a difference of a few splits is not one tree's luck.
const SPR_NOISE_LEAVES: usize = 2048;

/// Members left attached to the root when a linkage stops.
///
/// Three, not two. A binary dendrogram's root is degree two in the unrooted
/// sense, which carries no information and which `search::spr` refuses to prune
/// a child of. The greedy star primitive stops at three for the same reason.
const ROOT_MEMBERS: usize = 3;

/// Which starting tree a row measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    /// Search steps 1 and 2 as the pipeline runs them.
    Greedy,
    /// Average linkage, that is UPGMA, on Euclidean distance.
    Average,
    /// Ward's minimum-variance linkage on squared Euclidean distance.
    Ward,
    /// A uniformly random sequence of joins.
    Random,
}

impl Start {
    /// Column label.
    ///
    /// ### Returns
    ///
    /// The name printed in the table.
    fn name(self) -> &'static str {
        match self {
            Start::Greedy => "greedy",
            Start::Average => "average",
            Start::Ward => "ward",
            Start::Random => "random",
        }
    }
}

/////////////
// Linkage //
/////////////

/// Full squared-Euclidean distance matrix over the transformed means.
///
/// Row-major `n * n`, with the diagonal at infinity so an argmin never returns
/// the point itself. `O(n^2 p)`, which is the part of this that would have to
/// go through a neighbour graph if any of it ever shipped.
///
/// ### Params
///
/// * `means` - Transformed means, row-major `[cell][feature]`
/// * `n` - Number of cells
/// * `p` - Number of features
///
/// ### Returns
///
/// The matrix, flattened row-major.
fn squared_distances(means: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n * n];
    d.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a = &means[i * p..(i + 1) * p];
        for j in 0..n {
            if i == j {
                row[j] = f64::INFINITY;
                continue;
            }
            let b = &means[j * p..(j + 1) * p];
            let mut acc = 0.0f64;
            for g in 0..p {
                let diff = a[g] - b[g];
                acc += diff * diff;
            }
            row[j] = acc;
        }
    });
    d
}

/// Agglomerate by nearest-neighbour chaining under a Lance-Williams update.
///
/// The chain walks to a mutually nearest pair, merges it, and updates the row
/// of the surviving slot in place. Both objectives here are reducible, so the
/// pair the chain finds is the pair the naive scan would have found.
///
/// ### Params
///
/// * `means` - Transformed means, row-major
/// * `n` - Number of cells
/// * `p` - Number of features
/// * `ward` - Ward's update on squared distance, otherwise average linkage on
///   Euclidean distance
///
/// ### Returns
///
/// A tree whose leaves are the cells in order, with unit branch lengths for
/// step 4 to replace.
fn linkage_tree(means: &[f64], n: usize, p: usize, ward: bool) -> Tree {
    let mut d = squared_distances(means, n, p);
    if !ward {
        d.par_iter_mut().for_each(|x| *x = x.sqrt());
    }

    // Slot `i` holds a live cluster; `node_of[i]` is the arena node summarising
    // it. Merging writes into the first slot and retires the second.
    let mut active = vec![true; n];
    let mut size = vec![1.0f64; n];
    let mut node_of: Vec<u32> = (0..n as u32).collect();

    let mut parent = vec![NO_NODE; 2 * n - ROOT_MEMBERS + 1];
    let mut next_internal = n;
    let mut remaining = n;
    let mut chain: Vec<usize> = Vec::with_capacity(64);

    while remaining > ROOT_MEMBERS {
        if chain.is_empty() {
            let seed = (0..n).find(|&i| active[i]).expect("a live cluster");
            chain.push(seed);
        }
        // Walk until the last two entries are mutually nearest.
        let (a, b) = loop {
            let a = *chain.last().expect("a non-empty chain");
            let row = &d[a * n..(a + 1) * n];
            let mut best = f64::INFINITY;
            let mut b = usize::MAX;
            for j in 0..n {
                if active[j] && j != a && row[j] < best {
                    best = row[j];
                    b = j;
                }
            }
            if chain.len() >= 2 && b == chain[chain.len() - 2] {
                chain.pop();
                chain.pop();
                break (a, b);
            }
            chain.push(b);
        };

        let ancestor = next_internal as u32;
        next_internal += 1;
        parent[node_of[a] as usize] = ancestor;
        parent[node_of[b] as usize] = ancestor;

        let (na, nb, dab) = (size[a], size[b], d[a * n + b]);
        for k in 0..n {
            if !active[k] || k == a || k == b {
                continue;
            }
            let (dak, dbk) = (d[a * n + k], d[b * n + k]);
            let merged = if ward {
                let nk = size[k];
                ((na + nk) * dak + (nb + nk) * dbk - nk * dab) / (na + nb + nk)
            } else {
                (na * dak + nb * dbk) / (na + nb)
            };
            d[a * n + k] = merged;
            d[k * n + a] = merged;
        }

        active[b] = false;
        size[a] = na + nb;
        node_of[a] = ancestor;
        remaining -= 1;
    }

    let root = next_internal as u32;
    for i in 0..n {
        if active[i] {
            parent[node_of[i] as usize] = root;
        }
    }
    parent[root as usize] = NO_NODE;

    let branch = vec![1.0f64; parent.len()];
    Tree::from_parents(parent, branch, n).expect("linkage tree")
}

/// A uniformly random binary topology over `n` leaves.
///
/// The null: whatever refinement recovers from here is recovered from nothing.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `seed` - Stream offset for the shared counter-based generator
///
/// ### Returns
///
/// A tree with unit branch lengths.
fn random_tree(n: usize, seed: u64) -> Tree {
    let mut live: Vec<u32> = (0..n as u32).collect();
    let mut parent = vec![NO_NODE; 2 * n - ROOT_MEMBERS + 1];
    let mut next_internal = n;
    let mut draw = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);

    while live.len() > ROOT_MEMBERS {
        draw = draw.wrapping_add(1);
        let i = ((splitmix64_at(draw) * live.len() as f64) as usize).min(live.len() - 1);
        let a = live.swap_remove(i);
        draw = draw.wrapping_add(1);
        let j = ((splitmix64_at(draw) * live.len() as f64) as usize).min(live.len() - 1);
        let b = live.swap_remove(j);

        let ancestor = next_internal as u32;
        next_internal += 1;
        parent[a as usize] = ancestor;
        parent[b as usize] = ancestor;
        live.push(ancestor);
    }

    let root = next_internal as u32;
    for &node in &live {
        parent[node as usize] = root;
    }
    parent[root as usize] = NO_NODE;

    let branch = vec![1.0f64; parent.len()];
    Tree::from_parents(parent, branch, n).expect("random tree")
}

//////////
// Main //
//////////

/// Build one starting tree and report what it cost.
///
/// ### Params
///
/// * `start` - Which start to build
/// * `data` - The prepared cells
/// * `seed` - Stream offset, used by the random start only
///
/// ### Returns
///
/// The tree and the seconds spent building it.
fn build_start(start: Start, data: &PreparedData<f64>, seed: u64) -> (Tree, f64) {
    let (n, p) = (data.n_cells, data.n_features());
    let t0 = Instant::now();
    let tree = match start {
        Start::Greedy => {
            let mut parent = vec![n as u32; n];
            parent.push(NO_NODE);
            let mut branch = vec![1.0f64; n];
            branch.push(0.0);
            let mut star = Tree::from_parents(parent, branch, n).expect("star");
            let mut state = NodeState::new(
                star.n_nodes(),
                p,
                &data.transformed_means,
                &data.transformed_precisions,
            )
            .expect("state");
            optimise_branch_lengths(&mut star, &mut state, None).expect("step 1");

            let mut candidates = EllipsoidBounds::new(KnnCandidates::new(None), None);
            star_tree_with(
                Star {
                    means: &data.transformed_means,
                    precisions: &data.transformed_precisions,
                    branch: &star.branches()[..n],
                    n_features: p,
                },
                None,
                &mut candidates,
            )
            .expect("step 2")
            .0
        }
        Start::Average => linkage_tree(&data.transformed_means, n, p, false),
        Start::Ward => linkage_tree(&data.transformed_means, n, p, true),
        Start::Random => random_tree(n, seed),
    };
    (tree, t0.elapsed().as_secs_f64())
}

/// Every start through the full refinement at one configuration, averaged over
/// the seeds.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `p` - Number of features
/// * `noise` - Measurement noise in transformed units
///
/// ### Returns
///
/// One row per start: the start, mean build seconds, mean refine seconds, mean
/// Robinson-Foulds to the generating tree, and mean loglikelihood.
fn run(n: usize, p: usize, noise: f64) -> Vec<(Start, f64, f64, f64, f64)> {
    let mut acc: Vec<(Start, f64, f64, f64, f64)> = Vec::new();

    for &seed in SEEDS.iter() {
        let sim = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: noise,
            seed,
            ..Default::default()
        }))
        .expect("simulation");

        // `simulate` already returns transformed units, so ingest is bypassed
        // exactly as `benches/pipeline.rs` bypasses it.
        let data = PreparedData {
            transformed_means: sim.means.clone(),
            transformed_precisions: sim.precisions(),
            features: (0..p).collect(),
            variances: vec![1.0; p],
            signal_to_noise: vec![f64::INFINITY; p],
            n_cells: n,
            n_features_in: p,
        };

        for start in [Start::Greedy, Start::Average, Start::Ward, Start::Random] {
            let (tree, build) = build_start(start, &data, seed);
            let t0 = Instant::now();
            let out = refine(&tree, &data, None).expect("refine");
            let refine_s = t0.elapsed().as_secs_f64();
            assert!(out.loglik.is_finite(), "non-finite loglikelihood");
            let rf = robinson_foulds(&out.tree, &sim.tree).expect("rf") as f64;

            match acc.iter_mut().find(|row| row.0 == start) {
                Some(row) => {
                    row.1 += build;
                    row.2 += refine_s;
                    row.3 += rf;
                    row.4 += out.loglik;
                }
                None => acc.push((start, build, refine_s, rf, out.loglik)),
            }
        }
    }

    let reps = SEEDS.len() as f64;
    for row in acc.iter_mut() {
        row.1 /= reps;
        row.2 /= reps;
        row.3 /= reps;
        row.4 /= reps;
    }
    acc
}

/// Steps 3 to 7 timed one at a time, from a Ward start.
///
/// `bonsai::refine` runs these as one call, so the split has to be replicated
/// here the way `benches/steps.rs` replicates it. The order is the pipeline's
/// and is not negotiable: the interchange and regraft filters reject every
/// improvement available until the branch lengths are optimised.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `seed` - Simulation seed
///
/// ### Returns
///
/// Seconds for the linkage build and for steps 3, 4, 5, 6 and 7 in order, then
/// the Robinson-Foulds distance to the generating tree.
fn step_split(n: usize, seed: u64) -> ([f64; 6], f64) {
    let p = STEP_FEATURES;
    let sim = simulate_binary::<f64>(Some(SimulationParams {
        n_leaves: n,
        n_features: p,
        noise_sd: NOISE,
        seed,
        ..Default::default()
    }))
    .expect("simulation");
    let precisions = sim.precisions();
    let leaves = Leaves {
        means: &sim.means,
        precisions: &precisions,
        n_features: p,
    };

    let mut t = [0.0f64; 6];

    let t0 = Instant::now();
    let mut tree = linkage_tree(&sim.means, n, p, true);
    t[0] = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    tree = resolve_polytomies(&tree, leaves, None)
        .expect("step 3")
        .tree;
    t[1] = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut state =
        NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
    optimise_branch_lengths(&mut tree, &mut state, None).expect("step 4");
    t[2] = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    tree = spr(&tree, leaves, None).expect("step 5").tree;
    t[3] = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    tree = nni(&tree, leaves, None).expect("step 6").tree;
    t[4] = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut state =
        NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
    optimise_branch_lengths(&mut tree, &mut state, None).expect("step 7");
    t[5] = t0.elapsed().as_secs_f64();

    let rf = robinson_foulds(&tree, &sim.tree).expect("rf") as f64;
    (t, rf)
}

/// What one size costs and recovers with SPR and without it, plus the beam.
///
/// Both arms share the Ward start and steps 3 and 4, so the only difference is
/// whether step 5 runs. If they reach the same Robinson-Foulds then SPR is
/// buying nothing at this noise level and a tree-distance limit is the fix
/// rather than a rewrite.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `seed` - Simulation seed
///
/// ### Returns
///
/// Seconds, Robinson-Foulds and loglikelihood with SPR, the same three without
/// it, and the mean nodes the placement beam scored on the SPR arm's tree.
fn spr_ablation(n: usize, seed: u64, noise: f64) -> ([f64; 3], [f64; 3], f64) {
    let p = STEP_FEATURES;
    let sim = simulate_binary::<f64>(Some(SimulationParams {
        n_leaves: n,
        n_features: p,
        noise_sd: noise,
        seed,
        ..Default::default()
    }))
    .expect("simulation");
    let precisions = sim.precisions();
    let leaves = Leaves {
        means: &sim.means,
        precisions: &precisions,
        n_features: p,
    };

    // Shared prefix: the Ward start, polytomy resolution and step 4.
    let mut base = linkage_tree(&sim.means, n, p, true);
    base = resolve_polytomies(&base, leaves, None)
        .expect("step 3")
        .tree;
    let mut state =
        NodeState::new(base.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
    optimise_branch_lengths(&mut base, &mut state, None).expect("step 4");

    // Both arms finish with the interchanges and step 7, so the only thing that
    // differs is step 5.
    let finish = |mut tree: Tree| -> (f64, f64) {
        tree = nni(&tree, leaves, None).expect("step 6").tree;
        let mut state =
            NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
        let loglik = optimise_branch_lengths(&mut tree, &mut state, None).expect("step 7");
        let rf = robinson_foulds(&tree, &sim.tree).expect("rf") as f64;
        (rf, loglik)
    };

    let t0 = Instant::now();
    let with_tree = spr(&base, leaves, None).expect("step 5").tree;
    let (rf_with, ll_with) = finish(with_tree.clone());
    let with = [t0.elapsed().as_secs_f64(), rf_with, ll_with];

    let t0 = Instant::now();
    let (rf_without, ll_without) = finish(base);
    let without = [t0.elapsed().as_secs_f64(), rf_without, ll_without];

    // The beam, on the tree the full pipeline actually produced. The query is a
    // cell already in the tree, which is the fixture `model::place`'s own
    // tolerance measurement used.
    let (eff_m, eff_w) =
        collapse_onto_every_node(&with_tree, leaves.means, leaves.precisions, p).expect("collapse");
    let mut scored = 0usize;
    for q_index in 0..BEAM_QUERIES {
        let leaf = q_index * n / BEAM_QUERIES;
        let lo = leaf * p;
        let placement = place(
            &with_tree,
            EffLeaf {
                m: &leaves.means[lo..lo + p],
                w: &leaves.precisions[lo..lo + p],
            },
            |node| {
                let at = node as usize * p;
                EffLeaf {
                    m: &eff_m[at..at + p],
                    w: &eff_w[at..at + p],
                }
            },
            None,
        )
        .expect("placement");
        scored += placement.scored;
    }

    (with, without, scored as f64 / BEAM_QUERIES as f64)
}

fn main() {
    // Each block on its own, because the whole thing is hours and the three
    // answer different questions. No argument runs all of them.
    let want: Vec<String> = std::env::args().skip(1).collect();
    let size = want.is_empty() || want.iter().any(|a| a == "size");
    let noise_sweep = want.is_empty() || want.iter().any(|a| a == "noise");
    let steps_sweep = want.is_empty() || want.iter().any(|a| a == "steps");
    let spr_block = want.is_empty() || want.iter().any(|a| a == "spr");

    println!("threads {}", rayon::current_num_threads());

    if size {
        println!("\n=== size sweep, noise {NOISE} ===");
        println!(
            "{:>7} {:>6} {:>8} {:>9} {:>9} {:>9} {:>7} {:>14}",
            "leaves", "feat", "start", "build s", "refine s", "total s", "RF", "loglik"
        );
        for &p in FEATURES.iter() {
            for &n in LEAVES.iter() {
                for (start, build, refine_s, rf, loglik) in run(n, p, NOISE) {
                    println!(
                        "{n:>7} {p:>6} {:>8} {build:>9.2} {refine_s:>9.2} {:>9.2} {rf:>7.1} {loglik:>14.1}",
                        start.name(),
                        build + refine_s
                    );
                }
                println!("{:>7} {:>6} {:>8} (of {} splits)", "", "", "", 2 * (n - 3));
            }
        }
    }

    if noise_sweep {
        println!(
            "\n=== noise sweep, {NOISE_SWEEP_LEAVES} leaves by {NOISE_SWEEP_FEATURES} features ==="
        );
        println!(
            "{:>7} {:>8} {:>9} {:>9} {:>9} {:>7} {:>14}",
            "noise", "start", "build s", "refine s", "total s", "RF", "loglik"
        );
        for &noise in NOISES.iter() {
            for (start, build, refine_s, rf, loglik) in
                run(NOISE_SWEEP_LEAVES, NOISE_SWEEP_FEATURES, noise)
            {
                println!(
                    "{noise:>7.1} {:>8} {build:>9.2} {refine_s:>9.2} {:>9.2} {rf:>7.1} {loglik:>14.1}",
                    start.name(),
                    build + refine_s
                );
            }
        }
        println!(
            "{:>7} {:>8} (of {} splits)",
            "",
            "",
            2 * (NOISE_SWEEP_LEAVES - 3)
        );
    }

    if steps_sweep {
        println!("\n=== per-step split from a Ward start, {STEP_FEATURES} features ===");
        println!(
            "{:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
            "leaves", "ward", "3 poly", "4 branch", "5 spr", "6 nni", "7 branch", "total s", "RF"
        );
        let mut previous: Option<(usize, [f64; 6])> = None;
        for &n in STEP_LEAVES.iter() {
            let mut t = [0.0f64; 6];
            let mut rf = 0.0f64;
            for &seed in STEP_SEEDS.iter() {
                let (one, one_rf) = step_split(n, seed);
                for k in 0..6 {
                    t[k] += one[k] / STEP_SEEDS.len() as f64;
                }
                rf += one_rf / STEP_SEEDS.len() as f64;
            }
            let total: f64 = t.iter().sum();
            println!(
                "{n:>7} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {total:>9.2} {rf:>7.1}",
                t[0], t[1], t[2], t[3], t[4], t[5]
            );
            if let Some((prev_n, prev)) = previous {
                let factor = (n / prev_n) as f64;
                print!("{:>7}", "exps");
                for k in 0..6 {
                    // Below a tenth of a second the ratio is timer noise, not an
                    // exponent, and printing one invites it to be quoted.
                    let e = if prev[k] > 0.1 {
                        (t[k] / prev[k]).ln() / factor.ln()
                    } else {
                        f64::NAN
                    };
                    print!(" {e:>9.2}");
                }
                println!("   {:>9.2}   (n^exponent)", {
                    let prev_total: f64 = prev.iter().sum();
                    (total / prev_total).ln() / factor.ln()
                });
            }
            previous = Some((n, t));
            if total > STEP_BUDGET_SECONDS {
                println!("        (budget reached, stopping)");
                break;
            }
        }
    }

    if !spr_block {
        return;
    }
    println!("\n=== is SPR earning its 67 per cent? {STEP_FEATURES} features, Ward start ===");
    println!(
        "{:>7} {:>9} {:>7} {:>14} {:>9} {:>7} {:>14} {:>10} {:>9}",
        "leaves", "spr s", "RF", "loglik", "no-spr s", "RF", "loglik", "beam nodes", "of tree"
    );
    let mut previous: Option<(usize, f64)> = None;
    for &n in STEP_LEAVES.iter() {
        let reps = STEP_SEEDS.len() as f64;
        let (mut with, mut without, mut beam) = ([0.0f64; 3], [0.0f64; 3], 0.0f64);
        for &seed in STEP_SEEDS.iter() {
            let (w, wo, b) = spr_ablation(n, seed, NOISE);
            for k in 0..3 {
                with[k] += w[k] / reps;
                without[k] += wo[k] / reps;
            }
            beam += b / reps;
        }
        println!(
            "{n:>7} {:>9.2} {:>7.1} {:>14.1} {:>9.2} {:>7.1} {:>14.1} {beam:>10.1} {:>8.1}%",
            with[0],
            with[1],
            with[2],
            without[0],
            without[1],
            without[2],
            100.0 * beam / (2.0 * n as f64)
        );
        if let Some((prev_n, prev_beam)) = previous {
            let factor = (n / prev_n) as f64;
            println!(
                "{:>7} beam n^{:.2}",
                "",
                (beam / prev_beam).ln() / factor.ln()
            );
        }
        previous = Some((n, beam));
        if with[0] > STEP_BUDGET_SECONDS {
            println!("        (budget reached, stopping)");
            break;
        }
    }

    // Noise 0.3 recovers the whole topology from the Ward start alone, so it
    // cannot show SPR earning anything. Turn the noise up to where recovery
    // breaks and ask again.
    println!("\n=== the same ablation where recovery breaks, {SPR_NOISE_LEAVES} leaves ===");
    println!(
        "{:>7} {:>9} {:>7} {:>14} {:>9} {:>7} {:>14} {:>10}",
        "noise", "spr s", "RF", "loglik", "no-spr s", "RF", "loglik", "splits won"
    );
    for &noise in NOISES.iter() {
        let reps = STEP_SEEDS.len() as f64;
        let (mut with, mut without) = ([0.0f64; 3], [0.0f64; 3]);
        for &seed in STEP_SEEDS.iter() {
            let (w, wo, _) = spr_ablation(SPR_NOISE_LEAVES, seed, noise);
            for k in 0..3 {
                with[k] += w[k] / reps;
                without[k] += wo[k] / reps;
            }
        }
        println!(
            "{noise:>7.1} {:>9.2} {:>7.1} {:>14.1} {:>9.2} {:>7.1} {:>14.1} {:>10.1}",
            with[0],
            with[1],
            with[2],
            without[0],
            without[1],
            without[2],
            without[1] - with[1]
        );
    }
    println!("{:>7} (of {} splits)", "", 2 * (SPR_NOISE_LEAVES - 3));
}
