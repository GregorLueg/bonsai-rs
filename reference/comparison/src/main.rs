//! Comparison harness for bonsai-rs: builds test data, runs the search, and
//! scores the resulting tree against the generating tree and (optionally)
//! Newick output from another implementation.
//!
//! * `gen` writes one simulated dataset as CSV, along with the generating tree
//!   and the noise-free true positions.
//! * `prep` turns a Sanity-shaped subset folder (as `prep.py select` writes it)
//!   into the CSV `ours` reads: the S5 posterior-to-likelihood conversion and
//!   the per-feature scale transform, through the crate's own ingest.
//! * `ours` runs the eight search steps over the CSV one at a time, timing each,
//!   and emits a Newick string, a wall time and a per-step table.
//! * `backbone` runs `backbone::backbone` over the same CSV: search a random
//!   subset, place the rest, refine the whole tree. One wall time, since the
//!   crate does not split its phases.
//! * `score` loads every Newick it can find for a configuration, puts them all
//!   on the same leaf indexing, and reports distance recovery, Robinson-Foulds
//!   and loglikelihood, plus one layout CSV per tree. A Newick named
//!   `theirs.nwk`, if present, is scored under the label "reference": any
//!   second tree the caller wants compared against ours and the truth.
//!
//! Everything is deterministic: the simulator seed is a command-line argument
//! and the pair sample used by distance recovery is fixed at [`PAIR_SEED`].
//!
//! Units: `ours/means.csv` and `ours/sds.csv` are always in the transformed
//! units `PreparedData` holds, whether `gen` or `prep` wrote them, so `ours` and
//! `score` never rescale anything.

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bonsai_rs::backbone::BackboneParams;
use bonsai_rs::bonsai::BonsaiParams;
use bonsai_rs::ingest::{IngestParams, PreparedData, from_sanity, prepare};
use bonsai_rs::model::global::optimise_branch_lengths;
use bonsai_rs::model::likelihood::NodeState;
use bonsai_rs::search::Leaves;
use bonsai_rs::search::bounds::EllipsoidBounds;
use bonsai_rs::search::candidates::KnnCandidates;
use bonsai_rs::search::nni::nni;
use bonsai_rs::search::polytomy::resolve_polytomies;
use bonsai_rs::search::spr::spr;
use bonsai_rs::search::star::{Star, star_tree_with};
use bonsai_rs::tree::distance::{MAX_PAIRS, distance_recovery};
use bonsai_rs::tree::export::layout_csv;
use bonsai_rs::tree::layout::equal_angle;
use bonsai_rs::tree::newick::{parse_newick, write_newick};
use bonsai_rs::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};
use bonsai_rs::tree::{NO_NODE, Tree};

/// Seed for the leaf-pair sample inside distance recovery. Fixed so that the
/// two implementations are scored on exactly the same pairs.
const PAIR_SEED: u64 = 20260906;

/// Spread of the per-cell per-feature error bars about `noise_sd`, passed to
/// the simulator. Kept off `1.0` so the precision weighting is exercised.
const NOISE_SPREAD: f64 = 2.0;

/// Branch length of every edge in the generating tree.
const BRANCH_LENGTH: f64 = 1.0;

/// Branch every leaf hangs on before step 1 optimises it. One in transformed
/// units, which is what the crate's own pipeline starts from.
const INITIAL_STAR_BRANCH: f64 = 1.0;

type Fallible<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("harness: {e}");
        std::process::exit(1);
    }
}

/// Dispatch on the subcommand.
fn run() -> Fallible<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => {
            if args.len() != 7 {
                return Err("usage: harness gen <dir> <n_leaves> <n_features> <noise_sd> <seed>"
                    .into());
            }
            generate(
                Path::new(&args[2]),
                args[3].parse()?,
                args[4].parse()?,
                args[5].parse()?,
                args[6].parse()?,
            )
        }
        Some("prep") => {
            if args.len() != 3 {
                return Err("usage: harness prep <dir>".into());
            }
            prep(Path::new(&args[2]))
        }
        Some("ours") => {
            if args.len() != 3 {
                return Err("usage: harness ours <dir>".into());
            }
            ours(Path::new(&args[2]))
        }
        Some("backbone") => {
            if !(3..=4).contains(&args.len()) {
                return Err("usage: harness backbone <dir> [backbone_cells]".into());
            }
            let cells = args.get(3).map(|v| v.parse()).transpose()?;
            backbone(Path::new(&args[2]), cells)
        }
        Some("score") => {
            if args.len() != 3 {
                return Err("usage: harness score <dir>".into());
            }
            score(Path::new(&args[2]))
        }
        Some("score-tree") => {
            if args.len() != 4 {
                return Err("usage: harness score-tree <dir> <newick>".into());
            }
            score_tree(Path::new(&args[2]), Path::new(&args[3]))
        }
        _ => Err("usage: harness <gen|prep|ours|backbone|score|score-tree> ...".into()),
    }
}

/////////////////////
// Small utilities //
/////////////////////

/// Render a row-major `[row][col]` matrix as delimited text, one line per row.
fn matrix_text(values: &[f64], n_rows: usize, n_cols: usize, sep: char) -> String {
    let mut out = String::with_capacity(n_rows * n_cols * 12);
    for r in 0..n_rows {
        for c in 0..n_cols {
            if c > 0 {
                out.push(sep);
            }
            let _ = write!(out, "{}", values[r * n_cols + c]);
        }
        out.push('\n');
    }
    out
}

/// Read a headerless comma-separated numeric matrix as row-major values plus
/// its shape. Rejects ragged input rather than padding it.
fn read_csv(path: &Path) -> Fallible<(Vec<f64>, usize, usize)> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut values: Vec<f64> = Vec::new();
    let mut n_rows = 0usize;
    let mut n_cols = 0usize;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut cols = 0usize;
        for field in line.split(',') {
            let v: f64 = field.trim().parse().map_err(|_| {
                format!(
                    "{}: line {}: {:?} is not a number",
                    path.display(),
                    line_no + 1,
                    field
                )
            })?;
            values.push(v);
            cols += 1;
        }
        if n_rows == 0 {
            n_cols = cols;
        } else if cols != n_cols {
            return Err(format!(
                "{}: line {} has {cols} fields, expected {n_cols}",
                path.display(),
                line_no + 1
            )
            .into());
        }
        n_rows += 1;
    }
    if n_rows == 0 || n_cols == 0 {
        return Err(format!("{}: no data", path.display()).into());
    }
    Ok((values, n_rows, n_cols))
}

/// One label per line.
fn read_lines(path: &Path) -> Fallible<Vec<String>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// Cell labels, `cell0 .. cell{n-1}`, indexed by leaf index.
fn cell_labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("cell{i}")).collect()
}

/////////////////////////
// gen: write the data //
/////////////////////////

/// Simulate one dataset and write it as CSV.
///
/// Layout under `dir`:
///
/// * `ours/means.csv`, `ours/sds.csv`, `ours/truth.csv` - comma separated,
///   cells as rows and features as columns, no header
/// * `truth.nwk` - the generating tree
/// * `meta.tsv` - the parameters, for the results table
fn generate(dir: &Path, n_leaves: usize, n_features: usize, noise_sd: f64, seed: u64) -> Fallible<()> {
    let data = simulate_binary::<f64>(Some(SimulationParams {
        n_leaves,
        n_features,
        branch_length: BRANCH_LENGTH,
        noise_sd,
        noise_spread: NOISE_SPREAD,
        feature_mean_sd: 0.0,
        seed,
    }))?;

    let our_dir = dir.join("ours");
    fs::create_dir_all(&our_dir)?;
    let labels = cell_labels(n_leaves);

    fs::write(
        our_dir.join("means.csv"),
        matrix_text(&data.means, n_leaves, n_features, ','),
    )?;
    fs::write(
        our_dir.join("sds.csv"),
        matrix_text(&data.sds, n_leaves, n_features, ','),
    )?;
    fs::write(
        our_dir.join("truth.csv"),
        matrix_text(&data.truth, n_leaves, n_features, ','),
    )?;

    fs::write(dir.join("truth.nwk"), write_newick(&data.tree, &labels)? + "\n")?;
    fs::write(
        dir.join("meta.tsv"),
        format!(
            "n_leaves\t{n_leaves}\nn_features\t{n_features}\nnoise_sd\t{noise_sd}\n\
             noise_spread\t{NOISE_SPREAD}\nbranch_length\t{BRANCH_LENGTH}\nseed\t{seed}\n"
        ),
    )?;
    println!("gen: {n_leaves} cells by {n_features} features into {}", dir.display());
    Ok(())
}

/////////////////////////////////
// prep: Sanity subset to CSV  //
/////////////////////////////////

/// Read a Sanity genes-by-cells matrix, tab separated, no header, no index,
/// tolerating the trailing tab Sanity leaves on every line. Returned
/// transposed to row-major `[cell][gene]`.
fn read_sanity_matrix(path: &Path, n_genes: usize, n_cells: usize) -> Fallible<Vec<f64>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = vec![0.0f64; n_cells * n_genes];
    let mut g = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if g >= n_genes {
            return Err(format!("{}: more than {n_genes} rows", path.display()).into());
        }
        let mut c = 0usize;
        for field in line.split('\t') {
            if field.is_empty() {
                continue;
            }
            if c >= n_cells {
                return Err(format!(
                    "{}: row {g} has more than {n_cells} fields",
                    path.display()
                )
                .into());
            }
            out[c * n_genes + g] = field
                .parse()
                .map_err(|_| format!("{}: row {g}: {field:?} is not a number", path.display()))?;
            c += 1;
        }
        if c != n_cells {
            return Err(format!(
                "{}: row {g} has {c} fields, expected {n_cells}",
                path.display()
            )
            .into());
        }
        g += 1;
    }
    if g != n_genes {
        return Err(format!("{}: {g} rows, expected {n_genes}", path.display()).into());
    }
    Ok(out)
}

/// Read a one-value-per-line vector.
fn read_vector(path: &Path) -> Fallible<Vec<f64>> {
    read_lines(path)?
        .iter()
        .map(|l| {
            l.parse::<f64>()
                .map_err(|_| format!("{}: {l:?} is not a number", path.display()).into())
        })
        .collect()
}

/// Convert `dir/sanity_sel` into `dir/ours/means.csv` and `dir/ours/sds.csv`,
/// in the transformed units `PreparedData` holds, so that `ours` and `score`
/// run unchanged on this track.
///
/// Sanity's `delta.txt` and `d_delta.txt` are the posterior log fold changes
/// and their error bars, `variance.txt` its `v[g]`. `from_sanity` applies the
/// S5 conversion to likelihood means and error bars, `prepare` divides by
/// `sqrt(v[g])`. The signal-to-noise threshold is zero because the subset is
/// already the selected gene set; a feature `from_sanity` drops as
/// ill-conditioned is counted in `prep.tsv` and removed from our side only,
/// since the reference reads the subset folder as it is.
fn prep(dir: &Path) -> Fallible<()> {
    let sel = dir.join("sanity_sel");
    let genes = read_lines(&sel.join("geneID.txt"))?;
    let cells = read_lines(&sel.join("cellID.txt"))?;
    let (n_genes, n_cells) = (genes.len(), cells.len());
    if cells != cell_labels(n_cells) {
        return Err("cellID.txt is not cell0..cell{n-1} in order; the scorer relies on that".into());
    }
    let t0 = Instant::now();
    let delta = read_sanity_matrix(&sel.join("delta.txt"), n_genes, n_cells)?;
    let d_delta = read_sanity_matrix(&sel.join("d_delta.txt"), n_genes, n_cells)?;
    let variance = read_vector(&sel.join("variance.txt"))?;
    if variance.len() != n_genes {
        return Err(format!(
            "variance.txt has {} values for {n_genes} genes",
            variance.len()
        )
        .into());
    }
    let read_secs = t0.elapsed().as_secs_f64();

    let params = IngestParams {
        min_signal_to_noise: 0.0,
        ..Default::default()
    };
    let t0 = Instant::now();
    let lik = from_sanity::<f64>(&delta, &d_delta, n_cells, n_genes, &variance, Some(params))?;
    let k = lik.features.len();
    let prepared: PreparedData<f64> = prepare(
        &lik.means,
        &lik.sds,
        n_cells,
        k,
        Some(&lik.variances),
        Some(params),
    )?;
    let ingest_secs = t0.elapsed().as_secs_f64();
    if prepared.n_features() != k {
        return Err(format!(
            "prepare kept {} of {k} features at threshold zero",
            prepared.n_features()
        )
        .into());
    }

    let our_dir = dir.join("ours");
    fs::create_dir_all(&our_dir)?;
    let sds: Vec<f64> = prepared
        .transformed_precisions
        .iter()
        .map(|&w| 1.0 / w.sqrt())
        .collect();
    fs::write(
        our_dir.join("means.csv"),
        matrix_text(&prepared.transformed_means, n_cells, k, ','),
    )?;
    fs::write(our_dir.join("sds.csv"), matrix_text(&sds, n_cells, k, ','))?;
    fs::write(
        dir.join("prep.tsv"),
        format!(
            "n_cells\t{n_cells}\nn_genes_in\t{n_genes}\nn_features_ours\t{k}\n\
             dropped_ill_conditioned\t{}\nread_seconds\t{read_secs:.2}\ningest_seconds\t{ingest_secs:.2}\n",
            lik.dropped.len()
        ),
    )?;
    println!(
        "prep: {n_cells} cells, {k} of {n_genes} genes kept ({} ill-conditioned dropped), \
         read {read_secs:.1} s, ingest {ingest_secs:.2} s",
        lik.dropped.len()
    );
    Ok(())
}

///////////////////////////
// ours: run bonsai-rs   //
///////////////////////////

/// Loglikelihood of a tree from the leaf data, via the crate's own
/// `NodeState::prune`.
fn tree_loglik(tree: &Tree, leaves: Leaves<'_, f64>) -> Fallible<f64> {
    let mut state = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )?;
    Ok(state.prune(tree))
}

/// Run our implementation over the CSV, one search step at a time, and write
/// `ours.nwk`, `ours_seconds.txt` and `steps.tsv`.
///
/// The steps are the ones `bonsai_rs::bonsai::bonsai_prepared` runs, called
/// through the same public functions with the same default parameters, so the
/// tree is the one `bonsai_prepared` would return before its display reroot.
/// Not reproduced: the reroot and the per-node posteriors, neither of which
/// touches the topology or the loglikelihood. `ours_seconds.txt` is the sum of
/// the eight step timings; the loglikelihood evaluations between steps are
/// timed separately, reported in `steps.tsv`, and excluded from the sum.
///
/// The CSV is already in transformed units, so it goes straight in with every
/// feature retained, bypassing ingest. That is what makes the two sides
/// comparable: neither one gets to drop features the other kept.
fn ours(dir: &Path) -> Fallible<()> {
    let (means, n_cells, p) = read_csv(&dir.join("ours").join("means.csv"))?;
    let (sds, sd_cells, sd_p) = read_csv(&dir.join("ours").join("sds.csv"))?;
    if (n_cells, p) != (sd_cells, sd_p) {
        return Err(format!(
            "means are {n_cells}x{p} but standard deviations are {sd_cells}x{sd_p}"
        )
        .into());
    }
    let precisions: Vec<f64> = sds.iter().map(|&s| 1.0 / (s * s)).collect();
    if precisions.iter().any(|v| !v.is_finite()) {
        return Err("a standard deviation is zero or not finite".into());
    }
    let leaves = Leaves {
        means: &means,
        precisions: &precisions,
        n_features: p,
    };
    let mut params = BonsaiParams::default();
    // Knob overrides for the NNI attribution experiments, from the environment.
    // OURS_TAG names the outputs.
    let env_usize = |k: &str| env::var(k).ok().map(|v| v.parse::<usize>()).transpose();
    let env_f64 = |k: &str| env::var(k).ok().map(|v| v.parse::<f64>()).transpose();
    if let Some(v) = env_usize("NNI_RANDOM")? {
        params.nni.n_random = v;
    }
    if let Some(v) = env_usize("NNI_SEED")? {
        params.nni.seed = v as u64;
    }
    if let Some(v) = env_usize("NNI_MAX_ROUNDS")? {
        params.nni.max_rounds = v;
    }
    if let Some(v) = env_f64("NNI_MIN_GAIN")? {
        params.nni.star.min_gain = v;
    }
    // SPR's cap is documented as a runaway guard rather than a working limit,
    // justified on fixtures of 16 to 64 leaves. At 10k it binds: the 2026-09-13
    // run accepted 4843 moves and stopped at exactly 100 rounds, so it was
    // truncated rather than converged.
    if let Some(v) = env_usize("SPR_MAX_ROUNDS")? {
        params.spr.max_rounds = v;
    }
    // SPR's acceptance floor. The default 1e-9 was measured against the merge
    // score, magnitude O(p) with a rounding floor near 1e-12. SPR compares
    // whole-tree loglikelihoods, magnitude O(n p): at 10k cells that is 1.1e7,
    // whose f64 rounding floor is about 2.4e-9, so the default sits *below* the
    // noise and SPR accepts neutral moves for ever.
    if let Some(v) = env_f64("SPR_MIN_GAIN")? {
        params.spr.star.min_gain = v;
    }
    let tag = env::var("OURS_TAG").ok().filter(|t| !t.is_empty());
    let named = |base: &str, ext: &str| match &tag {
        Some(t) => format!("{base}_{t}.{ext}"),
        None => format!("{base}.{ext}"),
    };
    println!(
        "ours: nni n_random {} seed {} max_rounds {} min_gain {:e}",
        params.nni.n_random, params.nni.seed, params.nni.max_rounds, params.nni.star.min_gain
    );

    let mut table = String::from("step\tseconds\tloglik\tgain\tloglik_seconds\n");
    let mut total = 0.0f64;
    let mut last: Option<f64> = None;
    let mut record = |step: &str, secs: f64, loglik: f64, loglik_secs: f64| {
        let gain = last.map_or(0.0, |l| loglik - l);
        last = Some(loglik);
        total += secs;
        let _ = writeln!(
            table,
            "{step}\t{secs:.3}\t{loglik:.3}\t{gain:.3}\t{loglik_secs:.3}"
        );
        println!("ours: {step:<10} {secs:>9.2} s   loglik {loglik:>16.2}   gain {gain:>12.2}");
    };

    // START=linkage replaces steps 1 and 2 with the Ward linkage
    // (`StartTree::Linkage`); anything else is the greedy merge.
    let use_linkage = env::var("START").map(|s| s == "linkage").unwrap_or(false);

    let mut tree: Tree = if use_linkage {
        let t0 = Instant::now();
        let tree = bonsai_rs::tree::linkage::linkage_tree(leaves.means, n_cells, p, None)?;
        let secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let loglik = tree_loglik(&tree, leaves)?;
        record("1-2 linkage", secs, loglik, t1.elapsed().as_secs_f64());
        tree
    } else {
        // Step 1: a star with optimised branch lengths.
        let mut parent = vec![n_cells as u32; n_cells];
        parent.push(NO_NODE);
        let mut branch = vec![INITIAL_STAR_BRANCH; n_cells];
        branch.push(0.0);
        let mut star = Tree::from_parents(parent, branch, n_cells)?;
        let mut state = NodeState::new(star.n_nodes(), p, leaves.means, leaves.precisions)?;
        let t0 = Instant::now();
        let loglik = optimise_branch_lengths(&mut star, &mut state, Some(params.branch))?;
        record("1 star", t0.elapsed().as_secs_f64(), loglik, 0.0);

        // Step 2: greedy merging under the kNN restriction and ellipsoid
        // bounds, the same composition `bonsai_prepared` uses.
        let t0 = Instant::now();
        let mut candidates =
            EllipsoidBounds::new(KnnCandidates::new(Some(params.knn)), Some(params.bounds));
        let (tree, _) = star_tree_with(
            Star {
                means: leaves.means,
                precisions: leaves.precisions,
                branch: &star.branches()[..n_cells],
                n_features: p,
            },
            Some(params.star),
            &mut candidates,
        )?;
        let secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let loglik = tree_loglik(&tree, leaves)?;
        record("2 merge", secs, loglik, t1.elapsed().as_secs_f64());
        tree
    };
    let after_merge = tree.clone();

    // Step 3.
    let t0 = Instant::now();
    tree = resolve_polytomies(&tree, leaves, Some(params.star))?.tree;
    let secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let loglik = tree_loglik(&tree, leaves)?;
    record("3 polytomy", secs, loglik, t1.elapsed().as_secs_f64());

    // Step 4.
    let mut state = NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions)?;
    let t0 = Instant::now();
    let loglik = optimise_branch_lengths(&mut tree, &mut state, Some(params.branch))?;
    record("4 branch", t0.elapsed().as_secs_f64(), loglik, 0.0);

    // Step 5.
    let t0 = Instant::now();
    let spr_result = spr(&tree, leaves, Some(params.spr))?;
    tree = spr_result.tree;
    let secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let loglik = tree_loglik(&tree, leaves)?;
    record("5 spr", secs, loglik, t1.elapsed().as_secs_f64());
    println!(
        "ours: spr accepted {} moves over {} rounds",
        spr_result.gains.len(),
        spr_result.rounds
    );
    let after_spr = tree.clone();

    // Step 6.
    let t0 = Instant::now();
    let nni_result = nni(&tree, leaves, Some(params.nni))?;
    tree = nni_result.tree;
    let secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let loglik = tree_loglik(&tree, leaves)?;
    record("6 nni", secs, loglik, t1.elapsed().as_secs_f64());
    println!(
        "ours: nni performed {} moves over {} greedy rounds",
        nni_result.n_moves, nni_result.rounds
    );
    let after_nni = tree.clone();

    // Step 7.
    let mut state = NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions)?;
    let t0 = Instant::now();
    let loglik = optimise_branch_lengths(&mut tree, &mut state, Some(params.branch))?;
    record("7 branch", t0.elapsed().as_secs_f64(), loglik, 0.0);

    // Step 8, the crate's deviation: collapse the internal zero-length edges
    // steps 4 to 7 leave behind, then reoptimise. Step 3 is the only other
    // collapse and it runs before step 4, so nothing else removes these.
    let t0 = Instant::now();
    tree = resolve_polytomies(&tree, leaves, Some(params.star))?.tree;
    let mut state = NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions)?;
    let loglik = optimise_branch_lengths(&mut tree, &mut state, Some(params.branch))?;
    record("8 collapse", t0.elapsed().as_secs_f64(), loglik, 0.0);

    let labels = cell_labels(n_cells);
    // Trees at the stage boundaries, so that ours can be scored stage by stage.
    let stages_dir = dir.join(match &tag {
        Some(t) => format!("ours_stages_{t}"),
        None => "ours_stages".to_string(),
    });
    fs::create_dir_all(&stages_dir)?;
    for (name, t) in [("2_merge", &after_merge), ("5_spr", &after_spr), ("6_nni", &after_nni)] {
        fs::write(stages_dir.join(format!("{name}.nwk")), write_newick(t, &labels)? + "\n")?;
    }
    fs::write(dir.join(named("ours", "nwk")), write_newick(&tree, &labels)? + "\n")?;
    fs::write(dir.join(named("ours_seconds", "txt")), format!("{total}\n"))?;
    fs::write(dir.join(named("steps", "tsv")), &table)?;
    fs::write(
        dir.join(named("moves", "tsv")),
        format!(
            "spr_moves\t{}\nspr_rounds\t{}\nnni_moves\t{}\nnni_rounds\t{}\nnni_n_random\t{}\n\
             nni_max_rounds\t{}\nnni_min_gain\t{:e}\n",
            spr_result.gains.len(),
            spr_result.rounds,
            nni_result.n_moves,
            nni_result.rounds,
            params.nni.n_random,
            params.nni.max_rounds,
            params.nni.star.min_gain
        ),
    )?;
    println!("ours: {n_cells} cells by {p} features in {total:.2} s, loglik {loglik:.1}");
    Ok(())
}

/// Run backbone mode over the CSV and write `backbone.nwk`,
/// `backbone_seconds.txt` and `backbone_steps.tsv`.
///
/// Same input as [`ours`], wrapped as a `PreparedData` with every feature kept
/// and unit variances, the way `backbone`'s own subset step builds one, so
/// ingest never runs and both searches see identical leaves. The wall time
/// covers the whole call: backbone search, placement and the final refinement.
/// `backbone_steps.tsv` holds the refinement's per-step loglikelihoods and the
/// growth report.
///
/// With `BACKBONE_TAG` set, the outputs are suffixed `_<tag>`.
fn backbone(dir: &Path, backbone_cells: Option<usize>) -> Fallible<()> {
    let (means, n_cells, p) = read_csv(&dir.join("ours").join("means.csv"))?;
    let (sds, sd_cells, sd_p) = read_csv(&dir.join("ours").join("sds.csv"))?;
    if (n_cells, p) != (sd_cells, sd_p) {
        return Err(format!(
            "means are {n_cells}x{p} but standard deviations are {sd_cells}x{sd_p}"
        )
        .into());
    }
    let precisions: Vec<f64> = sds.iter().map(|&s| 1.0 / (s * s)).collect();
    if precisions.iter().any(|v| !v.is_finite()) {
        return Err("a standard deviation is zero or not finite".into());
    }
    let data = PreparedData {
        transformed_means: means,
        transformed_precisions: precisions,
        features: (0..p).collect(),
        variances: vec![1.0; p],
        signal_to_noise: vec![f64::INFINITY; p],
        n_cells,
        n_features_in: p,
    };
    let mut params = BackboneParams::default();
    if let Some(c) = backbone_cells {
        params.backbone_cells = c;
    }
    let tag = env::var("BACKBONE_TAG").ok().filter(|t| !t.is_empty());
    let named = |base: &str, ext: &str| match &tag {
        Some(t) => format!("{base}_{t}.{ext}"),
        None => format!("{base}.{ext}"),
    };

    let t0 = Instant::now();
    let (out, report) = bonsai_rs::backbone::backbone(&data, Some(params.clone()))?;
    let secs = t0.elapsed().as_secs_f64();

    let mut table = String::from("step\tloglik\tgain\n");
    for s in &out.steps {
        let _ = writeln!(table, "{}\t{:.3}\t{:.3}", s.step, s.loglik, s.gain);
    }
    let _ = writeln!(
        table,
        "# backbone_cells\t{}\n# placed\t{}\n# reoptimisations\t{}\n# mean_scored\t{:.2}",
        report.backbone_cells, report.placed, report.reoptimisations, report.mean_scored
    );

    fs::write(
        dir.join(named("backbone", "nwk")),
        write_newick(&out.tree, &cell_labels(n_cells))? + "\n",
    )?;
    fs::write(dir.join(named("backbone_seconds", "txt")), format!("{secs}\n"))?;
    fs::write(dir.join(named("backbone_steps", "tsv")), &table)?;
    println!(
        "backbone: {n_cells} cells by {p} features, backbone {} placed {} reopt {} \
         in {secs:.2} s, loglik {:.1}",
        report.backbone_cells, report.placed, report.reoptimisations, out.loglik
    );
    Ok(())
}

/// Score one Newick file, from anywhere, against a configuration's data:
/// loglikelihood under our scorer, Robinson-Foulds to the generating tree and
/// distance recovery. For stage trees or any other tree, so that searches can
/// be compared under one scorer.
fn score_tree(dir: &Path, nwk: &Path) -> Fallible<()> {
    let (means, n_cells, p) = read_csv(&dir.join("ours").join("means.csv"))?;
    let (sds, _, _) = read_csv(&dir.join("ours").join("sds.csv"))?;
    let (truth, _, _) = read_csv(&dir.join("ours").join("truth.csv"))?;
    let precisions: Vec<f64> = sds.iter().map(|&s| 1.0 / (s * s)).collect();
    let labels = cell_labels(n_cells);
    let index_of: HashMap<&str, usize> =
        labels.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let truth_tree = load_tree(&dir.join("truth.nwk"), &index_of)?;
    let tree = load_tree(nwk, &index_of)?;
    let recovery = distance_recovery(&tree, &truth, p, MAX_PAIRS, PAIR_SEED);
    let rf = robinson_foulds(&tree, &truth_tree)?;
    let mut state = NodeState::new(tree.n_nodes(), p, &means, &precisions)?;
    let loglik = state.prune(&tree);
    println!(
        "{}\tloglik\t{loglik:.3}\trf\t{rf}\trecovery\t{recovery:.6}\tnodes\t{}",
        nwk.display(),
        tree.n_nodes()
    );
    Ok(())
}

//////////////////////////////
// score: compare the trees //
//////////////////////////////

/// Rebuild a parsed tree on the canonical leaf indexing.
///
/// [`parse_newick`] numbers leaves in the order they appear in the string,
/// which is not the order the cells were simulated in, and every metric here
/// matches leaves by index. Only leaf slots move; internal nodes keep their
/// indices, so the arena's "parent index exceeds its children's" invariant
/// survives the permutation untouched.
fn relabel(tree: &Tree, labels: &[String], index_of: &HashMap<&str, usize>) -> Fallible<Tree> {
    let n_nodes = tree.n_nodes();
    let n_leaves = tree.n_leaves();
    if labels.len() != n_leaves {
        return Err(format!("{} labels for {n_leaves} leaves", labels.len()).into());
    }
    let mut parent = vec![NO_NODE; n_nodes];
    let mut branch = vec![0.0f64; n_nodes];
    let mut seen = vec![false; n_leaves];
    for node in 0..n_nodes {
        let dst = if node < n_leaves {
            let want = labels[node].as_str();
            let &i = index_of
                .get(want)
                .ok_or_else(|| format!("leaf label {want:?} is not one of the simulated cells"))?;
            if seen[i] {
                return Err(format!("leaf label {want:?} appears twice").into());
            }
            seen[i] = true;
            i
        } else {
            node
        };
        parent[dst] = tree.parent(node as u32).unwrap_or(NO_NODE);
        branch[dst] = tree.branch(node as u32);
    }
    Ok(Tree::from_parents(parent, branch, n_leaves)?)
}

/// Load a Newick file and put it on the canonical leaf indexing.
fn load_tree(path: &Path, index_of: &HashMap<&str, usize>) -> Fallible<Tree> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (tree, labels) =
        parse_newick(&text).map_err(|e| format!("{}: parse_newick refused it: {e}", path.display()))?;
    relabel(&tree, &labels, index_of).map_err(|e| format!("{}: {e}", path.display()).into())
}

/// The three numbers reported per tree.
struct Metrics {
    /// Pearson correlation of tree path distance against squared Euclidean
    /// distance in the **true** positions.
    recovery: f64,
    /// Robinson-Foulds to the generating tree.
    rf: usize,
    /// Tree loglikelihood under our scorer, up to the constants the crate
    /// drops, so only differences between trees are meaningful.
    loglik: f64,
}

/// Score one tree and write its layout CSV.
fn measure(
    tree: &Tree,
    truth_tree: &Tree,
    truth: &[f64],
    means: &[f64],
    precisions: &[f64],
    p: usize,
    out_csv: &Path,
    labels: &[String],
) -> Fallible<Metrics> {
    let recovery = distance_recovery(tree, truth, p, MAX_PAIRS, PAIR_SEED);
    let rf = robinson_foulds(tree, truth_tree)?;
    let mut state = NodeState::new(tree.n_nodes(), p, means, precisions)?;
    let loglik = state.prune(tree);

    // One layout function for both trees, so a visual difference is a real
    // difference and not a difference in the drawing code.
    let layout = equal_angle(tree, None)?;
    fs::write(out_csv, layout_csv(tree, &layout, labels)?)?;

    Ok(Metrics { recovery, rf, loglik })
}

/// Score every tree present for a configuration.
fn score(dir: &Path) -> Fallible<()> {
    let (means, n_cells, p) = read_csv(&dir.join("ours").join("means.csv"))?;
    let (sds, _, _) = read_csv(&dir.join("ours").join("sds.csv"))?;
    let (truth, truth_cells, truth_p) = read_csv(&dir.join("ours").join("truth.csv"))?;
    if (truth_cells, truth_p) != (n_cells, p) {
        return Err("truth.csv does not match means.csv in shape".into());
    }
    let precisions: Vec<f64> = sds.iter().map(|&s| 1.0 / (s * s)).collect();

    let labels = cell_labels(n_cells);
    let index_of: HashMap<&str, usize> =
        labels.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let truth_tree = load_tree(&dir.join("truth.nwk"), &index_of)?;

    let mut table = String::from(
        "implementation\tdistance_recovery\trobinson_foulds\trf_ours_vs_reference\tloglik\tseconds\n",
    );
    let candidates: [(&str, PathBuf, PathBuf, PathBuf); 3] = [
        (
            "truth",
            dir.join("truth.nwk"),
            dir.join("truth_layout.csv"),
            dir.join("nonexistent_seconds.txt"),
        ),
        (
            "bonsai-rs",
            dir.join("ours.nwk"),
            dir.join("ours_layout.csv"),
            dir.join("ours_seconds.txt"),
        ),
        (
            "reference",
            dir.join("theirs.nwk"),
            dir.join("theirs_layout.csv"),
            dir.join("theirs_seconds.txt"),
        ),
    ];

    // Loaded first, so that the two reconstructions can also be compared with
    // each other and not only with the generating tree. Identical scores against
    // the truth do not by themselves mean the same tree was found.
    let mut loaded: Vec<(&str, Option<Tree>, &PathBuf, &PathBuf)> = Vec::new();
    for (name, nwk, csv, secs_path) in candidates.iter() {
        if !nwk.exists() {
            println!("score: {name}: no Newick at {}, skipped", nwk.display());
            continue;
        }
        match load_tree(nwk, &index_of) {
            Ok(t) => {
                if t.n_leaves() != n_cells {
                    println!("score: {name}: {} leaves, expected {n_cells}", t.n_leaves());
                }
                loaded.push((name, Some(t), csv, secs_path));
            }
            Err(e) => {
                println!("score: {name}: FAILED: {e}");
                loaded.push((name, None, csv, secs_path));
            }
        }
    }

    let pairwise = {
        let get = |want: &str| {
            loaded
                .iter()
                .find(|(n, t, _, _)| *n == want && t.is_some())
                .and_then(|(_, t, _, _)| t.as_ref())
        };
        match (get("bonsai-rs"), get("reference")) {
            (Some(a), Some(b)) => Some(robinson_foulds(a, b)?),
            _ => None,
        }
    };

    for (name, tree, csv, secs_path) in loaded.iter() {
        let Some(tree) = tree else {
            let _ = writeln!(table, "{name}\tNA\tNA\tNA\tNA\tNA");
            continue;
        };
        let m = measure(
            tree, &truth_tree, &truth, &means, &precisions, p, csv, &labels,
        )?;
        let secs = fs::read_to_string(secs_path)
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok());
        let secs_col = match secs {
            Some(v) => format!("{v:.2}"),
            None => "NA".to_string(),
        };
        // The pairwise column is the same number on both reconstruction rows;
        // the generating tree has no counterpart, so it gets NA.
        let pair_col = match (*name, pairwise) {
            ("truth", _) | (_, None) => "NA".to_string(),
            (_, Some(v)) => v.to_string(),
        };
        let _ = writeln!(
            table,
            "{name}\t{:.6}\t{}\t{pair_col}\t{:.2}\t{secs_col}",
            m.recovery, m.rf, m.loglik
        );
        println!(
            "score: {name:>10}  recovery {:.4}  RF {:>5}  RF-pair {pair_col:>4}  \
             loglik {:>14.1}  {secs_col} s",
            m.recovery, m.rf, m.loglik
        );
    }
    println!("score: maximum possible RF for {n_cells} leaves is {}", 2 * (n_cells - 3));

    fs::write(dir.join("metrics.tsv"), &table)?;
    println!("score: wrote {}", dir.join("metrics.tsv").display());
    Ok(())
}
