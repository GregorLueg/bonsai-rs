//! The whole algorithm: matrix in, tree out.
//!
//! Runs the seven steps of SPEC.md section 9, then an eighth of our own:
//! collapse the zero-length internal edges that steps 4 to 7 leave behind
//! (step 3 is the only other collapse and runs before them). It runs last so
//! it cannot change what the search finds. The arithmetic lives in `model`,
//! `search` and `tree`.
//!
//! ### Ordering constraints (not in the specification)
//!
//! Steps 5 and 6 must follow step 4: SPR and NNI reject proposals whose
//! topology fingerprint is unchanged, which before the branch lengths are
//! optimised rejects the only available improvements. Step 3 must follow step 2,
//! since the centre moves after a polytomy is created.

use std::time::Instant;

use crate::model::global::{GlobalBranchParams, collapse_onto_every_node, optimise_branch_lengths};
use crate::model::likelihood::NodeState;
use crate::prelude::*;
use crate::search::bounds::EllipsoidBounds;
use crate::search::candidates::{KnnCandidates, KnnCandidatesParams};
use crate::search::nni::nni;
use crate::search::polytomy::resolve_polytomies;
use crate::search::spr::spr;
use crate::search::star::{Star, StarParams, star_tree_with};
use crate::search::{Leaves, tree_loglik};
use crate::tree::cluster::reroot_for_display;
use crate::tree::linkage::{LinkageParams, linkage_tree};
use crate::utils::traits::narrow;

////////////
// Consts //
////////////

/// Branch length the initial star starts from; step 1 replaces it. One is the
/// signal scale after the ingest transform (SPEC.md section 3.1).
const INITIAL_STAR_BRANCH: f64 = 1.0;

///////////////
// StartTree //
///////////////

/// How the initial topology is built.
///
/// Search steps 1 and 2, or a linkage in their place; steps 3 to 7 are
/// identical either way. The linkage is the default: on real
/// Sanity-preprocessed input it beats the specified start on loglikelihood and
/// Robinson-Foulds and is several times faster (`docs/PERFORMANCE.md`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StartTree {
    /// The star of SPEC.md section 9.1, agglomerated by the merge score.
    ///
    /// Chains on real input, increasingly with cell count, and steps 3 to 7 do
    /// not recover from it (`docs/PERFORMANCE.md`). Kept because the paper
    /// specifies it: choose it to reproduce the published method, not to build
    /// the best tree.
    GreedyMerge,
    /// Ward linkage over a neighbour graph, [`crate::tree::linkage`].
    ///
    /// The default.
    #[default]
    Linkage,
}

//////////////////
// BonsaiParams //
//////////////////

/// Knobs for the whole pipeline, one field per stage.
///
/// Every field has its own documented `Default`, so the usual call passes
/// `None` and never names any of them.
#[derive(Clone, Copy, Debug, Default)]
pub struct BonsaiParams {
    /// Feature selection and the scale transform.
    pub ingest: IngestParams,
    /// How the initial topology is built, steps 1 and 2 or a linkage.
    pub start: StartTree,
    /// Knobs for the linkage, ignored unless `start` selects it.
    pub linkage: LinkageParams,
    /// The greedy star primitive and polytomy resolution, steps 2 and 3 only.
    /// Steps 5 and 6 take theirs from `spr.star` and `nni.star`.
    pub star: StarParams,
    /// Candidate-pair restriction. Without it step 2 is `O(n^3 p)`, measured
    /// `n^2.9` and 94 per cent of the runtime.
    pub knn: KnnCandidatesParams,
    /// Global branch-length optimisation (section 6), steps 1, 4 and 7.
    pub branch: GlobalBranchParams,
    /// Subtree pruning and regrafting (section 9.3), step 5.
    pub spr: SprParams,
    /// Nearest-neighbour interchange (section 9.4), step 6.
    pub nni: NniParams,
    /// Whether to reroot for display once the search is done (section 9.7).
    ///
    /// The likelihood does not depend on the root (S14). Done last because a
    /// degree-two root is degenerate for branch-length optimisation.
    pub reroot: bool,
    /// Skip the per-node posteriors, leaving `node_means` and `node_sds` empty.
    ///
    /// They are `2 * n_nodes * n_features` values; the search does not depend on
    /// them.
    pub skip_posteriors: bool,
}

////////////////
// StepReport //
////////////////

/// What one step of the search cost and what it bought.
#[derive(Clone, Copy, Debug)]
pub struct StepReport {
    /// Which step this is, `"1 star"` through `"8 collapse"`.
    pub step: &'static str,
    /// Tree loglikelihood after the step, up to the constants SPEC.md section 3
    /// drops.
    pub loglik: f64,
    /// Change from the previous step.
    ///
    /// Expected non-negative but not asserted, so a regression shows as a
    /// negative here rather than a panic across FFI.
    pub gain: f64,
    /// Wall time of the step in seconds, including the loglikelihood
    /// evaluation that closes it.
    pub seconds: f64,
}

//////////////////
// BonsaiResult //
//////////////////

/// A finished reconstruction.
#[derive(Clone, Debug)]
pub struct BonsaiResult<T> {
    /// The tree. Leaves `0..n_cells` are the input cells in their original
    /// order; the rest are inferred ancestors.
    pub tree: Tree,
    /// Final tree loglikelihood, up to an additive constant.
    pub loglik: f64,
    /// Indices into the caller's original feature axis for the features that
    /// survived selection, ascending.
    pub features: Vec<usize>,
    /// Posterior mean position of every node, row-major `[node][retained
    /// feature]`, in **raw** units. Empty under
    /// [`BonsaiParams::skip_posteriors`].
    pub node_means: Vec<T>,
    /// Posterior standard deviation of every node, same layout, **raw** units.
    /// Empty under [`BonsaiParams::skip_posteriors`].
    pub node_sds: Vec<T>,
    /// Loglikelihood after each step, in order.
    pub steps: Vec<StepReport>,
}

impl<T: BonsaiFloat> BonsaiResult<T> {
    /// Number of retained features, which is the stride of `node_means`.
    ///
    /// ### Returns
    ///
    /// The retained feature count.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }
}

/////////////
// Helpers //
/////////////

/// Announce the start of a step.
///
/// ### Params
///
/// * `what` - The line to print, e.g. `"Step 5: SPR"`
/// * `verbosity` - Prints at [`Verbosity::Normal`] and above
///
/// ### Returns
///
/// The step's start time, for [`record`].
fn begin(what: &str, verbosity: Verbosity) -> Instant {
    if verbosity.normal_verbosity() {
        println!("{what}...");
    }
    Instant::now()
}

/// Append a step report, computing its gain from the previous one.
///
/// ### Params
///
/// * `step` - Which of the eight steps
/// * `loglik` - Loglikelihood after it
/// * `steps` - Reports so far
/// * `verbosity` - Prints the report at [`Verbosity::Normal`] and above
/// * `started` - When the step began, from [`begin`]
fn record(
    step: &'static str,
    loglik: f64,
    steps: &mut Vec<StepReport>,
    verbosity: Verbosity,
    started: Instant,
) {
    let gain = steps.last().map_or(0.0, |last| loglik - last.loglik);
    let elapsed = started.elapsed();
    steps.push(StepReport {
        step,
        loglik,
        gain,
        seconds: elapsed.as_secs_f64(),
    });
    if verbosity.normal_verbosity() {
        println!("  loglik {loglik:.6e}, gain {gain:+.3e} ({elapsed:.2?})");
    }
}

/// A star tree over `n_leaves` cells, every branch at [`INITIAL_STAR_BRANCH`].
///
/// ### Params
///
/// * `n_leaves` - Number of cells
///
/// ### Returns
///
/// The star, or `EmptyInput` if there are fewer than two cells.
fn star_of(n_leaves: usize) -> Result<Tree, BonsaiErrors> {
    if n_leaves < 2 {
        return Err(BonsaiErrors::EmptyInput {
            n_cells: n_leaves,
            n_features: 0,
        });
    }
    let mut parent = vec![n_leaves as u32; n_leaves];
    parent.push(crate::tree::NO_NODE);
    let mut branch = vec![INITIAL_STAR_BRANCH; n_leaves];
    branch.push(0.0);
    Tree::from_parents(parent, branch, n_leaves)
}

/// Optimise every branch length in place.
///
/// ### Params
///
/// * `tree` - Tree to optimise, modified in place
/// * `leaves` - Transformed leaf data
/// * `params` - Pipeline knobs, for the branch-length settings
///
/// ### Returns
///
/// The loglikelihood at the optimum.
fn optimise_all<T: BonsaiFloat>(
    tree: &mut Tree,
    leaves: Leaves<'_, T>,
    params: &BonsaiParams,
) -> Result<f64, BonsaiErrors> {
    let mut state = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )?;
    optimise_branch_lengths(tree, &mut state, Some(params.branch))
}

/// Posterior mean and standard deviation for every node, in raw units.
///
/// The posterior at a node is the whole tree collapsed onto it, which is what
/// [`collapse_onto_every_node`] computes; this turns its precisions into
/// standard deviations and both blocks back into the caller's units.
///
/// ### Params
///
/// * `tree` - The finished tree
/// * `leaves` - Transformed leaf data
/// * `data` - The ingest result, for converting back to raw units
///
/// ### Returns
///
/// Means and standard deviations, row-major `[node][feature]`, raw units.
fn posteriors<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    data: &PreparedData<T>,
) -> Result<(Vec<T>, Vec<T>), BonsaiErrors> {
    let (means, precisions) =
        collapse_onto_every_node(tree, leaves.means, leaves.precisions, leaves.n_features)?;
    let sds: Vec<T> = precisions.iter().map(|&w| narrow(1.0 / w.sqrt())).collect();
    Ok((data.restore_scale(&means)?, data.restore_scale(&sds)?))
}

/// Steps 3 to 7, shared by [`bonsai_prepared`] and [`refine`].
///
/// ### Params
///
/// * `tree` - Tree to refine, consumed
/// * `data` - Transformed means and precisions
/// * `params` - Knobs, already resolved
/// * `steps` - Step reports so far, appended to
/// * `verbosity` - How much to print while running
///
/// ### Returns
///
/// The refined tree with its posteriors.
fn refine_from<T: BonsaiFloat>(
    mut tree: Tree,
    data: &PreparedData<T>,
    params: &BonsaiParams,
    mut steps: Vec<StepReport>,
    verbosity: Verbosity,
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    let leaves = Leaves {
        means: &data.transformed_means,
        precisions: &data.transformed_precisions,
        n_features: data.n_features(),
    };

    // Step 3. The collapse that exposes polytomies lives in `search::polytomy`.
    let started = begin("Step 3: resolve polytomies", verbosity);
    let resolved = resolve_polytomies(&tree, leaves, Some(params.star))?;
    tree = resolved.tree;
    record(
        "3 polytomy",
        tree_loglik(&tree, leaves)?,
        &mut steps,
        verbosity,
        started,
    );

    // Step 4: steps 5 and 6 depend on this, see the module docs.
    let started = begin("Step 4: branch lengths", verbosity);
    let loglik = optimise_all(&mut tree, leaves, params)?;
    record("4 branch", loglik, &mut steps, verbosity, started);

    // Step 5.
    let started = begin("Step 5: SPR", verbosity);
    tree = spr(&tree, leaves, Some(params.spr), verbosity)?.tree;
    record(
        "5 spr",
        tree_loglik(&tree, leaves)?,
        &mut steps,
        verbosity,
        started,
    );

    // Step 6.
    let started = begin("Step 6: NNI", verbosity);
    tree = nni(&tree, leaves, Some(params.nni), verbosity)?.tree;
    record(
        "6 nni",
        tree_loglik(&tree, leaves)?,
        &mut steps,
        verbosity,
        started,
    );

    // Step 7.
    let started = begin("Step 7: branch lengths", verbosity);
    let loglik = optimise_all(&mut tree, leaves, params)?;
    record("7 branch", loglik, &mut steps, verbosity, started);

    // Step 8, our deviation from SPEC.md section 9. Collapsing before step 5
    // recovers slightly more topology at two orders of magnitude more time;
    // after the search it cannot change what the search finds. Zero-length leaf
    // edges survive by design.
    let started = begin("Step 8: collapse zero-length edges", verbosity);
    let (resolved, loglik) = collapse_step(&tree, leaves, params)?;
    tree = resolved.tree;
    record("8 collapse", loglik, &mut steps, verbosity, started);

    // Display only (S14), so after the last branch-length step.
    if params.reroot {
        tree = reroot_for_display(&tree)?;
    }

    let (node_means, node_sds) = if params.skip_posteriors {
        (Vec::new(), Vec::new())
    } else {
        let started = begin("Posteriors", verbosity);
        let out = posteriors(&tree, leaves, data)?;
        if verbosity.normal_verbosity() {
            println!("  {} nodes ({:.2?})", tree.n_nodes(), started.elapsed());
        }
        out
    };
    Ok(BonsaiResult {
        tree,
        loglik,
        features: data.features.clone(),
        node_means,
        node_sds,
        steps,
    })
}

////////////
// Bonsai //
////////////

/// Reconstruct a tree from a matrix of measurements and their error bars.
///
/// The whole pipeline: ingest, the seven search steps, and the extra collapse.
///
/// ### Params
///
/// * `means` - Measured means, row-major `[cell][feature]`, raw units
/// * `sds` - Standard deviations on those means, same layout and units
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `variances` - Per-feature variance in raw units, or `None` to estimate it
/// * `params` - Knobs, `None` for the defaults
/// * `verbosity` - How much to print while running
///
/// ### Returns
///
/// The tree, its loglikelihood, the retained features, and a posterior mean and
/// standard deviation for every node including the inferred ancestors.
pub fn bonsai<T: BonsaiFloat>(
    means: &[T],
    sds: &[T],
    n_cells: usize,
    n_features: usize,
    variances: Option<&[f64]>,
    params: Option<BonsaiParams>,
    verbosity: Verbosity,
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let started = begin("Ingest", verbosity);
    let prepared = prepare(
        means,
        sds,
        n_cells,
        n_features,
        variances,
        Some(params.ingest),
    )?;
    if verbosity.normal_verbosity() {
        println!(
            "  {n_cells} cells, {} / {n_features} features retained ({:.2?})",
            prepared.n_features(),
            started.elapsed()
        );
    }
    bonsai_prepared(&prepared, Some(params), verbosity)
}

/// Reconstruct a tree from data that has already been through
/// [`crate::ingest::prepare`].
///
/// For callers doing their own feature selection, or reusing one ingest across
/// several parameter settings.
///
/// ### Params
///
/// * `data` - Transformed means and precisions with their retained features
/// * `params` - Knobs, `None` for the defaults
/// * `verbosity` - How much to print while running
///
/// ### Returns
///
/// As [`bonsai`].
pub fn bonsai_prepared<T: BonsaiFloat>(
    data: &PreparedData<T>,
    params: Option<BonsaiParams>,
    verbosity: Verbosity,
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let p = data.n_features();
    let n_cells = data.n_cells;
    let leaves = Leaves {
        means: &data.transformed_means,
        precisions: &data.transformed_precisions,
        n_features: p,
    };

    let mut steps: Vec<StepReport> = Vec::with_capacity(7);

    let tree = match params.start {
        StartTree::GreedyMerge => {
            // Step 1: a star with optimised branch lengths.
            let started = begin("Step 1: star", verbosity);
            let mut star = star_of(n_cells)?;
            let mut state = NodeState::new(star.n_nodes(), p, leaves.means, leaves.precisions)?;
            let loglik = optimise_branch_lengths(&mut star, &mut state, Some(params.branch))?;
            record("1 star", loglik, &mut steps, verbosity, started);

            // Step 2: the star's branch lengths carry over as the members'
            // branches. The graph decides which pairs exist, the bounds which
            // need rescoring this round.
            let started = begin("Step 2: greedy merge", verbosity);
            let mut candidates = EllipsoidBounds::new(KnnCandidates::new(Some(params.knn)));
            let (tree, _) = star_tree_with(
                Star {
                    means: leaves.means,
                    precisions: leaves.precisions,
                    branch: &star.branches()[..n_cells],
                    n_features: p,
                },
                Some(params.star),
                &mut candidates,
                verbosity,
            )?;
            record(
                "2 merge",
                tree_loglik(&tree, leaves)?,
                &mut steps,
                verbosity,
                started,
            );
            tree
        }
        StartTree::Linkage => {
            // Steps 1 and 2 at once; step 4 supplies the branch lengths.
            let started = begin("Steps 1-2: linkage", verbosity);
            let tree = linkage_tree(leaves.means, n_cells, p, Some(params.linkage))?;
            record(
                "1-2 linkage",
                tree_loglik(&tree, leaves)?,
                &mut steps,
                verbosity,
                started,
            );
            tree
        }
    };

    refine_from(tree, data, &params, steps, verbosity)
}

/// Run the refinement steps on a tree that already exists.
///
/// Resolve polytomies, optimise the branch lengths, SPR, NNI, optimise again,
/// collapse. For a tree from elsewhere, such as an external clustering; the
/// returned `steps` start at step 3.
///
/// ### Params
///
/// * `tree` - Starting tree, whose leaves must be the cells of `data` in order
/// * `data` - Transformed means and precisions
/// * `params` - Knobs, `None` for the defaults
/// * `verbosity` - How much to print while running
///
/// ### Returns
///
/// As [`bonsai`], but having refined rather than reconstructed.
pub fn refine<T: BonsaiFloat>(
    tree: &Tree,
    data: &PreparedData<T>,
    params: Option<BonsaiParams>,
    verbosity: Verbosity,
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    if tree.n_leaves() != data.n_cells {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: data.n_cells,
            mean_features: data.n_features(),
            sd_cells: tree.n_leaves(),
            sd_features: data.n_features(),
        });
    }
    let params = params.unwrap_or_default();
    refine_from(
        tree.clone(),
        data,
        &params,
        Vec::with_capacity(5),
        verbosity,
    )
}

/// Step 8: collapse the zero-length internal edges steps 4 to 7 leave behind,
/// resolve what that exposes, reoptimise, and collapse once more.
///
/// The reoptimise can land an internal edge on `t = 0` (SPEC.md section 6); the
/// last collapse removes it without changing the loglikelihood or any other
/// edge's optimum, so no second solve is needed.
///
/// ### Params
///
/// * `tree` - Tree after step 7; not modified
/// * `leaves` - Transformed leaf data
/// * `params` - Pipeline knobs, for the star and branch-length settings
///
/// ### Returns
///
/// The resolver's report with its tree replaced by the final one, and the
/// loglikelihood, or the error the resolver, the solve or the arena failed
/// with.
pub fn collapse_step<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: &BonsaiParams,
) -> Result<(crate::search::polytomy::PolytomyResult, f64), BonsaiErrors> {
    let mut resolved = resolve_polytomies(tree, leaves, Some(params.star))?;
    let loglik = optimise_all(&mut resolved.tree, leaves, params)?;
    if let Some((collapsed, _)) = crate::search::polytomy::collapse_zero_edges(&resolved.tree)? {
        resolved.tree = collapsed;
    }
    Ok((resolved, loglik))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::prepare;
    use crate::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};

    /// A simulated dataset put back on the raw scale, which is what `bonsai`
    /// takes. `simulate` returns transformed units.
    fn raw_fixture(
        n_leaves: usize,
        n_features: usize,
        noise_sd: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Tree) {
        let data = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves,
            n_features,
            noise_sd,
            seed,
            ..Default::default()
        }))
        .expect("simulation");
        let scale: Vec<f64> = data.variances.iter().map(|v| v.sqrt()).collect();
        let mut means = Vec::with_capacity(n_leaves * n_features);
        let mut sds = Vec::with_capacity(n_leaves * n_features);
        for i in 0..n_leaves {
            for g in 0..n_features {
                means.push(data.means[i * n_features + g] * scale[g]);
                sds.push(data.sds[i * n_features + g] * scale[g]);
            }
        }
        (means, sds, data.variances.clone(), data.tree)
    }

    #[test]
    fn test_the_pipeline_recovers_a_simulated_tree() {
        let (n, p) = (32usize, 128usize);
        let (means, sds, variances, truth) = raw_fixture(n, p, 0.1, 7);
        let out =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");

        let rf = robinson_foulds(&out.tree, &truth).expect("rf");
        assert_eq!(
            rf, 0,
            "recovered tree differs from the truth by {rf} splits"
        );
        assert_eq!(out.tree.n_leaves(), n);
        assert_eq!(out.features.len(), p);
        assert!(out.loglik.is_finite());
    }

    #[test]
    fn test_every_step_is_monotone() {
        // The whole pipeline is a sequence of moves each of which is supposed
        // to be an improvement. If any step reports a loss, the ordering in the
        // module docs is wrong or a step is broken.
        let (n, p) = (32usize, 128usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.3, 11);

        // The linkage start reports steps 1 and 2 as one step, so the count is
        // start-dependent and both are checked.
        for (start, n_steps) in [(StartTree::Linkage, 7), (StartTree::GreedyMerge, 8)] {
            let params = BonsaiParams {
                start,
                ..Default::default()
            };
            let out = bonsai(
                &means,
                &sds,
                n,
                p,
                Some(&variances),
                Some(params),
                Verbosity::Quiet,
            )
            .expect("bonsai");

            assert_eq!(out.steps.len(), n_steps, "{start:?}");
            for step in out.steps.iter().skip(1) {
                assert!(
                    step.gain >= -1e-9,
                    "{start:?} step '{}' lost {} nats",
                    step.step,
                    -step.gain
                );
            }
            assert_relative_eq!(
                out.steps.last().expect("steps").loglik,
                out.loglik,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_posteriors_are_finite_and_leaves_stay_near_their_measurements() {
        // A leaf's posterior is its own measurement combined with what the rest
        // of the tree says, so it should sit near the measurement and be no
        // less certain than it.
        let (n, p) = (16usize, 64usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.1, 3);
        let out =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");

        assert_eq!(out.node_means.len(), out.tree.n_nodes() * p);
        assert_eq!(out.node_sds.len(), out.tree.n_nodes() * p);
        assert!(out.node_means.iter().all(|x| x.is_finite()));
        assert!(out.node_sds.iter().all(|x| x.is_finite() && *x > 0.0));

        for leaf in 0..n {
            for g in 0..p {
                let idx = leaf * p + g;
                assert!(
                    out.node_sds[idx] <= sds[idx] * 1.000_001,
                    "leaf {leaf} feature {g}: posterior sd {} exceeds the measurement's {}",
                    out.node_sds[idx],
                    sds[idx]
                );
            }
        }
    }

    #[test]
    fn test_skip_posteriors_leaves_the_tree_alone() {
        let (n, p) = (16usize, 64usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 7);

        let full =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");
        let skipped = bonsai(
            &means,
            &sds,
            n,
            p,
            Some(&variances),
            Some(BonsaiParams {
                skip_posteriors: true,
                ..Default::default()
            }),
            Verbosity::Quiet,
        )
        .expect("bonsai");

        assert!(skipped.node_means.is_empty() && skipped.node_sds.is_empty());
        assert_eq!(full.loglik, skipped.loglik);
        assert_eq!(full.tree.branches(), skipped.tree.branches());
        assert_eq!(
            robinson_foulds(&full.tree, &skipped.tree).expect("rf"),
            0,
            "skipping the posteriors changed the topology"
        );
    }

    #[test]
    fn test_rerooting_does_not_change_the_loglikelihood() {
        // S14. The reroot flag is a display choice and must cost nothing.
        let (n, p) = (16usize, 64usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 5);

        let plain =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");
        let rerooted = bonsai(
            &means,
            &sds,
            n,
            p,
            Some(&variances),
            Some(BonsaiParams {
                reroot: true,
                ..Default::default()
            }),
            Verbosity::Quiet,
        )
        .expect("bonsai");

        assert_relative_eq!(plain.loglik, rerooted.loglik, max_relative = 1e-12);
        assert_eq!(
            robinson_foulds(&plain.tree, &rerooted.tree).expect("rf"),
            0,
            "rerooting changed the unrooted topology"
        );
    }

    #[test]
    fn test_the_same_input_gives_the_same_tree() {
        let (n, p) = (16usize, 64usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 9);
        let first =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");
        let second =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");
        assert_eq!(first.tree.branches(), second.tree.branches());
        assert_eq!(first.loglik.to_bits(), second.loglik.to_bits());
    }

    #[test]
    fn test_refining_a_finished_tree_finds_nothing_left() {
        // Run on a tree the full pipeline already produced, `refine` should
        // have nothing to do, which is what says the two paths agree about when
        // the search is done.
        let (n, p) = (32usize, 128usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 13);
        let full =
            bonsai(&means, &sds, n, p, Some(&variances), None, Verbosity::Quiet).expect("bonsai");

        let prepared = prepare(&means, &sds, n, p, Some(&variances), None).expect("prepare");
        let again = refine(&full.tree, &prepared, None, Verbosity::Quiet).expect("refine");

        assert_relative_eq!(again.loglik, full.loglik, max_relative = 1e-9);
        assert_eq!(
            robinson_foulds(&again.tree, &full.tree).expect("rf"),
            0,
            "refining a finished tree changed its topology"
        );
        assert_eq!(again.steps.len(), 6, "refine should report steps 3 to 8");
    }

    #[test]
    fn test_refining_a_bad_tree_improves_it() {
        // The other half: given a deliberately wrong topology over the right
        // leaves, refinement has to move it towards the truth.
        let (n, p) = (32usize, 128usize);
        let (means, sds, variances, truth) = raw_fixture(n, p, 0.2, 17);
        let prepared = prepare(&means, &sds, n, p, Some(&variances), None).expect("prepare");

        let ladder = Tree::ladder(n, 1.0).expect("ladder");
        let before = {
            let mut state = NodeState::new(
                ladder.n_nodes(),
                p,
                &prepared.transformed_means,
                &prepared.transformed_precisions,
            )
            .expect("state");
            state.prune(&ladder)
        };
        let rf_before = robinson_foulds(&ladder, &truth).expect("rf");

        let out = refine(&ladder, &prepared, None, Verbosity::Quiet).expect("refine");
        let rf_after = robinson_foulds(&out.tree, &truth).expect("rf");

        assert!(
            out.loglik > before,
            "refinement lost likelihood: {} against {before}",
            out.loglik
        );
        assert!(
            rf_after < rf_before,
            "refinement moved away from the truth: {rf_before} to {rf_after}"
        );
    }

    #[test]
    fn test_refine_rejects_a_tree_with_the_wrong_leaves() {
        let (n, p) = (16usize, 32usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 19);
        let prepared = prepare(&means, &sds, n, p, Some(&variances), None).expect("prepare");
        let wrong = Tree::balanced_binary(8, 1.0).expect("balanced");
        assert!(refine(&wrong, &prepared, None, Verbosity::Quiet).is_err());
    }

    #[test]
    fn test_too_few_cells_is_an_error() {
        let means = vec![1.0f64; 4];
        let sds = vec![0.1f64; 4];
        assert!(bonsai(&means, &sds, 1, 4, None, None, Verbosity::Quiet).is_err());
    }

    #[test]
    fn test_collapse_step_leaves_no_zero_length_internal_edge() {
        // ((A,B)X,C)Y,D): A, B and C sit at equal, orthogonal offsets from one
        // centre, so nothing pairs A with B and the solve puts X on Y. The tree
        // is binary, so the resolver leaves it alone and the zero is the
        // reoptimise's own, which is the case the last collapse exists for.
        let (n, p) = (4usize, 60usize);
        let mut m = vec![0.0f64; n * p];
        for g in 0..p {
            m[(g / 20) * p + g] = 1.0;
            m[3 * p + g] = 8.0;
        }
        let w = vec![1.0f64; n * p];
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let tree = Tree::from_parents(
            vec![4, 4, 5, 6, 5, 6, crate::tree::NO_NODE],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            n,
        )
        .expect("tree");
        let params = BonsaiParams::default();
        let zero_internal = |t: &Tree| {
            (t.n_leaves()..t.n_nodes())
                .filter(|&i| i as u32 != t.root() && t.branch(i as u32) == 0.0)
                .count()
        };

        // The fixture has to hit the case, or the assertion below is vacuous.
        let mut unfixed = resolve_polytomies(&tree, leaves, Some(params.star))
            .expect("resolve")
            .tree;
        optimise_all(&mut unfixed, leaves, &params).expect("optimise");
        assert!(zero_internal(&unfixed) > 0, "the fixture made no zero edge");

        let (out, loglik) = collapse_step(&tree, leaves, &params).expect("collapse step");
        assert_eq!(zero_internal(&out.tree), 0);
        assert_eq!(out.tree.n_leaves(), n);
        assert_relative_eq!(
            tree_loglik(&out.tree, leaves).expect("loglik"),
            loglik,
            max_relative = 1e-12
        );
    }

    use approx::assert_relative_eq;
}
