//! The whole algorithm: matrix in, tree out.
//!
//! Runs the seven steps of SPEC.md section 9 in order, then an eighth of our
//! own. Everything below is orchestration; the arithmetic lives in `model`,
//! `search` and `tree`.
//!
//! **The eighth step is a deviation.** Step 3 is the only step that collapses
//! zero-length edges and it runs before step 4, so every zero-length edge the
//! branch solves create afterwards outlives the only pass that would remove
//! one. See the comment on step 8 for why it runs last rather than earlier.
//!
//! ### Why the order is not negotiable
//!
//! Two of the steps only work where they are placed, and neither constraint is
//! in the specification.
//!
//! Steps 5 and 6 must follow step 4. Both SPR and NNI reject a proposal whose
//! topology fingerprint is unchanged, because otherwise they accept moves that
//! only reoptimise branch lengths and never terminate on topology. Run before
//! the branch lengths are optimised, that filter rejects the only improvements
//! available and topology recovery gets *worse*.
//!
//! Step 3 must follow step 2 rather than being folded into it. Polytomies are
//! created by step 2 when an optimal branch length comes out at zero, and the
//! configuration that was optimal when it was created often is not once the
//! centre has moved.

use crate::errors::BonsaiErrors;
use crate::ingest::{IngestParams, PreparedData, prepare};
use crate::model::global::{GlobalBranchParams, collapse_onto_every_node, optimise_branch_lengths};
use crate::model::likelihood::NodeState;
use crate::search::bounds::{EllipsoidBounds, EllipsoidBoundsParams};
use crate::search::candidates::{KnnCandidates, KnnCandidatesParams};
use crate::search::nni::{NniParams, nni};
use crate::search::polytomy::resolve_polytomies;
use crate::search::spr::{SprParams, spr};
use crate::search::star::{Star, StarParams, star_tree_with};
use crate::search::{Leaves, tree_loglik};
use crate::tree::Tree;
use crate::tree::cluster::reroot_for_display;
use crate::tree::linkage::{LinkageParams, linkage_tree};
use crate::utils::traits::{BonsaiFloat, narrow};

////////////
// Consts //
////////////

/// Branch length the initial star hangs every leaf on, before step 1 optimises
/// it.
///
/// Only a starting point for the optimiser, which converges from anywhere; step
/// 1 replaces it entirely. One is the natural scale because the ingest
/// transform of SPEC.md section 3.1 sets the per-feature signal variance to one,
/// so a branch of one is a diffusion of one signal standard deviation.
const INITIAL_STAR_BRANCH: f64 = 1.0;

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
    /// The greedy star primitive and polytomy resolution.
    ///
    /// Steps 2 and 3 only. Steps 5 and 6 run the same primitive but take their
    /// settings from `spr.star` and `nni.star`, so raising `min_gain` here
    /// leaves their move-acceptance floor at the default.
    pub star: StarParams,
    /// Candidate-pair restriction.
    ///
    /// Not optional in practice. Scanning every pair makes step 2 `O(n^3 p)`
    /// and 94 per cent of the runtime; measured, it scales as `n^2.9` without
    /// this and the ellipsoid bounds below.
    pub knn: KnnCandidatesParams,
    /// Upper bounds on merge scores (section 10).
    pub bounds: EllipsoidBoundsParams,
    /// Global branch-length optimisation (section 6), steps 1, 4 and 7.
    pub branch: GlobalBranchParams,
    /// Subtree pruning and regrafting (section 9.3), step 5.
    pub spr: SprParams,
    /// Nearest-neighbour interchange (section 9.4), step 6.
    pub nni: NniParams,
    /// Whether to reroot for display once the search is done (section 9.7).
    ///
    /// The likelihood does not depend on the root (S14), so this changes only
    /// how the tree is drawn. It is done last because a degree-two root is
    /// degenerate for branch-length optimisation: only the sum of the two
    /// branches below it is identifiable.
    pub reroot: bool,
}

///////////////
// StartTree //
///////////////

/// How the initial topology is built.
///
/// Search steps 1 and 2, or a linkage in their place. The refinement of steps 3
/// to 7 is identical either way.
///
/// **The linkage is the default and the specified start is not.** On synthetic
/// data the two are interchangeable. On real Sanity-preprocessed input they are
/// not: the linkage wins on the loglikelihood and on Robinson-Foulds at every
/// size measured, and is several times faster. `docs/PERFORMANCE.md` has the
/// table and the mechanism.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StartTree {
    /// The star of SPEC.md section 9.1, agglomerated by the merge score.
    ///
    /// **This chains on real input, and the chaining grows with the cell
    /// count.** Leaf depth straight after step 2 runs an order of magnitude
    /// above `log2(n)`; steps 3 to 7 do not recover from it, and the finished
    /// tree is worse on the loglikelihood as well as on the topology.
    ///
    /// The cause is the merge criterion, not the order pairs are taken in. A
    /// cluster's effective leaf carries `1/size` of the noise, so a large one
    /// sits closer to every member than that member's own relatives do and
    /// absorbs them one at a time. A Ward distance grows with cluster size and
    /// has no such bias; the merge gain shrinks. Rescheduling the merges does
    /// not reach it, and `docs/PERFORMANCE.md` records what was tried.
    ///
    /// Kept because it is what the paper specifies, so it is what a
    /// reproduction of the published method has to run. Choose it to compare
    /// against that method, not to build the best tree.
    GreedyMerge,
    /// Ward linkage over a neighbour graph, [`crate::tree::linkage`].
    ///
    /// The default. Stays close to `log2(n)` mean leaf depth at every size
    /// measured, and hands steps 3 to 7 a tree they refine rather than repair.
    #[default]
    Linkage,
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
    /// Non-negative on every measured run, and `test_every_step_is_monotone`
    /// checks it, but `record` does not assert it: a step that lost ground
    /// should surface as a visible negative here rather than as a panic
    /// crossing an FFI boundary.
    pub gain: f64,
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
    /// feature]`, in **raw** units.
    pub node_means: Vec<T>,
    /// Posterior standard deviation of every node, same layout, **raw** units.
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

/// Append a step report, computing its gain from the previous one.
///
/// ### Params
///
/// * `step` - Which of the seven steps
/// * `loglik` - Loglikelihood after it
/// * `steps` - Reports so far
fn record(step: &'static str, loglik: f64, steps: &mut Vec<StepReport>) {
    let gain = steps.last().map_or(0.0, |last| loglik - last.loglik);
    steps.push(StepReport { step, loglik, gain });
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

////////////
// Bonsai //
////////////

/// Reconstruct a tree from a matrix of measurements and their error bars.
///
/// The whole pipeline: ingest, the seven search steps of, and the collapse this
/// crate adds after them.
///
/// ### Params
///
/// * `means` - Measured means, row-major `[cell][feature]`, raw units
/// * `sds` - Standard deviations on those means, same layout and units
/// * `n_cells` - Number of cells
/// * `n_features` - Number of features
/// * `variances` - Per-feature variance in raw units, or `None` to estimate it
/// * `params` - Knobs, `None` for the defaults
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
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let prepared = prepare(
        means,
        sds,
        n_cells,
        n_features,
        variances,
        Some(params.ingest),
    )?;
    bonsai_prepared(&prepared, Some(params))
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
///
/// ### Returns
///
/// As [`bonsai`].
pub fn bonsai_prepared<T: BonsaiFloat>(
    data: &PreparedData<T>,
    params: Option<BonsaiParams>,
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
            let mut star = star_of(n_cells)?;
            let mut state = NodeState::new(star.n_nodes(), p, leaves.means, leaves.precisions)?;
            let loglik = optimise_branch_lengths(&mut star, &mut state, Some(params.branch))?;
            record("1 star", loglik, &mut steps);

            // Step 2: greedily add ancestors. The star's optimised branch
            // lengths carry over as the members' branches to the centre.
            // Both restrictions on, which is what makes this step tractable.
            // The bounds sit outside the neighbour graph: the graph decides
            // which pairs exist, the bounds decide which of those need
            // rescoring this round.
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
            record("2 merge", tree_loglik(&tree, leaves)?, &mut steps);
            tree
        }
        StartTree::Linkage => {
            // Steps 1 and 2 at once, and neither of them scores anything with
            // the model: the linkage supplies a topology and step 4 supplies
            // the branch lengths.
            let tree = linkage_tree(leaves.means, n_cells, p, Some(params.linkage))?;
            record("1-2 linkage", tree_loglik(&tree, leaves)?, &mut steps);
            tree
        }
    };

    refine_from(tree, data, &params, steps)
}

/// Run the refinement steps on a tree that already exists.
///
/// Resolve polytomies, optimise the branch lengths, SPR, NNI, optimise again.
/// Steps 1 and 2 build a tree from nothing; this improves one that is already
/// there.
///
/// That is what a caller with a tree from elsewhere wants. The paper
/// recommends seeding the search with cells grouped by an external clustering,
/// which is the same entry point.
///
/// The returned `steps` start at step 3, since 1 and 2 did not happen.
///
/// ### Params
///
/// * `tree` - Starting tree, whose leaves must be the cells of `data` in order
/// * `data` - Transformed means and precisions
/// * `params` - Knobs, `None` for the defaults
///
/// ### Returns
///
/// As [`bonsai`], but having refined rather than reconstructed.
pub fn refine<T: BonsaiFloat>(
    tree: &Tree,
    data: &PreparedData<T>,
    params: Option<BonsaiParams>,
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
    refine_from(tree.clone(), data, &params, Vec::with_capacity(5))
}

//////////////
// Internal //
//////////////

/// Steps 3 to 7, shared by [`bonsai_prepared`] and [`refine`].
///
/// ### Params
///
/// * `tree` - Tree to refine, consumed
/// * `data` - Transformed means and precisions
/// * `params` - Knobs, already resolved
/// * `steps` - Step reports so far, appended to
///
/// ### Returns
///
/// The refined tree with its posteriors.
fn refine_from<T: BonsaiFloat>(
    mut tree: Tree,
    data: &PreparedData<T>,
    params: &BonsaiParams,
    mut steps: Vec<StepReport>,
) -> Result<BonsaiResult<T>, BonsaiErrors> {
    let leaves = Leaves {
        means: &data.transformed_means,
        precisions: &data.transformed_precisions,
        n_features: data.n_features(),
    };

    // Step 3: resolve the polytomies that a zero-length branch stands for. The
    // collapse that finds them lives in `search::polytomy`; without it this step
    // sees a structurally binary tree and does nothing.
    let resolved = resolve_polytomies(&tree, leaves, Some(params.star))?;
    tree = resolved.tree;
    record("3 polytomy", tree_loglik(&tree, leaves)?, &mut steps);

    // Step 4: all branch lengths at once. Steps 5 and 6 depend on this having
    // happened; see the module docs.
    let loglik = optimise_all(&mut tree, leaves, params)?;
    record("4 branch", loglik, &mut steps);

    // Step 5.
    tree = spr(&tree, leaves, Some(params.spr))?.tree;
    record("5 spr", tree_loglik(&tree, leaves)?, &mut steps);

    // Step 6.
    tree = nni(&tree, leaves, Some(params.nni))?.tree;
    record("6 nni", tree_loglik(&tree, leaves)?, &mut steps);

    // Step 7.
    let loglik = optimise_all(&mut tree, leaves, params)?;
    record("7 branch", loglik, &mut steps);

    // Step 8, a deviation. SPEC.md section 9 lists seven steps and this is an
    // eighth: collapse the internal zero-length edges the branch solves leave
    // behind, then reoptimise what the collapse changed.
    //
    // It exists because step 3 is the only step that collapses, and it runs
    // before step 4. SPEC.md section 6 is explicit that a branch solve landing
    // on `t = 0` is normal, so every zero-length edge steps 4 to 7 create
    // outlives the only pass that would remove it.
    //
    // It runs last on purpose. Collapsing before step 5 instead recovers
    // slightly more topology but costs two orders of magnitude more time,
    // because a collapsed tree gives every nearby regraft a higher-degree star
    // to resolve. Running after the search cannot change what the search finds,
    // which is the property that makes this safe to do unconditionally.
    //
    // The collapse only ever *removes* structure, so it cannot invent a split
    // the data does not support. What it cannot touch is a zero-length edge at
    // a *leaf*, which is not an internal edge: those are the model declining to
    // separate two cells it has no evidence to separate, and they survive this
    // step by design.
    let resolved = resolve_polytomies(&tree, leaves, Some(params.star))?;
    tree = resolved.tree;
    let loglik = optimise_all(&mut tree, leaves, params)?;
    record("8 collapse", loglik, &mut steps);

    // Rerooting is a display choice and carries no information (S14), so it
    // happens after the last thing that cares about branch lengths.
    if params.reroot {
        tree = reroot_for_display(&tree)?;
    }

    let (node_means, node_sds) = posteriors(&tree, leaves, data)?;
    Ok(BonsaiResult {
        tree,
        loglik,
        features: data.features.clone(),
        node_means,
        node_sds,
        steps,
    })
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
        let out = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");

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
            let out =
                bonsai(&means, &sds, n, p, Some(&variances), Some(params)).expect("bonsai");

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
        let out = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");

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
    fn test_rerooting_does_not_change_the_loglikelihood() {
        // S14. The reroot flag is a display choice and must cost nothing.
        let (n, p) = (16usize, 64usize);
        let (means, sds, variances, _) = raw_fixture(n, p, 0.2, 5);

        let plain = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");
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
        let first = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");
        let second = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");
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
        let full = bonsai(&means, &sds, n, p, Some(&variances), None).expect("bonsai");

        let prepared = prepare(&means, &sds, n, p, Some(&variances), None).expect("prepare");
        let again = refine(&full.tree, &prepared, None).expect("refine");

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

        let out = refine(&ladder, &prepared, None).expect("refine");
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
        assert!(refine(&wrong, &prepared, None).is_err());
    }

    #[test]
    fn test_too_few_cells_is_an_error() {
        let means = vec![1.0f64; 4];
        let sds = vec![0.1f64; 4];
        assert!(bonsai(&means, &sds, 1, 4, None, None).is_err());
    }

    use approx::assert_relative_eq;
}
