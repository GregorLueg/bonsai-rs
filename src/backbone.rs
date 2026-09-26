//! Backbone mode: reconstruct on a subset, then place the rest.
//!
//! Four steps: preprocess everything, run the standard algorithm on a random
//! subset to get a backbone, place the remaining cells on it in rounds, then
//! refine the whole thing.
//!
//! ### What this does and does not buy
//!
//! It replaces the standard search's cost on `n` cells with its cost on the
//! backbone plus one placement per remaining cell. The start is what it
//! avoids, and the start is not the expensive step: SPR is, and the final
//! refinement runs SPR over every cell however the backbone was built. **So on
//! its own this trades away the cheap step and keeps the expensive one.** It is
//! worth having because it composes with anything that makes SPR cheaper.
//!
//! Measured, 2026-09-25, Baron 10k: 336 s against 211 s for the linkage-start
//! search, 8k nats below it. The grown tree matches the linkage start on
//! loglikelihood at a 4096-cell backbone, but the refinement still costs more
//! from it (253 s against 205 s), NNI above all. What the refinement costs
//! tracks the quality of its start, 42 s from a converged tree, so this only
//! pays once a grown tree beats the linkage start outright.
//!
//! ### The one place this differs from the standard algorithm's answer
//!
//! Placement is a beam search with a tolerance, not an exhaustive scan, and a
//! cell cannot see cells placed in its own round or after it. So this is an
//! approximation and its trees are not guaranteed identical to the standard
//! algorithm's. The paper is explicit that backbone mode trades accuracy for
//! time. What that costs is measured in the tests rather than assumed.
//!
//! Nothing else in the search is exact either. `candidates` can miss a best
//! pair that is in no neighbour list, and section 10.2's bound is not strict.
//! Both are very good in practice rather than guaranteed, and their own modules
//! say so.

use crate::bonsai::{BonsaiParams, BonsaiResult, bonsai_prepared, refine};
use crate::errors::BonsaiErrors;
use crate::ingest::PreparedData;
use crate::model::global::{GlobalBranchParams, collapse_onto_every_node, optimise_branch_lengths};
use crate::model::likelihood::NodeState;
use crate::model::merge::EffLeaf;
use crate::model::place::{Placement, PlacementParams, place_from};
use crate::search::Leaves;
use crate::search::polytomy::resolve_centres;
use crate::tree::{NO_NODE, Tree};
use crate::utils::rng::SplitMix64;
use crate::utils::traits::{BonsaiFloat, narrow};
use ann_search_rs::{build_exhaustive_index, query_exhaustive_index};
use rayon::prelude::*;
use std::time::Instant;

////////////
// Consts //
////////////

/// Cells in the initial backbone, when the caller does not say.
///
/// The backbone has to be large enough to carry the structure the rest of the
/// cells will be placed against; too small and every placement is deciding
/// between branches that are not there yet. The paper suggests ten thousand.
///
/// Ours, measured 2026-09-25 on Baron 10k. Grown-tree loglikelihood after
/// branch optimisation, against the seed search's cost: 1024 cells -11,460,009
/// in 15 s, 2048 -11,378,038 in 40 s, 4096 -11,326,618 in 75 s, the last level
/// with the linkage start's -11,327,995. End to end 4096 took 336 s and 2048
/// 372 s, one run each.
pub const DEFAULT_BACKBONE_CELLS: usize = 4096;

/// Growth per placement round, as a fraction of the current leaf count.
///
/// Each round places its cells against one collapse of the tree as it stood, so
/// a cell cannot see its own round. Smaller rounds cost a collapse each and buy
/// sight of more of the tree. Ours, measured 2026-09-25 on Baron 10k with a
/// 2048-cell backbone: 0.05 grew a tree 16k nats better than 0.25 for 7 s more
/// placement, and a single round was 43k worse than 0.25.
pub const DEFAULT_REGROW_FRACTION: f64 = 0.05;

/// Passes of [`Growing::resolve_new`] per round. Each pass resolves every
/// centre whose parent is not also a centre, so nested centres need a second;
/// a runaway guard beyond that, not a tuned value.
const MAX_RESOLVE_PASSES: usize = 4;

/// Metric for [`Growing::nearest_leaves`], as `ann_search_rs` spells it.
const NEAREST_METRIC: &str = "euclidean";

/// Nearest leaves each cell's beam search starts from, on top of its spread
/// starts.
///
/// The beam's own starts are spread over the node index, and on a 10k-cell
/// tree they leave it in a local optimum: exhaustive placement grew a tree
/// 650k nats better than the beam at the shipped tolerance. Starting next to
/// the cell's nearest neighbours puts the search in the right basin.
///
/// Ours, measured 2026-09-25 on Baron 10k with a 2048-cell backbone, grown
/// loglikelihood after branch optimisation: none -12,031,580, 4 leaves
/// -11,411,956, 16 leaves -11,389,458, exhaustive -11,378,943. Placement stayed
/// near 5 s throughout; exhaustive took 314 s. Widening the beam instead gets
/// less: tolerance 100 with the spread starts reached -11,463,839.
pub const DEFAULT_LEAF_STARTS: usize = 16;

////////////////////
// BackboneParams //
////////////////////

/// Knobs for backbone mode.
#[derive(Clone, Copy, Debug)]
pub struct BackboneParams {
    /// Cells in the initial backbone.
    pub backbone_cells: usize,
    /// Growth per placement round as a fraction of the current leaf count.
    pub regrow_fraction: f64,
    /// Seed for choosing the backbone subset.
    pub seed: u64,
    /// Nearest current leaves, in Euclidean distance over the means, handed to
    /// the beam search as extra start points for each cell. Zero for the
    /// beam's own spread starts only.
    pub leaf_starts: usize,
    /// Resolve the polytomy at every node that received cells, after each
    /// round (SI.B.4.2: a cell attachment is always followed by resolving the
    /// polytomy it made). One settle per round serves every centre; see
    /// `search::polytomy::resolve_centres` for the approximation.
    pub resolve_attachments: bool,
    /// Run the full refinement over the cells placed so far each time the tree
    /// has grown by this factor since the last one, and grow on from the
    /// result (the Methods' multi-round growth). `f64::INFINITY` never does.
    pub stage_growth: f64,
    /// Placement search knobs for the growth phase.
    pub placement: PlacementParams,
    /// Stopping rule for reoptimising the branch lengths between rounds, which
    /// SPEC.md section 15 does and this skips by default: `max_iter: 0` skips
    /// it. Measured 2026-09-25 on Baron 10k, 2048-cell backbone, quarter
    /// rounds: the full reoptimisation cost 87 s and grew a tree 4k nats better
    /// than none. Resolving the polytomies between rounds as well (SPEC.md
    /// section 7.3) bought 6.5k nats for 107 s and is not offered.
    pub growth_branch: GlobalBranchParams,
    /// Everything the standard algorithm takes, used for the backbone and for
    /// the final refinement.
    pub bonsai: BonsaiParams,
}

impl Default for BackboneParams {
    /// See the constants above for how each default was arrived at.
    fn default() -> Self {
        Self {
            backbone_cells: DEFAULT_BACKBONE_CELLS,
            regrow_fraction: DEFAULT_REGROW_FRACTION,
            seed: 0,
            leaf_starts: DEFAULT_LEAF_STARTS,
            resolve_attachments: false,
            stage_growth: f64::INFINITY,
            placement: PlacementParams::default(),
            growth_branch: GlobalBranchParams {
                max_iter: 0,
                ..GlobalBranchParams::default()
            },
            bonsai: BonsaiParams::default(),
        }
    }
}

////////////////////
// BackboneReport //
////////////////////

/// What the growth phase did.
#[derive(Clone, Copy, Debug)]
pub struct BackboneReport {
    /// Cells in the initial backbone.
    pub backbone_cells: usize,
    /// Cells placed onto the backbone afterwards.
    pub placed: usize,
    /// Times the branch lengths were reoptimised during growth.
    pub reoptimisations: usize,
    /// Mean nodes scored per placement. The handle on whether the beam search's
    /// tolerance is doing anything; compare against the node count.
    pub mean_scored: f64,
    /// Wall time of the standard search on the backbone subset.
    pub seed_seconds: f64,
    /// Wall time of placing the remaining cells, reoptimisation excluded.
    pub place_seconds: f64,
    /// Wall time of the growth-phase branch-length reoptimisations.
    pub reoptimise_seconds: f64,
    /// Wall time of resolving the attachment polytomies between rounds.
    pub resolve_seconds: f64,
    /// Wall time of the intermediate refinements of multi-round growth.
    pub stage_seconds: f64,
    /// Wall time of the final refinement over every cell. Zero from [`grow`],
    /// which stops before it.
    pub refine_seconds: f64,
}

impl BackboneReport {
    /// A report for a backbone of `backbone_cells` with nothing done yet.
    ///
    /// ### Params
    ///
    /// * `backbone_cells` - Cells in the initial backbone
    ///
    /// ### Returns
    ///
    /// The report, every count and timing zero.
    fn empty(backbone_cells: usize) -> Self {
        Self {
            backbone_cells,
            placed: 0,
            reoptimisations: 0,
            mean_scored: 0.0,
            seed_seconds: 0.0,
            place_seconds: 0.0,
            reoptimise_seconds: 0.0,
            resolve_seconds: 0.0,
            stage_seconds: 0.0,
            refine_seconds: 0.0,
        }
    }
}

/// Reconstruct a tree by building a backbone and placing the rest onto it.
///
/// [`grow`] followed by [`crate::bonsai::refine`] over every cell.
///
/// ### Params
///
/// * `data` - Transformed means and precisions for every cell
/// * `params` - Knobs, `None` for the defaults
///
/// ### Returns
///
/// The tree and its posteriors, exactly as [`crate::bonsai::bonsai_prepared`]
/// returns them, plus what the growth phase did.
pub fn backbone<T: BonsaiFloat>(
    data: &PreparedData<T>,
    params: Option<BackboneParams>,
) -> Result<(BonsaiResult<T>, BackboneReport), BonsaiErrors> {
    let params = params.unwrap_or_default();
    let n_cells = data.n_cells;

    // A backbone at least as large as the dataset means there is nothing to
    // place, so this is the standard algorithm with extra steps.
    if n_cells >= 2 && params.backbone_cells >= n_cells {
        let t0 = Instant::now();
        let out = bonsai_prepared(data, Some(params.bonsai))?;
        let mut report = BackboneReport::empty(n_cells);
        report.seed_seconds = t0.elapsed().as_secs_f64();
        return Ok((out, report));
    }

    let (tree, mut report) = grow(data, Some(params))?;
    let t0 = Instant::now();
    let out = refine(&tree, data, Some(params.bonsai))?;
    report.refine_seconds = t0.elapsed().as_secs_f64();
    Ok((out, report))
}

/// Build the backbone and place every remaining cell on it, without the final
/// refinement.
///
/// Steps 2 and 3 of SPEC.md section 15. Public so that a caller can time or
/// replace the refinement; [`backbone`] is this plus [`crate::bonsai::refine`].
///
/// ### Params
///
/// * `data` - Transformed means and precisions for every cell
/// * `params` - Knobs, `None` for the defaults
///
/// ### Returns
///
/// The grown tree, leaves in cell order and polytomies unresolved, plus what
/// the growth phase did.
pub fn grow<T: BonsaiFloat>(
    data: &PreparedData<T>,
    params: Option<BackboneParams>,
) -> Result<(Tree, BackboneReport), BonsaiErrors> {
    let params = params.unwrap_or_default();
    let n_cells = data.n_cells;
    let p = data.n_features();

    if n_cells < 2 {
        return Err(BonsaiErrors::EmptyInput {
            n_cells,
            n_features: p,
        });
    }
    let n_backbone = params.backbone_cells.clamp(2, n_cells);
    let mut report = BackboneReport::empty(n_backbone);

    // Step 2: the standard algorithm on a random subset.
    let t0 = Instant::now();
    let order = shuffled_cells(n_cells, params.seed);
    let mut grown = Growing::seed(data, &order[..n_backbone])?;
    let seed_tree = bonsai_prepared(&grown.subset()?, Some(params.bonsai))?.tree;
    grown.adopt(seed_tree);
    report.seed_seconds = t0.elapsed().as_secs_f64();

    // Step 3: place the rest in rounds, each growing the tree by
    // `regrow_fraction` against one collapse, optionally reoptimising between
    // rounds. Not after the last: the final refinement's step 4 does that.
    let mut scored_total = 0usize;
    let mut next = n_backbone;
    let mut last_stage = n_backbone;
    while next < n_cells {
        let size = ((grown.n_leaves as f64 * params.regrow_fraction).ceil() as usize).max(1);
        let end = (next + size).min(n_cells);

        let t0 = Instant::now();
        scored_total += grown.place_round(data, &order[next..end], &params)?;
        report.place_seconds += t0.elapsed().as_secs_f64();
        let k = end - next;
        report.placed += k;
        next = end;

        if params.resolve_attachments {
            let t0 = Instant::now();
            grown.resolve_new(k, &params)?;
            report.resolve_seconds += t0.elapsed().as_secs_f64();
        }
        if next < n_cells && grown.n_leaves as f64 >= last_stage as f64 * params.stage_growth {
            let t0 = Instant::now();
            let sub = grown.subset()?;
            let refined = refine(&grown.tree()?, &sub, Some(params.bonsai))?.tree;
            grown.adopt(refined);
            report.stage_seconds += t0.elapsed().as_secs_f64();
            last_stage = grown.n_leaves;
        }

        if next < n_cells && params.growth_branch.max_iter > 0 {
            let t0 = Instant::now();
            grown.reoptimise(&params)?;
            report.reoptimise_seconds += t0.elapsed().as_secs_f64();
            report.reoptimisations += 1;
        }
    }
    report.mean_scored = if report.placed == 0 {
        0.0
    } else {
        scored_total as f64 / report.placed as f64
    };

    Ok((grown.finish(n_cells)?, report))
}

//////////////
// Internal //
//////////////

/// Cell indices in a deterministic shuffled order.
///
/// The backbone is the first `n_backbone` of these and the growth order is the
/// rest, so one shuffle fixes both. Fisher-Yates over the crate's own PRNG, so
/// the same seed gives the same subset on every platform and thread count.
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `seed` - Seed for the shuffle
///
/// ### Returns
///
/// A permutation of `0..n_cells`.
fn shuffled_cells(n_cells: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_cells).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n_cells).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    order
}

/// A tree being grown one round of leaves at a time.
///
/// Leaves are numbered in the order they were added, not by cell index, because
/// the arena needs its leaves contiguous from zero and a cell arriving later
/// cannot be given an index in the middle. [`Growing::finish`] permutes them
/// back into cell order at the end, which is a relabelling of leaves only and so
/// preserves the arena invariant.
struct Growing<T> {
    /// Parent of each local node, [`NO_NODE`] for the root.
    parent: Vec<u32>,
    /// Branch above each local node.
    branch: Vec<f64>,
    /// Number of leaves placed so far.
    n_leaves: usize,
    /// Cell index of each local leaf.
    cell_of: Vec<usize>,
    /// Leaf means in placement order, row-major.
    means: Vec<T>,
    /// Leaf precisions, same layout.
    precisions: Vec<T>,
    /// Features per row.
    p: usize,
}

impl<T: BonsaiFloat> Growing<T> {
    /// Start from a set of cells with no topology yet.
    ///
    /// ### Params
    ///
    /// * `data` - The full dataset
    /// * `cells` - Cell indices forming the backbone
    ///
    /// ### Returns
    ///
    /// The growing tree, with no parent structure until [`Growing::adopt`].
    fn seed(data: &PreparedData<T>, cells: &[usize]) -> Result<Self, BonsaiErrors> {
        let p = data.n_features();
        let mut means = Vec::with_capacity(cells.len() * p);
        let mut precisions = Vec::with_capacity(cells.len() * p);
        for &cell in cells {
            let lo = cell * p;
            means.extend_from_slice(&data.transformed_means[lo..lo + p]);
            precisions.extend_from_slice(&data.transformed_precisions[lo..lo + p]);
        }
        Ok(Self {
            parent: Vec::new(),
            branch: Vec::new(),
            n_leaves: cells.len(),
            cell_of: cells.to_vec(),
            means,
            precisions,
            p,
        })
    }

    /// The current leaves as a standalone dataset, for the backbone search.
    ///
    /// ### Returns
    ///
    /// A `PreparedData` over the placed cells, in placement order.
    fn subset(&self) -> Result<PreparedData<T>, BonsaiErrors> {
        Ok(PreparedData {
            transformed_means: self.means.clone(),
            transformed_precisions: self.precisions.clone(),
            features: (0..self.p).collect(),
            variances: vec![1.0; self.p],
            signal_to_noise: vec![f64::INFINITY; self.p],
            n_cells: self.n_leaves,
            n_features_in: self.p,
        })
    }

    /// Take the topology of a freshly built tree over the current leaves.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose leaves are this object's leaves, in order
    fn adopt(&mut self, tree: Tree) {
        self.parent = (0..tree.n_nodes() as u32)
            .map(|v| tree.parent(v).unwrap_or(NO_NODE))
            .collect();
        self.branch = tree.branches().to_vec();
    }

    /// The current topology as a `Tree`.
    ///
    /// ### Returns
    ///
    /// The tree, or the arena's error if the structure is malformed.
    fn tree(&self) -> Result<Tree, BonsaiErrors> {
        Tree::from_parents(self.parent.clone(), self.branch.clone(), self.n_leaves)
    }

    /// Place one round of cells against the tree as it stands, then attach
    /// them all.
    ///
    /// One collapse serves the whole round and every cell is placed against
    /// it in parallel, so a round costs one `O(n p)` sweep plus one beam search
    /// per cell, where placing one cell at a time paid the sweep per cell. The
    /// price is that cells in the same round cannot see each other: two that
    /// belong together land on the same node as siblings rather than one below
    /// the other, which is a polytomy the final refinement's step 3 resolves.
    ///
    /// Each cell attaches as another child of its chosen node. SPEC.md section
    /// 7.3 says that is how attaching to an *edge* is covered as well.
    ///
    /// ### Params
    ///
    /// * `data` - The full dataset
    /// * `cells` - Cell indices to place, in growth order
    /// * `params` - Backbone knobs, for the placement tolerance
    ///
    /// ### Returns
    ///
    /// Total nodes the beam searches scored.
    fn place_round(
        &mut self,
        data: &PreparedData<T>,
        cells: &[usize],
        params: &BackboneParams,
    ) -> Result<usize, BonsaiErrors> {
        let p = self.p;
        let tree = self.tree()?;

        // The effective leaf of the whole tree seen from each node, which is
        // what `place` documents as its contract. That is the same quantity as
        // the posterior at that node: the subtree below combined with
        // everything above, reached across the node's own branch.
        let (eff_m, eff_w) = collapse_onto_every_node(&tree, &self.means, &self.precisions, p)?;
        let eff_w: Vec<T> = eff_w.iter().map(|&x| narrow(x)).collect();

        let starts = self.nearest_leaves(data, cells, params.leaf_starts)?;

        // Independent read-only searches, collected in input order, so the
        // result does not depend on the thread count.
        let placements: Vec<Placement> = cells
            .par_iter()
            .zip(starts.par_iter())
            .map(|(&cell, extra)| {
                let lo = cell * p;
                let q = EffLeaf {
                    m: &data.transformed_means[lo..lo + p],
                    w: &data.transformed_precisions[lo..lo + p],
                };
                place_from(
                    &tree,
                    q,
                    |node| {
                        let at = node as usize * p;
                        EffLeaf {
                            m: &eff_m[at..at + p],
                            w: &eff_w[at..at + p],
                        }
                    },
                    extra,
                    Some(params.placement),
                )
            })
            .collect::<Result<_, _>>()?;

        self.attach(data, cells, &placements);
        Ok(placements.iter().map(|x| x.scored).sum())
    }

    /// Resolve the polytomies at the parents of the newest leaves.
    ///
    /// Up to [`MAX_RESOLVE_PASSES`] passes, because a centre whose parent is
    /// also a centre waits for the next pass.
    ///
    /// ### Params
    ///
    /// * `k` - How many of the most recent leaves were just attached
    /// * `params` - Backbone knobs, for the star primitive
    fn resolve_new(&mut self, k: usize, params: &BackboneParams) -> Result<(), BonsaiErrors> {
        let leaves = Leaves {
            means: &self.means,
            precisions: &self.precisions,
            n_features: self.p,
        };
        let mut tree = self.tree()?;
        for _ in 0..MAX_RESOLVE_PASSES {
            let centres: Vec<u32> = (self.n_leaves - k..self.n_leaves)
                .filter_map(|leaf| tree.parent(leaf as u32))
                .collect();
            let (resolved, _, skipped) =
                resolve_centres(&tree, leaves, &centres, Some(params.bonsai.star))?;
            tree = resolved;
            if skipped == 0 {
                break;
            }
        }
        self.adopt(tree);
        Ok(())
    }

    /// The `k` current leaves nearest each cell, as start points for its beam
    /// search.
    ///
    /// Plain Euclidean over the transformed means in `f32`, the same metric and
    /// precision [`crate::tree::linkage`] builds its graph in, and for the same
    /// reason: this only decides where a search starts, and the search scores
    /// with the model. Exhaustive, one index per round over the leaves placed
    /// so far.
    ///
    /// ### Params
    ///
    /// * `data` - The full dataset
    /// * `cells` - Cell indices about to be placed
    /// * `k` - Leaves wanted per cell; zero skips the search
    ///
    /// ### Returns
    ///
    /// One list of leaf node indices per cell, nearest first.
    fn nearest_leaves(
        &self,
        data: &PreparedData<T>,
        cells: &[usize],
        k: usize,
    ) -> Result<Vec<Vec<u32>>, BonsaiErrors> {
        let k = k.min(self.n_leaves);
        if k == 0 {
            return Ok(vec![Vec::new(); cells.len()]);
        }
        let p = self.p;
        let to_f32 = |x: &T| x.to_f32().unwrap_or(0.0);
        let leaves: Vec<f32> = self.means.iter().map(to_f32).collect();
        let mut queries: Vec<f32> = Vec::with_capacity(cells.len() * p);
        for &cell in cells {
            queries.extend(
                data.transformed_means[cell * p..(cell + 1) * p]
                    .iter()
                    .map(to_f32),
            );
        }
        let index = build_exhaustive_index((leaves.as_slice(), self.n_leaves, p), NEAREST_METRIC);
        let (rows, _) = query_exhaustive_index(
            (queries.as_slice(), cells.len(), p),
            &index,
            k,
            false,
            false,
        )
        .map_err(|e| BonsaiErrors::NeighbourGraph {
            reason: e.to_string(),
        })?;
        Ok(rows
            .into_iter()
            .map(|row| row.into_iter().map(|i| i as u32).collect())
            .collect())
    }

    /// Append one round of leaves, each below the node its placement chose.
    ///
    /// Two cases. A new leaf below an *internal* node just becomes another
    /// child. A new leaf below another *leaf* cannot: this arena has no
    /// data-carrying internal node, so a fresh internal node takes the target's
    /// place with the target hanging off it on a zero-length branch. That is the
    /// same unrooted tree and the same point the placement was scored at, and it
    /// is what `search::spr` does for the identical reason. Every cell of the
    /// round that chose the same leaf shares that one fresh node.
    ///
    /// Indices are rebuilt rather than patched. Inserting `k` leaves shifts
    /// every internal node up by `k` and the fresh nodes land at the end, so
    /// the parent-above-child invariant is restored by [`Growing::renumber`]
    /// rather than reasoned about case by case.
    ///
    /// ### Params
    ///
    /// * `data` - The full dataset
    /// * `cells` - Cell indices the new leaves carry, in order
    /// * `placements` - Where each cell goes, in the current numbering
    fn attach(&mut self, data: &PreparedData<T>, cells: &[usize], placements: &[Placement]) {
        let p = self.p;
        let old_leaves = self.n_leaves;
        let k = cells.len();
        // Every internal node moves up by `k` to make room for the new leaves.
        let shift = |v: u32| -> u32 {
            if v == NO_NODE || (v as usize) < old_leaves {
                v
            } else {
                v + k as u32
            }
        };

        let mut parent = Vec::with_capacity(self.parent.len() + 2 * k);
        let mut branches = Vec::with_capacity(self.branch.len() + 2 * k);
        parent.extend(self.parent[..old_leaves].iter().map(|&v| shift(v)));
        branches.extend_from_slice(&self.branch[..old_leaves]);
        parent.extend(std::iter::repeat_n(NO_NODE, k)); // the new leaves, wired below
        branches.extend(placements.iter().map(|x| x.branch));
        parent.extend(self.parent[old_leaves..].iter().map(|&v| shift(v)));
        branches.extend_from_slice(&self.branch[old_leaves..]);

        let mut joint_of = vec![NO_NODE; old_leaves];
        for (i, at) in placements.iter().enumerate() {
            let target = at.node as usize;
            parent[old_leaves + i] = if target < old_leaves {
                // Leaf target: a new internal node takes its place, once.
                if joint_of[target] == NO_NODE {
                    let joint = parent.len() as u32;
                    parent.push(parent[target]);
                    branches.push(branches[target]);
                    parent[target] = joint;
                    branches[target] = 0.0;
                    joint_of[target] = joint;
                }
                joint_of[target]
            } else {
                shift(at.node)
            };
        }

        for &cell in cells {
            let lo = cell * p;
            self.means
                .extend_from_slice(&data.transformed_means[lo..lo + p]);
            self.precisions
                .extend_from_slice(&data.transformed_precisions[lo..lo + p]);
        }
        self.cell_of.extend_from_slice(cells);
        self.n_leaves += k;
        let (parent, branches) = Self::renumber(parent, branches, self.n_leaves);
        self.parent = parent;
        self.branch = branches;
    }

    /// Reorder internal nodes so every parent index exceeds its children's.
    ///
    /// The arena requires it and `Tree::from_parents` checks rather than fixes
    /// it. Leaves keep their indices; internal nodes are numbered in the order
    /// their children finish, which is a post-order, by a Kahn-style sweep over
    /// the remaining child counts. No recursion, so a hundred-thousand-leaf
    /// ladder is a flat scan.
    ///
    /// ### Params
    ///
    /// * `parent` - Parent per node, in any internal ordering
    /// * `branch` - Branch above each node, same indexing
    /// * `n_leaves` - Leaf count; leaves occupy `0..n_leaves` already
    ///
    /// ### Returns
    ///
    /// The same tree with its internal nodes renumbered.
    fn renumber(parent: Vec<u32>, branch: Vec<f64>, n_leaves: usize) -> (Vec<u32>, Vec<f64>) {
        let n = parent.len();
        let mut pending = vec![0usize; n];
        for &v in &parent {
            if v != NO_NODE {
                pending[v as usize] += 1;
            }
        }

        let mut relabel = vec![u32::MAX; n];
        let mut queue: Vec<u32> = Vec::with_capacity(n);
        for (leaf, slot) in relabel.iter_mut().enumerate().take(n_leaves) {
            *slot = leaf as u32;
            queue.push(leaf as u32);
        }

        let mut next = n_leaves as u32;
        let mut head = 0usize;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            let up = parent[node as usize];
            if up == NO_NODE {
                continue;
            }
            pending[up as usize] -= 1;
            if pending[up as usize] == 0 {
                relabel[up as usize] = next;
                next += 1;
                queue.push(up);
            }
        }

        let mut out_parent = vec![NO_NODE; n];
        let mut out_branch = vec![0.0f64; n];
        for old in 0..n {
            let to = relabel[old] as usize;
            out_parent[to] = match parent[old] {
                NO_NODE => NO_NODE,
                up => relabel[up as usize],
            };
            out_branch[to] = branch[old];
        }
        (out_parent, out_branch)
    }

    /// Reoptimise every branch length of the grown tree.
    ///
    /// ### Params
    ///
    /// * `params` - Backbone knobs, for the branch-length settings
    fn reoptimise(&mut self, params: &BackboneParams) -> Result<(), BonsaiErrors> {
        let mut tree = self.tree()?;
        let mut state = NodeState::new(tree.n_nodes(), self.p, &self.means, &self.precisions)?;
        optimise_branch_lengths(&mut tree, &mut state, Some(params.growth_branch))?;
        self.branch = tree.branches().to_vec();
        Ok(())
    }

    /// The finished tree, with leaves back in cell order.
    ///
    /// Leaves were numbered as they arrived; the caller's contract is that leaf
    /// `i` is cell `i`. Permuting leaves among themselves leaves every internal
    /// node's index alone, so the arena invariant survives untouched.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells, which must equal the leaf count
    ///
    /// ### Returns
    ///
    /// The tree, leaves in cell order.
    fn finish(&self, n_cells: usize) -> Result<Tree, BonsaiErrors> {
        if self.n_leaves != n_cells {
            return Err(BonsaiErrors::ShapeMismatch {
                mean_cells: n_cells,
                mean_features: self.p,
                sd_cells: self.n_leaves,
                sd_features: self.p,
            });
        }
        // `to_cell[local] = cell`, so the leaf currently at `local` belongs at
        // index `cell`.
        let mut parent = vec![NO_NODE; self.parent.len()];
        let mut branch = vec![0.0f64; self.branch.len()];
        let relabel = |v: u32| -> u32 {
            if v == NO_NODE {
                NO_NODE
            } else if (v as usize) < self.n_leaves {
                self.cell_of[v as usize] as u32
            } else {
                v
            }
        };
        for local in 0..self.parent.len() {
            let to = relabel(local as u32) as usize;
            parent[to] = relabel(self.parent[local]);
            branch[to] = self.branch[local];
        }
        Tree::from_parents(parent, branch, n_cells)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};
    use approx::assert_relative_eq;

    /// A simulated dataset already in transformed units, which is what
    /// `PreparedData` holds.
    fn fixture(n: usize, p: usize, noise: f64, seed: u64) -> (PreparedData<f64>, Tree) {
        let d = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: noise,
            seed,
            ..Default::default()
        }))
        .expect("simulation");
        (
            PreparedData {
                transformed_means: d.means.clone(),
                transformed_precisions: d.precisions(),
                features: (0..p).collect(),
                variances: vec![1.0; p],
                signal_to_noise: vec![f64::INFINITY; p],
                n_cells: n,
                n_features_in: p,
            },
            d.tree,
        )
    }

    #[test]
    fn test_a_backbone_covering_everything_is_the_standard_algorithm() {
        let (data, _) = fixture(32, 64, 0.2, 3);
        let (out, report) = backbone(
            &data,
            Some(BackboneParams {
                backbone_cells: 1000,
                ..Default::default()
            }),
        )
        .expect("backbone");
        let plain = bonsai_prepared(&data, None).expect("bonsai");

        assert_eq!(report.placed, 0);
        assert_relative_eq!(out.loglik, plain.loglik, max_relative = 1e-12);
        assert_eq!(robinson_foulds(&out.tree, &plain.tree).expect("rf"), 0);
    }

    #[test]
    fn test_growth_puts_every_cell_in_and_keeps_them_in_order() {
        // The contract is that leaf `i` is cell `i`, and growth numbers leaves
        // in arrival order, so the permutation at the end is what makes it right.
        let (n, p) = (64usize, 64usize);
        let (data, _) = fixture(n, p, 0.2, 5);
        let (out, report) = backbone(
            &data,
            Some(BackboneParams {
                backbone_cells: 16,
                ..Default::default()
            }),
        )
        .expect("backbone");

        assert_eq!(report.backbone_cells, 16);
        assert_eq!(report.placed, n - 16);
        assert_eq!(out.tree.n_leaves(), n);

        // Every leaf's posterior must sit near its own measurement, which is
        // only true if leaf `i` really is cell `i`.
        for leaf in 0..n {
            for g in 0..p {
                let idx = leaf * p + g;
                let measured = data.transformed_means[idx];
                let posterior = out.node_means[idx];
                let sd = (1.0 / data.transformed_precisions[idx]).sqrt();
                assert!(
                    (posterior - measured).abs() < 6.0 * sd,
                    "leaf {leaf} feature {g} sits {} sds from its measurement",
                    (posterior - measured).abs() / sd
                );
            }
        }
    }

    #[test]
    fn test_backbone_recovers_the_tree_and_costs_little_against_the_standard_search() {
        // The question the mode exists to answer: how much accuracy does the
        // approximation give up? Reported rather than merely asserted.
        let (n, p) = (64usize, 128usize);
        let (data, truth) = fixture(n, p, 0.2, 7);

        let plain = bonsai_prepared(&data, None).expect("bonsai");
        let (grown, _) = backbone(
            &data,
            Some(BackboneParams {
                backbone_cells: 16,
                ..Default::default()
            }),
        )
        .expect("backbone");

        let rf_plain = robinson_foulds(&plain.tree, &truth).expect("rf");
        let rf_grown = robinson_foulds(&grown.tree, &truth).expect("rf");
        assert!(
            rf_grown <= rf_plain + 6,
            "backbone gave up too much: RF {rf_grown} against the standard search's {rf_plain}"
        );
        assert!(grown.loglik.is_finite());
    }

    #[test]
    fn test_the_same_seed_gives_the_same_tree() {
        let (data, _) = fixture(32, 64, 0.2, 11);
        let params = BackboneParams {
            backbone_cells: 12,
            seed: 99,
            ..Default::default()
        };
        let first = backbone(&data, Some(params)).expect("backbone").0;
        let second = backbone(&data, Some(params)).expect("backbone").0;
        assert_eq!(first.loglik.to_bits(), second.loglik.to_bits());
        assert_eq!(first.tree.branches(), second.tree.branches());
    }

    #[test]
    fn test_a_different_seed_picks_a_different_backbone() {
        let (data, _) = fixture(32, 64, 0.2, 13);
        let base = BackboneParams {
            backbone_cells: 12,
            ..Default::default()
        };
        let a = shuffled_cells(32, 1);
        let b = shuffled_cells(32, 2);
        assert_ne!(a[..12], b[..12], "the seed did not reach the subset");

        // Both must still produce a usable tree.
        for seed in [1u64, 2] {
            let out = backbone(&data, Some(BackboneParams { seed, ..base }))
                .expect("backbone")
                .0;
            assert!(out.loglik.is_finite());
        }
    }

    #[test]
    fn test_reoptimisation_cadence_changes_the_result() {
        // Pins that `regrow_fraction` is wired through rather than ignored. A
        // constant nothing reads is worse than no constant.
        let (data, _) = fixture(64, 64, 0.3, 17);
        let mut reports = Vec::new();
        for fraction in [0.1f64, 10.0] {
            let (_, report) = backbone(
                &data,
                Some(BackboneParams {
                    backbone_cells: 12,
                    regrow_fraction: fraction,
                    growth_branch: GlobalBranchParams::default(),
                    ..Default::default()
                }),
            )
            .expect("backbone");
            reports.push(report.reoptimisations);
        }
        assert!(
            reports[0] > reports[1],
            "a tighter cadence did not reoptimise more often: {reports:?}"
        );
    }

    #[test]
    fn test_too_few_cells_is_an_error() {
        let (data, _) = fixture(2, 8, 0.2, 19);
        let one = PreparedData {
            transformed_means: data.transformed_means[..8].to_vec(),
            transformed_precisions: data.transformed_precisions[..8].to_vec(),
            n_cells: 1,
            ..data
        };
        assert!(backbone(&one, None).is_err());
    }
}
