//! Backbone mode: reconstruct on a subset, then place the rest.
//!
//! SPEC.md section 15. Four steps: preprocess everything, run the standard
//! algorithm on a random subset to get a backbone, place the remaining cells on
//! it one at a time, then refine the whole thing.
//!
//! ### What this does and does not buy
//!
//! It replaces the standard search's cost on `n` cells with its cost on the
//! backbone plus one placement per remaining cell. The merge step is what it
//! avoids, and measured at 200 features the merge is `n^1.64` while step 5 is
//! `n^1.98` and ninety per cent of the runtime. **So on the current search this
//! trades away the cheap step and keeps the expensive one**, because the final
//! refinement runs SPR over every cell however the backbone was built. It is
//! written now because it composes: when SPR is sub-quadratic, this is what
//! turns the remaining cost into something atlas-scale.
//!
//! Be honest about that when quoting numbers for it.
//!
//! ### The one place this differs from the standard algorithm's answer
//!
//! Placement is a beam search with a tolerance, not an exhaustive scan, and a
//! cell placed early cannot see cells placed after it. So this is an
//! approximation and its trees are not guaranteed identical to the standard
//! algorithm's, unlike SPEC sections 10 and 11 whose whole point is that they
//! are. The paper is explicit that backbone mode trades accuracy for time. What
//! that costs is measured in the tests rather than assumed.

use crate::bonsai::{BonsaiParams, BonsaiResult, bonsai_prepared, refine};
use crate::errors::BonsaiErrors;
use crate::ingest::PreparedData;
use crate::model::global::{UpState, optimise_branch_lengths};
use crate::model::likelihood::NodeState;
use crate::model::merge::EffLeaf;
use crate::model::place::{PlacementParams, place};
use crate::tree::{NO_NODE, Tree};
use crate::utils::rng::SplitMix64;
use crate::utils::traits::{BonsaiFloat, narrow, wide};

/// Cells in the initial backbone, when the caller does not say.
///
/// The backbone has to be large enough to carry the structure the rest of the
/// cells will be placed against; too small and every placement is deciding
/// between branches that are not there yet. The reference suggests ten
/// thousand. This crate's default is smaller because the standard search is
/// still `n^1.8`, so a ten thousand cell backbone is most of the total cost;
/// raise it once that changes. Chosen 2026-09-05 as a starting point, not from
/// a recovery measurement.
pub const DEFAULT_BACKBONE_CELLS: usize = 2048;

/// Fraction of growth after which the branch lengths are reoptimised.
///
/// The Methods note the backbone changes appreciably as cells are added, so a
/// tree grown far past its last optimisation is being placed against stale
/// branch lengths. A quarter means four reoptimisations per doubling.
/// Chosen 2026-09-05; see `test_reoptimisation_cadence_changes_the_result`.
pub const DEFAULT_REGROW_FRACTION: f64 = 0.25;

/// Knobs for backbone mode.
#[derive(Clone, Copy, Debug)]
pub struct BackboneParams {
    /// Cells in the initial backbone.
    pub backbone_cells: usize,
    /// Reoptimise the branch lengths once the tree has grown by this fraction
    /// since the last time.
    pub regrow_fraction: f64,
    /// Seed for choosing the backbone subset.
    pub seed: u64,
    /// Placement search knobs for the growth phase.
    pub placement: PlacementParams,
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
            placement: PlacementParams::default(),
            bonsai: BonsaiParams::default(),
        }
    }
}

/// What the growth phase did.
#[derive(Clone, Copy, Debug)]
pub struct BackboneReport {
    /// Cells in the initial backbone.
    pub backbone_cells: usize,
    /// Cells added one at a time afterwards.
    pub placed: usize,
    /// Times the branch lengths were reoptimised during growth.
    pub reoptimisations: usize,
    /// Mean nodes scored per placement. The handle on whether the beam search's
    /// tolerance is doing anything; compare against the node count.
    pub mean_scored: f64,
}

/// Reconstruct a tree by building a backbone and placing the rest onto it.
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
    let p = data.n_features();

    if n_cells < 2 {
        return Err(BonsaiErrors::EmptyInput {
            n_cells,
            n_features: p,
        });
    }

    // A backbone at least as large as the dataset means there is nothing to
    // place, so this is the standard algorithm with extra steps.
    let n_backbone = params.backbone_cells.clamp(2, n_cells);
    if n_backbone == n_cells {
        let out = bonsai_prepared(data, Some(params.bonsai))?;
        return Ok((
            out,
            BackboneReport {
                backbone_cells: n_cells,
                placed: 0,
                reoptimisations: 0,
                mean_scored: 0.0,
            },
        ));
    }

    // Step 2: the standard algorithm on a random subset.
    let order = shuffled_cells(n_cells, params.seed);
    let mut grown = Growing::seed(data, &order[..n_backbone])?;
    let seed_tree = bonsai_prepared(&grown.subset()?, Some(params.bonsai))?.tree;
    grown.adopt(seed_tree);

    // Step 3: place the rest, reoptimising as the tree grows.
    let mut report = BackboneReport {
        backbone_cells: n_backbone,
        placed: 0,
        reoptimisations: 0,
        mean_scored: 0.0,
    };
    let mut scored_total = 0usize;
    let mut since_optimised = n_backbone;

    for &cell in &order[n_backbone..] {
        let scored = grown.place_cell(data, cell, &params)?;
        scored_total += scored;
        report.placed += 1;

        if grown.n_leaves as f64 >= since_optimised as f64 * (1.0 + params.regrow_fraction) {
            grown.reoptimise(&params)?;
            report.reoptimisations += 1;
            since_optimised = grown.n_leaves;
        }
    }
    report.mean_scored = if report.placed == 0 {
        0.0
    } else {
        scored_total as f64 / report.placed as f64
    };

    // Step 4: refine with every cell, which is the standard algorithm's steps 3
    // to 7 over the grown tree.
    let tree = grown.finish(n_cells)?;
    let out = refine(&tree, data, Some(params.bonsai))?;
    Ok((out, report))
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

/// A tree being grown one leaf at a time.
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

    /// Place one cell and attach it.
    ///
    /// The cell attaches as another child of the chosen node, which makes a
    /// polytomy there. SPEC.md section 7.3 says that is how attaching to an
    /// *edge* is covered as well, and the polytomy is resolved by step 3 of the
    /// final refinement rather than here: resolving after every placement would
    /// pay a star primitive per cell for a configuration the next cell may
    /// change anyway.
    ///
    /// ### Params
    ///
    /// * `data` - The full dataset
    /// * `cell` - Cell index to place
    /// * `params` - Backbone knobs, for the placement tolerance
    ///
    /// ### Returns
    ///
    /// How many nodes the beam search scored.
    fn place_cell(
        &mut self,
        data: &PreparedData<T>,
        cell: usize,
        params: &BackboneParams,
    ) -> Result<usize, BonsaiErrors> {
        let p = self.p;
        let tree = self.tree()?;

        // The effective leaf of the whole tree seen from each node, which is
        // what `place` documents as its contract. That is the same quantity as
        // the posterior at that node: the subtree below combined with
        // everything above, reached across the node's own branch.
        let (eff_m, eff_w) = collapsed_onto_every_node(&tree, &self.means, &self.precisions, p)?;

        let lo = cell * p;
        let q = EffLeaf {
            m: &data.transformed_means[lo..lo + p],
            w: &data.transformed_precisions[lo..lo + p],
        };
        let placement = place(
            &tree,
            q,
            |node| {
                let at = node as usize * p;
                EffLeaf {
                    m: &eff_m[at..at + p],
                    w: &eff_w[at..at + p],
                }
            },
            Some(params.placement),
        )?;

        self.attach(cell, placement.node, placement.branch, q);
        Ok(placement.scored)
    }

    /// Append a leaf below an existing node.
    ///
    /// Two cases. A new leaf below an *internal* node just becomes another
    /// child, which makes a polytomy there for the final refinement's step 3 to
    /// resolve. A new leaf below another *leaf* cannot: this arena has no
    /// data-carrying internal node, so a fresh internal node takes the target's
    /// place with the target hanging off it on a zero-length branch. That is the
    /// same unrooted tree and the same point the placement was scored at, and it
    /// is what `search::spr` does for the identical reason.
    ///
    /// Indices are rebuilt rather than patched. Inserting a leaf shifts every
    /// internal node up by one, and the leaf case adds a node in the middle of
    /// the ordering, so the parent-above-child invariant is restored by
    /// [`Growing::renumber`] rather than reasoned about case by case.
    ///
    /// ### Params
    ///
    /// * `cell` - Cell index the new leaf carries
    /// * `target` - Node to attach below, in the current numbering
    /// * `branch` - Length of the new edge
    /// * `q` - The cell's own effective leaf
    fn attach(&mut self, cell: usize, target: u32, branch: f64, q: EffLeaf<'_, T>) {
        let old_leaves = self.n_leaves;
        let new_leaf = old_leaves as u32;
        // Every internal node moves up one to make room for the new leaf.
        let shift = |v: u32| -> u32 {
            if v == NO_NODE || (v as usize) < old_leaves {
                v
            } else {
                v + 1
            }
        };

        let mut parent = Vec::with_capacity(self.parent.len() + 2);
        let mut branches = Vec::with_capacity(self.branch.len() + 2);
        parent.extend(self.parent[..old_leaves].iter().map(|&v| shift(v)));
        branches.extend_from_slice(&self.branch[..old_leaves]);
        parent.push(NO_NODE); // the new leaf, wired below
        branches.push(branch);
        parent.extend(self.parent[old_leaves..].iter().map(|&v| shift(v)));
        branches.extend_from_slice(&self.branch[old_leaves..]);

        if (target as usize) < old_leaves {
            // Leaf target: a new internal node takes its place.
            let joint = parent.len() as u32;
            parent.push(shift(self.parent[target as usize]));
            branches.push(self.branch[target as usize]);
            parent[target as usize] = joint;
            branches[target as usize] = 0.0;
            parent[new_leaf as usize] = joint;
        } else {
            parent[new_leaf as usize] = shift(target);
        }

        self.n_leaves += 1;
        self.cell_of.push(cell);
        self.means.extend_from_slice(q.m);
        self.precisions.extend_from_slice(q.w);
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
        optimise_branch_lengths(&mut tree, &mut state, Some(params.bonsai.branch))?;
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

/// The whole tree collapsed onto each node in turn.
///
/// This is `place`'s contract: for node `a`, the effective leaf you get by
/// rooting there and marginalising everything else. Two sweeps give it for every
/// node at once, which is what makes a placement search affordable.
///
/// ### Params
///
/// * `tree` - The tree
/// * `means` - Leaf means, row-major
/// * `precisions` - Leaf precisions, same layout
/// * `p` - Features per row
///
/// ### Returns
///
/// Means and precisions, row-major `[node][feature]`.
fn collapsed_onto_every_node<T: BonsaiFloat>(
    tree: &Tree,
    means: &[T],
    precisions: &[T],
    p: usize,
) -> Result<(Vec<T>, Vec<T>), BonsaiErrors> {
    let n_nodes = tree.n_nodes();
    let mut down = NodeState::new(n_nodes, p, means, precisions)?;
    down.prune(tree);
    let mut up = UpState::new(n_nodes, p);
    up.sweep(tree, &down);

    let mut m = vec![T::zero(); n_nodes * p];
    let mut w = vec![T::zero(); n_nodes * p];
    for node in 0..n_nodes as u32 {
        let lo = node as usize * p;
        let (m_down, w_down) = (down.means(node), down.precisions(node));
        if tree.parent(node).is_none() {
            m[lo..lo + p].copy_from_slice(m_down);
            w[lo..lo + p].copy_from_slice(w_down);
            continue;
        }
        let (m_up, w_up) = (up.means(node), up.precisions(node));
        let t = tree.branch(node);
        for g in 0..p {
            let w_u = wide(w_up[g]);
            let up_here = w_u / (1.0 + t * w_u);
            let w_d = wide(w_down[g]);
            let total = w_d + up_here;
            let md = wide(m_down[g]);
            m[lo + g] = narrow(md + (wide(m_up[g]) - md) * (up_here / total));
            w[lo + g] = narrow(total);
        }
    }
    Ok((m, w))
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
        // in arrival order, so the permutation at the end is load-bearing.
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
