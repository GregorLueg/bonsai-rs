//! A starting tree from a neighbour-graph linkage.
//!
//! Search step 2 builds the initial topology by greedy likelihood-driven
//! agglomeration (SPEC.md section 9.1). It does not earn its keep: a Ward
//! linkage over the same transformed means reaches the same Robinson-Foulds
//! distance and the same loglikelihood after steps 3 to 7, at a fraction of the
//! cost. `docs/PERFORMANCE.md` has the numbers.
//!
//! What this module does *not* do is score anything with the model. It hands
//! the refinement a structurally sensible tree and gets out of the way. That is
//! a low bar on purpose: refinement reaches Robinson-Foulds 0 even from a
//! uniformly random topology, so the linkage only has to beat random, which it
//! does in wall time rather than in recovery.
//!
//! ### Why Ward and not the merge score
//!
//! Ward is reducible, so every mutually nearest pair is a pair the naive scan
//! merges at some point, and merging all of them at once builds the same
//! dendrogram (Bruynooghe 1977, Murtagh 1983). The Bonsai merge gain is
//! **not**: merging changes the peeled remainder that every other pair's score
//! depends on, and about one round in five lifts some other pair above the
//! score the merged pair had, by up to 13 nats. A linkage driven by the gain
//! would silently build a different dendrogram from the one the round-by-round
//! scan builds.
//!
//! Ward is also defined through centroids, which is what makes it work on a
//! sparse graph: a merged cluster's position is one `O(p)` update and needs no
//! distance matrix.
//!
//! ### The graph, and why it is an approximation
//!
//! Each round needs the nearest live *cluster* to every live cluster. A
//! neighbour graph over the original cells answers that for leaves and only
//! approximately for clusters, so this inherits the union of the two children's
//! neighbour lists on every merge and rebuilds the graph over the live
//! centroids when the live count has halved. That is the same device SPEC.md
//! section 11 uses for the merge scan, for the same reason, and
//! [`crate::search::candidates`] documents the measurement behind `k = 16`.
//!
//! The union has to be symmetric. Giving the merged cluster its children's
//! lists without replacing the children in everyone else's leaves the new
//! cluster invisible to the clusters it should be adjacent to, and every list
//! runs dry. `search::candidates` records the same trap.
//!
//! ### Rounds, not a chain
//!
//! A nearest-neighbour chain builds caterpillars here, an order of magnitude
//! deeper than `log2(n)`. The chain is depth-first, so one cluster runs away
//! while everything else is still a singleton. A big centroid carries `1/size`
//! of the noise, which at a few thousand features puts it closer to every leaf
//! than that leaf's own third cousins are, so it takes a slot in every list;
//! once a block has merged all its listed relatives the big cluster is its only
//! edge left, the mutual test passes trivially, and the block is absorbed.
//! Merging every mutual pair per round keeps the live clusters at similar
//! sizes. `benches/start_tree.rs` is the sweep.

use crate::errors::BonsaiErrors;
use crate::tree::{NO_NODE, Tree};
use crate::utils::traits::BonsaiFloat;
use ann_search_rs::{
    build_exhaustive_index, build_kmknn_index, build_nndescent_index, extract_nndescent_knn,
    query_exhaustive_self, query_kmknn_self,
};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

////////////////
// Parameters //
////////////////

/// Neighbours kept per cluster.
///
/// Ours, chosen by measurement; matches
/// [`crate::search::candidates::KnnCandidatesParams`]. At every `k` from 8 to
/// 128 the rounds reproduce the dense Ward tree exactly, at depth `log2(n)`, so
/// the value is set by the cost of the graph and not by recovery. The `drift`
/// block of `benches/start_tree.rs` is the sweep.
const DEFAULT_K: usize = 16;

/// Rebuild the graph once the live cluster count has fallen to this fraction of
/// what it was at the last rebuild.
///
/// Inheritance propagates the old adjacency, and a merged cluster's centroid is
/// neither child's, so the graph goes stale as a description of the current
/// geometry. Halving gives `log2(n)` rebuilds over a whole linkage, each over a
/// set half the size of the last, so the rebuilds are a geometric series and
/// cost a constant multiple of the first one.
///
/// Ours, chosen by measurement. Every cadence swept, including never redrawing
/// on the count and leaving only the dry-list redraw, reproduces the dense tree
/// at depth `log2(n)`, so the cadence does not decide recovery. Halving is kept
/// as the bound on staleness.
const DEFAULT_REBUILD_FRACTION: f64 = 0.5;

/// Cells up to which the exhaustive backend is used; NN-descent above it.
///
/// Ours, and deliberately conservative. NN-descent's graph produces the
/// identical tree to the exhaustive one at every size measured, so nothing the
/// linkage can see is lost by switching and the crossover is a pure cost
/// decision. kmknn is out of the automatic path: k-means pruning buys nothing
/// at this dimension. The `backend` block of `benches/start_tree.rs` is where
/// to re-place this if the exhaustive build starts to show.
const EXHAUSTIVE_MAX_CELLS: usize = 4_096;

/// Metric the graph is built in. Plain Euclidean on the transformed means: the
/// graph decides which pairs are *considered*, and the linkage that decides
/// which one wins is precision-free by construction.
const ANN_METRIC: &str = "euclidean";

/// NN-descent convergence threshold, as a fraction of the graph's edges updated
/// in an iteration. The backend's own default.
const NNDESCENT_DELTA: f32 = 0.001;

/// NN-descent diversification probability. One, meaning no pruning: the graph
/// is small in `k` and wanted at full recall.
const NNDESCENT_DIVERSIFY: f32 = 1.0;

/// Which neighbour-search backend builds the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnBackend {
    /// Every pair, `O(n^2 p)`. Exact, no build phase.
    Exhaustive,
    /// k-means pruned exact search. Exact, sublinear in distance computations.
    Kmknn,
    /// NN-descent. Approximate, and the cheapest route to a self-kNN graph.
    NnDescent,
}

/// Tuning knobs for the linkage.
#[derive(Clone, Copy, Debug)]
pub struct LinkageParams {
    /// Neighbours kept per cluster.
    pub k: usize,
    /// Backend, or `None` to pick one from the problem size.
    pub backend: Option<KnnBackend>,
    /// Rebuild once the live count has fallen to this fraction of its value at
    /// the last rebuild.
    pub rebuild_fraction: f64,
    /// Seed for the backends that take one.
    pub seed: usize,
}

impl Default for LinkageParams {
    /// `k = 16`, an automatic backend and a rebuild every halving.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            backend: None,
            rebuild_fraction: DEFAULT_REBUILD_FRACTION,
            seed: 0,
        }
    }
}

/// Pick a backend from the problem size.
///
/// Exhaustive up to [`EXHAUSTIVE_MAX_CELLS`], NN-descent above it; the
/// state of the measurement is on the constant.
///
/// ### Params
///
/// * `n_cells` - Number of cells the graph is built over
///
/// ### Returns
///
/// The backend to use.
fn resolve_backend(n_cells: usize) -> KnnBackend {
    if n_cells <= EXHAUSTIVE_MAX_CELLS {
        KnnBackend::Exhaustive
    } else {
        KnnBackend::NnDescent
    }
}

//////////////////
// The linkage //
//////////////////

/// Squared Euclidean distance between two centroid rows.
///
/// Accumulated in `f64` over `f32` storage, which is this crate's rule
/// everywhere: the sum is `O(p)` while the differences that matter are `O(1)`.
///
/// ### Params
///
/// * `a` - First centroid, length `p`
/// * `b` - Second centroid, same length
///
/// ### Returns
///
/// The squared distance.
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for g in 0..a.len() {
        let diff = a[g] as f64 - b[g] as f64;
        acc += diff * diff;
    }
    acc
}

/// Ward's dissimilarity between two live clusters.
///
/// The increase in within-cluster sum of squares that merging them would cause,
/// `(|A| |B| / (|A| + |B|)) * ||cA - cB||^2`. Expressed through centroids rather
/// than through a Lance-Williams recurrence over a distance matrix, which is
/// what lets the linkage run on a neighbour graph with no matrix at all.
///
/// ### Params
///
/// * `centroids` - Centroid rows, `[slot][feature]`, row-major
/// * `size` - Cluster sizes by slot
/// * `p` - Features per row
/// * `a`, `b` - The two slots
///
/// ### Returns
///
/// Ward's dissimilarity.
#[inline]
fn ward(centroids: &[f32], size: &[f64], p: usize, a: usize, b: usize) -> f64 {
    let d2 = squared_distance(
        &centroids[a * p..(a + 1) * p],
        &centroids[b * p..(b + 1) * p],
    );
    let (sa, sb) = (size[a], size[b]);
    sa * sb / (sa + sb) * d2
}

/// Build the neighbour graph over the live clusters.
///
/// Returns a symmetrised adjacency keyed by slot, so a slot's list holds every
/// slot it is adjacent to in either direction. Built in `f32` whatever the
/// storage type, for the reason `search::candidates` gives: the graph only
/// filters which pairs are considered, so its precision does not reach the
/// answer, and `f32` halves the working set.
///
/// ### Params
///
/// * `centroids` - Centroid rows, `[slot][feature]`, row-major
/// * `live` - Slots currently holding a cluster, ascending
/// * `p` - Features per row
/// * `params` - Linkage knobs
///
/// ### Returns
///
/// Adjacency lists indexed by slot; slots not in `live` have empty lists.
fn build_graph(
    centroids: &[f32],
    live: &[usize],
    p: usize,
    params: &LinkageParams,
) -> Result<Vec<Vec<u32>>, BonsaiErrors> {
    let m = live.len();
    let k = params.k.min(m.saturating_sub(1)).max(1);

    let mut packed = Vec::with_capacity(m * p);
    for &slot in live {
        packed.extend_from_slice(&centroids[slot * p..(slot + 1) * p]);
    }

    let backend = params.backend.unwrap_or_else(|| resolve_backend(m));
    let matrix = (packed.as_slice(), m, p);
    let rows = match backend {
        KnnBackend::Exhaustive => {
            let index = build_exhaustive_index(matrix, ANN_METRIC);
            query_exhaustive_self(&index, k + 1, false, false)
        }
        KnnBackend::Kmknn => {
            let index = build_kmknn_index(matrix, ANN_METRIC, None, None, params.seed, false)
                .map_err(|e| BonsaiErrors::NeighbourGraph {
                    reason: e.to_string(),
                })?;
            query_kmknn_self(&index, k + 1, false, false)
        }
        KnnBackend::NnDescent => {
            let index = build_nndescent_index(
                matrix,
                ANN_METRIC,
                NNDESCENT_DELTA,
                NNDESCENT_DIVERSIFY,
                Some(k + 1),
                None,
                None,
                None,
                params.seed,
                false,
            )
            .map_err(|e| BonsaiErrors::NeighbourGraph {
                reason: e.to_string(),
            })?;
            // Reshape the graph NN-descent already built rather than running a
            // beam search per point, which the backend documents as orders of
            // magnitude more cost for higher recall. A linkage start does not
            // need that recall: it only has to beat a random topology, and
            // steps 3 to 7 fix what it gets wrong.
            extract_nndescent_knn(&index, Some(k + 1), false, false)
        }
    };
    let (neighbours, _) = rows.map_err(|e| BonsaiErrors::NeighbourGraph {
        reason: e.to_string(),
    })?;

    // The backends index into `live`, and every one of them counts a point as
    // its own nearest neighbour, so the self edge is dropped here rather than
    // guarded against in the scan.
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); centroids.len() / p];
    let mut seen: FxHashSet<(u32, u32)> = FxHashSet::default();
    for (i, row) in neighbours.iter().enumerate() {
        let a = live[i];
        for &j in row.iter() {
            if j >= m {
                continue;
            }
            let b = live[j];
            if a == b {
                continue;
            }
            let key = (a.min(b) as u32, a.max(b) as u32);
            if !seen.insert(key) {
                continue;
            }
            adjacency[a].push(b as u32);
            adjacency[b].push(a as u32);
        }
    }
    Ok(adjacency)
}

/// Build a starting tree by Ward linkage over a neighbour graph.
///
/// Rounds of mutual nearest-neighbour merges, with the candidate neighbours of
/// each cluster restricted to a graph that is inherited across merges and
/// rebuilt as it goes stale. `O(n k p)` per round plus the graph builds,
/// against `O(n^2 p)` for the dense form.
///
/// Leaves come back as `0..n_cells` in the caller's order and every branch is
/// one, since only the topology is meant: search step 4 replaces the lengths.
/// The root is left with three children rather than two, because a binary
/// dendrogram's root is degree two in the unrooted sense and carries no
/// information, and `search::spr` refuses to prune a child of such a root.
///
/// ### Params
///
/// * `means` - Transformed means, row-major `[cell][feature]`
/// * `n_cells` - Number of cells
/// * `n_features` - Features per cell
/// * `params` - Knobs, or `None` for [`LinkageParams::default`]
///
/// ### Returns
///
/// The starting tree, or `ShapeMismatch` if `means` is not `n_cells` rows of
/// `n_features`, or `NeighbourGraph` if a backend fails.
pub fn linkage_tree<T: BonsaiFloat>(
    means: &[T],
    n_cells: usize,
    n_features: usize,
    params: Option<LinkageParams>,
) -> Result<Tree, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let p = n_features;
    if means.len() != n_cells * p {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: n_cells,
            mean_features: p,
            sd_cells: if p == 0 { 0 } else { means.len() / p.max(1) },
            sd_features: p,
        });
    }
    if n_cells < ROOT_MEMBERS {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("{n_cells} cells is fewer than the {ROOT_MEMBERS} a root needs"),
        });
    }

    // One slot per original cell. A merge writes the union into the lower of
    // the two slots and retires the higher, so the slot count never grows.
    let mut centroids: Vec<f32> = means.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
    let mut size = vec![1.0f64; n_cells];
    let mut node_of: Vec<u32> = (0..n_cells as u32).collect();
    let mut live: Vec<usize> = (0..n_cells).collect();

    let mut parent = vec![NO_NODE; 2 * n_cells - ROOT_MEMBERS + 1];
    let mut next_internal = n_cells;

    let mut adjacency = build_graph(&centroids, &live, p, &params)?;
    let mut live_at_rebuild = live.len();
    let mut fresh = true;

    while live.len() > ROOT_MEMBERS {
        let stale = (live.len() as f64) <= params.rebuild_fraction * live_at_rebuild as f64;
        if stale {
            adjacency = build_graph(&centroids, &live, p, &params)?;
            live_at_rebuild = live.len();
            fresh = true;
        }

        let mut pairs = mutual_pairs(&adjacency, &centroids, &size, &live, p);
        if pairs.is_empty() {
            if !fresh {
                // Inheritance has run every list dry before the halving is due.
                // Redraw rather than join arbitrarily.
                adjacency = build_graph(&centroids, &live, p, &params)?;
                live_at_rebuild = live.len();
                fresh = true;
                continue;
            }
            // A fresh symmetric graph always has a mutual pair unless every
            // remaining edge ties; join the best edge and carry on.
            pairs.push(
                best_edge(&adjacency, &centroids, &size, &live, p).ok_or_else(|| {
                    BonsaiErrors::NeighbourGraph {
                        reason: format!("no edges left among {} live clusters", live.len()),
                    }
                })?,
            );
        }
        fresh = false;

        for (merged, (a, b)) in pairs.into_iter().enumerate() {
            if live.len() - merged == ROOT_MEMBERS {
                break;
            }
            let ancestor = next_internal as u32;
            next_internal += 1;
            parent[node_of[a] as usize] = ancestor;
            parent[node_of[b] as usize] = ancestor;

            merge_slots(&mut centroids, &mut size, p, a, b);
            merge_adjacency(&mut adjacency, a, b);
            node_of[a] = ancestor;
        }
        // Retired slots leave `live` once per round, so a round is `O(n)` in
        // bookkeeping rather than `O(n)` per merge.
        live.retain(|&slot| size[slot] > 0.0);
    }

    let root = next_internal as u32;
    for &slot in &live {
        parent[node_of[slot] as usize] = root;
    }
    parent[root as usize] = NO_NODE;

    let branch = vec![1.0f64; parent.len()];
    Tree::from_parents(parent, branch, n_cells)
}

/// Members left on the root when the linkage stops.
///
/// Three, for the reason [`linkage_tree`] gives.
const ROOT_MEMBERS: usize = 3;

/// Nearest live neighbour of one slot under Ward's dissimilarity.
///
/// ### Params
///
/// * `slot` - The slot whose list is scanned
/// * `adjacency` - Neighbour lists by slot
/// * `centroids` - Centroid rows
/// * `size` - Cluster sizes by slot; zero marks a retired slot
/// * `p` - Features per row
///
/// ### Returns
///
/// The nearest slot and its dissimilarity, or `None` if the list holds no live
/// cluster.
fn nearest(
    slot: usize,
    adjacency: &[Vec<u32>],
    centroids: &[f32],
    size: &[f64],
    p: usize,
) -> Option<(usize, f64)> {
    let mut best = f64::INFINITY;
    let mut who = usize::MAX;
    for &candidate in &adjacency[slot] {
        let other = candidate as usize;
        if other == slot || size[other] == 0.0 {
            continue;
        }
        let d = ward(centroids, size, p, slot, other);
        // Ties go to the lower slot, so the answer does not depend on the
        // order the graph happened to list neighbours in.
        if d < best || (d == best && other < who) {
            best = d;
            who = other;
        }
    }
    (who != usize::MAX).then_some((who, best))
}

/// Every mutually nearest pair among the live clusters, `(lower, higher)` in
/// ascending order of the lower slot.
///
/// Ward is reducible, so a mutually nearest pair is a pair the sequential
/// linkage merges at some point and merging all of them at once builds the same
/// dendrogram (Murtagh 1983). Taking them all per round is what keeps the live
/// clusters level-synchronous: nothing grows far ahead of its neighbours, so no
/// centroid becomes the low-noise attractor that crowds the true relatives out
/// of every list. The depth-first chain this replaced did exactly that, and
/// ended several times deeper than a dense linkage.
///
/// Per-slot searches are independent and run in parallel; the collection order
/// is the live order, so the result is the same at any thread count.
///
/// ### Params
///
/// * `adjacency` - Neighbour lists by slot
/// * `centroids` - Centroid rows
/// * `size` - Cluster sizes by slot
/// * `live` - Slots currently holding a cluster, ascending
/// * `p` - Features per row
///
/// ### Returns
///
/// The mutual pairs.
fn mutual_pairs(
    adjacency: &[Vec<u32>],
    centroids: &[f32],
    size: &[f64],
    live: &[usize],
    p: usize,
) -> Vec<(usize, usize)> {
    let mut nn = vec![usize::MAX; adjacency.len()];
    let found: Vec<usize> = live
        .par_iter()
        .map(|&slot| {
            nearest(slot, adjacency, centroids, size, p)
                .map(|(who, _)| who)
                .unwrap_or(usize::MAX)
        })
        .collect();
    for (&slot, &who) in live.iter().zip(&found) {
        nn[slot] = who;
    }
    live.iter()
        .filter_map(|&a| {
            let b = nn[a];
            (b != usize::MAX && a < b && nn[b] == a).then_some((a, b))
        })
        .collect()
}

/// The cheapest edge in the graph, for a round with no mutual pair.
///
/// ### Params
///
/// * `adjacency` - Neighbour lists by slot
/// * `centroids` - Centroid rows
/// * `size` - Cluster sizes by slot
/// * `live` - Slots currently holding a cluster, ascending
/// * `p` - Features per row
///
/// ### Returns
///
/// The pair `(lower, higher)`, or `None` if no live slot has a live neighbour.
fn best_edge(
    adjacency: &[Vec<u32>],
    centroids: &[f32],
    size: &[f64],
    live: &[usize],
    p: usize,
) -> Option<(usize, usize)> {
    let mut best = f64::INFINITY;
    let mut pair = None;
    for &slot in live {
        if let Some((who, d)) = nearest(slot, adjacency, centroids, size, p)
            && d < best
        {
            best = d;
            pair = Some((slot.min(who), slot.max(who)));
        }
    }
    pair
}

/// Fold slot `b`'s cluster into slot `a` and retire `b`.
///
/// ### Params
///
/// * `centroids` - Centroid rows, edited in place
/// * `size` - Cluster sizes, edited in place
/// * `p` - Features per row
/// * `a`, `b` - The two slots, `a < b`
fn merge_slots(centroids: &mut [f32], size: &mut [f64], p: usize, a: usize, b: usize) {
    let (sa, sb) = (size[a], size[b]);
    let total = sa + sb;
    for g in 0..p {
        let ca = centroids[a * p + g] as f64;
        let cb = centroids[b * p + g] as f64;
        // A convex combination rather than a ratio of weighted sums, so the
        // result is pinned between the two inputs and cannot cancel. The same
        // choice the pruning recursion makes; see `CLAUDE.md`.
        centroids[a * p + g] = (ca + (cb - ca) * sb / total) as f32;
    }
    size[a] = total;
    size[b] = 0.0;
}

/// Give slot `a` the union of both slots' neighbours and retire `b`.
///
/// The union is maintained symmetrically: every list that held either child is
/// rewritten to hold `a` instead. Doing only half of that leaves the merged
/// cluster invisible to its own neighbours and its list runs dry, which is the
/// trap `search::candidates` records.
///
/// ### Params
///
/// * `adjacency` - Neighbour lists by slot, edited in place
/// * `a`, `b` - The surviving and the retired slot
fn merge_adjacency(adjacency: &mut [Vec<u32>], a: usize, b: usize) {
    let theirs = std::mem::take(&mut adjacency[b]);
    for &other in &theirs {
        let list = &mut adjacency[other as usize];
        for entry in list.iter_mut() {
            if *entry == b as u32 {
                *entry = a as u32;
            }
        }
        list.sort_unstable();
        list.dedup();
        list.retain(|&x| x != other);
    }

    let mut merged = std::mem::take(&mut adjacency[a]);
    merged.extend(theirs);
    merged.sort_unstable();
    merged.dedup();
    merged.retain(|&x| x as usize != a && x as usize != b);
    adjacency[a] = merged;
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary};

    /// Ward linkage over a full distance matrix, the answer the graph version
    /// has to reproduce when the graph is complete.
    ///
    /// Deliberately the naive scan over an `O(n^2)` matrix with Lance-Williams
    /// updates, which reaches the same dendrogram by a different route than
    /// [`linkage_tree`]'s centroid arithmetic. Two independent formulations
    /// agreeing is worth more than one agreeing with itself.
    ///
    /// ### Params
    ///
    /// * `means` - Transformed means, row-major
    /// * `n` - Number of cells
    /// * `p` - Features per cell
    ///
    /// ### Returns
    ///
    /// The dense Ward tree.
    fn dense_ward(means: &[f64], n: usize, p: usize) -> Tree {
        let coords: Vec<f32> = means.iter().map(|&x| x as f32).collect();
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = if i == j {
                    f64::INFINITY
                } else {
                    // Ward at unit sizes is half the squared distance, which is
                    // what makes the Lance-Williams recurrence below the
                    // ordinary one.
                    0.5 * squared_distance(&coords[i * p..(i + 1) * p], &coords[j * p..(j + 1) * p])
                };
            }
        }

        let mut active = vec![true; n];
        let mut size = vec![1.0f64; n];
        let mut node_of: Vec<u32> = (0..n as u32).collect();
        let mut parent = vec![NO_NODE; 2 * n - ROOT_MEMBERS + 1];
        let mut next_internal = n;
        let mut remaining = n;

        while remaining > ROOT_MEMBERS {
            let (mut best, mut pair) = (f64::INFINITY, (usize::MAX, usize::MAX));
            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if active[j] && d[i * n + j] < best {
                        best = d[i * n + j];
                        pair = (i, j);
                    }
                }
            }
            let (a, b) = pair;
            let ancestor = next_internal as u32;
            next_internal += 1;
            parent[node_of[a] as usize] = ancestor;
            parent[node_of[b] as usize] = ancestor;

            let (na, nb, dab) = (size[a], size[b], d[a * n + b]);
            for k in 0..n {
                if !active[k] || k == a || k == b {
                    continue;
                }
                let nk = size[k];
                let merged = ((na + nk) * d[a * n + k] + (nb + nk) * d[b * n + k] - nk * dab)
                    / (na + nb + nk);
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
        Tree::from_parents(parent, branch, n).expect("dense ward")
    }

    /// Transformed means of a simulated binary tree.
    ///
    /// ### Params
    ///
    /// * `n` - Number of cells
    /// * `p` - Number of features
    /// * `seed` - Simulation seed
    ///
    /// ### Returns
    ///
    /// The means, row-major.
    fn fixture(n: usize, p: usize, seed: u64) -> Vec<f64> {
        simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: 0.3,
            seed,
            ..Default::default()
        }))
        .expect("simulation")
        .means
    }

    #[test]
    fn test_the_linkage_returns_a_well_formed_tree() {
        let (n, p) = (64usize, 32usize);
        let means = fixture(n, p, 7);
        let tree = linkage_tree(&means, n, p, None).expect("linkage");
        assert_eq!(tree.n_leaves(), n);
        let root = tree.root();
        assert_eq!(tree.children(root).len(), ROOT_MEMBERS);
        for node in tree.internal_postorder() {
            if node != root {
                assert_eq!(tree.children(node).len(), 2, "node {node} is not binary");
            }
        }
    }

    #[test]
    fn test_a_complete_graph_reproduces_the_dense_linkage() {
        // The whole approximation is the sparsity of the graph. At `k = n - 1`
        // there is none, so the rounds have to find exactly the pairs the dense
        // scan finds, every round. This is what says the inheritance and the
        // symmetry bookkeeping are right, and it is the gate
        // `search::candidates` keeps for the same reason.
        let (n, p) = (64usize, 24usize);
        for seed in [1u64, 2, 3] {
            let means = fixture(n, p, seed);
            let params = LinkageParams {
                k: n - 1,
                backend: Some(KnnBackend::Exhaustive),
                ..LinkageParams::default()
            };
            let got = linkage_tree(&means, n, p, Some(params)).expect("linkage");
            let want = dense_ward(&means, n, p);
            assert_eq!(
                robinson_foulds(&got, &want).expect("rf"),
                0,
                "seed {seed}: a complete graph did not reproduce the dense linkage"
            );
        }
    }

    #[test]
    fn test_the_sparse_graph_stays_close_to_the_dense_linkage() {
        // At the shipped `k` the answer may differ and how much is the whole
        // question, so this pins a measurement rather than asserting an
        // equality it does not have. A large drift means the inheritance has
        // gone wrong, not that the approximation has bitten.
        let (n, p) = (64usize, 24usize);
        let mut total = 0usize;
        for seed in [1u64, 2, 3] {
            let means = fixture(n, p, seed);
            let got = linkage_tree(&means, n, p, None).expect("linkage");
            total += robinson_foulds(&got, &dense_ward(&means, n, p)).expect("rf");
        }
        assert!(
            total <= 3 * (n / 4),
            "sparse linkage drifted {total} splits from dense over three seeds"
        );
    }

    #[test]
    fn test_the_exact_backends_agree() {
        // Exhaustive and kmknn are both exact, so at the same `k` they hand the
        // rounds the identical graph and therefore the identical tree.
        // NN-descent is approximate and is not held to this.
        let (n, p) = (64usize, 32usize);
        let means = fixture(n, p, 11);
        let mut trees = Vec::new();
        for backend in [KnnBackend::Exhaustive, KnnBackend::Kmknn] {
            let params = LinkageParams {
                backend: Some(backend),
                ..LinkageParams::default()
            };
            trees.push(linkage_tree(&means, n, p, Some(params)).expect("linkage"));
        }
        assert_eq!(robinson_foulds(&trees[0], &trees[1]).expect("rf"), 0);
    }

    #[test]
    fn test_the_linkage_is_deterministic_whatever_the_thread_count() {
        let (n, p) = (64usize, 32usize);
        let means = fixture(n, p, 5);
        let reference = linkage_tree(&means, n, p, None).expect("linkage");
        for threads in [1usize, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| linkage_tree(&means, n, p, None).expect("linkage"));
            assert_eq!(
                robinson_foulds(&got, &reference).expect("rf"),
                0,
                "topology moved at {threads} threads"
            );
        }
    }

    #[test]
    fn test_too_few_cells_is_an_error() {
        assert!(linkage_tree(&[1.0f64, 2.0], 2, 1, None).is_err());
    }
}
