//! A starting tree from a neighbour-graph linkage.
//!
//! Replaces the greedy merge start of SPEC.md section 9.1 and is the default
//! ([`crate::bonsai::StartTree`]). On real Sanity-preprocessed input it wins on
//! loglikelihood and Robinson-Foulds and is several times faster;
//! `docs/PERFORMANCE.md` has the table. The specified merge score chains on
//! real data because of the size bias [`crate::bonsai::StartTree::GreedyMerge`]
//! sets out. Nothing here is scored with the model.
//!
//! ### Why Ward and not the merge score
//!
//! Ward is reducible, so merging every mutually nearest pair at once builds the
//! same dendrogram as the naive scan (Bruynooghe 1977, Murtagh 1983). The
//! Bonsai merge gain is not: about one round in five lifts another pair above
//! the merged pair's score, by up to 13 nats. Ward is defined through
//! centroids, so a merged cluster is one `O(p)` update and needs no distance
//! matrix.
//!
//! ### The graph
//!
//! Each round needs the nearest live cluster to every live cluster. A
//! neighbour graph over the cells answers that only approximately for clusters,
//! so each merge inherits the union of the children's neighbour lists
//! (rewritten symmetrically, else the merged cluster is invisible to its
//! neighbours and lists run dry), and the graph is rebuilt over the live
//! centroids when the live count has halved (as SPEC.md section 11, see
//! [`crate::search::candidates`]).
//!
//! ### Rounds, not a chain
//!
//! A nearest-neighbour chain builds caterpillars, an order of magnitude deeper
//! than `log2(n)`: one big centroid carries `1/size` of the noise, sits close to
//! every leaf and absorbs blocks. Merging every mutual pair per round keeps
//! cluster sizes similar. `benches/start_tree.rs` is the sweep.

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
/// Matches [`crate::search::candidates::KnnCandidatesParams`]. Every `k` from
/// 8 to 128 reproduces the dense Ward tree, so cost sets the value
/// (`drift` block of `benches/start_tree.rs`).
const DEFAULT_K: usize = 16;

/// Rebuild the graph once the live cluster count has fallen to this fraction of
/// what it was at the last rebuild.
///
/// Halving gives `log2(n)` rebuilds costing a geometric series. Every cadence
/// swept reproduces the dense tree, so this only bounds staleness.
const DEFAULT_REBUILD_FRACTION: f64 = 0.5;

/// Cells up to which the exhaustive backend is used; NN-descent above it.
///
/// Conservative: NN-descent gives the identical tree at every size measured
/// (`backend` block of `benches/start_tree.rs`). kmknn is not chosen
/// automatically.
const EXHAUSTIVE_MAX_CELLS: usize = 4_096;

/// Metric the graph is built in; the graph only decides which pairs are
/// considered.
const ANN_METRIC: &str = "euclidean";

/// NN-descent convergence threshold, as a fraction of the graph's edges updated
/// in an iteration. The backend's own default.
const NNDESCENT_DELTA: f32 = 0.001;

/// NN-descent diversification probability. One means no pruning.
const NNDESCENT_DIVERSIFY: f32 = 1.0;

/// Members left on the root when the linkage stops.
///
/// Three, so the root is not degree two in the unrooted sense (`search::spr`
/// refuses to prune a child of such a root).
const ROOT_MEMBERS: usize = 3;

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

/////////////////////
// Private helpers //
/////////////////////

/// Pick a backend from the problem size.
///
/// Exhaustive up to [`EXHAUSTIVE_MAX_CELLS`], NN-descent above it.
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

/// Squared Euclidean distance between two centroid rows.
///
/// Accumulated in `f64` over `f32` storage.
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
/// `(|A| |B| / (|A| + |B|)) * ||cA - cB||^2`, through centroids so no distance
/// matrix is needed.
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
/// Returns a symmetrised adjacency keyed by slot. Built in `f32` whatever the
/// storage type, as `search::candidates` does.
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
            // Reshape the existing graph; a beam search per point costs orders
            // of magnitude more and the start does not need the recall.
            extract_nndescent_knn(&index, Some(k + 1), false, false)
        }
    };
    let (neighbours, _) = rows.map_err(|e| BonsaiErrors::NeighbourGraph {
        reason: e.to_string(),
    })?;

    // Backends index into `live` and count a point as its own neighbour.
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
        // Ties go to the lower slot, independent of list order.
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
/// Merging all of them at once builds the dendrogram of the sequential linkage
/// (Murtagh 1983). Per-slot searches run in parallel and are collected in live
/// order, so the result is thread-count independent.
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
        // Convex combination, as in the pruning recursion (see `CLAUDE.md`).
        centroids[a * p + g] = (ca + (cb - ca) * sb / total) as f32;
    }
    size[a] = total;
    size[b] = 0.0;
}

/// Give slot `a` the union of both slots' neighbours and retire `b`.
///
/// Symmetric: every list that held either child is rewritten to hold `a`.
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

///////////////////
// Main function //
///////////////////

/// Build a starting tree by Ward linkage over a neighbour graph.
///
/// Rounds of mutual nearest-neighbour merges over an inherited, periodically
/// rebuilt graph: `O(n k p)` per round plus the graph builds.
///
/// Leaves are `0..n_cells` in the caller's order and every branch is one (search
/// step 4 replaces the lengths). The root keeps three children, see
/// [`ROOT_MEMBERS`].
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

    // One slot per cell; a merge writes into the lower slot and retires the higher.
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
                // Inheritance ran every list dry before the halving; redraw.
                adjacency = build_graph(&centroids, &live, p, &params)?;
                live_at_rebuild = live.len();
                fresh = true;
                continue;
            }
            // A fresh graph has a mutual pair unless every edge ties.
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
    /// Naive scan over an `O(n^2)` matrix with Lance-Williams updates, an
    /// independent route to the same dendrogram.
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
                    // Ward at unit sizes is half the squared distance.
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
        // At `k = n - 1` the graph is complete, so the rounds must match the
        // dense scan; this checks the inheritance and symmetry bookkeeping.
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
        // Pins a bound on drift at the default `k`; large drift means broken
        // inheritance.
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
        // Both exact, so identical graph and tree; NN-descent is not held to this.
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
