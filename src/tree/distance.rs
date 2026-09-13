//! Distances along a tree, and how well they recover the distances in the data.
//!
//! This is the property the method exists for. A tree can have the right
//! topology and useless branch lengths, and Robinson-Foulds will report zero
//! either way; what the paper claims over UMAP and tSNE is that path distances
//! along the tree track high-dimensional distances *at every scale*, which is
//! the relation its Fig. S8 plots and Fig. S9 scores.
//!
//! So this is the measurement to compare two implementations on, and the one
//! that would catch a search that found a plausible topology by accident.
//!
//! Nothing in the search calls any of this. It is scored once, from
//! `benches/recovery.rs`, so it is not a target for vectorisation however much
//! [`data_distances`] and [`pearson`] look like one: a kernel the pipeline never
//! runs cannot be worth a lane. [`data_distances`] is parallel because it is
//! `O(pairs * p)` and the ceiling is two million pairs; [`pearson`] is `O(n)`
//! against that and is left alone.

use rayon::prelude::*;

use crate::tree::Tree;
use crate::utils::rng::SplitMix64;
use crate::utils::traits::{BonsaiFloat, wide};

////////////
// Consts //
////////////

/// Pair count above which the metrics subsample rather than enumerate.
///
/// All pairs is `n * (n - 1) / 2`, which is 2 million at 2048 leaves and 450
/// million at 30 thousand, so the full set stops being storable well before the
/// search stops being runnable. Two million pairs estimates a correlation far
/// past any precision the comparison needs.
pub const MAX_PAIRS: usize = 2_000_000;

///////////////
// Functions //
///////////////

/// Path distances from one node to every other, along the tree.
///
/// Breadth-first over the *undirected* tree, so it walks through the parent as
/// well as the children: the arena stores a rooted representation of an unrooted
/// tree and the root is a bookkeeping choice (S14). Iterative, so a
/// hundred-thousand-leaf ladder does not touch the stack.
///
/// ### Params
///
/// * `tree` - The tree
/// * `source` - Node to measure from
///
/// ### Returns
///
/// Distance to every node, indexed by node id.
pub fn distances_from(tree: &Tree, source: u32) -> Vec<f64> {
    let n = tree.n_nodes();
    let mut dist = vec![f64::INFINITY; n];
    let mut queue: Vec<u32> = Vec::with_capacity(n);
    dist[source as usize] = 0.0;
    queue.push(source);

    let mut head = 0usize;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        let here = dist[node as usize];

        for &child in tree.children(node) {
            if dist[child as usize].is_infinite() {
                dist[child as usize] = here + tree.branch(child);
                queue.push(child);
            }
        }
        if let Some(up) = tree.parent(node)
            && dist[up as usize].is_infinite()
        {
            dist[up as usize] = here + tree.branch(node);
            queue.push(up);
        }
    }
    dist
}

/// Leaf pairs to score, all of them or a deterministic sample.
///
/// ### Params
///
/// * `n_leaves` - Number of leaves
/// * `max_pairs` - Ceiling on how many pairs to return
/// * `seed` - Seed for the sample, unused when every pair fits
///
/// ### Returns
///
/// Leaf index pairs, `i < j`.
pub fn leaf_pairs(n_leaves: usize, max_pairs: usize, seed: u64) -> Vec<(usize, usize)> {
    let total = n_leaves.saturating_mul(n_leaves.saturating_sub(1)) / 2;
    if total <= max_pairs {
        let mut out = Vec::with_capacity(total);
        for i in 0..n_leaves {
            for j in i + 1..n_leaves {
                out.push((i, j));
            }
        }
        return out;
    }

    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(max_pairs);
    while out.len() < max_pairs {
        let i = rng.below(n_leaves);
        let j = rng.below(n_leaves);
        if i != j {
            out.push((i.min(j), i.max(j)));
        }
    }
    out
}

/// Path distance along the tree for each given leaf pair.
///
/// ### Params
///
/// * `tree` - The tree
/// * `pairs` - Leaf pairs to measure
///
/// ### Returns
///
/// One distance per pair, in the order given.
pub fn tree_distances(tree: &Tree, pairs: &[(usize, usize)]) -> Vec<f64> {
    // One traversal per distinct source, reused across every pair sharing it,
    // which turns the all-pairs case from `O(n^2)` traversals into `O(n)`.
    let mut by_source: Vec<Vec<usize>> = vec![Vec::new(); tree.n_leaves()];
    for (idx, &(i, _)) in pairs.iter().enumerate() {
        by_source[i].push(idx);
    }

    let mut out = vec![0.0f64; pairs.len()];
    for (source, slots) in by_source.iter().enumerate() {
        if slots.is_empty() {
            continue;
        }
        let dist = distances_from(tree, source as u32);
        for &idx in slots {
            out[idx] = dist[pairs[idx].1];
        }
    }
    out
}

/// Squared Euclidean distance in the data for each given leaf pair.
///
/// ### Params
///
/// * `means` - Cell positions, row-major `[cell][feature]`
/// * `n_features` - Features per row
/// * `pairs` - Cell pairs to measure
///
/// ### Returns
///
/// One squared distance per pair, in the order given.
pub fn data_distances<T: BonsaiFloat>(
    means: &[T],
    n_features: usize,
    pairs: &[(usize, usize)],
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|&(i, j)| {
            let (a, b) = (i * n_features, j * n_features);
            let mut acc = 0.0f64;
            for g in 0..n_features {
                let d = wide(means[a + g]) - wide(means[b + g]);
                acc += d * d;
            }
            acc
        })
        .collect()
}

/// Pearson correlation of two equal-length samples.
///
/// Two passes, means first, rather than the `E[xy] - E[x]E[y]` shortcut, which
/// cancels catastrophically when the values are large and their spread is not.
/// Tree distances at atlas scale are exactly that shape.
///
/// ### Params
///
/// * `a` - First sample
/// * `b` - Second sample, same length
///
/// ### Returns
///
/// The correlation, or `NaN` if either sample is constant or they differ in
/// length.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let (mut cov, mut var_a, mut var_b) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - mean_a, y - mean_b);
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return f64::NAN;
    }
    cov / (var_a * var_b).sqrt()
}

/// How well a tree's path distances recover the distances in the data.
///
/// The headline number for comparing two reconstructions of the same dataset,
/// and the one Robinson-Foulds cannot see: a tree with the right topology and
/// wrong branch lengths scores zero on RF and poorly here.
///
/// ### Params
///
/// * `tree` - The reconstruction
/// * `means` - Cell positions the tree was built from, row-major
/// * `n_features` - Features per row
/// * `max_pairs` - Ceiling on pairs scored; [`MAX_PAIRS`] is the usual choice
/// * `seed` - Seed for the sample, when there are more pairs than the ceiling
///
/// ### Returns
///
/// Pearson correlation between tree path distance and squared Euclidean
/// distance, over the scored pairs.
pub fn distance_recovery<T: BonsaiFloat>(
    tree: &Tree,
    means: &[T],
    n_features: usize,
    max_pairs: usize,
    seed: u64,
) -> f64 {
    let pairs = leaf_pairs(tree.n_leaves(), max_pairs, seed);
    let on_tree = tree_distances(tree, &pairs);
    let in_data = data_distances(means, n_features, &pairs);
    pearson(&on_tree, &in_data)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::NO_NODE;
    use crate::tree::simulate::{SimulationParams, simulate_binary};
    use approx::assert_relative_eq;

    #[test]
    fn test_distances_match_a_hand_computed_path() {
        // ((0,1)4, (2,3)5)6, every branch 1, so leaves within a cherry are 2
        // apart and leaves across are 4.
        let tree = Tree::from_parents(
            vec![4, 4, 5, 5, 6, 6, NO_NODE],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            4,
        )
        .expect("fixture");
        let d = distances_from(&tree, 0);
        assert_relative_eq!(d[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(d[2], 4.0, epsilon = 1e-12);
        assert_relative_eq!(d[3], 4.0, epsilon = 1e-12);
        assert_relative_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_distance_is_symmetric_and_reaches_every_node() {
        let tree = Tree::balanced_binary(16, 0.7).expect("balanced");
        for source in [0u32, 5, 11] {
            let d = distances_from(&tree, source);
            assert!(d.iter().all(|x| x.is_finite()), "a node was unreachable");
            for target in 0..tree.n_nodes() as u32 {
                let back = distances_from(&tree, target);
                assert_relative_eq!(
                    d[target as usize],
                    back[source as usize],
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_tree_distances_agree_with_per_source_traversals() {
        // The batched routine reuses one traversal per source; check it against
        // the obvious per-pair version.
        let tree = Tree::ladder(12, 0.9).expect("ladder");
        let pairs = leaf_pairs(12, MAX_PAIRS, 0);
        let batched = tree_distances(&tree, &pairs);
        for (k, &(i, j)) in pairs.iter().enumerate() {
            let want = distances_from(&tree, i as u32)[j];
            assert_relative_eq!(batched[k], want, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_pearson_against_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(pearson(&a, &[2.0, 4.0, 6.0, 8.0]), 1.0, epsilon = 1e-12);
        assert_relative_eq!(pearson(&a, &[4.0, 3.0, 2.0, 1.0]), -1.0, epsilon = 1e-12);
        assert!(pearson(&a, &[1.0; 4]).is_nan(), "constant input");
        assert!(pearson(&a, &[1.0, 2.0]).is_nan(), "length mismatch");
    }

    #[test]
    fn test_the_generating_tree_recovers_its_own_distances() {
        // The load-bearing test, and the paper's Fig. S8 relation. Under the
        // model a branch of length `t` diffuses with variance `t` per feature,
        // so summed branch length should track squared Euclidean displacement.
        // On noise-free simulated data the generating tree must show that
        // strongly, or the simulator and the model disagree about what a branch
        // length means.
        let (n, p) = (64usize, 2000usize);
        let d = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: 1e-6,
            seed: 5,
            ..Default::default()
        }))
        .expect("simulation");

        let r = distance_recovery(&d.tree, &d.means, p, MAX_PAIRS, 0);
        assert!(
            r > 0.9,
            "the generating tree recovered its own distances at only r = {r:.4}"
        );
    }

    #[test]
    fn test_a_shuffled_tree_recovers_distances_far_worse() {
        // The metric has to discriminate, or it says nothing about a
        // reconstruction.
        let (n, p) = (64usize, 500usize);
        let d = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: 1e-6,
            seed: 5,
            ..Default::default()
        }))
        .expect("simulation");

        let truth = distance_recovery(&d.tree, &d.means, p, MAX_PAIRS, 0);
        // A ladder over the same leaves keeps every branch length in the tree
        // but throws the topology away.
        let ladder = Tree::ladder(n, 1.0).expect("ladder");
        let shuffled = distance_recovery(&ladder, &d.means, p, MAX_PAIRS, 0);

        assert!(
            truth > shuffled + 0.2,
            "truth {truth:.4} barely beat a ladder at {shuffled:.4}"
        );
    }

    #[test]
    fn test_sampling_estimates_the_full_correlation() {
        let (n, p) = (64usize, 200usize);
        let d = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: 0.1,
            seed: 9,
            ..Default::default()
        }))
        .expect("simulation");

        let full = distance_recovery(&d.tree, &d.means, p, MAX_PAIRS, 0);
        let sampled = distance_recovery(&d.tree, &d.means, p, 500, 0);
        assert_relative_eq!(sampled, full, epsilon = 0.05);
    }

    #[test]
    fn test_sampling_is_deterministic_and_respects_its_ceiling() {
        let first = leaf_pairs(1000, 250, 42);
        let second = leaf_pairs(1000, 250, 42);
        assert_eq!(first, second);
        assert_eq!(first.len(), 250);
        assert!(first.iter().all(|&(i, j)| i < j && j < 1000));
        assert_ne!(first, leaf_pairs(1000, 250, 43));
    }
}
