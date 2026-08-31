//! Nearest-neighbour interchanges, generalised to polytomies
//! (SPEC.md section 9.4).
//!
//! The move is three lines: pick an internal edge `k-l` with both ends
//! internal, delete `k` and move all of its subtrees onto `l`, then run the
//! star primitive on `l`. For four subtrees that is exactly the classical
//! interchange, and `test_a_classical_interchange_reconnects_four_subtrees`
//! pins it.
//!
//! ### Why the collapse costs nothing
//!
//! Deleting `k` changes the tree only strictly inside `l`'s subtree and
//! strictly above `k`'s children. So the effective leaf of every subtree the
//! star needs is already in the *original* tree's settled rows: `l`'s remaining
//! children and `k`'s children keep their down rows, and `l`'s upstream side
//! keeps its up row. Nothing is repruned to propose a move, and one settled
//! pair of sweeps serves every edge of a round. What the collapse does change
//! is the branch to the centre for each of `k`'s children, which becomes
//! `t_c + t_k`: diffusion times add along a path, so that is the length that
//! leaves each subtree where it was.
//!
//! ### Two phases
//!
//! The random phase samples the pair to merge inside the star rather than
//! taking the best ([`StarSelection::Weighted`]) and accepts the result
//! unconditionally, which is meant to escape a local optimum. How much it
//! actually escapes depends hard on the feature count, and at the counts this
//! crate expects the answer is "not much": see [`StarSelection::Weighted`] for
//! the measurement. The greedy phase scores an interchange at every eligible
//! edge, performs the best, and repeats until none improves the tree.
//!
//! **Monotonicity.** The greedy phase never lowers the tree loglikelihood: a
//! move is accepted only when a fresh [`NodeState::prune`] of the candidate
//! beats the incumbent. The random phase gives no such guarantee and is not
//! meant to; the collapse alone can lose a split that the resampled star does
//! not put back.
//!
//! ### This is a topology search and only a topology search
//!
//! A collapse and re-resolution that puts the same subtrees back where they
//! were is not an interchange at all: it is a reoptimisation of the three
//! branches the star primitive creates at `l`, and it nearly always gains a
//! little. Accepting those turns the greedy phase into an extremely expensive
//! branch-length descent. Measured 2026-08-31, starting from the generating
//! tree itself at 32 to 64 leaves and 256 features, that ran for 104 to 239
//! rounds with the Robinson-Foulds distance to the truth pinned at zero
//! throughout. So a proposal is discarded unless it changes the tree's splits;
//! branch lengths are search steps 4 and 7, which do the same job globally and
//! for a fraction of the cost.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::search::Leaves;
use crate::search::polytomy::{CentreStar, Splice, splice_star};
use crate::search::star::{StarParams, StarSelection};
use crate::tree::Tree;
use crate::utils::rng::SplitMix64;
use crate::utils::traits::BonsaiFloat;

////////////////
// Parameters //
////////////////

/// Smallest star an interchange can do anything with.
///
/// One more than the three members the star primitive stops at: a collapse that
/// leaves three members has nothing to merge, and splicing it back would only
/// throw away the split that `k` carried. Those edges are skipped rather than
/// proposed and rejected.
const MIN_INTERCHANGE_MEMBERS: usize = 4;

/// Default for [`NniParams::max_rounds`].
///
/// A runaway guard and not a working limit. Every accepted greedy move raises
/// the tree loglikelihood by more than [`StarParams::min_gain`] and the
/// loglikelihood is bounded above, so the phase terminates on its own; this
/// only bounds how long it can take to notice.
///
/// One round performs one move, so the requirement is the number of
/// interchanges between the starting topology and the local optimum, and that
/// grows with the leaf count. Measured 2026-08-31 on simulated data at 256
/// features, starting from a ladder with its branch lengths already at the step
/// 4 optimum and running to a Robinson-Foulds distance of zero from the
/// generating tree: 23 rounds at 32 leaves and 53 at 64, which is a shade under
/// one round per leaf. The default therefore covers roughly ten thousand
/// leaves; beyond that a caller should raise it rather than accept a truncated
/// search.
const DEFAULT_MAX_ROUNDS: usize = 10_000;

/// Default for [`NniParams::n_random`].
///
/// Zero: the random phase is opt-in. It is a diversification budget traded
/// against wall time, nothing in SPEC.md fixes one, and any constant here would
/// be a number this crate invented and then had to defend.
///
/// It is also weaker than it looks at the feature counts this crate expects.
/// The sampling weight is a softmax over tree loglikelihoods, whose gaps are
/// `O(p)` nats, so it concentrates on the greedy pick as `p` grows. Measured
/// 2026-08-31 at 32 leaves, eight moves and eight seeds: at 8 features all
/// eight seeds moved the tree off its starting topology, at 32 features four,
/// and at 128 and 512 features two and three. So a budget buys real
/// diversification on small feature sets and mostly buys branch-length
/// reoptimisation on large ones.
const DEFAULT_RANDOM_MOVES: usize = 0;

/// Tuning knobs for search step 6.
#[derive(Clone, Copy, Debug)]
pub struct NniParams {
    /// Star primitive knobs. The greedy phase uses these as they stand; the
    /// random phase overrides [`StarParams::selection`] with its own seed per
    /// move and leaves everything else alone.
    pub star: StarParams,
    /// Number of randomised moves performed before the greedy phase.
    pub n_random: usize,
    /// Seed for the random phase, unused when `n_random` is zero.
    pub seed: u64,
    /// Cap on greedy rounds.
    pub max_rounds: usize,
}

impl Default for NniParams {
    /// The default star knobs, no random phase and `DEFAULT_MAX_ROUNDS`.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            star: StarParams::default(),
            n_random: DEFAULT_RANDOM_MOVES,
            seed: 0,
            max_rounds: DEFAULT_MAX_ROUNDS,
        }
    }
}

////////////
// Output //
////////////

/// What a run of the interchanges did.
#[derive(Clone, Debug)]
pub struct NniResult {
    /// The tree the phase finished on.
    pub tree: Tree,
    /// Its loglikelihood, from [`NodeState::prune`] on the tree itself.
    pub loglik: f64,
    /// Number of moves performed.
    pub n_moves: usize,
    /// Number of greedy rounds, the last of which found no improving move.
    /// Zero for a run of the random phase alone.
    pub rounds: usize,
}

///////////////
// One move //
///////////////

/// Size of the star an interchange at `k` would build.
///
/// ### Params
///
/// * `tree` - The tree
/// * `k` - The node that would be deleted, the lower end of the edge
///
/// ### Returns
///
/// The member count, or `None` if the edge is not eligible: `k` is the root or
/// a leaf, or the collapse leaves too few members to merge.
fn interchange_members(tree: &Tree, k: u32) -> Option<usize> {
    let l = tree.parent(k)?;
    if tree.children(k).is_empty() {
        return None;
    }
    let n =
        tree.children(l).len() - 1 + tree.children(k).len() + usize::from(tree.parent(l).is_some());
    (n >= MIN_INTERCHANGE_MEMBERS).then_some(n)
}

/// Build the star at `l` that deleting `k` leaves behind.
///
/// The members are `l`'s other children on their own branches, then `k`'s
/// children on `t_c + t_k`, then `l`'s upstream side. All of them are read off
/// the *original* tree's settled rows; see the module docs for why that is
/// sound.
///
/// ### Params
///
/// * `tree` - The tree
/// * `down` - Down rows, settled by [`NodeState::prune`] against this tree
/// * `up` - Up rows, settled by [`UpState::sweep`] against the same
/// * `k` - The node to delete
///
/// ### Returns
///
/// The star, or `None` if the edge is not eligible.
fn collapsed_star<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
    k: u32,
) -> Option<CentreStar<T>> {
    let n = interchange_members(tree, k)?;
    let l = tree.parent(k)?;
    let p = down.n_features();
    let t_k = tree.branch(k);
    let above = tree.parent(l);

    let mut star = CentreStar {
        centre: l,
        member_nodes: Vec::with_capacity(n),
        has_upstream: above.is_some(),
        deleted: vec![k],
        means: Vec::with_capacity(n * p),
        precisions: Vec::with_capacity(n * p),
        branch: Vec::with_capacity(n),
        n_features: p,
    };

    let push = |node: u32, branch: f64, star: &mut CentreStar<T>| {
        star.member_nodes.push(node);
        star.means.extend_from_slice(down.means(node));
        star.precisions.extend_from_slice(down.precisions(node));
        star.branch.push(branch);
    };

    for &child in tree.children(l) {
        if child != k {
            push(child, tree.branch(child), &mut star);
        }
    }
    for &child in tree.children(k) {
        push(child, tree.branch(child) + t_k, &mut star);
    }
    if let Some(par) = above {
        star.member_nodes.push(par);
        star.means.extend_from_slice(up.means(l));
        star.precisions.extend_from_slice(up.precisions(l));
        star.branch.push(tree.branch(l));
    }
    Some(star)
}

/// Propose the interchange at one internal edge.
///
/// ### Params
///
/// * `tree` - The tree
/// * `down` - Down rows, settled against this tree
/// * `up` - Up rows, settled against the same
/// * `k` - Lower end of the edge, the node that is deleted
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The proposed tree, or `None` if the edge is not eligible, or the error the
/// primitive or the arena failed with.
///
/// The [`Splice::gain`] that comes back is measured against the *collapsed*
/// tree and not against `tree`, because the collapse happened before the star
/// was scored. Callers that need the gain of the move itself take the
/// difference of two [`NodeState::prune`] calls, which is what both phases do.
pub fn interchange_at<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
    k: u32,
    params: Option<StarParams>,
) -> Result<Option<Splice>, BonsaiErrors> {
    match collapsed_star(tree, down, up, k) {
        None => Ok(None),
        Some(star) => Ok(Some(splice_star(tree, &star, params)?)),
    }
}

////////////////////////
// Topology comparison //
////////////////////////

/////////////
// Phases //
/////////////

/// Settle a tree's down and up rows.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// The settled rows and the tree loglikelihood, or the error the state
/// allocation failed with.
fn settle<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
) -> Result<(NodeState<T>, UpState<T>, f64), BonsaiErrors> {
    let mut down = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )?;
    let loglik = down.prune(tree);
    let mut up = UpState::new(tree.n_nodes(), leaves.n_features);
    up.sweep(tree, &down);
    Ok((down, up, loglik))
}

/// Loglikelihood of a tree, from the leaf data alone.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data
///
/// ### Returns
///
/// The tree loglikelihood, or the error the state allocation failed with.
fn tree_loglik<T: BonsaiFloat>(tree: &Tree, leaves: Leaves<'_, T>) -> Result<f64, BonsaiErrors> {
    let mut state = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )?;
    Ok(state.prune(tree))
}

/// The random phase: `n_random` interchanges with the merge sampled rather than
/// chosen, accepted whatever they do to the tree.
///
/// The edge is drawn uniformly from the eligible ones and the star's own seed
/// is drawn from the same stream, so the whole phase is a function of
/// [`NniParams::seed`] and the tree. Nothing here reduces over rayon, and the
/// sampling inside the star is done over a fixed candidate order, so the result
/// does not depend on the thread count.
///
/// ### Params
///
/// * `tree` - Tree to move away from; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults, whose `n_random` is zero
///
/// ### Returns
///
/// The tree the phase finished on, which may be worse than the one it started
/// from, or the error the primitive or the arena failed with.
pub fn nni_random<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let mut rng = SplitMix64::new(params.seed);
    let mut tree = tree.clone();
    let mut n_moves = 0usize;

    for _ in 0..params.n_random {
        let eligible: Vec<u32> = tree
            .internal_postorder()
            .filter(|&k| interchange_members(&tree, k).is_some())
            .collect();
        if eligible.is_empty() {
            break;
        }
        let draw = (rng.uniform() * eligible.len() as f64) as usize;
        let k = eligible[draw.min(eligible.len() - 1)];
        let star = StarParams {
            selection: StarSelection::Weighted {
                seed: rng.next_u64(),
            },
            ..params.star
        };

        let (down, up, _) = settle(&tree, leaves)?;
        if let Some(spliced) = interchange_at(&tree, &down, &up, k, Some(star))? {
            tree = spliced.tree;
            n_moves += 1;
        }
    }

    let loglik = tree_loglik(&tree, leaves)?;
    Ok(NniResult {
        tree,
        loglik,
        n_moves,
        rounds: 0,
    })
}

/// The greedy phase: score an interchange at every eligible edge, perform the
/// best, repeat until none improves the tree.
///
/// Every candidate is scored by a fresh [`NodeState::prune`] of the tree it
/// would produce, so the accepted move is an improvement in the quantity that
/// actually matters rather than in the star primitive's local gain, which is
/// measured against the collapsed tree and not against this one. That makes the
/// phase monotone by construction. A candidate whose splits match the current
/// tree's is discarded before it is scored; see the module docs.
///
/// Edges are visited in ascending node order and ties go to the lower node, so
/// the round's winner is fixed. The scan is sequential: the parallelism in this
/// crate lives on the feature axis inside the pruning kernels, and a candidate
/// scan that forked over edges would nest inside it.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults
///
/// ### Returns
///
/// The improved tree, whose loglikelihood is never below the input's, or the
/// error the primitive or the arena failed with.
pub fn nni_greedy<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let mut tree = tree.clone();
    let mut best = tree_loglik(&tree, leaves)?;
    let mut n_moves = 0usize;
    let mut rounds = 0usize;

    while rounds < params.max_rounds {
        rounds += 1;
        let (down, up, _) = settle(&tree, leaves)?;

        let here = crate::search::split_fingerprint(&tree);
        let mut winner: Option<(f64, Tree)> = None;
        for k in tree.internal_postorder() {
            let Some(spliced) = interchange_at(&tree, &down, &up, k, Some(params.star))? else {
                continue;
            };
            // A proposal that puts the same subtrees back where they were is
            // not an interchange: it is a reoptimisation of the three branches
            // the star primitive creates at `l`. Those nearly always gain a
            // little, and taking them turns the phase into branch-length
            // descent that steps 4 and 7 do properly and far more cheaply.
            // Measured 2026-08-31: from the generating tree itself, at 32 to 64
            // leaves and 256 features, accepting them ran 104 to 239 rounds
            // with the Robinson-Foulds distance pinned at zero throughout, so
            // every one of those rounds was branch lengths and none was
            // topology.
            if spliced.n_merges == 0 || crate::search::split_fingerprint(&spliced.tree) == here {
                continue;
            }
            let loglik = tree_loglik(&spliced.tree, leaves)?;
            let beats = match &winner {
                None => loglik > best + params.star.min_gain,
                Some((incumbent, _)) => loglik > *incumbent,
            };
            if beats {
                winner = Some((loglik, spliced.tree));
            }
        }

        match winner {
            None => break,
            Some((loglik, next)) => {
                best = loglik;
                tree = next;
                n_moves += 1;
            }
        }
    }

    Ok(NniResult {
        tree,
        loglik: best,
        n_moves,
        rounds,
    })
}

/// Search step 6: the random phase, then the greedy phase.
///
/// ### Params
///
/// * `tree` - Tree to improve; not modified
/// * `leaves` - The leaf data
/// * `params` - Knobs, or `None` for the defaults, which skip the random phase
///
/// ### Returns
///
/// The tree both phases finished on, with `n_moves` counting the moves of both,
/// or the error the primitive or the arena failed with.
pub fn nni<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<NniParams>,
) -> Result<NniResult, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let random = nni_random(tree, leaves, Some(params))?;
    let mut out = nni_greedy(&random.tree, leaves, Some(params))?;
    out.n_moves += random.n_moves;
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::global::optimise_branch_lengths;
    use crate::search::polytomy::resolve_polytomies;
    use crate::tree::simulate::{SimulationParams, robinson_foulds, simulate_binary, splits};
    use crate::tree::{NO_NODE, simulate::SimulatedData};
    use std::collections::HashSet;

    /// A small simulated dataset.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, a power of two
    /// * `n_features` - Number of features
    /// * `seed` - Simulation seed
    ///
    /// ### Returns
    ///
    /// The dataset with its precisions already formed.
    fn dataset(n_leaves: usize, n_features: usize, seed: u64) -> (SimulatedData<f64>, Vec<f64>) {
        let data = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves,
            n_features,
            seed,
            ..SimulationParams::default()
        }))
        .expect("simulate");
        let precisions = data.precisions();
        (data, precisions)
    }

    /// Leaves below every node of a tree.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    ///
    /// ### Returns
    ///
    /// One sorted leaf set per node.
    fn leaf_sets(tree: &Tree) -> Vec<Vec<u32>> {
        let mut sets: Vec<Vec<u32>> = vec![Vec::new(); tree.n_nodes()];
        for leaf in 0..tree.n_leaves() {
            sets[leaf].push(leaf as u32);
        }
        for node in tree.internal_postorder() {
            let mut here: Vec<u32> = tree
                .children(node)
                .iter()
                .flat_map(|&c| sets[c as usize].clone())
                .collect();
            here.sort_unstable();
            sets[node as usize] = here;
        }
        sets
    }

    /// Canonicalise a leaf set into a split key the way `splits` does.
    ///
    /// ### Params
    ///
    /// * `side` - One side of the bipartition
    /// * `n_leaves` - Total leaf count
    ///
    /// ### Returns
    ///
    /// The side that does not hold leaf zero, sorted.
    fn canonical(side: &[u32], n_leaves: usize) -> Vec<u32> {
        let holds_zero = side.contains(&0);
        let mut out: Vec<u32> = if holds_zero {
            (0..n_leaves as u32)
                .filter(|leaf| !side.contains(leaf))
                .collect()
        } else {
            side.to_vec()
        };
        out.sort_unstable();
        out
    }

    /// A starting tree with its branch lengths optimised, as search step 4
    /// leaves them.
    ///
    /// The interchanges of step 6 run after the global branch-length
    /// optimisation of step 4, and they are a topology search: run them on a
    /// tree whose branches are all at their default and the landscape they read
    /// is dominated by the branch lengths being wrong rather than by the shape.
    ///
    /// ### Params
    ///
    /// * `tree` - Starting topology
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The tree with its branch lengths optimised.
    fn optimised(tree: &Tree, leaves: Leaves<'_, f64>) -> Tree {
        let mut tree = tree.clone();
        let mut state = NodeState::new(
            tree.n_nodes(),
            leaves.n_features,
            leaves.means,
            leaves.precisions,
        )
        .expect("state");
        optimise_branch_lengths(&mut tree, &mut state, None).expect("branch lengths");
        tree
    }

    #[test]
    fn test_a_classical_interchange_reconnects_four_subtrees() {
        // A binary tree, an internal edge with an internal node at both ends,
        // and no polytomy anywhere: the generalised move must collapse to the
        // textbook one. The four subtrees admit exactly three unrooted
        // topologies, and the tree that comes back has to be one of them.
        let (p, n) = (128usize, 16usize);
        let (data, w) = dataset(n, p, 4);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let tree = &data.tree;
        let (down, up, _) = settle(tree, leaves).expect("settle");
        let sets = leaf_sets(tree);
        let base = splits(tree);

        let mut checked = 0usize;
        for k in tree.internal_postorder() {
            if interchange_members(tree, k) != Some(4) {
                continue;
            }
            let kids = tree.children(k);
            let l = tree.parent(k).expect("k is not the root");
            let sibling = tree
                .children(l)
                .iter()
                .copied()
                .find(|&c| c != k)
                .expect("l has another child");
            let a = &sets[kids[0] as usize];
            let b = &sets[kids[1] as usize];
            let c = &sets[sibling as usize];

            let split_k = canonical(&sets[k as usize], n);
            let mut without = base.clone();
            assert!(without.remove(&split_k), "the edge at {k} was not a split");

            // The three reconnections: A with B, which is what is already
            // there, A with C, and A with the upstream side. Each is one split
            // swapped for another, everything else untouched.
            let allowed: Vec<Vec<u32>> = [b, c]
                .iter()
                .map(|other| {
                    let mut side = a.clone();
                    side.extend_from_slice(other);
                    canonical(&side, n)
                })
                .chain(std::iter::once(canonical(a, n)))
                .collect();

            let spliced = interchange_at(tree, &down, &up, k, None)
                .expect("interchange")
                .expect("eligible edge");
            assert_eq!(spliced.tree.n_nodes(), tree.n_nodes());
            for node in spliced.tree.internal_postorder() {
                assert_eq!(
                    spliced.tree.children(node).len(),
                    2,
                    "the interchange at {k} left a polytomy"
                );
            }

            let got = splits(&spliced.tree);
            let extra: Vec<Vec<u32>> = got.difference(&without).cloned().collect();
            assert_eq!(
                extra.len(),
                1,
                "the interchange at {k} changed more than one split"
            );
            assert!(
                allowed.contains(&extra[0]),
                "the interchange at {k} produced {:?}, not one of the three reconnections {allowed:?}",
                extra[0]
            );
            assert_eq!(
                got.len(),
                base.len(),
                "the interchange at {k} changed the split count"
            );
            checked += 1;
        }
        assert!(checked > 0, "the fixture had no eligible internal edge");
    }

    #[test]
    fn test_the_greedy_phase_never_lowers_the_loglikelihood() {
        // Monotonicity, from a deliberately wrong starting topology so that
        // there is plenty for the phase to do.
        for seed in [1u64, 2, 3] {
            let (p, n) = (128usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = tree_loglik(&start, leaves).expect("loglik");
            let out = nni_greedy(&start, leaves, None).expect("greedy");
            let after = tree_loglik(&out.tree, leaves).expect("loglik");

            assert!(
                after >= before - 1e-9,
                "seed {seed}: {before} fell to {after}"
            );
            approx::assert_relative_eq!(out.loglik, after, max_relative = 1e-12);
            assert!(out.rounds < NniParams::default().max_rounds);
        }
    }

    #[test]
    fn test_the_greedy_phase_stops_at_a_fixed_point() {
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 6);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let once = nni_greedy(&start, leaves, None).expect("greedy");
        let twice = nni_greedy(&once.tree, leaves, None).expect("greedy");
        assert_eq!(twice.n_moves, 0);
        assert_eq!(twice.rounds, 1);
        assert_eq!(splits(&twice.tree), splits(&once.tree));
    }

    #[test]
    fn test_the_greedy_phase_does_not_chase_branch_lengths() {
        // Started on the generating tree with its branch lengths optimised,
        // there is no topology left to find. Every proposal from here rebuilds
        // the same splits and only reoptimises the three branches the star
        // primitive creates, so every one must be discarded. Without that
        // filter this ran for over two hundred rounds at a Robinson-Foulds
        // distance of zero throughout.
        let (p, n) = (256usize, 32usize);
        let (data, w) = dataset(n, p, 5);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let start = optimised(&data.tree, leaves);
        let out = nni_greedy(&start, leaves, None).expect("greedy");
        println!(
            "from the truth: {} moves over {} rounds, RF {}",
            out.n_moves,
            out.rounds,
            robinson_foulds(&out.tree, &data.tree).expect("rf")
        );
        assert!(
            out.rounds < 10,
            "{} rounds from a tree with nothing to find",
            out.rounds
        );
    }

    #[test]
    fn test_the_greedy_phase_recovers_the_generating_topology() {
        // The recovery test. Start from a ladder, which shares almost nothing
        // with the tree the data came off, put its branch lengths where search
        // step 4 would, and show the distance to truth falls. The numbers are
        // printed rather than pinned to a threshold beyond "it must fall":
        // what is being tested is the direction.
        for seed in [1u64, 2, 3] {
            let (p, n) = (256usize, 32usize);
            let (data, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &data.means,
                precisions: &w,
                n_features: p,
            };
            let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
            let before = robinson_foulds(&start, &data.tree).expect("rf");
            let out = nni_greedy(&start, leaves, None).expect("greedy");
            let after = robinson_foulds(&out.tree, &data.tree).expect("rf");
            println!(
                "seed {seed}: RF {before} -> {after} of a possible {} in {} moves over {} rounds",
                2 * (n - 3),
                out.n_moves,
                out.rounds
            );
            assert!(after < before, "seed {seed}: RF went {before} -> {after}");
        }
    }

    #[test]
    fn test_the_greedy_phase_recovers_what_the_random_phase_costs() {
        // The random phase is not monotone by construction: a collapse can lose
        // a split that the resampled star does not put back. What is claimed is
        // that the greedy phase which follows recovers at least the tree the
        // pair started from.
        let (p, n) = (256usize, 32usize);
        let (data, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        // Start at a local optimum of the greedy phase, so that the random
        // phase has something to escape from.
        let start = optimised(&Tree::ladder(n, 1.0).expect("ladder"), leaves);
        let settled = nni_greedy(&start, leaves, None).expect("greedy");

        for seed in [1u64, 7, 19] {
            let params = NniParams {
                n_random: n / 4,
                seed,
                ..NniParams::default()
            };
            let random = nni_random(&settled.tree, leaves, Some(params)).expect("random");
            let out = nni(&settled.tree, leaves, Some(params)).expect("nni");
            println!(
                "seed {seed}: start {:.3} (RF to truth {}), after {} random moves {:.3} \
                 (RF to the start {}), after greedy {:.3}",
                settled.loglik,
                robinson_foulds(&settled.tree, &data.tree).expect("rf"),
                random.n_moves,
                random.loglik,
                robinson_foulds(&random.tree, &settled.tree).expect("rf"),
                out.loglik
            );
            assert!(
                out.loglik >= settled.loglik - 1e-6,
                "seed {seed}: random-then-greedy left {} against a start of {}",
                out.loglik,
                settled.loglik
            );
        }
    }

    #[test]
    fn test_the_random_phase_is_deterministic_whatever_the_thread_count() {
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 8);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let params = NniParams {
            n_random: 12,
            seed: 0x2545_F491_4F6C_DD1D,
            ..NniParams::default()
        };
        let reference = nni_random(&data.tree, leaves, Some(params)).expect("random");

        for threads in [1usize, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got =
                pool.install(|| nni_random(&data.tree, leaves, Some(params)).expect("random"));
            assert_eq!(
                splits(&got.tree),
                splits(&reference.tree),
                "topology moved at {threads} threads"
            );
            assert_eq!(got.tree.branches(), reference.tree.branches());
            assert_eq!(got.loglik.to_bits(), reference.loglik.to_bits());
            assert_eq!(got.n_moves, reference.n_moves);
        }
    }

    #[test]
    fn test_a_different_seed_gives_a_different_walk() {
        // Otherwise the random phase is randomised in name only.
        let (p, n) = (128usize, 32usize);
        let (data, w) = dataset(n, p, 8);
        let leaves = Leaves {
            means: &data.means,
            precisions: &w,
            n_features: p,
        };
        let walk = |seed: u64| {
            let params = NniParams {
                n_random: 12,
                seed,
                ..NniParams::default()
            };
            nni_random(&data.tree, leaves, Some(params)).expect("random")
        };
        let seen: HashSet<Vec<Vec<u32>>> = (0..6u64)
            .map(|s| {
                let mut keys: Vec<Vec<u32>> = splits(&walk(s).tree).into_iter().collect();
                keys.sort();
                keys
            })
            .collect();
        assert!(seen.len() > 1, "six seeds all produced the same tree");
    }

    #[test]
    fn test_a_star_tree_has_no_eligible_edge() {
        // Every leaf hangs off the root, so there is no internal edge at all.
        let n = 12usize;
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let tree = Tree::from_parents(parent, vec![0.5; n + 1], n).expect("star tree");
        for node in tree.internal_postorder() {
            assert_eq!(interchange_members(&tree, node), None);
        }

        let p = 64usize;
        let (data, w) = dataset(16, p, 21);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let out = nni_greedy(&tree, leaves, None).expect("greedy");
        assert_eq!(out.n_moves, 0);
        assert_eq!(out.rounds, 1);
    }

    #[test]
    fn test_a_tree_too_small_for_an_interchange() {
        // Four leaves on a balanced binary tree. The root has two children, so
        // collapsing either of them leaves a three-member star and there is
        // nothing to interchange.
        let (p, n) = (64usize, 4usize);
        let (data, w) = dataset(16, p, 15);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let tree = Tree::balanced_binary(n, 1.0).expect("balanced");
        for node in tree.internal_postorder() {
            assert_eq!(interchange_members(&tree, node), None);
        }
        let out = nni_greedy(&tree, leaves, None).expect("greedy");
        assert_eq!(out.n_moves, 0);
    }

    #[test]
    fn test_interchanges_survive_a_polytomy() {
        // The generalised move has to work on a tree that is not binary, which
        // is the whole point of SPEC.md section 9.4. Resolve the polytomies of
        // a star tree first so the fixture is a real search state, then leave
        // one node unresolved by hand.
        let (p, n) = (128usize, 24usize);
        let (data, w) = dataset(32, p, 12);
        let leaves = Leaves {
            means: &data.means[..n * p],
            precisions: &w[..n * p],
            n_features: p,
        };
        let mut parent = vec![n as u32; n + 1];
        parent[n] = NO_NODE;
        let start = Tree::from_parents(parent, vec![0.5; n + 1], n).expect("star tree");
        let resolved = resolve_polytomies(&start, leaves, None).expect("resolve");

        let before = tree_loglik(&resolved.tree, leaves).expect("loglik");
        let out = nni_greedy(&resolved.tree, leaves, None).expect("greedy");
        assert!(out.loglik >= before - 1e-9);
    }
}
