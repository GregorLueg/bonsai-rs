//! Candidate restriction by a nearest-neighbour graph (SPEC.md section 11).
//!
//! The exhaustive provider scores every pair of star members every round, which
//! is `O(n^2 p)` per round and `O(n^3 p)` over a whole star. Nodes far apart in
//! plain Euclidean distance are never the best merge, so a `k`-nearest-neighbour
//! graph over the members is enough to decide which pairs are worth scoring:
//! `n*k` pairs a round instead of `n^2/2`.
//!
//! Four rules, from SPEC.md section 11:
//!
//! 1. Build a `k`-nearest-neighbour graph over the members before the first
//!    round, plain Euclidean on the transformed effective means. Precisions are
//!    ignored on purpose: this decides which pairs get *considered*, not which
//!    one wins, and the score that decides that is precision weighted already.
//! 2. Emit each member paired with its neighbours, once per unordered pair.
//! 3. A new ancestor inherits the union of its two children's lists, minus the
//!    two children.
//! 4. Because the lists therefore grow, rebuild the graph every so often.
//!
//! Rule 4's premise does not survive measurement: under the symmetric union
//! below the lists thin out rather than grow, and a rebuild replenishes them
//! rather than trimming them. It is still worth doing, for the opposite reason.
//! See [`KnnCandidatesParams::default`].
//!
//! ### The graph is kept symmetric, and has to be
//!
//! A `k`-nearest-neighbour graph is directed: `j` can be one of `i`'s nearest
//! neighbours without `i` being one of `j`'s. The pairs it induces are
//! undirected all the same, so this module symmetrises it at every rebuild and
//! maintains that symmetry across merges, replacing both children by the
//! ancestor in every list that held them.
//!
//! Doing only what rule 3 literally says, and giving the ancestor its inherited
//! list without touching anyone else's, is wrong and quietly so. The ancestor
//! then knows its partners but no old member knows the ancestor, so a *second*
//! ancestor inherits lists that never mention the first, and no pair of
//! ancestors is ever offered. The star stalls several merges early with the
//! ancestors mutually invisible. `k >= n - 1` catches it, which is why that
//! test exists.
//!
//! **This is an approximation and the only honest test of it is measurement.**
//! The best pair in the star can fail to be in any neighbour list. What that
//! costs in recovered topology, measured against the exhaustive scan on
//! simulated trees, is in the [`KnnCandidatesParams::default`] documentation,
//! and it is the entire justification for the constants shipped here.
//!
//! ### Precision
//!
//! The graph is built in `f32` whatever the storage type. It only filters
//! pairs, so precision is irrelevant to it, `f32` halves the working set that
//! thrashes the cache during the search, and it keeps `ann-search-rs`'s
//! `SimdDistance + ComplexField` bounds out of [`BonsaiFloat`], which would
//! otherwise drag `faer-traits` into every generic in this crate.

use crate::errors::BonsaiErrors;
use crate::model::merge::EffLeaf;
use crate::search::star::{CandidatePairs, Round, StarMerge};
use crate::utils::traits::{BonsaiFloat, wide};
use ann_search_rs::{build_exhaustive_index, query_exhaustive_self};

///////////////
// Constants //
///////////////

/// Distance metric handed to `ann-search-rs`.
///
/// It parses this to a *squared* Euclidean distance, which orders neighbours
/// identically to the Euclidean distance SPEC.md section 11 asks for.
const ANN_METRIC: &str = "euclidean";

/// Default for [`KnnCandidatesParams::k`].
///
/// See [`KnnCandidatesParams::default`] for the measurement that pins it.
const DEFAULT_K: usize = 16;

/// Default for [`KnnCandidatesParams::rebuild_every`].
///
/// See [`KnnCandidatesParams::default`] for the measurement that pins it.
const DEFAULT_REBUILD_EVERY: usize = 64;

////////////////
// Parameters //
////////////////

/// Tuning knobs for [`KnnCandidates`].
///
/// Both are pure speed against accuracy. Neither changes the model: a larger
/// `k` and a shorter rebuild cadence both converge on the exhaustive scan.
#[derive(Clone, Copy, Debug)]
pub struct KnnCandidatesParams {
    /// Neighbours per member in the graph, excluding the member itself.
    ///
    /// At `k >= n_members - 1` the graph is complete and the provider returns
    /// exactly what [`crate::search::star::AllPairs`] returns. At `k = 0` it
    /// returns nothing and the star is left unresolved.
    pub k: usize,
    /// Rebuild the graph after this many new ancestors.
    ///
    /// Zero rebuilds every round, which is correct but pays a graph build per
    /// merge. A rebuild costs `O(n^2 p)` against a round's `O(n k p)`, so the
    /// rebuilds are a fraction `n / (rebuild_every * k)` of the scan they sit
    /// inside: scale this with the number of members, and see
    /// [`KnnCandidatesParams::default`].
    pub rebuild_every: usize,
}

impl Default for KnnCandidatesParams {
    /// `k = 16` neighbours, rebuilt every `64` ancestors.
    ///
    /// ### Where these came from
    ///
    /// Measured on 2026-08-28 on simulated data (SPEC.md section 13.1) over
    /// three tree shapes, balanced with constant branches, balanced with random
    /// branches and unbalanced, eight seeds each, so 24 replicates per point.
    /// 128 leaves by 200 features, error bars a tenth of the data spread, star
    /// branch lengths optimised first as search step 1 does. The metric is the
    /// Robinson-Foulds distance to the generating tree, out of 125 non-trivial
    /// splits, and the comparison is the exhaustive scan on the identical
    /// fixture, which recovers 3.67 on average. Also counted: replicates whose
    /// topology comes out *bit-identical* to the exhaustive one, which is the
    /// stronger question of whether the restriction changed the answer at all.
    ///
    /// | `k` | mean RF | excess over exhaustive | identical to exhaustive |
    /// |---|---|---|---|
    /// | 4 | 7.62 | +3.96 | 0 / 24 |
    /// | 6 | 4.25 | +0.58 | 5 / 24 |
    /// | 8 | 4.08 | +0.42 | 12 / 24 |
    /// | 12 | 3.67 | +0.00 | 21 / 24 |
    /// | 16 | 3.67 | +0.00 | 24 / 24 |
    /// | 24 | 3.67 | +0.00 | 24 / 24 |
    /// | 32 | 3.67 | +0.00 | 24 / 24 |
    /// | 127, complete | 3.67 | 0 | 24 / 24 |
    ///
    /// The same sweep at 64 leaves (24 replicates) and at 256 leaves (12
    /// replicates) puts the knee in the same place: zero excess RF from
    /// `k = 12` at all three sizes, and every replicate identical to the
    /// exhaustive tree from `k = 16` at all three. The `k` that matters is not
    /// a fraction of the star, it is a small constant.
    ///
    /// **What this says about the paper's 5 to 10.** It is not free, but it is
    /// not far off either. At `k = 8` the restriction costs four tenths of a
    /// split out of 125, which is real but small, and it reproduces the
    /// exhaustive topology in only half the replicates. `16` is where both
    /// measures stop moving, and it is the value this crate ships. It scores
    /// a fifth of the exhaustive scan's pairs at 128 members and a tenth at
    /// 256, so paying a factor of two over `k = 8` for an approximation that
    /// stops being visible is a good trade. Below `k = 6` the curve turns
    /// sharply and the restriction does real damage.
    ///
    /// ### The cadence, and a result that contradicts the premise
    ///
    /// Swept at `k = 16` over the same 24 replicates, alongside the number of
    /// pairs a whole star actually scored, counted at the seam over 12 of them
    /// against the exhaustive scan's 349,500:
    ///
    /// | `rebuild_every` | rebuilds per star | mean RF | pairs scored |
    /// |---|---|---|---|
    /// | 1 | 125 | 3.67 | 79,495 |
    /// | 8 | 16 | 3.67 | 77,294 |
    /// | 16 | 8 | 3.67 | 75,035 |
    /// | 32 | 4 | 3.67 | 71,458 |
    /// | 64 | 2 | 3.67 | 65,865 |
    /// | never | 1 | 3.83 | 59,957 |
    ///
    /// **Rebuilding makes the candidate set larger, not smaller.** SPEC.md
    /// section 11 motivates rule 4 by the lists growing, and with the symmetric
    /// maintenance this module needs they do the opposite: a merge removes two
    /// nodes' worth of edges and adds back the deduplicated union, so the graph
    /// thins out over a star and the widest round is always the first. What a
    /// rebuild does is replenish it, which is why rebuilding every round costs
    /// a third more pairs over a star than never rebuilding, and buys back the
    /// recovery the thinning loses. The widest round was 1312 pairs at every
    /// cadence.
    ///
    /// The thinning costs nothing at 64 members, `+0.17` splits at 128 and
    /// `+0.75` at 256, so it is a real effect that grows with the star. Any
    /// cadence from one rebuild per merge down to `n/2` recovers all of it, and
    /// the cheapest of those is the one to take: `64` is two rebuilds for the
    /// 128-member stars measured here and four for 256-member ones, and both
    /// match the fully rebuilt tree exactly.
    ///
    /// **Scale it with the star.** A rebuild is `O(n^2 p)` against a round's
    /// `O(n k p)`, so the rebuilds are a fraction `n / (rebuild_every * k)` of
    /// the scan they sit inside, and holding that fixed means `rebuild_every`
    /// of about `n/2`. `64` is that value at 128 members. For a much larger
    /// star raise it, and see [`KnnCandidates`] on when to stop building the
    /// graph exhaustively at all.
    ///
    /// ### Returns
    ///
    /// The default parameters.
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            rebuild_every: DEFAULT_REBUILD_EVERY,
        }
    }
}

//////////////////
// The provider //
//////////////////

/// Candidate pairs restricted to a `k`-nearest-neighbour graph.
///
/// ### Lifetime of an instance
///
/// The graph belongs to one star. An instance detects the first round of a new
/// star, which is the only round in which no ancestors exist yet, and rebuilds
/// from scratch, so reusing one instance across several
/// [`crate::search::star::resolve_star_with`] calls is safe. It is still
/// clearer to hand each star its own.
///
/// ### Which backend
///
/// The graph is built exhaustively, at `O(n^2 p)` per rebuild. That is the
/// right default while the search is being built: it makes "did the
/// approximation miss it" a question about `k` alone and never about the
/// index. It is also the term that stops scaling first. With a cadence of
/// `n/2` a star pays two exhaustive rebuilds against a restricted scan of
/// `O(n^2 k p)`, so at `k = 16` the rebuilds are an eighth of the scan and
/// stay there; the trouble is that the ratio is `n / (rebuild_every * k)`, so
/// holding it fixed means growing the cadence with `n`, and a single `n^2 p`
/// distance matrix stops being a sensible thing to compute somewhere around
/// `10^5` members whatever the cadence.
///
/// At that point reach for one of `ann-search-rs`'s approximate backends,
/// which sit behind the same shape: `build_nndescent_index`,
/// `build_hnsw_index`, `build_ivf_index`, plus GPU variants. NN-descent is the
/// one to want here, because it builds a `k`-nearest-neighbour graph directly,
/// which is exactly this structure, at near-linear cost, and it is never
/// queried again afterwards. HNSW pays for its construction over many later
/// queries, and there are none. IVF wants a natural cluster structure and a
/// tuned probe count, which is one more knob to justify.
///
/// **Adding any of them needs one new `BonsaiErrors` variant.** The exhaustive
/// path's only failure is a dimension mismatch this module cannot produce, so
/// it borrows `MalformedTree` for something that cannot happen. An approximate
/// backend fails for real reasons and deserves to say so.
#[derive(Clone, Debug, Default)]
pub struct KnnCandidates {
    /// Tuning knobs.
    params: KnnCandidatesParams,
    /// Neighbour ids per node id, ascending, self excluded.
    ///
    /// Indexed by node id and sized to every node created so far, so a member's
    /// list is found without a search. Rows of nodes that have been swallowed
    /// are emptied rather than removed, which keeps the indexing trivial. The
    /// relation is symmetric: `j` is in row `i` exactly when `i` is in row `j`.
    neighbours: Vec<Vec<u32>>,
    /// Ancestors created since the last rebuild.
    since_rebuild: usize,
    /// `f32` copy of the current members' means, held across rebuilds so the
    /// allocation is paid once.
    coords: Vec<f32>,
}

impl KnnCandidates {
    /// A provider over a fresh graph.
    ///
    /// ### Params
    ///
    /// * `params` - Tuning knobs, or `None` for [`KnnCandidatesParams::default`]
    ///
    /// ### Returns
    ///
    /// The provider. The graph itself is built on the first round, when the
    /// members are known.
    pub fn new(params: Option<KnnCandidatesParams>) -> Self {
        Self {
            params: params.unwrap_or_default(),
            neighbours: Vec::new(),
            since_rebuild: 0,
            coords: Vec::new(),
        }
    }

    /// Rebuild the graph over the current members.
    ///
    /// ### Params
    ///
    /// * `round` - The round's view of the star
    ///
    /// ### Returns
    ///
    /// Nothing, or the error the neighbour search failed with.
    fn rebuild<T: BonsaiFloat>(&mut self, round: &Round<'_, T>) -> Result<(), BonsaiErrors> {
        let p = round.n_features;
        let n = round.members.len();
        let n_nodes = round.means.len() / p.max(1);

        self.coords.clear();
        self.coords.reserve(n * p);
        for &id in round.members {
            let base = id as usize * p;
            for g in 0..p {
                self.coords.push(wide(round.means[base + g]) as f32);
            }
        }

        // One more than `k` because the self-query returns each point as one of
        // its own neighbours. The backend clamps to `n` itself.
        let want = self.params.k.saturating_add(1);
        let index = build_exhaustive_index((self.coords.as_slice(), n, p), ANN_METRIC);
        let (rows, _) = query_exhaustive_self(&index, want, false, false).map_err(|e| {
            BonsaiErrors::MalformedTree {
                reason: format!("the {n} by {p} neighbour graph could not be built: {e}"),
            }
        })?;

        self.neighbours.clear();
        self.neighbours.resize(n_nodes, Vec::new());
        for (i, row) in rows.iter().enumerate() {
            // Filter self by value, not by position: the row is not guaranteed
            // to lead with it. Truncate before anything reorders the row, so
            // what goes is the farthest neighbour and not the largest id.
            let id = round.members[i] as usize;
            for &j in row.iter().filter(|&&j| j != i).take(self.params.k) {
                let neighbour = round.members[j];
                self.neighbours[id].push(neighbour);
                self.neighbours[neighbour as usize].push(round.members[i]);
            }
        }
        for list in self.neighbours.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        self.since_rebuild = 0;
        Ok(())
    }
}

impl<T: BonsaiFloat> CandidatePairs<T> for KnnCandidates {
    /// Each member paired with its neighbours, deduplicated.
    ///
    /// Rebuilds the graph first if this is the first round of a star or if the
    /// cadence has come round. Every neighbour is a live member, because a
    /// merge replaces its two children throughout the graph, so the lookup is
    /// only there to turn a node id into a position.
    ///
    /// ### Params
    ///
    /// * `round` - Read-only view of the current round
    /// * `out` - Destination for the pairs
    ///
    /// ### Returns
    ///
    /// Nothing, or the error the neighbour search failed with.
    fn candidates(
        &mut self,
        round: Round<'_, T>,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<(), BonsaiErrors> {
        if round.n_features == 0 || round.members.is_empty() {
            return Ok(());
        }
        // No ancestor exists yet exactly in the first round of a star, whoever
        // owned this provider before. Members shrink and node ids grow with
        // every merge, so the two can agree in no other round.
        let n_nodes = round.means.len() / round.n_features;
        let first_round = round.members.len() == n_nodes;
        if first_round || self.since_rebuild >= self.params.rebuild_every {
            self.rebuild(&round)?;
        }

        for (i, &id) in round.members.iter().enumerate() {
            for &neighbour in &self.neighbours[id as usize] {
                // Members are ascending, which is the primitive's invariant.
                if let Ok(j) = round.members.binary_search(&neighbour) {
                    match i.cmp(&j) {
                        std::cmp::Ordering::Less => out.push((i, j)),
                        std::cmp::Ordering::Greater => out.push((j, i)),
                        std::cmp::Ordering::Equal => {}
                    }
                }
            }
        }
        // The graph is symmetric, so both spellings of a pair reach here.
        // Emitting both and deduplicating, rather than emitting only the
        // ascending one, means a broken symmetry would cost a duplicate rather
        // than silently lose a candidate. Sorting also makes the emitted order
        // independent of the neighbour lists' history.
        out.sort_unstable();
        out.dedup();
        Ok(())
    }

    /// Replace the two children by their ancestor throughout the graph.
    ///
    /// SPEC.md section 11 rule 3, done symmetrically: the ancestor takes the
    /// union of its children's lists minus the children themselves, and every
    /// node in that union has the two children swapped for the ancestor. See
    /// the module documentation for why the one-sided version does not work.
    ///
    /// The ancestor's own effective mean is not consulted. The union is what
    /// the specification asks for, and it is exact where a neighbour query for
    /// a node that did not exist when the graph was built would itself be an
    /// approximation.
    ///
    /// ### Params
    ///
    /// * `merge` - The merge that was performed
    /// * `ancestor` - The new ancestor's effective leaf, unused here
    fn merged(&mut self, merge: &StarMerge, ancestor: EffLeaf<'_, T>) {
        let _ = ancestor;
        let (k, l, a) = (
            merge.left as usize,
            merge.right as usize,
            merge.ancestor as usize,
        );

        let mut list = std::mem::take(&mut self.neighbours[k]);
        list.extend_from_slice(&self.neighbours[l]);
        self.neighbours[l] = Vec::new();
        list.retain(|&x| x != merge.left && x != merge.right);
        list.sort_unstable();
        list.dedup();

        for &x in &list {
            let row = &mut self.neighbours[x as usize];
            row.retain(|&y| y != merge.left && y != merge.right);
            // Ancestor ids are allocated in increasing order and exceed every
            // member's, so appending keeps the list sorted.
            row.push(merge.ancestor);
        }

        if a >= self.neighbours.len() {
            self.neighbours.resize(a + 1, Vec::new());
        }
        self.neighbours[a] = list;
        self.since_rebuild += 1;
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::star::{Star, StarResult, resolve_star, resolve_star_with};
    use crate::utils::rng::SplitMix64;

    /// Leaves in tight groups, far apart from each other.
    ///
    /// ### Params
    ///
    /// * `n_clusters` - Number of groups
    /// * `per_cluster` - Leaves per group
    /// * `p` - Number of features
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Row-major means and precisions.
    fn clustered(
        n_clusters: usize,
        per_cluster: usize,
        p: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let centres: Vec<Vec<f64>> = (0..n_clusters)
            .map(|_| (0..p).map(|_| 8.0 * (rng.uniform() - 0.5)).collect())
            .collect();
        let mut m = Vec::with_capacity(n_clusters * per_cluster * p);
        let mut w = Vec::with_capacity(n_clusters * per_cluster * p);
        for c in 0..n_clusters {
            for _ in 0..per_cluster {
                for g in 0..p {
                    m.push(centres[c][g] + 0.4 * (rng.uniform() - 0.5));
                    w.push(0.5 + rng.uniform());
                }
            }
        }
        (m, w)
    }

    /// The pairs a provider emits in the first round of a star over members
    /// `0..n`.
    ///
    /// ### Params
    ///
    /// * `provider` - Provider under test
    /// * `m` - Means, row-major
    /// * `w` - Precisions, row-major
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// The emitted pairs, as positions into the members.
    fn first_round(
        provider: &mut KnnCandidates,
        m: &[f64],
        w: &[f64],
        p: usize,
    ) -> Vec<(usize, usize)> {
        let members: Vec<u32> = (0..(m.len() / p) as u32).collect();
        let mut pairs = Vec::new();
        provider
            .candidates(
                Round {
                    members: &members,
                    means: m,
                    precisions: w,
                    n_features: p,
                    best_gain: f64::NEG_INFINITY,
                },
                &mut pairs,
            )
            .expect("candidates");
        pairs
    }

    /// Run the primitive with the restricted provider.
    ///
    /// ### Params
    ///
    /// * `m` - Means, row-major
    /// * `w` - Precisions, row-major
    /// * `t` - Branches to the centre
    /// * `p` - Number of features
    /// * `params` - Provider knobs
    ///
    /// ### Returns
    ///
    /// What the primitive built.
    fn restricted(
        m: &[f64],
        w: &[f64],
        t: &[f64],
        p: usize,
        params: KnnCandidatesParams,
    ) -> StarResult<f64> {
        let mut provider = KnnCandidates::new(Some(params));
        resolve_star_with(
            Star {
                means: m,
                precisions: w,
                branch: t,
                n_features: p,
            },
            None,
            &mut provider,
        )
        .expect("resolve")
    }

    #[test]
    fn test_a_complete_graph_reproduces_the_exhaustive_tree_exactly() {
        // The plumbing test. With every member a neighbour of every other the
        // restriction is not a restriction, so anything but an identical tree
        // is a bug in the emission, the union rule or the position mapping,
        // not an approximation.
        for (n_clusters, per) in [(3usize, 5usize), (4, 4), (5, 3)] {
            let (p, n) = (24usize, n_clusters * per);
            let (m, w) = clustered(n_clusters, per, p, 0x2545_F491_4F6C_DD1D);
            let t0: Vec<f64> = (0..n).map(|i| 0.3 + 0.02 * i as f64).collect();

            let exhaustive = resolve_star(
                Star {
                    means: &m,
                    precisions: &w,
                    branch: &t0,
                    n_features: p,
                },
                None,
            )
            .expect("resolve");

            // Every cadence, because a rebuild mid-star must land on the same
            // complete graph the union rule was maintaining.
            for rebuild_every in [0usize, 1, 3, usize::MAX] {
                let got = restricted(
                    &m,
                    &w,
                    &t0,
                    p,
                    KnnCandidatesParams {
                        k: n - 1,
                        rebuild_every,
                    },
                );
                assert_eq!(got.parent, exhaustive.parent, "topology at n {n}");
                assert_eq!(got.branch, exhaustive.branch, "branches at n {n}");
                let gains: Vec<f64> = got.merges.iter().map(|x| x.gain).collect();
                let want: Vec<f64> = exhaustive.merges.iter().map(|x| x.gain).collect();
                assert_eq!(gains, want, "gains at n {n}");
            }
        }
    }

    #[test]
    fn test_a_restricted_graph_scores_fewer_pairs_and_still_resolves() {
        let (p, n) = (20usize, 24usize);
        let (m, w) = clustered(6, 4, p, 0x9E37_79B9_7F4A_7C15);
        let t0 = vec![0.4f64; n];

        let mut provider = KnnCandidates::new(Some(KnnCandidatesParams {
            k: 4,
            rebuild_every: 8,
        }));
        let pairs = first_round(&mut provider, &m, &w, p);

        // At most n*k directed edges, so at most that many undirected pairs,
        // and far below the n*(n-1)/2 the exhaustive scan would offer.
        assert!(pairs.len() <= n * 4, "{} pairs", pairs.len());
        assert!(pairs.len() < n * (n - 1) / 2);
        // Every emitted pair is ordered, in range and unique.
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, pairs);
        assert!(pairs.iter().all(|&(i, j)| i < j && j < n));

        let got = restricted(
            &m,
            &w,
            &t0,
            p,
            KnnCandidatesParams {
                k: 4,
                rebuild_every: 8,
            },
        );
        assert_eq!(got.centre_children.len(), 3);
    }

    #[test]
    fn test_the_graph_finds_the_clusters() {
        // Tight groups a long way apart: the neighbour lists must be inside the
        // groups, or the restriction is filtering on nothing.
        let (n_clusters, per, p) = (4usize, 6usize, 32usize);
        let (m, w) = clustered(n_clusters, per, p, 0xDEAD_BEEF_CAFE_F00D);

        let mut provider = KnnCandidates::new(Some(KnnCandidatesParams {
            k: per - 1,
            rebuild_every: usize::MAX,
        }));
        for (i, j) in first_round(&mut provider, &m, &w, p) {
            assert_eq!(i / per, j / per, "pair ({i}, {j}) crosses a cluster");
        }
    }

    #[test]
    fn test_an_ancestor_inherits_the_union_minus_its_children() {
        let (p, n) = (16usize, 12usize);
        let (m, w) = clustered(4, 3, p, 0x0BAD_C0DE_0BAD_C0DE);
        let mut provider = KnnCandidates::new(Some(KnnCandidatesParams {
            k: 3,
            rebuild_every: usize::MAX,
        }));
        let _ = first_round(&mut provider, &m, &w, p);

        let (left, right) = (2u32, 5u32);
        let mut want: Vec<u32> = provider.neighbours[left as usize]
            .iter()
            .chain(&provider.neighbours[right as usize])
            .copied()
            .filter(|&x| x != left && x != right)
            .collect();
        want.sort_unstable();
        want.dedup();

        let merge = StarMerge {
            left,
            right,
            ancestor: n as u32,
            t_left: 0.1,
            t_right: 0.1,
            t_centre: 0.1,
            gain: 1.0,
        };
        let row = vec![0.0f64; p];
        CandidatePairs::<f64>::merged(&mut provider, &merge, EffLeaf { m: &row, w: &row });

        assert_eq!(provider.neighbours[n], want);
        assert!(provider.neighbours[left as usize].is_empty());
        assert!(provider.neighbours[right as usize].is_empty());
        assert_eq!(provider.since_rebuild, 1);

        // The graph is undirected, so the ancestor has to appear on the other
        // side of every edge it inherited, and the children on neither.
        for (x, row) in provider.neighbours.iter().enumerate() {
            assert!(row.is_sorted(), "list of {x} is not sorted");
            assert!(!row.contains(&left) && !row.contains(&right));
            assert_eq!(
                row.contains(&(n as u32)),
                want.contains(&(x as u32)),
                "edge to the ancestor is one-sided at {x}"
            );
        }
    }

    #[test]
    fn test_the_same_tree_whatever_the_thread_count() {
        // The graph is built under rayon and the emitted order must not depend
        // on it, nor on which thread finished the pair scan first.
        let (p, n) = (24usize, 20usize);
        let (m, w) = clustered(5, 4, p, 0xC0FF_EE00_C0FF_EE00);
        let t0 = vec![0.35f64; n];
        let params = KnnCandidatesParams {
            k: 6,
            rebuild_every: 5,
        };

        let reference = restricted(&m, &w, &t0, p, params);
        for threads in [1usize, 2, 5, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| restricted(&m, &w, &t0, p, params));
            assert_eq!(
                got.parent, reference.parent,
                "topology at {threads} threads"
            );
            assert_eq!(
                got.branch, reference.branch,
                "branches at {threads} threads"
            );
        }
    }

    #[test]
    fn test_a_provider_reused_across_stars_rebuilds_for_each() {
        // The graph belongs to a star. A second, unrelated star handed to the
        // same instance must not inherit the first one's neighbour lists.
        let (p, n) = (20usize, 15usize);
        let (m1, w1) = clustered(5, 3, p, 0x1111_2222_3333_4444);
        let (m2, w2) = clustered(3, 5, p, 0x5555_6666_7777_8888);
        let t0 = vec![0.4f64; n];
        let params = KnnCandidatesParams {
            k: 5,
            rebuild_every: 4,
        };

        let mut shared = KnnCandidates::new(Some(params));
        let run = |provider: &mut KnnCandidates, m: &[f64], w: &[f64]| {
            resolve_star_with(
                Star {
                    means: m,
                    precisions: w,
                    branch: &t0,
                    n_features: p,
                },
                None,
                provider,
            )
            .expect("resolve")
        };

        let _ = run(&mut shared, &m1, &w1);
        let reused = run(&mut shared, &m2, &w2);
        let fresh = restricted(&m2, &w2, &t0, p, params);
        assert_eq!(reused.parent, fresh.parent);
    }

    #[test]
    fn test_f32_storage_builds_the_same_graph_as_f64() {
        // The graph is built in `f32` whatever the storage type, so the two
        // must agree on the pairs even though the scan does not agree bitwise.
        let (p, n) = (28usize, 18usize);
        let (m, w) = clustered(6, 3, p, 0xABCD_EF01_2345_6789);
        let members: Vec<u32> = (0..n as u32).collect();
        let params = KnnCandidatesParams {
            k: 5,
            rebuild_every: usize::MAX,
        };

        let wide_pairs = first_round(&mut KnnCandidates::new(Some(params)), &m, &w, p);

        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let mut narrow_provider = KnnCandidates::new(Some(params));
        let mut narrow_pairs = Vec::new();
        narrow_provider
            .candidates(
                Round {
                    members: &members,
                    means: &m32,
                    precisions: &w32,
                    n_features: p,
                    best_gain: f64::NEG_INFINITY,
                },
                &mut narrow_pairs,
            )
            .expect("candidates");

        assert_eq!(wide_pairs, narrow_pairs);
    }
}
