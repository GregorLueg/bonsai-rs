//! Candidate restriction by a nearest-neighbour graph (SPEC.md section 11).
//!
//! Scoring every pair of star members is `O(n^3 p)` over a star; a
//! `k`-nearest-neighbour graph over the members gives `n*k` pairs a round.
//!
//! Rules, from SPEC.md section 11:
//!
//! 1. Build a `k`-NN graph over the members before the first round, plain
//!    Euclidean on the effective means. Precisions are ignored on purpose: the
//!    graph only decides which pairs are considered.
//! 2. Emit each member paired with its neighbours, once per unordered pair.
//! 3. A new ancestor inherits the union of its children's lists, minus the
//!    children.
//! 4. Rebuild the graph every so often. Measurement contradicts the premise
//!    (lists thin out rather than grow); see [`KnnCandidatesParams::default`].
//!
//! The graph is kept symmetric at every rebuild and across merges: both
//! children are replaced by the ancestor in every list that held them. Rule 3
//! taken one-sidedly leaves ancestors mutually invisible and stalls the star
//! early (`k >= n - 1` catches it). The restriction is an approximation; the
//! best pair can miss every neighbour list.
//!
//! The graph is built in `f32` whatever the storage type: it only filters
//! pairs, and it keeps `ann-search-rs`'s bounds out of [`BonsaiFloat`].

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

/// Default for [`KnnCandidatesParams::k`], see [`KnnCandidatesParams::default`].
const DEFAULT_K: usize = 16;

/// Default for [`KnnCandidatesParams::rebuild_every`], see
/// [`KnnCandidatesParams::default`].
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
    /// Ours, measured on simulated trees (SPEC.md section 13.1, 24 replicates at
    /// 128 leaves by 200 features): from `k = 12` the mean Robinson-Foulds
    /// distance matches the exhaustive scan (3.67), and from `k = 16` every
    /// replicate is bit-identical to it. Below `k = 6` the restriction does real
    /// damage. Same knee at 64 and 256 leaves.
    ///
    /// Rebuilding makes the candidate set larger, not smaller: the graph thins
    /// over a star and a rebuild replenishes it. Any cadence down to about `n/2`
    /// recovers the fully rebuilt tree, so `64` (two rebuilds at 128 members) is
    /// the cheapest. A rebuild costs `O(n^2 p)` against a round's `O(n k p)`, so
    /// raise `rebuild_every` with the star.
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
/// The graph belongs to one star: the first round of a new star (the only round
/// with no ancestors) rebuilds it, so an instance can be reused across
/// [`crate::search::star::resolve_star_with`] calls.
///
/// The graph is built exhaustively at `O(n^2 p)` per rebuild, so `k` is the only
/// approximation. That stops being sensible around `10^5` members; an
/// approximate backend (NN-descent builds a `k`-NN graph directly) would also
/// need its own `BonsaiErrors` variant.
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
    /// Rebuilds the graph first on the first round of a star or when the
    /// cadence is due.
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
        // Both spellings of a pair arrive: a broken symmetry then costs a
        // duplicate, not a lost candidate. Sorting fixes the emitted order.
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
    use crate::model::merge::MergeParams;
    use crate::search::star::{Star, StarResult, resolve_star, resolve_star_with};
    use crate::utils::rng::SplitMix64;
    use crate::utils::verbosity::Verbosity;

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
        // The restricted provider reads means alone, so the rest of the view
        // is only there to satisfy the type.
        let branch = vec![0.0f64; members.len()];
        let (mc, wc) = (vec![0.0f64; p], vec![1.0f64; p]);
        let mut pairs = Vec::new();
        provider
            .candidates(
                Round {
                    members: &members,
                    means: m,
                    precisions: w,
                    n_features: p,
                    branch: &branch,
                    centre_means: &mc,
                    centre_precisions: &wc,
                    merge: MergeParams::default(),
                    best_gain: f64::NEG_INFINITY,
                    scored_last_round: 0,
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
            Verbosity::Quiet,
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
                Verbosity::Quiet,
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

        let branch = vec![0.0f64; n];
        let (mc, wc) = (vec![0.0f64; p], vec![1.0f64; p]);
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
                    branch: &branch,
                    centre_means: &mc,
                    centre_precisions: &wc,
                    merge: MergeParams::default(),
                    best_gain: f64::NEG_INFINITY,
                    scored_last_round: 0,
                },
                &mut narrow_pairs,
            )
            .expect("candidates");

        assert_eq!(wide_pairs, narrow_pairs);
    }
}
