# Changelog

## 0.2.0

**Breaking**

- Backbone mode removed: `backbone`, `grow`, `BackboneParams`,
  `BackboneReport`, and `model::place::place_from`. With the Ward linkage start
  it saved about 3 per cent of a run and kept the whole-tree refinement that is
  the rest; measured on Baron 5k, 10k and 25k and two synthetic sets it was below
  the full search's loglikelihood on every dataset and seed and slower up to 25k.
  SPEC section 15 has the numbers.
- `equal_daylight` returns the `Layout` alone; `DaylightReport` is gone.
- `write_newick_labelled` is gone; `write_newick` never writes internal labels.
  The parser still reads and discards them.
- Removed as analytics or test-only: `EllipsoidBoundsParams`,
  `EllipsoidBoundsStats` and `EllipsoidBounds::{stats, nsteps}`
  (`EllipsoidBounds::new` takes the inner provider only), `BonsaiParams.bounds`,
  `NniRound` and `NniResult::trace`, `PolytomyResult::gain`,
  `SprResult::n_moves`, the `new` constructors of `SprApprox`, `NniApprox` and
  `PlacementParams` (build them as structs), `merge::gain_at`,
  `cluster_centres`, `summed_leaf_distance`, `Clustering::n_clusters`, the
  `objective` and `cuts` fields of `Clustering`, and `export::posteriors_csv`.
- The `DAYLIGHT_*` constants, `DEFAULT_LEAF_SPACING` and `has_edge_crossing`
  are private.

**Search**

- SPR and NNI's lazy phase run on a live tree with stable node ids. SPR scores a
  move on views of the pruned and regrafted trees, accepts on a fixed-point
  loglikelihood total, and applies an accepted move to the paths it changes
  only. SPR is close to linear in `n` up to 131,072 cells.
- NNI and polytomy resolution keep their rows across moves instead of settling
  the whole tree every round.
- Approximate SPR re-applies the rest of a chunk after an acceptance instead of
  proposing it again (`SprApprox::recheck`, on by default).
- The prune and the up sweep run the nodes of a level in parallel, which is
  most of the global branch solve (steps 4, 7 and 8).
- New nodes are numbered by assembly rather than traversal order, so trees
  differ from 0.1 within the run-to-run spread.

## 0.1.1

**Search**

- `SprSearch` and `NniSearch`: `Exact` is SPEC 9.3 and 9.4, `Approximate` is
  the new default. Approximate SPR re-proposes only subtrees within five edges
  of a clade the last sweep created; approximate NNI caches each edge's gain
  and rescores only near the last move (lazy greedy).
- SPR accepts a move into a slot store instead of copying the whole state, and
  builds its arenas already in level order (`Tree::from_level_ordered`).

**Ingest**

- `DEFAULT_MIN_SIGNAL_TO_NOISE` is 1, the paper's threshold, up from 0.25.
  Default runs select fewer genes and give different trees.
- `gpu` feature: Sanity through `sanity-sc-rs` 0.1.0 on CubeCL/wgpu; its output
  goes into `from_sanity_output` unchanged.

**Other**

- The comparison harness is published under `reference/comparison`.

## 0.1.0

First release. Clean-room, built from the paper and its CC-BY-4.0
Supplementary Information; see [provenance](PROVENANCE.md).

**Model**

- Ingest: the SPEC 3.1 scale transform, signal-to-noise feature selection,
  Sanity posteriors to likelihood parameters.
- `sanity` feature: `ingest::from_sanity_output` takes a `sanity-sc-rs` run
  straight to Bonsai input. Raw UMIs to tree without leaving Rust.
- Pruning recursion, row-major `[node][feature]`, parallel over features, so
  tree shape doesn't matter.
- Branch lengths by safeguarded Newton on the closed-form stationarity
  condition.
- Merge scoring with the two-stage split solve; node placement by beam search.

**Search**

- SPEC section 9's seven steps plus an eighth that collapses internal
  zero-length edges. [Design](docs/DESIGN.md) says why the order is fixed.
- kNN candidate restriction (SPEC 11) and ellipsoid upper bounds (SPEC 10).
  Without them it's cubic.
- Ward linkage start: same answer as the greedy merge, a fraction of the time.
- SPR with lazy proposal rows and a scale-relative acceptance floor; NNI
  generalised to polytomies with a structural move filter.
- Backbone mode (SPEC 15).

**Output**

- Newick in and out, equal-angle, equal-daylight and dendrogram layouts, path
  distances, posterior mean and SD for every node including ancestors.

**Numerics**

- Storage generic over `BonsaiFloat`, reductions in `f64`.
- Deterministic whatever the thread count.
- SIMD for `f32` storage and the branch solve, via `wide`.
