# Changelog

## 0.2.0

**Breaking**

- Backbone mode removed: `backbone`, `grow`, `BackboneParams`,
  `BackboneReport`, and `model::place::place_from`. With the Ward linkage start
  it saved about 3 per cent of a run and kept the whole-tree refinement that is
  the rest; measured on Baron 5k, 10k and 25k and two synthetic sets it was below
  the full search's loglikelihood on every dataset and seed and slower up to 25k.
  SPEC section 15 has the numbers.

**Search**

- NNI keeps its rows across moves instead of settling the tree every round:
  identical trees, step 6 from 295 s to 18 s at 25k cells.
- Polytomy resolution (steps 3 and 8) does the same across sweeps: identical
  trees, step 3 from 69 s to 3.5 s at 25k.
- SPR decides whether a proposal changes any split on views of the pruned and
  regrafted trees, without building either: identical trees, step 5 from 395 s
  to 301 s at 25k.
- Full search at 25k cells, 1,009 s to about 460 s, same tree.

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
