# Changelog

## 0.1.0

First release. Clean-room, built from the paper and its CC-BY-4.0
Supplementary Information; see `PROVENANCE.md`.

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
  zero-length edges. `docs/DESIGN.md` says why the order is fixed.
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
