# Changelog

## 0.1.0

First release. A clean-room implementation of the Bonsai algorithm, built from
the paper and its CC-BY-4.0 Supplementary Information; `PROVENANCE.md` sets out
the licence position.

**The model**

- Ingest: the scale transform of SPEC 3.1, feature selection by
  signal-to-noise, and conversion from Sanity posteriors to likelihood
  parameters.
- Optional `sanity` feature: `ingest::from_sanity_output` takes a
  `sanity-sc-rs` run straight to Bonsai's input, so raw UMI counts to tree
  needs nothing outside Rust.
- Pruning recursion and tree loglikelihood, row-major `[node][feature]`,
  parallel over features rather than tree levels, so the cost is indifferent to
  tree shape.
- Branch-length optimisation by safeguarded Newton on the closed-form
  stationarity condition, bracketed from above during the same pass that
  prepares the edge constants.
- Merge scoring with the constrained two-stage split solve, and node placement
  by beam search over attachment points.

**The search**

- The seven steps of SPEC section 9, plus an eighth of our own that collapses
  the internal zero-length edges steps 4 to 7 leave behind. `docs/DESIGN.md`
  says why the order is not negotiable.
- Candidate restriction by nearest neighbours (SPEC 11) and upper bounds over an
  ellipsoid (SPEC 10), with an online schedule for the ellipsoid size. Without
  them the search is cubic.
- A Ward linkage over a neighbour graph as an alternative starting tree, which
  reaches the same answer as the greedy merge for a fraction of the time.
- Subtree pruning and regrafting with lazily formed proposal rows and a
  scale-relative acceptance floor, and nearest-neighbour interchange generalised
  to polytomies with a structural move filter.
- Backbone mode (SPEC 15): reconstruct on a subset, place the rest, refine.

**Output**

- Newick read and write, equal-angle and equal-daylight radial layouts and a
  dendrogram, tree path distances, and a posterior mean and standard deviation
  for every node including the inferred ancestors.

**Numerics**

- Storage generic over `BonsaiFloat`; every reduction accumulates in `f64`.
- Deterministic: same input, same tree, whatever the thread count. Parallel
  reductions collect per-unit contributions and sum in a fixed order.
- SIMD tier for `f32` storage and for the branch-solve kernel, through `wide`.

`docs/PERFORMANCE.md` records what was optimised, what was tried and thrown
away, and the rules that came out of it. `docs/COMPARISON.md` has the black-box
comparison against the published implementation.
