# News

## 0.2.1

Vendors `bonsai-rs` 0.2.1 and `sanity-sc-rs` 0.2.1.

- Sanity runs silently again. `sanity-sc-rs` 0.2 prints progress by default,
  and `verbose` covers the tree search only, as documented.

## 0.2.0

Vendors `bonsai-rs` 0.2.0.

- **Breaking:** `backbone` removed. It lost to the plain search on every dataset
  measured; see the core crate's changelog.
- `BonsaiResult.plot()` draws the radial tree; matplotlib comes with the `plot`
  extra.
- The search is faster, and SPR is close to linear in the cell count. Trees
  differ from 0.1 within the run-to-run spread; see the core crate's
  changelog.

## 0.1.1

Vendors `bonsai-rs` 0.1.1 and `sanity-sc-rs` 0.1.0.

- `gpu=True` on `sanity` and `bonsai_from_counts` runs Sanity on the GPU
  through wgpu, in `float32`; `gpu_available()` says whether one can be
  reached. Asking for it where none is raises rather than falling back. The
  tree search stays on the CPU.
- `search="approximate"` (the default) or `"exact"` on `bonsai`,
  `bonsai_from_counts` and `backbone`. Exact runs SPR and NNI as the paper
  specifies them; approximate revisits only what the last moves touched.
- Feature selection defaults to the paper's signal-to-noise threshold of 1,
  up from 0.25, so a default run sees the gene panel the published method
  would. Trees on the same input change accordingly.
- The search is faster: see the core crate's changelog.

## 0.1.0

First release, vendoring `bonsai-rs` 0.1.0 and `sanity-sc-rs` 0.0.1.

- `bonsai_from_counts`: raw UMI counts to tree, Sanity included, without the
  posteriors leaving Rust. Dense numpy or any scipy sparse input.
- `sanity`, `from_sanity` and `bonsai` for the same chain one step at a time,
  or for means and error bars from elsewhere. `backbone` for large datasets.
- Tree helpers: Newick in and out, equal-daylight, equal-angle and dendrogram
  layouts with an optional hyperbolic projection, clustering, path distances
  and Robinson-Foulds.
- `datasets.simulate` and `datasets.simulate_counts`, both on a known tree.
