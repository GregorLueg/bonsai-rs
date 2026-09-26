# News

## 0.2.0

Vendors `bonsai-rs` 0.2.0.

- **Breaking:** `backbone` removed. It lost to the plain search on every dataset
  measured; see the core crate's changelog.
- The search is faster with the same trees: about half the time at 25,000 cells,
  from NNI, polytomy resolution and SPR no longer re-settling the whole tree.

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
