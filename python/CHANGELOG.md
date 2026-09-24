# News

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
