# bonsai-rs

Tree representations of single-cell data under Brownian motion. The
[Rust crate](https://github.com/GregorLueg/bonsai-rs) does the work; this is a
thin functional layer over it.

A clean-room implementation of Bonsai: de Groot, Morillo Leonardo, Pachkov and
van Nimwegen, *Bonsai reconstructs tree representations for distortion-free
visualization and exploration of high-dimensional data*, Nature Biotechnology
2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2).
Raw counts go through a clean-room Sanity first
([sanity-sc-rs](https://github.com/GregorLueg/sanity-sc-rs)), so UMIs in, tree
out, no R or C in the loop.

## Why a tree

Every node carries a latent position in feature space and every edge is a
Brownian step. The objective is the marginal likelihood with every internal
position integrated out. Three things follow:

- **Distances hold at every scale.** Path distances along the tree track the
  high-dimensional distances globally, not just locally, which is where UMAP
  and tSNE fall over.
- **It consumes uncertainty.** Per-cell per-feature error bars go in and are
  marginalised over. A method that takes a matrix treats it as exact.
- **Nothing to tune.** No perplexity, no neighbours, no minimum distance.

The honest caveat: it assumes the data is tree-like. Cells that diverged along
a branching lineage are fine. Cell cycle, dose response, anything that loops
back or reconverges, still gets a tree, and a confident-looking one.

## Install

```bash
uv pip install bonsai-rs            # numpy only; dense counts
uv pip install "bonsai-rs[sparse]"  # adds scipy, for sparse counts
```

Wheels target Python 3.10 and up, on Linux x86_64 and macOS.

## Thirty seconds

```python
import bonsai_rs as bs

sim = bs.datasets.simulate_counts(256, 500, seed=0)
res = bs.bonsai_from_counts(sim.counts, cell_totals=sim.cell_totals)

bs.robinson_foulds(res.tree, sim.tree)  # 0: the generating topology, exactly
xy = bs.layout(res.tree)  # (n_nodes, 2), ready to draw
newick = bs.to_newick(res.tree)
```

## Input contract

Means **and** standard deviations, per cell per feature. Not a matrix. Without
real error bars it's a different and worse method; the paper's supplement
shows accuracy dropping hard on conventional preprocessing.

For scRNA-seq that means raw UMI counts into `bonsai_from_counts`. Anything
log-normalised is refused, because Sanity models the Poisson sampling in the
raw counts.

## Where to go next

- [Quickstart](quickstart.md) for counts to tree, means to tree, and drawing
  the result.
- [Guide](guide.md) for the Sanity handover, units, determinism, large
  datasets and threads.
- [API reference](api/core.md) for every parameter and what each `None`
  resolves to.
- [Changelog](https://github.com/GregorLueg/bonsai-rs/blob/main/python/CHANGELOG.md)
  for this package. It versions separately from the Rust crate.

Cite the paper. This is an independent implementation of their method and
claims none of the science.
