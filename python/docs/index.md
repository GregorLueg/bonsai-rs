# bonsai-rs

Tree representations of single-cell data under Brownian motion. The
[Rust crate](https://github.com/GregorLueg/bonsai-rs) does the work; this is a
thin layer over it.

A clean-room implementation of Bonsai: de Groot, Morillo Leonardo, Pachkov and
van Nimwegen, *Bonsai reconstructs tree representations for distortion-free
visualization and exploration of high-dimensional data*, Nature Biotechnology
2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2).
Counts go through a clean-room Sanity first
([sanity-sc-rs](https://github.com/GregorLueg/sanity-sc-rs)): UMIs in, tree out,
no R or C.

## Why a tree

Every node has a latent position, every edge is a Brownian step, and internal
positions are integrated out.

- **Distances hold at every scale**, not just locally. That's where UMAP and
  tSNE fall over.
- **It consumes uncertainty.** Per-cell per-feature error bars go in and get
  marginalised over.
- **Nothing to tune.** No perplexity, no neighbours, no minimum distance.

The catch: it assumes a tree. Cell cycle, dose response, anything that loops
back still gets one, and a confident-looking one.

## Install

```bash
uv pip install bonsai-rs            # numpy only; dense counts
uv pip install "bonsai-rs[sparse]"  # adds scipy, for sparse counts
```

Wheels for Python 3.10+ on Linux x86_64 and macOS.

## Thirty seconds

```python
import bonsai_rs as bs

sim = bs.datasets.simulate_counts(256, 500, seed=0)
res = bs.bonsai_from_counts(sim.counts, cell_totals=sim.cell_totals)

bs.robinson_foulds(res.tree, sim.tree)  # 0: the generating topology, exactly
xy = bs.layout(res.tree)  # (n_nodes, 2), ready to draw
newick = bs.to_newick(res.tree)
```

## Input

Means **and** standard deviations per cell per feature. Without real error bars
it's a worse method; the paper's supplement shows accuracy dropping hard on
conventional preprocessing. For scRNA-seq: raw UMI counts into
`bonsai_from_counts`. Log-normalised input is refused, since Sanity models the
Poisson sampling in raw counts.

## Next

- [Quickstart](quickstart.md): counts to tree, means to tree, drawing it.
- [Guide](guide.md): the Sanity handover, units, determinism, threads, big data.
- [API reference](api/core.md): every parameter and what each `None` becomes.
- [Changelog](https://github.com/GregorLueg/bonsai-rs/blob/main/python/CHANGELOG.md),
  versioned separately from the crate.

Cite the paper. This is an independent implementation and claims none of the
science.
