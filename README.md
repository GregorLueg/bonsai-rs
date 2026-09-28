[![CI](https://github.com/GregorLueg/bonsai-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/bonsai-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/bonsai-rs.svg)](https://crates.io/crates/bonsai-rs)
[![docs.rs](https://img.shields.io/docsrs/bonsai-rs)](https://docs.rs/bonsai-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/bonsai-rs.svg)](https://pypi.org/project/bonsai-rs/)

# bonsai-rs

Tree representations of high-dimensional data under Brownian motion. A
clean-room Rust implementation of Bonsai: de Groot, Morillo Leonardo, Pachkov
and van Nimwegen, *Bonsai reconstructs tree representations for distortion-free
visualization and exploration of high-dimensional data*, Nature Biotechnology
2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2).

## Why

Every node carries a latent position in feature space; an edge of length `t` is
a Gaussian step of variance `t` per feature. Cells come with their own error
bars. The objective is the marginal likelihood with every internal position
integrated out: Felsenstein's pruning algorithm for continuous traits.

- **Distances hold at every scale**, not just locally. UMAP and tSNE don't
  manage that; PHATE only partly.
- **It consumes uncertainty.** Per-cell per-feature error bars go in and get
  marginalised over.
- **Nothing to tune** until the picture matches your prior.

The catch: it assumes the data is a tree. Cell cycle or dose response still
gets one, and a confident-looking one.

## Input

Means **and** standard deviations per cell per feature, plus a per-feature
variance. Without real error bars it's a different and worse method; the
paper's supplement shows accuracy dropping hard on conventional preprocessing.
For scRNA-seq: Sanity on raw UMI counts, then this.

## Usage

```rust
use bonsai_rs::bonsai::bonsai;
use bonsai_rs::prelude::Verbosity;
use bonsai_rs::tree::newick::write_newick;

// means and sds are row-major [cell][gene]; None estimates the per-gene variance
let out = bonsai::<f32>(&means, &sds, n_cells, n_genes, None, None, Verbosity::Normal)?;
println!("{}", write_newick(&out.tree, &cell_names)?);
```

`out.steps` holds the loglikelihood after each of the eight search steps, so a
dud run tells you which step did nothing. Tune via `BonsaiParams`; `None` takes
the documented defaults. `ingest::prepare` plus `bonsai_prepared` splits ingest
from search, for your own feature selection or several parameter settings on
one ingest.

`Verbosity` sets what gets printed while it runs: `Quiet` (the default) prints
nothing, `Normal` one line per step with its loglikelihood, gain and time,
`Detailed` adds progress within the merge, SPR and NNI steps.

### From raw counts

The `sanity` feature pulls in [`sanity-sc-rs`](https://github.com/GregorLueg/sanity-sc-rs)
and adds `ingest::from_sanity_output`: transpose to `[cell][gene]`, undo the
prior shrinkage (S5), drop and report ill-conditioned genes.

```toml
bonsai-rs = { version = "0.2", features = ["sanity"] }
```

```rust
use bonsai_rs::bonsai::bonsai;
use bonsai_rs::ingest::from_sanity_output;
use bonsai_rs::prelude::Verbosity;
use sanity_sc_rs::sanity;

let post = sanity::<f32>(&counts, &cell_totals, None)?;
let lik = from_sanity_output(&post, None)?;
let out = bonsai(
    &lik.means,
    &lik.sds,
    lik.n_cells,
    lik.features.len(),
    Some(&lik.variances),
    None,
    Verbosity::Quiet,
)?;
```

The `gpu` feature runs Sanity through CubeCL/wgpu; the output goes into
`from_sanity_output` unchanged:

```rust
use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use sanity_sc_rs::gpu::sanity_gpu;

let client = WgpuRuntime::client(&WgpuDevice::default());
let post = sanity_gpu::<f32, WgpuRuntime>(&counts, &cell_totals, None, &client)?;
let lik = from_sanity_output(&post, None)?;
```

It reads Sanity's log fold changes, not the log transcription quotients: S5
inverts a zero-mean prior. On 64 simulated cells by 300 genes, fold changes give
back the generating tree exactly (Robinson-Foulds 0, recovery 0.950); quotients
give 100 of 122 and 0.016.

## Python

```bash
uv pip install bonsai-rs
```

```python
import bonsai_rs as bs

res = bs.bonsai_from_counts(counts, verbose=1)  # cells x genes, dense or scipy sparse
xy = bs.layout(res.tree)
```

`verbose` is `0`, `1` or `2`, as `Quiet`, `Normal` and `Detailed` above. It
prints from Rust to the process's stdout, so in a notebook it lands in the
kernel's terminal, not the cell. It covers the tree search, not Sanity.

Docs at [gregorlueg.github.io/bonsai-rs](https://gregorlueg.github.io/bonsai-rs/).
Bindings live in `python/` and version separately.

## Performance

Against the published implementation on Sanity-preprocessed Baron pancreas,
same input, same scorer:

| cells | genes | | seconds | Robinson-Foulds | distance recovery |
|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 2.9 | 136 | 0.652 |
| 512 | 2,382 | published | 386 | 146 | 0.633 |
| 5,000 | 2,701 | bonsai-rs | 32 | 1,287 | 0.666 |
| 5,000 | 2,701 | published | 4,868 | 1,937 | 0.496 |
| 5,000 | 2,701 | published, backbone 2,048 | 2,514 | 1,899 | 0.578 |
| 5,000 | 2,701 | published, backbone 1,000 | 1,794 | 1,959 | 0.569 |
| 10,000 | 2,767 | bonsai-rs | 88 | 2,613 | 0.469 |
| 10,000 | 2,767 | published | 16,062 | 5,149 | 0.281 |
| 10,000 | 2,767 | published, backbone 2,048 | 3,529 | 3,946 | 0.368 |
| 10,000 | 2,767 | published, backbone 1,000 | 3,594 | 4,191 | 0.347 |

The published backbone mode is its fast route for large data: built on a
subset of 2,048 or 1,000 cells, the rest placed onto it. bonsai-rs is 40 to 80
times faster than that at 5,000 and 10,000 cells, 150 and 180 times faster than
the published standard run, and closer to the truth on both metrics. At 512
it's a tie. Results differ with the order SPR visits subtrees in; the 10,000
row is one random order. Timings aren't like for like (ten threads against one process);
[comparison](docs/COMPARISON.md) has the caveats and the loglikelihoods.

### Exact or approximate search

The default search is **approximate**. SPR and NNI (steps 5 and 6) are most of
the run. As the paper specifies them, every sweep revisits every subtree and
every edge. The default skips that:

- **SPR** re-proposes only subtrees within five edges of what the last sweep
  changed, and after an acceptance re-applies the rest of its chunk instead of
  proposing it again.
- **NNI** caches each edge's gain and rescores only near the last move. Same
  finished tree as exact on all thirteen datasets measured.

Full search on the headline data above, 2026-09-27:

| cells | exact | approximate | loglikelihood, exact / approximate | Robinson-Foulds | recovery |
|---|---|---|---|---|---|
| 512 | 4.9 s | 2.9 s | -527,187 / -527,187 | 136 / 136 | 0.656 / 0.652 |
| 5,000 | 107 s | 32 s | -5,558,395 / -5,557,911 | 1,324 / 1,287 | 0.538 / 0.666 |
| 10,000 | 533 s | 88 s | -11,252,144 / -11,253,220 | 2,623 / 2,613 | 0.480 / 0.469 |

One run each. SPR's subtree order alone moves results by more than these gaps
([performance](docs/PERFORMANCE.md#how-much-one-real-data-run-says)).

Want the search exactly as the paper specifies it? It's one setting away:

```rust
use bonsai_rs::bonsai::BonsaiParams;
use bonsai_rs::prelude::Verbosity;
use bonsai_rs::search::{nni::NniSearch, spr::SprSearch};

let mut params = BonsaiParams::default();
params.spr.search = SprSearch::Exact;
params.nni.search = NniSearch::Exact;
let out = bonsai::<f32>(&means, &sds, n_cells, n_genes, None, Some(params), Verbosity::Quiet)?;
```

In Python: `bs.bonsai(..., search="exact")`.

The default also starts from a Ward linkage rather than the paper's greedy merge
(SPEC 9.1). The greedy merge chains on real data, and the results are worse and
3 to 5x slower. `StartTree::GreedyMerge` (Python `start="greedy"`) brings it
back for like-for-like reproduction.

[Performance](docs/PERFORMANCE.md) has the numbers and the failures,
[design](docs/DESIGN.md) the build.

## Licence

MIT. See `LICENSE`.

The reference implementation is CC-BY-NC-4.0. A port would be Adapted Material
under section 1(a) and couldn't be MIT. So this crate is built only from the
paper and its CC-BY-4.0 Supplementary Information, via [specs](docs/SPEC.md).
[Provenance](PROVENANCE.md) has the full position; contributors, read it first.

## Citing

Cite the paper. This is an independent implementation and claims none of the
science.
