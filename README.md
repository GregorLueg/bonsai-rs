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
use bonsai_rs::tree::newick::write_newick;

// means and sds are row-major [cell][gene]; None estimates the per-gene variance
let out = bonsai::<f32>(&means, &sds, n_cells, n_genes, None, None)?;
println!("{}", write_newick(&out.tree, &cell_names)?);
```

`out.steps` holds the loglikelihood after each of the eight search steps, so a
dud run tells you which step did nothing. Tune via `BonsaiParams`; `None` takes
the documented defaults. `ingest::prepare` plus `bonsai_prepared` splits ingest
from search, for your own feature selection or several parameter settings on
one ingest.

### From raw counts

The `sanity` feature pulls in [`sanity-sc-rs`](https://github.com/GregorLueg/sanity-sc-rs)
and adds `ingest::from_sanity_output`: transpose to `[cell][gene]`, undo the
prior shrinkage (S5), drop and report ill-conditioned genes.

```toml
bonsai-rs = { version = "0.1", features = ["sanity"] }
```

```rust
use bonsai_rs::bonsai::bonsai;
use bonsai_rs::ingest::from_sanity_output;
use sanity_sc_rs::sanity;

let post = sanity::<f32>(&counts, &cell_totals, None)?;
let lik = from_sanity_output(&post, None)?;
let out = bonsai(&lik.means, &lik.sds, lik.n_cells, lik.features.len(), Some(&lik.variances), None)?;
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

res = bs.bonsai_from_counts(counts)  # cells x genes, dense or scipy sparse
xy = bs.layout(res.tree)
```

Docs at [gregorlueg.github.io/bonsai-rs](https://gregorlueg.github.io/bonsai-rs/).
Bindings live in `python/` and version separately.

## Performance

Against the published implementation on Sanity-preprocessed Baron pancreas,
same input, same scorer:

| cells | genes | | seconds | Robinson-Foulds | distance recovery |
|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 3.5 | 147 | 0.591 |
| 512 | 2,382 | published | 386 | 146 | 0.633 |
| 5,000 | 2,701 | bonsai-rs | 70 | 1288 | 0.666 |
| 5,000 | 2,701 | published | 4,868 | 1937 | 0.496 |
| 10,000 | 2,767 | bonsai-rs | 210 | 2624 | 0.476 |
| 10,000 | 2,767 | published | 16,062 | 5149 | 0.281 |

69x and 76x faster at the larger sizes and closer to the truth on both metrics.
At 512 it's a tie on topology and the published one is ahead on recovery.
Timings aren't like for like; [comparison](docs/COMPARISON.md) has the caveats.

### Exact or approximate search

The default search is **approximate**. SPR and NNI (steps 5 and 6) are most of
the run. As the paper specifies them, every sweep revisits every subtree and
every edge. The default skips that:

- **SPR** re-proposes only subtrees within five edges of what the last sweep
  changed. Within a few nats of exact on every dataset measured.
- **NNI** caches each edge's gain and rescores only near the last move. Same
  finished tree as exact on all thirteen datasets measured.

Search time on the same input (counts to tree, see
[comparison](docs/COMPARISON.md)):

| cells | exact | approximate | Robinson-Foulds, exact / approximate |
|---|---|---|---|
| 512 | 4.2 s | 3.5 s | 136 / 137 |
| 5,000 | 117.2 s | 60.4 s | 1,285 / 1,277 |
| 10,000 | 631.0 s | 177.1 s | 2,607 / 2,627 |

The quality differences are inside the run-to-run spread on real data.

Want the search exactly as the paper specifies it? It's one setting away:

```rust
use bonsai_rs::bonsai::BonsaiParams;
use bonsai_rs::search::{nni::NniSearch, spr::SprSearch};

let mut params = BonsaiParams::default();
params.spr.search = SprSearch::Exact;
params.nni.search = NniSearch::Exact;
let out = bonsai::<f32>(&means, &sds, n_cells, n_genes, None, Some(params))?;
```

In Python: `bs.bonsai(..., search="exact")`.

The default also starts from a Ward linkage rather than the paper's greedy merge
(SPEC 9.1). The greedy merge chains on real data, and the results are worse and
3 to 5x slower. `StartTree::GreedyMerge` (Python `start="greedy"`) brings it
back for like-for-like reproduction.

[Performance](docs/PERFORMANCE.md) has the numbers and the failures, [design](docs/DESIGN.md) the
build.

## Licence

MIT. See `LICENSE`.

The reference implementation is CC-BY-NC-4.0. A port would be Adapted Material
under section 1(a) and couldn't be MIT. So this crate is built only from the
paper and its CC-BY-4.0 Supplementary Information, via [specs](docs/SPEC.md).
[Provenance](PROVENANCE.md) has the full position; contributors, read it first.

## Citing

Cite the paper. This is an independent implementation and claims none of the
science.
