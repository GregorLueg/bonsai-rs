[![CI](https://github.com/GregorLueg/bonsai-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/bonsai-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/bonsai-rs.svg)](https://crates.io/crates/bonsai-rs)
[![docs.rs](https://img.shields.io/docsrs/bonsai-rs)](https://docs.rs/bonsai-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/bonsai-rs.svg)](https://pypi.org/project/bonsai-rs/)

# bonsai-rs

Tree representations of high-dimensional data under Brownian motion.

A clean-room Rust implementation of the Bonsai algorithm: de Groot, Morillo
Leonardo, Pachkov and van Nimwegen, *Bonsai reconstructs tree representations for
distortion-free visualization and exploration of high-dimensional data*, Nature
Biotechnology 2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2).

## What it does, and why you might want it

Every node in the tree carries a latent position in feature space. An edge of
length `t` says the child is drawn from a Gaussian centred on the parent with
variance `t` per feature. Each observed cell contributes a Gaussian measurement
likelihood with its own error bars. The objective is the marginal likelihood
with every latent internal position integrated out, which stays tractable
because it is all Gaussian and factorises over features. That integration is the
continuous-trait form of Felsenstein's pruning algorithm.

Three things follow, and they are the reason to reach for this over a manifold
embedding:

- **Distances hold at every scale.** Tree path distances match high-dimensional
  distances globally, not just locally. That is a documented failure of UMAP and
  tSNE, and PHATE only partly fixes it.
- **It consumes uncertainty.** Per-cell per-feature error bars go in and are
  marginalised over. Methods that take a matrix treat it as exact.
- **Nothing to tune.** There are no parameters to turn until the picture matches
  your prior.

The honest caveat: it assumes the data is tree-like. Cells that diverged along a
branching lineage are fine. Anything cyclic or reconverging, cell cycle or dose
response, gets a tree anyway, and a confident-looking one. That is the
UMAP-hallucinates-clusters critique relocated, not solved.

## Input contract

Means **and** standard deviations, per cell per feature, plus a per-feature
variance. Not a matrix.

That is not a detail. The model marginalises over measurement noise, so without
real error bars it is a different and worse method; the paper's own supplementary
material shows accuracy dropping hard on conventional preprocessing. For
scRNA-seq the intended chain is Sanity from raw UMI counts, then this.

## Using it

```rust
use bonsai_rs::bonsai::bonsai;
use bonsai_rs::tree::newick::write_newick;

// means and sds are row-major [cell][gene]; the per-gene variance is
// estimated from the data when you pass None.
let out = bonsai::<f32>(&means, &sds, n_cells, n_genes, None, None)?;
println!("{}", write_newick(&out.tree, &cell_names)?);
```

`out.steps` carries the loglikelihood after each of the eight search steps, so a
run that went nowhere says which step did nothing. Everything is tunable through
`BonsaiParams`, and passing `None` uses defaults that are documented one by one.
`ingest::prepare` plus `bonsai_prepared` is the same thing split in two, for
callers doing their own feature selection or reusing one ingest across several
parameter settings.

For datasets too large to search directly, `backbone::backbone` reconstructs on a
random subset and places the rest against it (SPEC 15).

### From raw counts

The `sanity` feature pulls in [`sanity-sc-rs`](https://github.com/GregorLueg/sanity-sc-rs)
and adds `ingest::from_sanity_output`, which turns a Sanity run into Bonsai's
input: transposed to `[cell][gene]`, the prior shrinkage undone (S5), and
ill-conditioned genes dropped and reported.

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

It reads Sanity's log fold changes, not the log transcription quotients. S5
inverts a zero-mean prior, so the gene mean has to stay out. Measured on 64
simulated cells by 300 genes: log fold changes give back the generating tree
exactly (Robinson-Foulds 0, distance recovery 0.950), quotients land at 100 of
a possible 122 and 0.016.

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
The bindings live in `python/` and version separately from the crate.

## How it performs

Against the published implementation on Sanity-preprocessed Baron pancreas data,
both scored the same way against the same ground truth:

| cells | genes | | seconds | Robinson-Foulds | distance recovery |
|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 4 | 147 | 0.591 |
| 512 | 2,382 | published | 386 | 146 | 0.633 |
| 5,000 | 2,701 | bonsai-rs | 101 | 1288 | 0.666 |
| 5,000 | 2,701 | published | 4,868 | 1937 | 0.496 |
| 10,000 | 2,767 | bonsai-rs | 326 | 2624 | 0.476 |
| 10,000 | 2,767 | published | 16,062 | 5149 | 0.281 |

48x and 49x faster at the two larger sizes, and closer to the generating tree on
both metrics at both. At 512 cells the two land on the same topology quality and
the published implementation is mildly ahead on distance recovery, in about a
hundredth of the time. The two are not timed alike and do not get the same
hardware; `docs/COMPARISON.md` has the full tables from 2026-09-13, the figures
and the caveats.

These are `BonsaiParams::default()` as of 2026-09-24. Two defaults depart from
the paper, and the paper's version of each is one setting away:

- The start is a Ward linkage rather than the greedy merge of SPEC.md section
  9.1. `StartTree::GreedyMerge` is the specified start.
- SPR revisits only the neighbourhood of the previous sweep's moves rather than
  every subtree every sweep. `SprSearch::Exact` is the specified search, and
  lands within a few nats of the default on every dataset measured.

`docs/PERFORMANCE.md` has the numbers behind both, how it got this fast and the
things that did not work. `docs/DESIGN.md` says how it is built.

## Licence

MIT. See `LICENSE`.

The reference implementation is CC-BY-NC-4.0, which cannot produce an MIT crate:
a port is a translation, and translations are Adapted Material under section 1(a)
of that licence. This crate is therefore built from the paper and its CC-BY-4.0
Supplementary Information only, which are separately licensed from the code and
free to implement. Copyright covers expression, not algorithms.

`docs/SPEC.md` is the transcription and is the only thing the implementation
reads. `PROVENANCE.md` records the position in full, including what was read,
when, and by whom. Contributors: read it before opening the editor.

## Citing

Cite the paper. This crate is an independent implementation of their method and
claims none of the science.
