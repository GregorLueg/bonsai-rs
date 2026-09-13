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

## How it performs

Against the published implementation on Sanity-preprocessed Baron pancreas data,
both scored the same way against the same ground truth:

| cells | genes | | seconds | Robinson-Foulds | distance recovery |
|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 5 | 147 | 0.591 |
| 512 | 2,382 | published | 554 | 146 | 0.633 |
| 5,000 | 2,701 | bonsai-rs | 169 | 1293 | 0.666 |
| 5,000 | 2,701 | published | 4,880 | 1937 | 0.496 |
| 10,000 | 2,767 | bonsai-rs | 630 | 2632 | 0.466 |
| 10,000 | 2,767 | published | 17,389 | 5149 | 0.281 |

29x and 28x faster at the two larger sizes, and closer to the generating tree on
both metrics at both. At 512 cells the two land on the same topology quality and
the published implementation is mildly ahead on distance recovery, in a hundredth
of the time. The two are not timed alike and do not get the same hardware;
`docs/COMPARISON.md` has the full tables, the figures and the caveats.

These are `BonsaiParams::default()`, which starts from a Ward linkage rather than
the greedy merge of SPEC.md section 9.1. The specified start is still there as
`StartTree::GreedyMerge` and is what to use for a like-for-like reproduction of
the published method; `docs/PERFORMANCE.md` has both sets of numbers and why the
default is what it is.

`docs/PERFORMANCE.md` says how it got there, including the things that did not
work. `docs/DESIGN.md` says how it is built.

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
