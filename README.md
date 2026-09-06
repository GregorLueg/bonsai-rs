# bonsai-rs

Tree representations of high-dimensional data under Brownian motion.

A clean-room Rust implementation of the Bonsai algorithm: de Groot, Morillo
Leonardo, Pachkov and van Nimwegen, *Bonsai reconstructs tree representations for
distortion-free visualization and exploration of high-dimensional data*, Nature
Biotechnology 2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2).

**Status: complete against the specification, not yet validated against real
data.** All seven search steps, ingest, backbone mode, Newick, 2D layouts and
per-node posteriors are implemented and tested. What has not happened is a run
on a real dataset or a comparison against the reference implementation.

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

## Performance

End to end, `bonsai()` on one M1 Max, simulated data:

| cells | features | seconds |
|---|---|---|
| 512 | 2000 | 9.1 |
| 2048 | 200 | 15.7 |
| 2048 | 2000 | 70.6 |

Overall scaling is about `n^1.5`. By step, exponents fitted over 256 to 2048
leaves: the greedy merge `n^1.65`, subtree pruning and regrafting `n^1.47`,
branch-length optimisation linear, nearest-neighbour interchange linear per
round. Polytomy resolution is still `n^2.8`; it is four per cent of the runtime
and is the last structural item.

Extrapolating `n^1.5` to thirty thousand cells by two thousand features gives
roughly an hour on one machine. That is an extrapolation over a fourteen-fold
jump, not a measurement, and the paper reports under a day on ten CPUs for the
same size. A like-for-like comparison has not been run.

Reproduce with `cargo bench --bench pipeline` on an otherwise idle machine, and
`--bench steps` for the per-step attribution.

### What the search cost before it was tuned

Same data, same trees, identical loglikelihoods at every stage, 512 cells by
2000 features:

| | seconds |
|---|---|
| exhaustive candidate scan | 1478.8 |
| with the kNN restriction and ellipsoid bounds | 37.4 |
| with the lazy SPR proposal and structural NNI filter | 9.1 |

None of that traded accuracy: every step is exact and returns byte-identical
trees, which is what the correctness gates on SPEC sections 10 and 11 exist to
guarantee.

### Where the speed actually is

The likelihood kernel, which everything else calls, at 8192 cells by 2000
features:

| | time |
|---|---|
| numpy reference, same equations | 468 ms |
| Rust, `f64` storage | 67 ms |
| Rust, `f32` storage | 40 ms |

Loglikelihoods agree with the numpy reference to twelve significant figures. The
baseline is `reference/bonsai_ref.py`, written independently from the same
specification; the published implementation has not been run.

**That is the kernel in isolation and not an end-to-end claim.** The search has
costs the kernel benchmark never touches, which is why the table above it is the
honest one to quote.

The parallel axis is the feature axis, not the tree level, because the model
factorises over features. That makes it indifferent to tree shape: a pathological
ladder tree runs in 9.0 ms where level-parallelism takes 67.7 ms.

## Reconstruction quality

Against simulated data with known ground truth, correlation between tree path
distance and true squared Euclidean distance, which is the relation the paper's
Fig. S8 plots and the property the method claims over UMAP and tSNE:

| cells | features | noise | generating tree | this crate | Robinson-Foulds |
|---|---|---|---|---|---|
| 64 | 500 | 0.1 | 0.9650 | 0.9804 | 0 |
| 256 | 500 | 0.5 | 0.9418 | 0.9527 | 0 |
| 256 | 2000 | 0.1 | 0.9817 | 0.9901 | 0 |

Topology is recovered exactly at these noise levels. Scoring above the
generating tree is expected rather than suspicious: that tree's branch lengths
are diffusion times, the *expected* squared displacement, while these are fitted
to what was realised.

Recovery improves with more features, 0.9418 to 0.9817 at 256 cells going from
500 to 2000, which is the blessing of dimensionality the paper reports in
Figs. S12 and S13.

## Citing

Cite the paper. This crate is an independent implementation of their method and
claims none of the science.
