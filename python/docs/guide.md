# Guide

## The Sanity handover

Sanity reports posteriors: each cell's log fold change shrunk towards the gene
mean under a `N(0, v)` prior. Bonsai wants the measurement before shrinkage, so
`from_sanity` undoes it (SI eq. S5):

```text
mu    = x   * v / (v - eps^2)
sig^2 = eps^2 * v / (v - eps^2)
```

**Feed it log fold changes, not log transcription quotients.** The formula
inverts a zero-mean prior; add the gene mean and the per-cell amplification
invents structure. On 64 simulated cells by 300 genes, fold changes recover the
generating tree exactly (Robinson-Foulds 0, recovery 0.950); quotients give 100
of a possible 122 and 0.016. `bonsai_from_counts` gets this right; it only bites
if you wire it up by hand.

**Some genes get dropped.** `v / (v - eps^2)` blows up as a cell's error bar
nears the gene's variance, i.e. a gene with almost no signal. Any gene where a
cell exceeds `max_amplification` (default 1000) is dropped whole and listed in
`dropped`, not clamped into a fake error bar. A handful is normal; losing most
means the counts are too shallow.

Genes below `min_signal_to_noise` (default 1, the paper's threshold) go next.
`features` lists the survivors.

### cell_totals

Sanity normalises by each cell's UMI total over the whole transcriptome. The
default is the row sums of what you passed, right only if you passed every gene.
On a subset, pass the full-matrix totals.

It matters. The simulator draws 300 high-variance genes, so row sums carry a
per-cell shift the true library sizes don't. True sizes: exact tree on four
seeds of four. Row sums: Robinson-Foulds 2, 38, 28 and 42.

## Units

Loglikelihoods drop the `2 pi` and variance terms, which don't depend on
topology. Compare two trees on the same data; never two datasets, never against
the published implementation.

`node_means` and `node_sds` are in input units, one row per node. Rows
`n_leaves:` are the inferred ancestors, which no manifold embedding gives you.

## Determinism

Same input, same tree, any thread count. `bonsai` on the same arrays twice gives
bitwise the same loglikelihood.

## Threads

Rayon's global pool, sized to the machine. Set `RAYON_NUM_THREADS` before the
first call to cap it; after that it's fixed.

## Large datasets

The largest run measured is 10,000 cells by 2,767 genes of Sanity-preprocessed
Baron pancreas in 210 seconds; see the [comparison](https://github.com/GregorLueg/bonsai-rs/blob/main/docs/COMPARISON.md).

Beyond that, `backbone` reconstructs on a random subset (`backbone_cells`,
default 2048), places every other cell onto it and refines the whole tree.

## The start tree

`start="linkage"` (default) is Ward linkage over a neighbour graph.
`start="greedy"` is the paper's greedy merge. On real data it chains and the
refinement doesn't recover, so use it to reproduce the published method, not to
build the best tree.

## What it does not do

Tell you your data isn't a tree. A cycle gets cut somewhere and drawn as two
branches with confident lengths. Look at the data before you believe the
picture.
