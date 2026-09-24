# Guide

## The Sanity handover

Sanity reports posteriors: each cell's log fold change is shrunk towards the
gene mean under a `N(0, v)` prior. Bonsai wants the measurement before that
shrinkage, so `from_sanity` undoes it (SI eq. S5):

```text
mu    = x   * v / (v - eps^2)
sig^2 = eps^2 * v / (v - eps^2)
```

Two things follow.

**Feed it the log fold changes, not the log transcription quotients.** The
formula inverts a zero-mean prior. Add the gene mean first and the per-cell
amplification scales it differently in every cell, which invents structure
out of nothing. On 64 simulated cells by 300 genes, log fold changes recover the
generating tree exactly (Robinson-Foulds 0, distance recovery 0.950). The
quotients land at Robinson-Foulds 100 of a possible 122 and a distance
recovery of 0.016. `bonsai_from_counts` does the right thing; this only bites
if you wire it up by hand.

**Some genes get dropped.** The amplification `v / (v - eps^2)` blows up as a
cell's error bar approaches the gene's variance, which is what a gene with
almost no signal looks like. A gene where any cell exceeds `max_amplification`
(default 1000) is dropped whole and reported in `dropped`, rather than clamped
into a fabricated error bar. A handful in a large panel is normal. Losing most
of them says the counts are too shallow for those genes to say anything.

After the conversion, genes below a signal-to-noise floor are dropped too
(`min_signal_to_noise`, default 0.25). `features` lists what survived both.

### cell_totals

Sanity normalises by each cell's total UMI count over the whole transcriptome.
The default is the row sums of what you passed, which is right only if you
passed every gene. On a subset, pass the totals from the full matrix.

It matters more than it looks. The simulator draws 300 high-variance genes, so
the row sums carry a per-cell shift that the true library sizes do not. With
the true sizes the tree comes back exact on four seeds out of four; with row
sums, Robinson-Foulds 2, 38, 28 and 42.

## Units

Loglikelihoods drop the `2 pi` and variance terms, which do not depend on the
topology. They are meaningful up to an additive constant: compare two trees on
the same data, never two datasets, and never against the published
implementation's numbers.

`node_means` and `node_sds` are in the input's units, one row per node. Rows
`n_leaves:` are the inferred ancestors, which is the part a manifold embedding
cannot give you.

## Determinism

Same input, same tree, whatever the thread count. Parallel reductions sum in
a fixed order rather than rayon's split order. `bonsai` on the same arrays
twice gives bitwise the same loglikelihood.

## Threads

Rayon's global pool, sized to the machine. Set `RAYON_NUM_THREADS` before the
first call to cap it; the pool is built once and fixed after that.

## Large datasets

The largest run measured is 10,000 cells by 2,767 genes of Sanity-preprocessed
Baron pancreas data, which took 611 seconds; the [comparison](https://github.com/GregorLueg/bonsai-rs/blob/main/docs/COMPARISON.md)
has the details and the caveats.

Past that, `backbone` reconstructs on a random subset (`backbone_cells`,
default 2048), places every other cell onto it one at a time, and refines the
whole tree.

## The start tree

`start="linkage"` (the default) builds the initial tree by Ward linkage over a
neighbour graph. `start="greedy"` is the greedy merge the paper specifies. On
real data the greedy merge chains, and the refinement does not recover from it,
so it is there for like-for-like reproduction of the published method, not for
building the best tree.

## What it does not do

It won't tell you your data isn't a tree. A cycle gets cut somewhere and drawn
as two branches, with confident branch lengths either side. Look at the data
before you believe the picture.
