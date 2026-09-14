# Against the published implementation

A black-box comparison. Both implementations were run on the same
Sanity-preprocessed input and scored the same way against a ground truth neither
produced.

**What may and may not be said here.** Every claim below is anchored to either a
measurement of inputs and outputs, or to a statement in the paper. Nothing here
is anchored to the published source. `PROVENANCE.md` sets out why, and why that
does not change with time. Running a program to observe its output is black-box
observation. Explaining why its code behaves as it does is not, and there is no
version of this crate at which that becomes acceptable.

## The arrangement

The harness lives outside this repository and is not part of the crate. It
generates the input once and hands the identical Sanity-shaped subset to both
implementations, so both see exactly the same cells and the same gene set.
Reference trees are generated on demand, never committed, and were generated on
a private machine, which is what keeps the NonCommercial clause out of it.

Data is Baron pancreas (GSE84133), Sanity from raw UMI counts, then the gene
selection applied identically to both. The `512_s32` configuration is the same
512 cells at a different seed and a harder gene panel.

## Speed and quality


| cells | genes | method | seconds | peak RSS | Robinson-Foulds | distance recovery | loglikelihood |
|---|---|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 4.7 | | 147 | 0.591 | -527,499 |
| 512 | 2,382 | published | 386 | 566 MB | 146 | 0.633 | -527,255 |
| 512 | 2,382 | truth | | | 0 | 0.733 | |
| 512 s32 | 2,302 | bonsai-rs | 6.1 | | 220 | 0.318 | -513,567 |
| 512 s32 | 2,302 | published | 368 | 551 MB | 240 | 0.303 | -513,553 |
| 512 s32 | 2,302 | truth | | | 0 | 0.500 | |
| 5,000 | 2,701 | bonsai-rs | 166 | | 1,293 | 0.666 | -5,557,975 |
| 5,000 | 2,701 | published | 4,868 | 2,965 MB | 1,937 | 0.496 | -5,563,770 |
| 5,000 | 2,701 | truth | | | 0 | 0.679 | |
| 10,000 | 2,767 | bonsai-rs | 611 | | 2,632 | 0.466 | -11,253,188 |
| 10,000 | 2,767 | published | 16,062 | 5,686 MB | 5,149 | 0.281 | -11,280,721 |
| 10,000 | 2,767 | truth | | | 0 | 0.461 | |

bonsai-rs rows are `BonsaiParams::default()` as of 2026-09-13, which starts from
a Ward linkage. An earlier version of this table mixed starts without saying so:
the 512 rows were the greedy merge of SPEC.md section 9.1 and the 5,000 and
10,000 rows were the linkage. `docs/PERFORMANCE.md` has both starts at every
size and why the default changed.

Robinson-Foulds is to the generating tree, so lower is better. Distance recovery
is the correlation between tree path distance and true squared Euclidean
distance, the relation the paper's Fig. S8 plots, so higher is better; the truth
row is the generating tree's own value and is the ceiling. Loglikelihoods are
each implementation's own and are not comparable across the two, since the
constants dropped at ingest differ.

**The reversal is narrower than it looked, and the 512 end is a tie.** At 512
cells the two are level on topology, 147 splits from the truth against 146, and
the published implementation is ahead on distance recovery, 0.633 against 0.591.
At the 512 replicate it goes the other way on both, 220 against 240 and 0.318
against 0.303. By 5,000 bonsai-rs is ahead on both, and at 10,000 the gap is
wide: 2,632 against 5,149 splits, and 0.466 against 0.281.

An earlier version of this section had the published implementation ahead on
both metrics at 512. That was the greedy-merge start, not the default.

Robinson-Foulds between the two reconstructions: 57 at 512, 108 at 512 s32,
1,474 at 5,000 and 4,221 at 10,000. Equal scores against the truth do not on
their own mean the same tree was found, which is why this is reported too.

**The published implementation is deterministic across runs.** Every
configuration was run a second time on 2026-09-14, on a quieter machine, and
returned a byte-identical Newick file at all four sizes. So its quality
numbers here are reproduced rather than merely recorded. Its wall times moved
by under half a per cent where the first run had been taken on an idle machine,
and by 30 per cent at 512 and 8 per cent at 10,000 where it had not, which is
the load caveat doing what it said it would.

**At 10,000 cells both are near the ceiling on distance recovery and neither is
near it on topology.** The generating tree itself scores 0.461, and bonsai-rs
scores 0.466. Scoring above the generating tree is expected rather than
suspicious: that tree's branch lengths are diffusion times, the *expected*
squared displacement, while these are fitted to what was realised.

**Memory.** Peak resident set is not recorded for bonsai-rs in this run, which
is a gap in the harness rather than a result. The published implementation's
5,686 MB at 10,000 cells is what a dense `n x n` working set costs.

## What the trees look like


![radial layouts at 10,000 cells](https://raw.githubusercontent.com/GregorLueg/bonsai-rs/main/docs/figures/n10000_tree_layout.png)

Equal-angle radial layout, radius is cumulative branch length from the root,
leaves coloured by cutting the *true* tree's topology into ten clades so the
same colour marks the same cells in every panel.

**Clade fragmentation** counts the contiguous same-colour runs around the
circle, against an ideal of ten. A clade that survives reconstruction intact
contributes one run; a clade broken up by misplaced leaves contributes several.
It is a cheap read on tree quality that neither Robinson-Foulds nor the
loglikelihood surfaces.

| config | bonsai-rs | published |
|---|---|---|
| 512 | **14** | 15 |
| 512 s32 | **17** | 19 |
| 5,000 | **22** | 29 |
| 10,000 | **26** | 40 |

Unlike Robinson-Foulds and distance recovery, this one favours bonsai-rs at
every size, including the two where the published implementation is level or
ahead on the headline metrics. The gap widens with the cell count, from one
fragment at 512 to fourteen at 10,000.

![distance recovery at 10,000 cells](https://raw.githubusercontent.com/GregorLueg/bonsai-rs/main/docs/figures/n10000_distance_recovery.png)

Tree path distance against true squared Euclidean distance, 20,000 randomly
sampled pairs, the same pairs in both panels. Both show the same qualitative
shape: a tight near-linear rise at small true distance, saturating to a plateau
at large true distance, which is expected since path distance is bounded by the
tree's diameter and squared Euclidean distance is not. The correlation gap shows
as a wider vertical spread at every true distance rather than as a handful of
stray points, so the headline numbers are a pervasive difference and not a
subset of pairs carrying the average.

## Degenerate branches

Both implementations produce trees with zero-length branches, and they produce
them in different places.

![degenerate regions at 10,000 cells](https://raw.githubusercontent.com/GregorLueg/bonsai-rs/main/docs/figures/n10000_degenerate_regions.png)

| tree | zero-length leaf edges | zero-length internal edges | polytomies |
|---|---|---|---|
| bonsai-rs 512 | 0 | 0 | 3 |
| bonsai-rs 512 s32 | 8 | 0 | 3 |
| bonsai-rs 5,000 | 51 | 0 | 17 |
| bonsai-rs 10,000 | 129 | 0 | 40 |
| published 512 | 0 | 0 | 3 |
| published 512 s32 | 5 | 0 | 7 |
| published 5,000 | 40 | 0 | 107 |
| published 10,000 | 99 | 0 | 335 |

Two different phenomena, and conflating them is a mistake worth avoiding.

**Zero-length leaf edges** are a SPEC 6 boundary optimum: the model found no
evidence separating that cell from its parent, so the branch-length optimum sits
at `t = 0`. That is intended behaviour and a statement about noise in the input
rather than about topology. The generating tree has none. They carry modestly
noisier input than the rest, median per-cell standard deviation 1.25 against
1.14 at 10,000 cells, which is the mechanism doing what it says.

**Zero-length internal edges and polytomies** come from splices during SPR and
NNI. Step 8 exists to collapse the internal ones on the finished tree, and on
these trees it does: the internal count is exactly zero at every configuration,
against 28 at 10,000 cells before the collapse was added. It does not touch the
leaf-edge half, by construction. The stage snapshots show where the internal
ones come from: zero after step 2, 32 after step 5 at 10,000 cells.

Both implementations now show an internal count of exactly zero at every
configuration, so the two differ in the leaf-edge half and in how much of the
tree sits under a multifurcation. The published implementation carries a far
larger degenerate-leaf fraction, 34.9 per cent at 10,000 cells against
bonsai-rs's 4.7, with up to 513 leaves in a single multifurcation against 253,
and 335 polytomies against 40.

## Which picture is more faithful to the data

A tree is a claim about distances. If two cells sit far apart in the original
high-dimensional space, the path between them through the tree should be long.
That is what the distance-recovery number measures, and it is the thing to
judge a layout on if the layout is meant to be read rather than admired.

Both implementations run into the same wall: groups of cells the data cannot
order. The evidence separating them is weaker than the noise. **They write that
down in two different ways, and the difference is the single biggest reason the
two pictures do not look alike.**

The reference writes it as a **fan**. It hangs the whole group off one node as
siblings, saying in effect "these are unordered, here they all are". At 10,000
cells it makes 335 multifurcations and its largest holds 513 leaves.

This crate writes it as a **ladder**. It picks an order anyway and strings the
group out one behind the next, each on its own small but non-zero branch. At
10,000 cells it makes 40 multifurcations, the largest holding 253.

That is the whole of the visual difference. In a radial layout a leaf's radius
is the sum of every branch length above it, so a fan adds nothing and lands the
group in a blob, while a ladder adds up: 28 hops of small branches sum to a
median radius of 0.967 against the tree's typical 0.621, and because a ladder is
narrow the sum accumulates inside one thin angular slice. Long and thin is what
a spike is. The reference's picture is tidier because it has flattened the same
cells, not because it placed them better.

**Measured, the ladder is the more faithful of the two, and by the widest margin
exactly where a fan looks most defensible.** Pearson correlation between true
squared Euclidean distance and tree path distance at 10,000 cells, `faithfulness.py`
in the harness. The two subset rows use every pair in the subset, so they carry
no sampling error; the first row samples 200,000 pairs, which is why it does
not match the headline table's 0.466 to three decimals: that one is the
harness's own sampler at its own seed, and the gap between them is the sampling
error on the whole-tree figure.

| pairs drawn from | cells | in bonsai-rs's tree | in the reference's tree |
|---|---|---|---|
| all cells | 10,000 | 0.470 | 0.278 |
| the cells bonsai-rs strings into its longest ladder | 250 | 0.604 | 0.576 |
| the cells the reference fans into its largest multifurcation | 513 | 0.127 | 0.004 |

Three things to take from it.

**A multifurcation is not a way of declining to answer.** It is a positive claim
that every cell in it is equidistant from every other, and it carries no distance
information whatever: 0.004 over 513 cells is zero. Over those same 513 cells
the ladder still recovers 0.127. Refusing to order a group does not preserve
what you knew about it, it discards it.

**The order the ladder imposes is not noise.** Over the 250 cells in the spike,
bonsai-rs scores 0.604, above its own whole-tree 0.470 and above the reference's
0.576 on the same cells. If the chain were an arbitrary ordering of
indistinguishable cells it would score near zero there, as the fan does. It does
not.

**So the spike is a feature for anyone reading distances off the picture.** It
is small positive branches carrying real structure, drawn by a radius that adds
them up. Flattening it would mean forcing a fan where the branch-length solve
found positive optima, which is the discard above, performed deliberately.

Two limits on all of this. Path distance saturates against squared Euclidean
distance at large true distances in both trees, which is the plateau in the
recovery figure and a property of trees rather than of either implementation.
And at 10,000 cells the generating tree itself only scores 0.461, so there is
little headroom left globally however the degenerate regions are drawn.

## The arm

**This section was measured on the 2026-09-13 trees, which the default start
change has replaced. The spike is still there and its shape is unchanged; the
per-cell claims below have not been re-derived and one of them is already known
not to carry over.** On the current 10,000-cell tree the long-radius tail is 250
leaves against the 241 reported here, at a median depth of 28 hops against 17
elsewhere and a median radius of 0.967 against 0.621, which is the same
signature. But only 12 of the tree's 129 zero-length leaf edges fall in it,
where this section reports 75 of 122. Either the zero-length concentration is
genuinely gone or the radius cut used to check it is not the clade-based
definition used below, and that has not been separated. Treat the paragraphs
that follow as describing the earlier trees until it is.

The bonsai-rs panel at 5,000 and 10,000 cells has a spike sticking out of the
main mass, and the reference's does not. It is one clade: 241 cells at 10,000,
112 at 5,000. Four measurements settle what it is.

**It is not a misplaced clade.** In the generating tree those 241 cells are
spread over seven of the ten top-level clades, 66 here, 58 there, 51, 47, and no
clade holds even a third of them. At 5,000 cells it is nine of ten. There is no
true group that was put in the wrong place, because there is no true group.

**The reference builds the same group.** 217 of the 241 arm cells, ninety per
cent, sit inside a single reference clade of 520 leaves. At 5,000 it is 94 of
112 inside one clade of 176. Both implementations pull the same cells together,
independently. That is the strongest evidence that this is a property of the
data.

**These are the cells the model declines to separate.** 75 of the arm's 241 have
a zero-length branch of their own, which is 61 per cent of every zero-length leaf
edge in the tree, against the arm being 2.4 per cent of the cells. 66 of the
reference's 99 zero-length leaves are arm cells too, and 82 cells are flagged by
both implementations.

**It is not simply the noisiest cells.** Median per-cell standard deviation is
1.195 in the arm against 1.138 elsewhere, and the noisiest decile is 12.0 per
cent of the arm against 9.9 per cent of the rest. At 5,000 cells there is no
enrichment at all, 8.9 against 10.0. What makes a cell join the arm is its mean
sitting close to its neighbours' relative to its error bars, which is not the
same thing as having the largest error bars.

**Why it draws as a spike, and why the reference's does not.** Inside the arm the
search builds a deep ladder: 31 hops from leaf to root against 17 elsewhere, each
rung a small but non-zero branch, summing to a root-to-tip length of 0.900
against 0.621. The radial layout puts radius at cumulative branch length, so the
chain accumulates outward inside one narrow angular slice. The reference ends the
same group in multifurcations instead, so nothing accumulates. Two renderings of
the same statement, that these cells cannot be told apart.

Step 8 is not the answer to it. Collapsing on the finished tree takes the
internal zero-length edges from 32 to 0 and moves the arm's median radius from
0.902 to 0.902 and its depth by one hop. The chain is made of small *positive*
edges, not zero ones, so a collapse cannot reach it by construction.

Nothing here is a defect in the search. Making the arm go away would mean either
forcing a flat multifurcation where the branch solve found positive optima, or
moving points away from where their branch lengths put them. Both invent a
picture the model did not produce.

## Reproducing

The harness is not part of this crate. What it does is run the published
implementation once per configuration on a Sanity-shaped subset that this crate's
simulation and ingest also consume, score both trees against the truth, and draw
the figures above from Newick and CSV files only. `faithfulness.py` there
produces the distance-recovery table above; `degeneracy.py` the zero-length and
polytomy counts; `figures.py` the layouts, the scatters and the clade
fragmentation.

## Honest rendering

If a leaf genuinely sits at zero distance from its parent, drawing it at its true
radius is correct and drawing it anywhere else is not, however much better the
anywhere-else version would look. These figures never jitter coincident points
apart. Jittering a zero-length branch to look separated manufactures apparent
structure the branch-length optimisation explicitly did not find, and a reader
would walk away believing the tree resolved cells the model says it could not.
That is the same fabrication problem that motivates the method in the first
place, run in the opposite direction.

Overlapping markers at equal radius are leaves with a zero-length branch, not a
rendering glitch. This matters more for the published implementation's picture
than for ours, since seven times as many of its leaves are affected.

The same principle cuts the other way for the spike, and it is worth being
explicit because the temptation runs in the opposite direction. A ladder of
small positive branches is ugly and a reader may well ask for it to be
collapsed. Collapsing it would replace measured positive branch lengths with a
flat multifurcation, and the section above shows what that costs: a
multifurcation carries no distance information at all. Tidying the picture would
mean deleting the part of it that was true.
