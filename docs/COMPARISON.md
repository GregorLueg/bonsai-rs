# Against the published implementation

Black-box comparison: same Sanity-preprocessed input to both, scored the same
way against a ground truth neither produced.

Every claim here rests on a measurement of inputs and outputs, or on the paper.
None rests on the published source, and none ever will;
[provenance](../PROVENANCE.md) says why.

## Setup

The harness lives outside this repository. It builds the input once and hands
the identical cells and genes to both. Reference trees are generated on demand,
never committed, on a private machine (which keeps NonCommercial out of it).

Data: Baron pancreas (GSE84133), Sanity on raw UMIs, same gene selection for
both. `512 s32` is the same 512 cells at another seed with a harder gene panel.

## Speed and quality

| cells | genes | method | seconds | Robinson-Foulds | distance recovery | loglikelihood, as given | loglikelihood, refit |
|---|---|---|---|---|---|---|---|
| 512 | 2,382 | bonsai-rs | 2.9 | 136 | 0.652 | -527,187 | -527,187 |
| 512 | 2,382 | published | 386 | 146 | 0.633 | -527,255 | -527,248 |
| 512 | 2,382 | truth | | 0 | 0.733 | | -528,223 |
| 512 s32 | 2,302 | bonsai-rs | 3.4 | 222 | 0.319 | -513,568 | -513,568 |
| 512 s32 | 2,302 | published | 368 | 240 | 0.303 | -513,553 | -513,536 |
| 512 s32 | 2,302 | truth | | 0 | 0.500 | | -515,853 |
| 5,000 | 2,701 | bonsai-rs | 32 | 1,287 | 0.666 | -5,557,911 | -5,557,911 |
| 5,000 | 2,701 | published | 4,868 | 1,937 | 0.496 | -5,563,770 | -5,561,535 |
| 5,000 | 2,701 | published, backbone 2,048 | 2,514 | 1,899 | 0.578 | -5,571,002 | -5,565,342 |
| 5,000 | 2,701 | published, backbone 1,000 | 1,794 | 1,959 | 0.569 | -5,572,684 | -5,566,018 |
| 5,000 | 2,701 | truth | | 0 | 0.679 | | -5,571,449 |
| 10,000 | 2,767 | bonsai-rs | 72 | 2,618 | 0.409 | -11,253,736 | -11,253,736 |
| 10,000 | 2,767 | published | 16,062 | 5,149 | 0.281 | -11,280,721 | -11,274,900 |
| 10,000 | 2,767 | published, backbone 2,048 | 3,529 | 3,946 | 0.368 | -11,280,324 | -11,264,651 |
| 10,000 | 2,767 | published, backbone 1,000 | 3,594 | 4,191 | 0.347 | -11,284,311 | -11,268,274 |
| 10,000 | 2,767 | truth | | 0 | 0.461 | | -11,282,910 |

bonsai-rs is `BonsaiParams::default()` in 0.2.0, 2026-09-27, ten threads. Published is
its standard run on one core; the published backbone is its backbone-based mode
with `n_initial_cells` 2,048 or 1,000 and `growth_factor_guide` 10, one
process. Its default of 10,000 initial cells needs more cells than these sets
have.

Robinson-Foulds is to the generating tree, lower is better. Distance recovery
correlates tree path distance with true squared Euclidean distance (the paper's
Fig. S8), higher is better; the truth row is the ceiling for Robinson-Foulds,
and recovery can pass it (last bullet).

**Two loglikelihood columns, both from our scorer on the same data**, so the
dropped constants are the same and the numbers line up. What differs is the
branch lengths:

- *As given* scores each tree with its own branch lengths. The published
  implementation fitted its lengths to its own preprocessing, so for its trees
  this column mixes the topology with that mismatch.
- *Refit* first optimises the branch lengths of each topology on this data with
  our global solve (search steps 4 and 7), then scores. That compares the
  topologies alone, and it is the column to read. bonsai-rs doesn't move,
  because its lengths already are that optimum. The truth has no as-given value:
  its lengths are expected displacements in simulation units.

Higher is better, but the truth is no ceiling here. A maximum-likelihood tree
fits the noise that was realised, so both searches score above the true
topology. The loglikelihood is what the search optimises; Robinson-Foulds and
recovery are what it's for.

- **512 is a tie.** 136 splits against 146, recovery 0.652 against 0.633, 61
  nats ahead after the refit. The replicate flips it: 222 against 240 and 0.319
  against 0.303, but 32 nats behind.
- **From 5,000 bonsai-rs pulls ahead**, and at 10,000 by a lot: 2,618 against
  5,149 splits, 21,000 nats after the refit, and 220 times faster.
- **The refit matters for the published trees.** It recovers 2,200 nats at 5,000
  and 5,800 at 10,000; 15,700 for the 10,000 backbone. The as-given column
  overstates bonsai-rs's lead by 62 per cent at 5,000 and 28 at 10,000.
- **The published backbone pays off in the published implementation**: 2 to 4.5
  times faster than its standard run, and at 10,000 better on splits and on the
  refit loglikelihood. It is still 50 to 80 times slower than bonsai-rs's
  standard run at 5,000 and 10,000, and behind it on every column.
- **The two trees differ from each other** by 26, 110, 1,498 and 4,235 splits.
  Equal scores against the truth don't mean the same tree.
- **The published implementation is deterministic.** A 2026-09-14 rerun gave
  byte-identical Newick at all four sizes. Wall times moved under half a per
  cent where the first run had an idle machine, 30 per cent at 512 and 8 at
  10,000 where it hadn't.
- **Results differ with SPR's subtree order.** At 10,000 cells recovery lands
  anywhere from 0.36 to 0.49 across orders
  ([performance](PERFORMANCE.md#how-much-one-real-data-run-says)).
  Some orders pass the truth's 0.461. That's possible because the truth's branch
  lengths are expected displacements and ours are fitted to what was realised.

## Counts to tree

Raw UMIs to finished tree on **one node**: ten-core M1 Max, 64 GB, each run
alone. The Rust route is 0.2.0, 2026-09-29; the original route is from
2026-09-25. The published implementation can run MPI across nodes; nothing here
speaks to a cluster.

Data: the harness's simulated Baron counts, 17,499 genes. Original route: the
Sanity binary (10 threads, `-v_m MAP`), the harness's selection at `S >= 1`,
then published Bonsai on one core and under MPI with 10 ranks. Rust route,
driven through the Python bindings: genes with no counts dropped (the Sanity
binary drops them itself), `sanity-sc-rs` on CPU or GPU in `f32`, the S5
conversion, `bonsai()` at defaults (also `S >= 1`). Rust Sanity's default
`marginalise` variance rule does more work per gene than `MAP`.

Sanity, seconds:

| cells | original | Rust, CPU | Rust, GPU |
|---|---|---|---|
| 512 | 71.9 | 31.0 | 0.5 |
| 512 s32 | 72.1 | 30.6 | 0.4 |
| 5,000 | 766.9 | 286.0 | 3.6 |
| 10,000 | 1,604.7 | 560.2 | 6.8 |

Bonsai, seconds:

| cells | published, 1 core | published, 10 ranks | Rust, exact search | Rust, approximate search |
|---|---|---|---|---|
| 512 | 371.7 | 170.2 | 4.4 | 2.1 |
| 512 s32 | 371.0 | | 4.9 | 2.4 |
| 5,000 | 4,868 | 3,208.6 | 86.7 | 29.5 |
| 10,000 | 16,062 | 9,439.3 | 417.2 | 71.8 |

Exact is SPR and NNI as the paper specifies them; approximate is the default
(see the README). Rust search times are on CPU-Sanity input. On GPU-Sanity
input, exact took 82.2 s and 406.7 s at 5,000 and 10,000, approximate 28.8 s
and 75.5 s. Published single-core times at 5,000 and 10,000 are the harness's,
reproduced within half a per cent on 2026-09-14; the rest are reruns. The
original route also spends 2.0, 0.8, 29.7 and 66.5 s on gene selection, and the
Rust route 0.1 to 2.0 s on the S5 conversion, both included below.

End to end, seconds:

| cells | original, 1 core | original, 10 ranks | Rust CPU + exact | Rust GPU + approximate | speed-up over the original on 1 core / 10 ranks |
|---|---|---|---|---|---|
| 512 | 445.6 | 244.1 | 35.5 | 2.7 | 166x / 91x |
| 512 s32 | 443.9 | | 35.5 | 2.7 | 162x / |
| 5,000 | 5,664.2 | 4,005.2 | 373.9 | 33.4 | 170x / 120x |
| 10,000 | 17,733.2 | 11,110.5 | 979.5 | 83.9 | 211x / 132x |

Quality against the generating tree. Recovery here uses all 17,499 genes' true
positions, not the selected genes of the headline table:

| cells | original, 1 core | original, 10 ranks | Rust CPU + exact | Rust GPU + approximate |
|---|---|---|---|---|
| 512 | RF 146, 0.492 | RF 141, 0.496 | RF 137, 0.632 | RF 138, 0.631 |
| 512 s32 | RF 240, 0.165 | | RF 241, 0.233 | RF 243, 0.222 |
| 5,000 | RF 1,937, 0.352 | RF 2,016, 0.261 | RF 1,278, 0.573 | RF 1,292, 0.449 |
| 10,000 | RF 5,149, 0.162 | RF 5,151, 0.226 | RF 2,636, 0.353 | RF 2,669, 0.265 |

Published memory, peak RSS summed over processes: 543 MB, 2,965 MB, 5,686 MB at
512, 5,000, 10,000 on one core; 3,009 MB, 10,033 MB, 18,737 MB with 10 ranks.
That's 5.5, 3.4 and 3.3x the memory for 2.2, 1.5 and 1.7x the speed, one Python
process per rank. MPI also changes its answer: RF 141 against 146 at 512, 2,016
against 1,937 at 5,000, 5,151 against 5,149 at 10,000.

Two cautions. Differences between the Rust Sanity paths and search modes sit
inside the run-to-run spread ([performance](PERFORMANCE.md#how-much-one-real-data-run-says)).
GPU input with approximate search is lowest at 5,000 and 10,000, 0.449 and
0.265, but the same GPU input with exact search gives 0.570 and 0.384, and CPU
input with approximate search 0.571 and 0.354. That's a basin, not the GPU
making worse trees. The gaps to the published trees, over 600 splits from 5,000
up, are far outside it.

Same trees drawn as in [What the trees look like](#what-the-trees-look-like):
clade fragments (ideal ten) and recovery on the selected genes, same 20,000
pairs for every tree.

| cells | GPU + approximate | CPU + exact | reference, 1 core | reference, 10 ranks |
|---|---|---|---|---|
| 512 | 13, 0.744 | 13, 0.745 | 15, 0.638 | 14, 0.643 |
| 512 s32 | 18, 0.364 | 17, 0.374 | 19, 0.311 | |
| 5,000 | 23, 0.591 | 24, 0.674 | 29, 0.500 | 36, 0.402 |
| 10,000 | 25, 0.360 | 24, 0.444 | 40, 0.258 | 58, 0.323 |

![radial layouts at 10,000 cells, counts to tree](figures/n10000_e2e_tree_layout.png)

![distance recovery at 10,000 cells, counts to tree](figures/n10000_e2e_distance_recovery.png)

## What the trees look like

Measured on the 0.2.0 trees of the [headline table](#speed-and-quality),
2026-09-29.

![radial layouts at 10,000 cells](figures/n10000_tree_layout.png)

Equal-angle radial layout, radius is branch length from the root. Leaves are
coloured by cutting the *true* tree into ten clades, so a colour marks the same
cells in every panel.

**Clade fragmentation** counts same-colour runs around the circle, ideal ten. An
intact clade is one run; a broken one is several. Cheap, and it catches things
neither Robinson-Foulds nor the loglikelihood does.

| config | bonsai-rs | published |
|---|---|---|
| 512 | 15 | 15 |
| 512 s32 | **16** | 19 |
| 5,000 | **21** | 29 |
| 10,000 | **26** | 40 |

A tie at 512, then bonsai-rs wins, including on the replicate where it's behind
on the loglikelihood. The gap grows to fourteen fragments at 10,000.

![distance recovery at 10,000 cells](figures/n10000_distance_recovery.png)

Path distance against true squared Euclidean distance, the same 20,000 random
pairs in both panels. Same shape in both: tight near-linear at small distances,
plateau at large ones (path distance is bounded by the tree's diameter). The
correlation gap is a wider spread everywhere, not a few stray pairs.

## Degenerate branches

![degenerate regions at 10,000 cells](figures/n10000_degenerate_regions.png)

| tree | zero-length leaf edges | zero-length internal edges | polytomies |
|---|---|---|---|
| bonsai-rs 512 | 0 | 0 | 2 |
| bonsai-rs 512 s32 | 9 | 0 | 3 |
| bonsai-rs 5,000 | 51 | 0 | 18 |
| bonsai-rs 10,000 | 119 | 0 | 35 |
| published 512 | 0 | 0 | 3 |
| published 512 s32 | 5 | 0 | 7 |
| published 5,000 | 40 | 0 | 107 |
| published 10,000 | 99 | 0 | 335 |

Two different things; don't conflate them.

**Zero-length leaf edges** are a SPEC 6 boundary optimum: no evidence separating
the cell from its parent. Intended, and about input noise, not topology. The
generating tree has none. These cells are noisier: median of each cell's mean
SD 1.35 against 1.14 at 10,000, and 31 per cent of them in the noisiest decile.

**Zero-length internal edges and polytomies** come from SPR and NNI splices. At
10,000 there are 72 internal ones after step 5, 57 after step 6 and 31 after
step 7. Step 8 collapses them, and collapses once more after its closing branch
solve, which can land an edge on exactly zero again (it did once here). So none
survive at any size. Leaf edges step 8 leaves alone by design.

So the difference is in the leaf edges and in how much tree sits under a
multifurcation: 34.9 per cent degenerate leaves at 10,000 for the published
tree against 3.7, a largest multifurcation of 513 leaves against 187, and 335
polytomies against 35.

## Fans and ladders

Both implementations hit groups of cells the data can't order: the evidence
separating them is weaker than the noise. **How they write that down is the
biggest reason the pictures look different.**

The reference draws a **fan**: the whole group as siblings off one node. At
10,000 cells, 335 multifurcations, the largest 513 leaves.

bonsai-rs draws a **ladder**: an order anyway, each cell on its own small,
non-zero branch. 35 multifurcations, the largest 187.

In a radial layout, radius is the sum of branch lengths above a leaf. A fan adds
nothing and lands as a blob. A ladder adds up (28 hops, median radius 0.958
against the tree's typical 0.627) inside one thin angular slice. Long and thin
is a spike. The reference's picture is tidier because it flattened those cells,
not because it placed them better.

**The ladder is more faithful, most of all exactly where a fan looks
defensible.** Pearson correlation of true squared distance with path distance
at 10,000 cells (`faithfulness.py` in the harness). The subset rows use every
pair; the first samples 200,000, which is why it isn't the headline table's
0.409.

| pairs drawn from | cells | in bonsai-rs's tree | in the reference's tree |
|---|---|---|---|
| all cells | 10,000 | 0.412 | 0.278 |
| the cells bonsai-rs strings into its longest ladder | 250 | 0.850 | 0.811 |
| the cells the reference fans into its largest multifurcation | 513 | 0.109 | 0.004 |

- **A multifurcation isn't declining to answer.** It claims every member is
  equidistant from every other, and carries no distance information: 0.004 is
  zero. The ladder still gets 0.109 on the same cells.
- **The ladder's order isn't noise.** 0.850 on the spike's 250 cells, above its
  own whole-tree 0.412 and the reference's 0.811 there. An arbitrary order would
  score near zero, like the fan.
- **So the spike is a feature** for anyone reading distances off the picture.
  Flattening it would force a fan where the branch solve found positive optima.

Limits: path distance saturates at large true distances in both trees, and at
10,000 cells the generating tree only scores 0.461, so there's little global
headroom either way.

## The arm

The spike, taken as the 2.5 per cent of cells furthest from the root: 250 at
10,000, 125 at 5,000. Median 28 hops against 17 at 10,000 and 28 against 15 at
5,000; radius 0.958 against 0.627 and 1.045 against 0.609.

- **At 10,000 it's mostly one true clade**: 231 of its 250 cells sit in one of
  the generating tree's ten top-level clades. At 5,000 it isn't: it spans all
  ten, none holding more than 34 of 125. So at 5,000 the spike groups cells the
  truth doesn't; at 10,000 it's one clade drawn long.
- **The reference half agrees.** 128 of the 250 sit in one reference clade of
  393 leaves at 10,000; 77 of 125 in one of 114 at 5,000.
- **Zero-length leaf edges** gather in it at 5,000 (35 of 51) but not at 10,000
  (1 of 119). 82 cells have a zero-length leaf edge in both trees at 10,000.
- **Not the noisiest cells.** Median of each cell's mean SD 1.12 against 1.14 at
  10,000 and 1.15 against 1.12 at 5,000; 12.8 and 12.0 per cent in the noisiest
  decile against 9.9.

## Honest rendering

These figures never jitter coincident points apart. A leaf at zero distance from
its parent drawn anywhere else manufactures structure the model said it couldn't
find. Overlapping markers at equal radius are zero-length leaf edges, not a
glitch, and seven times as many of the published tree's leaves are affected.

The same goes the other way for the spike. It's ugly and someone will ask to
collapse it. That would replace measured positive branch lengths with a
multifurcation, which carries no distance information. Tidying the picture would
delete the part that's true.

## Reproducing

`reference/comparison/` has the harness: the Baron count simulator, Sanity and
gene selection, the scorer, and every figure and table script here.
`faithfulness.py` makes the recovery table, `degeneracy.py` the zero-length
counts, `figures.py` the layouts, scatters and fragmentation. Its README has
the commands.

What it leaves out is running the published implementation. Get a tree from it
by its own documentation on the same Sanity output, save it as `theirs.nwk` in
the configuration, and `harness score` puts it next to ours.
