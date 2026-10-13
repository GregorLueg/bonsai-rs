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
| 512 | 2,382 | bonsai-rs | 2.4 | 136 | 0.652 | -527,187 | -527,187 |
| 512 | 2,382 | published | 386 | 146 | 0.633 | -527,255 | -527,248 |
| 512 | 2,382 | truth | | 0 | 0.733 | | -528,223 |
| 512 s32 | 2,302 | bonsai-rs | 2.8 | 222 | 0.319 | -513,568 | -513,568 |
| 512 s32 | 2,302 | published | 368 | 240 | 0.303 | -513,553 | -513,536 |
| 512 s32 | 2,302 | truth | | 0 | 0.500 | | -515,853 |
| 5,000 | 2,701 | bonsai-rs | 32 | 1,304 | 0.543 | -5,558,390 | -5,558,390 |
| 5,000 | 2,701 | published | 4,868 | 1,937 | 0.496 | -5,563,770 | -5,561,535 |
| 5,000 | 2,701 | published, backbone 2,048 | 2,514 | 1,899 | 0.578 | -5,571,002 | -5,565,342 |
| 5,000 | 2,701 | published, backbone 1,000 | 1,794 | 1,959 | 0.569 | -5,572,684 | -5,566,018 |
| 5,000 | 2,701 | truth | | 0 | 0.679 | | -5,571,449 |
| 10,000 | 2,767 | bonsai-rs | 83 | 2,575 | 0.539 | -11,253,540 | -11,253,540 |
| 10,000 | 2,767 | published | 16,062 | 5,149 | 0.281 | -11,280,721 | -11,274,900 |
| 10,000 | 2,767 | published, backbone 2,048 | 3,529 | 3,946 | 0.368 | -11,280,324 | -11,264,651 |
| 10,000 | 2,767 | published, backbone 1,000 | 3,594 | 4,191 | 0.347 | -11,284,311 | -11,268,274 |
| 10,000 | 2,767 | truth | | 0 | 0.461 | | -11,282,910 |

bonsai-rs is `BonsaiParams::default()` in 0.3.0, 2026-10-13, ten threads. Published is
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
- **From 5,000 bonsai-rs pulls ahead**, and at 10,000 by a lot: 2,575 against
  5,149 splits, 21,000 nats after the refit, and 190 times faster.
- **The refit matters for the published trees.** It recovers 2,200 nats at 5,000
  and 5,800 at 10,000; 15,700 for the 10,000 backbone. The as-given column
  overstates bonsai-rs's lead by 71 per cent at 5,000 and 27 at 10,000.
- **The published backbone pays off in the published implementation**: 2 to 4.5
  times faster than its standard run, and at 10,000 better on splits and on the
  refit loglikelihood. It is still 40 to 80 times slower than bonsai-rs's
  standard run at 5,000 and 10,000, and behind it on every column but one: at
  5,000 both backbone runs edge recovery, 0.578 and 0.569 against 0.543. That
  run of ours sits in the poorer of two basins; see the last bullet.
- **The two trees differ from each other** by 26, 110, 1,463 and 4,288 splits.
  Equal scores against the truth don't mean the same tree.
- **The published implementation is deterministic.** A 2026-09-14 rerun gave
  byte-identical Newick at all four sizes. Wall times moved under half a per
  cent where the first run had an idle machine, 30 per cent at 512 and 8 at
  10,000 where it hadn't.
- **Results differ with SPR's subtree order.** Recovery lands anywhere from
  0.54 to 0.67 at 5,000 cells and 0.36 to 0.54 at 10,000 across orders
  ([performance](PERFORMANCE.md#how-much-one-real-data-run-says)).
  Some orders pass the truth's 0.461, the default among them. That's possible because the truth's branch
  lengths are expected displacements and ours are fitted to what was realised.

## Counts to tree

Raw UMIs to finished tree on **one node**: ten-core M1 Max, 64 GB, each run
alone. The Rust route is 0.3.0 with `sanity-sc-rs` 0.4.0 and `ann-search-rs`
0.11.0, 2026-10-13; the
original route is from 2026-09-25, its backbone runs from 2026-09-26. The
published implementation can run MPI across nodes; nothing here speaks to a
cluster.

Data: the harness's simulated Baron counts, 17,499 genes. Original route: the
Sanity binary (10 threads, `-v_m MAP`), the harness's selection at `S >= 1`,
then published Bonsai. Rust route, driven through the Python bindings: genes
with no counts dropped (the Sanity binary drops them itself), `sanity-sc-rs` on
CPU or GPU in `f32`, the S5 conversion, `bonsai()` at defaults (also `S >= 1`).
Rust Sanity's default `marginalise` variance rule does more work per gene than
`MAP`.

Each search is paired with its counterpart. **Exact** is SPR and NNI as the
paper specifies them: ours against the published standard run, on one core and
under MPI with 10 ranks. **Approximate** is our default (see the README) against
the published backbone mode, its fastest, with `n_initial_cells` 2,048 and
`growth_factor_guide` 10, one process. Its default of 10,000 initial cells
needs more cells than these sets have, so it's untested at the setting it
ships with. At 512 the backbone (128 initial cells) took 758 s, twice its
standard run, so it has no row there. Whether it combines with MPI wasn't run.

Sanity, seconds:

| cells | original | Rust, CPU | Rust, GPU |
|---|---|---|---|
| 512 | 71.9 | 4.6 | 0.8 |
| 512 s32 | 72.1 | 4.4 | 0.6 |
| 5,000 | 766.9 | 29.7 | 1.7 |
| 10,000 | 1,604.7 | 55.2 | 3.2 |

Bonsai, seconds:

| cells | published, 1 core | published, 10 ranks | Rust, exact | published, backbone | Rust, approximate |
|---|---|---|---|---|---|
| 512 | 371.7 | 170.2 | 4.2 | | 1.9 |
| 512 s32 | 371.0 | | 4.5 | | 2.2 |
| 5,000 | 4,868 | 3,208.6 | 94.3 | 2,514.0 | 28.0 |
| 10,000 | 16,062 | 9,439.3 | 458.7 | 3,528.7 | 75.1 |

Rust search times are on CPU-Sanity input. On GPU-Sanity input, exact took
110.2 s and 436.5 s at 5,000 and 10,000, approximate 28.8 s and 78.2 s.
Published single-core times at 5,000 and 10,000 are the harness's, reproduced
within half a per cent on 2026-09-14; the rest are reruns. The backbone with
1,000 initial cells took 1,794.3 s and 3,594.3 s, with worse trees at both
sizes (RF 1,959 and 4,191). The original route also spends 2.0, 0.8, 29.7 and
66.5 s on gene selection, and the Rust route 0.1 to 1.6 s on the S5
conversion, both included below.

End to end, exact, seconds:

| cells | original, 1 core | original, 10 ranks | Rust CPU + exact | speed-up on 1 core / 10 ranks |
|---|---|---|---|---|
| 512 | 445.6 | 244.1 | 8.9 | 50x / 27x |
| 512 s32 | 443.9 | | 9.0 | 49x / |
| 5,000 | 5,664.2 | 4,005.2 | 124.7 | 45x / 32x |
| 10,000 | 17,733.2 | 11,110.5 | 515.5 | 34x / 22x |

End to end, approximate, seconds:

| cells | original, backbone | Rust CPU + approximate | Rust GPU + approximate | speed-up, CPU / GPU |
|---|---|---|---|---|
| 5,000 | 3,310.6 | 58.5 | 31.2 | 57x / 106x |
| 10,000 | 5,199.9 | 131.9 | 83.0 | 39x / 63x |

The original route is CPU only, so the CPU columns are the like-for-like
reading; GPU is what a Rust user gets with an adapter.

Quality against the generating tree. Recovery here uses all 17,499 genes' true
positions, not the selected genes of the headline table:

| cells | original, 1 core | original, 10 ranks | Rust CPU + exact | original, backbone | Rust GPU + approximate |
|---|---|---|---|---|---|
| 512 | RF 146, 0.492 | RF 141, 0.496 | RF 137, 0.632 | | RF 139, 0.632 |
| 512 s32 | RF 240, 0.165 | | RF 241, 0.233 | | RF 237, 0.224 |
| 5,000 | RF 1,937, 0.352 | RF 2,016, 0.261 | RF 1,306, 0.406 | RF 1,899, 0.456 | RF 1,279, 0.569 |
| 10,000 | RF 5,149, 0.162 | RF 5,151, 0.226 | RF 2,642, 0.295 | RF 3,946, 0.252 | RF 2,633, 0.390 |

Published memory, peak RSS summed over processes: 543 MB, 2,965 MB, 5,686 MB at
512, 5,000, 10,000 on one core; 3,009 MB, 10,033 MB, 18,737 MB with 10 ranks;
2,809 MB and 5,418 MB for the backbone at 5,000 and 10,000. MPI costs 5.5, 3.4
and 3.3x the memory for 2.2, 1.5 and 1.7x the speed, one Python process per
rank. It also changes the answer: RF 141 against 146 at 512, 2,016 against
1,937 at 5,000, 5,151 against 5,149 at 10,000. The backbone is the best of the
published routes at both sizes, and at 10,000 by 1,200 splits.

Two cautions. Differences between the Rust Sanity paths and search modes sit
inside the run-to-run spread ([performance](PERFORMANCE.md#how-much-one-real-data-run-says)).
Across the four Sanity and search pairings recovery spans 0.395 to 0.571 at
5,000 and 0.295 to 0.390 at 10,000, lowest with exact search at both sizes.
That's the search basin, not Sanity and not the neighbour graph. At 5,000, CPU
Sanity and exact search, we ran the starting tree from `ann-search-rs` 0.9 (an
approximate kNN graph) and from 0.11 (the exact one) through four random SPR
orders each. Both split two and two: recovery near 0.41 or near 0.57,
loglikelihoods over 985 nats. The default order happens to put 0.9 in the good
basin and 0.11 in the poor one, 0.570 against 0.406. The kNN seed changes
nothing in either. 10,000 is one run. The gaps to the published trees, over 600
splits from 5,000 up, the backbone included, are far outside it.

Same trees drawn as in [What the trees look like](#what-the-trees-look-like):
clade fragments (ideal ten) and recovery on the selected genes, same 20,000
pairs for every tree.

| cells | GPU + approximate | CPU + exact | reference, backbone | reference, 1 core | reference, 10 ranks |
|---|---|---|---|---|---|
| 512 | 13, 0.744 | 13, 0.745 | | 15, 0.638 | 14, 0.643 |
| 512 s32 | 17, 0.366 | 17, 0.374 | | 19, 0.311 | |
| 5,000 | 25, 0.673 | 26, 0.556 | 24, 0.589 | 29, 0.500 | 36, 0.402 |
| 10,000 | 21, 0.474 | 23, 0.386 | 33, 0.348 | 40, 0.258 | 58, 0.323 |

![radial layouts at 10,000 cells, counts to tree](figures/n10000_e2e_tree_layout.png)

![distance recovery at 10,000 cells, counts to tree](figures/n10000_e2e_distance_recovery.png)

## What the trees look like

Measured on the 0.3.0 trees of the [headline table](#speed-and-quality),
2026-10-13.

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
| 5,000 | **23** | 29 |
| 10,000 | **22** | 40 |

A tie at 512, then bonsai-rs wins, including on the replicate where it's behind
on the loglikelihood. The gap grows to eighteen fragments at 10,000.

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
| bonsai-rs 5,000 | 48 | 0 | 18 |
| bonsai-rs 10,000 | 130 | 0 | 37 |
| published 512 | 0 | 0 | 3 |
| published 512 s32 | 5 | 0 | 7 |
| published 5,000 | 40 | 0 | 107 |
| published 10,000 | 99 | 0 | 335 |

Two different things; don't conflate them.

**Zero-length leaf edges** are a SPEC 6 boundary optimum: no evidence separating
the cell from its parent. Intended, and about input noise, not topology. The
generating tree has none. These cells are noisier: median of each cell's mean
SD 1.34 against 1.14 at 10,000, and 31 per cent of them in the noisiest decile.

**Zero-length internal edges and polytomies** come from SPR and NNI splices. At
10,000 there are 66 internal ones after step 5 and 58 after step 6. Step 8
collapses them, and collapses once more after its closing branch solve, which
can land an edge on exactly zero again. None survive at any size. Leaf edges step 8 leaves alone by design.

So the difference is in the leaf edges and in how much tree sits under a
multifurcation: 34.9 per cent degenerate leaves at 10,000 for the published
tree against 3.4, a largest multifurcation of 513 leaves against 71, and 335
polytomies against 37.

## Fans and ladders

Both implementations hit groups of cells the data can't order: the evidence
separating them is weaker than the noise. **How they write that down is the
biggest reason the pictures look different.**

The reference draws a **fan**: the whole group as siblings off one node. At
10,000 cells, 335 multifurcations, the largest 513 leaves.

bonsai-rs draws a **ladder**: an order anyway, each cell on its own small,
non-zero branch. 37 multifurcations, the largest 71.

In a radial layout, radius is the sum of branch lengths above a leaf. A fan adds
nothing and lands as a blob. A ladder adds up (37 hops, median radius 1.086
against the tree's typical 0.627) inside one thin angular slice. Long and thin
is a spike. The reference's picture is tidier because it flattened those cells,
not because it placed them better.

**The ladder is more faithful exactly where a fan looks defensible.** Pearson correlation of true squared distance with path distance
at 10,000 cells (`faithfulness.py` in the harness). The subset rows use every
pair; the first samples 200,000, which is why it isn't the headline table's
0.539.

| pairs drawn from | cells | in bonsai-rs's tree | in the reference's tree |
|---|---|---|---|
| all cells | 10,000 | 0.540 | 0.278 |
| the cells bonsai-rs strings into its longest ladder | 250 | 0.359 | 0.359 |
| the cells the reference fans into its largest multifurcation | 513 | 0.141 | 0.004 |

- **A multifurcation isn't declining to answer.** It claims every member is
  equidistant from every other, and carries no distance information: 0.004 is
  zero. The ladder still gets 0.141 on the same cells.
- **The spike itself is no better than the reference.** 0.359 on its 250 cells
  in both trees, well below bonsai-rs's whole-tree 0.540. On the 0.2.0 tree it
  was 0.850 against 0.811: that spike was one true clade drawn long, this one
  mixes clades ([the arm](#the-arm)). Read it as cells the search couldn't
  place, not as structure.
- **Flattening it still wouldn't help.** It would force a fan where the branch
  solve found positive optima, and a fan scores zero.

Limits: path distance saturates at large true distances in both trees, and at
10,000 cells the generating tree only scores 0.461, so there's little global
headroom either way.

## The arm

The spike, taken as the 2.5 per cent of cells furthest from the root: 250 at
10,000, 125 at 5,000. Median 37 hops against 17 at 10,000 and 22 against 15 at
5,000; radius 1.086 against 0.627 and 0.797 against 0.604.

- **It isn't one true clade at either size.** Of the generating tree's ten
  top-level clades it spans seven at 10,000, the largest holding 117 of 250, and
  nine at 5,000, the largest 44 of 125. So the spike groups cells the truth
  doesn't. The 0.2.0 tree's spike at 10,000 was different: 231 of 250 in one
  clade.
- **The reference half agrees at 10,000.** 139 of the 250 sit in one reference
  subtree of 212 leaves. At 5,000 it doesn't: the smallest reference subtree
  holding half the arm has 1,937 leaves.
- **Zero-length leaf edges** gather in it at 10,000 (45 of 130) but not at 5,000
  (0 of 48). 82 cells have a zero-length leaf edge in both trees at 10,000.
- **A little noisier than the rest.** Median of each cell's mean SD 1.19 against
  1.14 at 10,000 and 1.15 against 1.12 at 5,000; 11.2 and 15.2 per cent in the
  noisiest decile against 10.0 and 9.9.

## Honest rendering

These figures never jitter coincident points apart. A leaf at zero distance from
its parent drawn anywhere else manufactures structure the model said it couldn't
find. Overlapping markers at equal radius are zero-length leaf edges, not a
glitch, and seven times as many of the published tree's leaves are affected.

The same goes the other way for the spike. It's ugly and someone will ask to
collapse it. That would replace measured positive branch lengths with a
multifurcation, which carries no distance information. Tidying the picture would
swap measured branch lengths for a structure that scores zero.

## Reproducing

`reference/comparison/` has the harness: the Baron count simulator, Sanity and
gene selection, the scorer, and every figure and table script here.
`faithfulness.py` makes the recovery table, `degeneracy.py` the zero-length
counts, `figures.py` the layouts, scatters and fragmentation. Its README has
the commands.

What it leaves out is running the published implementation. Get a tree from it
by its own documentation on the same Sanity output, save it as `theirs.nwk` in
the configuration, and `harness score` puts it next to ours.
