# Performance

Where the time goes now, the rules that came out of getting it there, and a log
of what worked and what did not. `docs/DESIGN.md` says how the crate is built.

Timings are from one ten-core M1 Max unless stated. Dates in the log are the day
a result was first committed; 2026-09-12 is when this file was created, so a row
dated then was measured on or before it.

## Now

`BonsaiParams::default()` on Sanity-preprocessed Baron data, every step timed,
2026-09-25, load average 6 to 7 at the start of each run:

| step | 512 | 5,000 | 10,000 | share at 10,000 |
|---|---|---|---|---|
| 1-2 linkage | 0.04 s | 1.7 s | 5.9 s | 3% |
| 3 polytomy | 0.01 s | 0.1 s | 0.3 s | 0% |
| 4 branch | 0.7 s | 8.9 s | 26.1 s | 12% |
| 5 SPR | 1.9 s | 42.1 s | 128.0 s | 61% |
| 6 NNI | 0.1 s | 5.9 s | 17.0 s | 8% |
| 7 branch | 0.7 s | 7.2 s | 20.3 s | 10% |
| 8 collapse | 0.05 s | 4.5 s | 12.6 s | 6% |
| total | 3.5 s | 70.4 s | 210.2 s | |

On 2026-09-13 the same configurations took 4.7 s, 166 s and 611 s. From 5,000
to 10,000 cells the total goes as `n^1.58`, SPR `n^1.60` and NNI `n^1.53`,
two-point exponents over two different gene panels, so a slope rather than a law.

What that says:

- SPR is three fifths of the run and branch optimisation, steps 4, 7 and 8,
  over a quarter. Neither has an approximation left that measured safe; see the
  log for the ones that did not.
- The full-tree sweeps are still small. Sampled at 5,000 cells, the prune
  kernels and the up-sweep are 2.8 per cent of busy thread time.
- A third of SPR's thread time is idle. Round one accepts a move every few
  candidates, so its chunks run at the floor of eight proposals on ten threads,
  and a larger floor measured slower because it discards more proposals.
- The hottest code is the branch solve inside merges and placements:
  `edge_newton_simd`, `split_derivative` and `log` are the top three by self
  time.

## Rules

1. **Measure a component's share of the whole before optimising it.** A
   component's share is a hard ceiling on what any change to it returns.
   `BlockedState` was 6x on a prune that is 2.4 per cent of a run.
2. **Check that the pipeline calls it at all.** `BlockedState` was only ever
   built by a benchmark, and the kNN restriction and ellipsoid bounds sat
   unwired for a week while the search ran cubically.
3. **Pick the kernel by call count, not by how vectorisable it looks.**
   `edge_newton` runs 35 to 45 times per candidate pair and earned a SIMD tier;
   `split_derivative` reads like the hot loop, runs 0.07 times, and measured
   flat.
4. **Measure an exponent at the parameter value you ship.** At 200 features
   polytomy resolution and NNI scaled as `n^2.83` and `n^2.60`; at 2,000 they
   are `n^1.08` and `n^0.97`. Both were round counts, which collapse as the
   feature axis cleans up the landscape.
5. **Diagnose before fixing, and expect the first diagnosis to be wrong.** Four
   for four on SPR and NNI: the re-prune, the NNI re-prune, the placement beam
   and the 5k-to-10k slowdown were all blamed and none was the cost. See
   [Diagnoses that were wrong](#diagnoses-that-were-wrong).
6. **Ablate the step, and at more than one noise level.** At low noise SPR costs
   20x the wall clock and changes nothing; at noise 1.6 it wins 27.5 splits of
   4,090 and is 7.3x *faster* than not having it.
7. **A justification expires with the measurement behind it.** "X is fine
   because Y is small" wants re-reading whenever Y's denominator changes. When
   the linkage replaced step 2, the prune-based steps went from a minority of
   the run to most of it. The prune kernel itself turned out still to be small,
   2.8 per cent in 2026-09-24's sample, but that had to be measured again, not
   assumed.
8. **A smaller input is not automatically a cheaper one.** Collapsing zero-length
   edges before step 5 removes an eighth of the nodes and makes SPR 30 per cent
   slower, because a regraft then lands next to a bigger star.
9. **A generator is not the data, and a tie on it is not a tie.** The Ward start
   tied on synthetic data and wins on real data. Subtree splice pairs cost 5 to
   11 nats on synthetic data and 437 to 737 on real. Confirm on the input you
   ship against.
10. **Sweep sizes as well as shapes.** SPR revisit radius 3 held on every shape
    and at 5,000 cells, then failed at 10,000.
11. **Look at core utilisation, not only wall time.** Once the linkage replaced
    step 2 the pipeline got 1.26x out of ten cores, which nothing in the
    wall-clock numbers showed until a run at `RAYON_NUM_THREADS=1`.
12. **Time on an idle machine against a kept baseline binary, alternating.** On
    2026-09-24 the load average wandered from 2 to 27 while other work ran, and
    the same build's step 6 at 10,000 cells measured 98 s and 156 s. Quality
    numbers are deterministic and survive load; timings do not.
13. **A single real-data run is one draw.** Equally valid searches spread
    over 460 nats at 5,000 cells and 1,700 at 10,000; see
    [How much one real-data run says](#how-much-one-real-data-run-says). Trust
    an approximation that reproduces the exact tree; treat a few hundred nats
    either way as noise unless it repeats across sizes.

## Log

### What worked

| recorded | change | effect |
|---|---|---|
| 2026-08-28 | Adaptive ellipsoid sizing on redraw-versus-walk cost | 0.109 s against 0.170 and 0.175 for fixed schedules either side |
| 2026-09-05 | Lazy SPR proposal rows | 93.19 s to 6.71 s at 2048 by 200; `n^1.98` to `n^1.47` |
| 2026-09-06 | Wire in the kNN restriction and ellipsoid bounds | 1478.8 s to 37.4 s at 512 by 2000; `n^2.9` to `n^1.8` |
| 2026-09-12 | Structural NNI filter over the star result | 1.98 s to 0.75 s at 2048 by 200; per round `n^1.53` to `n^1.02` |
| 2026-09-12 | Ward linkage start replacing search step 2 | 66.08 s to 17.06 s at 2048 by 2000; better on real data, see [Starting tree](#starting-tree) |
| 2026-09-12 | SPR proposals in parallel chunks, restarted on acceptance | 200.04 s to 24.44 s at 16384 by 2000; `n^1.45` to `n^1.31` |
| 2026-09-12 | SPR acceptance on the candidate's own terms, rows assembled not swept | 127.9 s to 19.1 s at 2048 by 2000, noise 1.6, with the above |
| 2026-09-12 | Graph linkage by mutual-nearest rounds instead of a chain | depth 131 to 12 at 4096 by 2000, 248 splits from the dense tree to 0 |
| 2026-09-12 | Edge solve started at `mean(d - s)`, stopped on the Newton correction | 22.4 to 4.6 passes per attachment |
| 2026-09-12 | Merge split by Illinois regula falsi instead of bisection | 58 to 20.5 derivative passes per pair |
| 2026-09-12 | Parallel edge solve in `optimise_branch_lengths` | 2.4x on ten cores, bit-identical |
| 2026-09-12 | Parallel edge scan in `nni_greedy` | 2.0x on top of the 3.8x it borrowed, bit-identical |
| 2026-09-12 | `f64x4` tier on `edge_newton` | 0.95 to 0.71 ns per feature; 14 per cent of the merge scan, 8 per cent of a run |
| 2026-09-13 | Scale-relative SPR acceptance floor | 2383.75 s to 616.26 s at 10,000 by 2,767, and a better tree |
| 2026-09-24 | SPR acceptance into a slot store instead of an `O(n p)` state copy | step 5 134.1 s to 101.4 s at 5,000 by 2,701, byte-identical tree |
| 2026-09-24 | SPR revisit radius 5, the default `SprSearch::Approximate` | steps 5 to 8 132.7 s to 84.8 s at 5,000 and 461.0 s to 257.8 s at 10,000, within a few nats everywhere; see [Revisit radius](#revisit-radius) |
| 2026-09-25 | Lazy NNI greedy phase, radius 5, the default `NniSearch::Approximate` | step 6 29.6 s to 5.9 s at 5,000 and 99.4 s to 16.5 s at 10,000, finished tree identical to the exact phase on all thirteen datasets; see [Lazy NNI](#lazy-nni) |
| 2026-09-25 | SPR arenas built by `Tree::from_level_ordered`, skipping the relabel `from_parents` does | step 5 42.3 s to 41.4 s at 5,000 and 125.3 s to 121.3 s at 10,000, byte-identical tree |

What the big ones have in common is not materialising things. The lazy rows
form what a proposal reads and no more; the NNI filter tests the star result
rather than building a tree and walking it; the slot store writes the rows a
move changed rather than copying the rest. None made a kernel faster.

The parallel changes are bit-identical on purpose. The branch solve writes one
slot per node with no reduction; the NNI scan reduces to a running best with
ties broken on the lower node id, which is what the sequential scan did
implicitly. `test_the_greedy_phase_is_deterministic_whatever_the_thread_count`
pins it at 1, 3 and 8 threads.

### What did not work

| recorded | attempt | why not |
|---|---|---|
| 2026-08-27 | `BlockedState`, feature-blocked parallel prune | 6x on a kernel that is 2.4 per cent of a run |
| 2026-09-12 | SIMD on `split_derivative` | runs 0.07 times per candidate pair; measured flat |
| 2026-09-12 | `edge_newton` with the division replaced by a multiply | 0.92 against 0.95 ns per feature, inside noise |
| 2026-09-12 | `edge_newton` with eight accumulator chains | 1.00 against 0.95, slower |
| 2026-09-12 | `edge_newton` with one division per four features | 1.23 against 0.95, much slower |
| 2026-09-12 | Ellipsoid sizing on walk depth | the walk is chunked, so every round reads shallow and the cap chose the answer |
| 2026-09-12 | Feature subsampling to rank merge candidates | argmax survives 4 of 15 checkpoints at a quarter of the features; see [Feature subsampling](#feature-subsampling) |
| 2026-09-12 | Chain or Boruvka agglomeration on the Bonsai merge gain | the score is not reducible; see [Reducibility](#reducibility) |
| 2026-09-12 | Tree-distance limit on SPR regrafts | the beam is `n^0.06`, so there is nothing to limit |
| 2026-09-12 | Warm-starting the beam's root find from the neighbour's optimum | 8.3 passes per solve against 5.8 from the mean start |
| 2026-09-12 | Counting sort in SPR's `assemble` | the height sort was never the cost; `Tree::from_parents` is |
| 2026-09-12 | Parallel pair scan on the four-member star inside the SPR proposal loop | 3.0 ms a resolution summed over threads against 0.7 sequential |
| 2026-09-12 | Redrawing the linkage graph more often to stop it chaining | the chain was the cause, not the cadence |
| 2026-09-13 | Collapsing zero-length internal edges before step 5 | 589 s against 451 in SPR, on an eighth fewer nodes |
| 2026-09-13 | Merging every mutually-best pair a round instead of the single best | step 2 seven to eight times faster, the tree no better; see [Starting tree](#starting-tree) |
| 2026-09-24 | Starting the SPR beam at the pruned subtree's origin only | 64 and 449 nats and 6 and 15 splits lost at 512; the 3 per cent of accepted moves that travel far carry real gain |
| 2026-09-24 | SPR revisit radius 3 | fine to 5,000 cells; at 10,000 step 6 inherits the work, 106 s to 273 s |
| 2026-09-24 | Scoring only the three splice pairs through the regrafted subtree | 437 and 737 nats and distance recovery 0.67 to 0.54 on real data, and slower; see [Splice pairs](#splice-pairs-through-the-regrafted-subtree) |
| 2026-09-25 | Expanding a beam start point's neighbours only when the start is within tolerance of the best | SPR 3 to 9 per cent faster, quality within 2 nats either way; too small to carry a knob |
| 2026-09-25 | SPR proposal chunk floor other than 8 | 2 to 6 flat, 10 to 32 slower (46.8 s to 70.4 s against 42.1 s at 5,000); the tree is identical at every floor, so the existing 8 stands |
| 2026-09-25 | Branch tolerance 1e-8 or 1e-7 on steps 4 and 7 instead of 1e-10 | saves 5 to 25 s; at 1e-8 landed in the worse basin at both 5,000 and 10,000, recovery 0.67 to 0.54 and 0.48 to 0.38; one draw each, see [How much one real-data run says](#how-much-one-real-data-run-says) |
| 2026-09-25 | Repeating a discarded SPR proposal from its previous target alone | step 5 43.3 s to 37.1 s at 5,000, 490 nats worse after step 5; one draw, inside the spread, so not proven harmful but not worth 14 per cent |

## Notes

### Diagnoses that were wrong

- SPR was assumed to be paying for its `O(n p)` acceptance re-prune. That
  re-prune was 0.03 per cent of the step, because the split fingerprint discards
  99.7 per cent of candidates before it runs. The cost was *proposing*: five
  `O(n p)` sweeps per candidate for rows of which a few dozen are read.
- NNI was assumed to be paying for its re-prune too. The cost was the filter
  itself, which built a whole tree and walked it, `O(n)` each, per candidate.
- SPR's remaining `n^1.46` was assumed to be the placement beam. The beam is
  flat: 41 to 49 nodes over a sixteenfold growth in `n`, `n^0.06`.
- The 5k-to-10k slowdown was assumed to be SPR leaving work for NNI. It was SPR
  accepting rounding noise and cycling.

The acceptance path came back later on a different axis. Once proposals were
cheap, the `O(n p)` copy of the state on each accepted move was 23 of the 42
seconds of the first SPR round at 5,000 cells, measured 2026-09-24, all of it on
the sequential path with nine threads waiting. The slot store removed it.

### Revisit radius

After the first SPR sweep, only subtrees within `r` edges of a clade the
previous sweep created are proposed again: the don't-look bits of TSP local
search. Before it, rounds after the first cost two thirds of step 5 for a
quarter of its moves.

Measured 2026-09-24 over thirteen datasets: balanced, random-branch and
unbalanced trees at noise 0.4, 1.0 and 1.6 at 4,096 by 1,000, and the four
Sanity-preprocessed configurations. At `r = 5` every one is within a few nats of
the exact search. At `r = 3` all of them are too, except 10,000 cells, where
step 6 inherits what step 5 skipped. `SprSearch::Exact` keeps the specified
search one line away, and `DEFAULT_REVISIT_RADIUS` in `src/search/spr.rs` has
the table.

### Lazy NNI

The greedy phase performs one interchange per round and, as specified, rescores
every edge every round, so it costs a full scan per move: 1.1 s a round at
5,000 cells and 2.2 s at 10,000, for 27 and 44 moves. `NniSearch::Approximate`
caches each edge's gain under the leaf set below it, rescores after a move only
the edges within five edges of the clades the move created, and rescores the
leading cached gain on the current tree before taking it, the lazy greedy
evaluation of Minoux (1978). When nothing cached improves, a full scan runs, so
the phase stops on exactly the condition the exact phase stops on.

Measured 2026-09-25 on the thirteen-dataset grid of the revisit radius, steps 5
to 8 with the default SPR: the finished tree matched the exact phase on all
thirteen, loglikelihood, Robinson-Foulds and recovery to the last printed digit.
On step 6 alone radius two and three drifted by 0.01 and 7.5 nats at 5,000
cells; five was identical at both sizes. What is left per round is the settle,
about 0.28 s at 10,000 cells, which a lazy row provider like SPR's could take
away.

### How much one real-data run says

The finished tree on real data is sensitive to small upstream changes. Runs that
differ only in the order SPR visits subtrees are equally valid searches, and
they spread widely, measured 2026-09-25:

| cells | SPR order | loglikelihood | Robinson-Foulds | recovery |
|---|---|---|---|---|
| 5,000 | longest branch, the default | -5,557,978 | 1,288 | 0.666 |
| 5,000 | random, seed 1 | -5,558,223 | 1,274 | 0.581 |
| 5,000 | random, seed 2 | -5,557,825 | 1,258 | 0.668 |
| 5,000 | random, seed 3 | -5,557,834 | 1,275 | 0.665 |
| 5,000 | random, seed 4 | -5,557,764 | 1,267 | 0.666 |
| 10,000 | longest branch, the default | -11,252,721 | 2,624 | 0.476 |
| 10,000 | random, seed 1 | -11,254,387 | 2,668 | 0.403 |
| 10,000 | random, seed 2 | -11,253,896 | 2,563 | 0.479 |
| 10,000 | random, seed 3 | -11,254,362 | 2,571 | 0.404 |

About 460 nats and 0.58 to 0.67 in recovery at 5,000 cells, 1,700 nats and 0.40
to 0.48 at 10,000. So a single-run difference of a few hundred nats is inside
the noise at these sizes, and an approximation is only safe to call harmless
when it reproduces the exact result outright, as lazy NNI does on all thirteen
datasets. The revisit radius reproduces it on the 512-cell and low-noise sets;
at 5,000 and 10,000 cells it lands 3 nats below and 467 above the exact search,
both inside this spread. The same caution runs the other way: the 1,281-nat,
0.38-recovery loss of a loose branch tolerance at 10,000 cells looks like one
of the bad basins above rather than a measured cost of the tolerance.

Two things about the search itself. The runs fall into two basins, one with
visibly worse recovery, and the loglikelihood separates them every time, so a
best-of-several run chosen by loglikelihood is a cheap way to avoid the bad one
now that a run is a minute or three. And the specified longest-branch order is
not dominated: random order did better at 5,000 cells and worse at 10,000.

### Splice pairs through the regrafted subtree

A regraft onto an internal node leaves a four-member star, which the star
primitive resolves by scoring all six pairs. A pair and its complement build the
same unrooted tree, so the three pairs containing the regrafted subtree already
offer every topology and only the three branch lengths the merge optimises
differ. It looked like half the splice for free. Measured 2026-09-24 against the
radius-5 default, it saved nothing in step 5 and cost everywhere else:

| data | steps 5 to 8 | loglikelihood | Robinson-Foulds | recovery |
|---|---|---|---|---|
| 5,000 real, all six pairs | 84.8 s | -5,557,978 | 1,288 | 0.666 |
| 5,000 real, three pairs | 131.3 s | -5,558,415 | 1,310 | 0.545 |
| 10,000 real, all six pairs | 257.8 s | -11,252,721 | 2,624 | 0.476 |
| 10,000 real, three pairs | 566.0 s | -11,253,458 | 2,567 | 0.384 |

With the worse branch lengths SPR accepts 5 to 20 per cent more moves, each
worth less, and NNI then runs two to four times longer cleaning up. Step 7's
global branch optimisation does not repair the recovery loss, so the damage is
in the topology the search walked into.

### Feature subsampling

The merge gain is a sum over `p` features, so estimating it on `p'` of them and
rescaling looks like a free 30x. Measured over five checkpoints through a
128-member star at 2000 features, three seeds, with a shared column set per
checkpoint so the correlated part of the error is already cancelled:

| subsample | argmax agreed | mean relative error | mean nats lost |
|---|---|---|---|
| 32 | 1 / 15 | 9.4e-1 | 94.6 |
| 128 | 3 / 15 | 3.7e-1 | 26.1 |
| 512 | 4 / 15 | 1.2e-1 | 12.0 |

The estimator's standard deviation is `p * sigma / sqrt(p')`, about 11 per cent
of the gain at `p' = 512`, and competing merges in a real round differ by far
less than that. A random *projection* is a different mechanism and is still
open, since it summarises every feature rather than discarding all but `p'`. It
needs the precisions to be approximately rank-1 in gene by cell, which has not
been checked.

### Reducibility

Merging changes the peeled remainder that every other pair's score depends on,
so a merge can lift another pair above the score the merged pair had. Measured
at 64 and 128 members: 10 to 32 inversions over 61 to 125 rounds, worst excess
3.6 to 13.2 nats, up to 60 per cent relative. Ward has no global remainder term
and is reducible by construction, which is a second reason to prefer it for the
starting tree.

### Starting tree

`StartTree::Linkage` is the default. `StartTree::GreedyMerge` is search steps 1
and 2 as SPEC.md section 9.1 specifies them. Use the linkage for any real work
and the greedy merge to reproduce the published method.

Measured 2026-09-13 on Sanity-preprocessed Baron pancreas data, the same gene
set to both, scored against the generating tree:

| cells | start | Robinson-Foulds | distance recovery | loglikelihood | seconds |
|---|---|---|---|---|---|
| 512 | greedy | 175 | 0.600 | -527,518 | 19.9 |
| 512 | linkage | 147 | 0.591 | -527,499 | 4.8 |
| 512 s32 | greedy | 310 | 0.228 | -514,593 | 19.5 |
| 512 s32 | linkage | 220 | 0.318 | -513,567 | 6.2 |
| 5,000 | greedy | 1,368 | 0.518 | -5,560,648 | 723 |
| 5,000 | linkage | 1,293 | 0.666 | -5,557,975 | 169 |
| 10,000 | greedy | 3,305 | 0.260 | -11,270,741 | 3,680 |
| 10,000 | linkage | 2,632 | 0.466 | -11,253,188 | 630 |

The linkage wins on the loglikelihood and on Robinson-Foulds at all four sizes,
on distance recovery at three of four, and is three to five times faster. The
exception is recovery at 512, where the replicate at the same size goes the
other way by more than the gap. It is not a size threshold: the linkage wins at
512 too.

The mechanism is chaining, and it is in the criterion. Mean leaf depth straight
after step 2, against `log2(n)`:

| cells | greedy | linkage | log2(n) |
|---|---|---|---|
| 512 | 18.7 | 8.9 | 9.0 |
| 5,000 | 114.5 | 12.5 | 12.3 |
| 10,000 | 151.7 | 13.5 | 13.3 |

The greedy overshoot grows with the cell count, and steps 3 to 7 spend their
budget repairing it: at 10,000 cells the greedy start leaves SPR 11,330 accepted
moves over 32 rounds and NNI 457 over 458, against 4,755 and 47 from the
linkage. A cluster's effective leaf carries `1/size` of the noise, so at a few
thousand features a large cluster sits closer to every member than that
member's own relatives do and absorbs them one at a time. Ward has the opposite
bias, since a Lance-Williams distance to a cluster grows with its size.

Rescheduling does not reach it. Merging every mutually-best pair a round made
step 2 seven to eight times faster and reached a *better* merge score, and the
finished tree got worse: Robinson-Foulds 175 to 186 at 512 and 1,368 to 1,425 at
5,000, depth only down to 16.1 and 74.1. The mutual test stops a large cluster
taking more than one partner a round, but not from being every singleton's
preferred partner; those singletons are then mutual with nothing, the round
starves, and the chain builds anyway. So on real input the specified criterion
is a poor proxy for the tree you end up with.

### Earlier snapshots

Where the time went on 2026-09-12, Ward start, 2000 features, 16384 synthetic
leaves, before the slot store and the revisit radius: linkage 22 per cent, step 4
4, SPR 73, NNI 2. Exponents then: linkage 2.00, polytomy 0.98, branch 1.03, SPR
1.45, NNI 0.98, total 1.52.

The arc at 512 cells by 2000 features, identical loglikelihoods at every stage:
the exhaustive candidate scan took 1478.8 s, the kNN restriction and ellipsoid
bounds took it to 37.4 s, and the lazy SPR proposal and structural NNI filter to
9.1 s. None of that traded accuracy.

## Reference

### Memory

**Dense `n x n` anything is dead above about 20k cells.** A Ward distance matrix
is 2.1 GB at 16384 in `f64` and 4 TB at a million. A `k = 16` neighbour graph at
a million is 128 MB. That, and not the flop count, is what forces the linkage
through `ann-search-rs`.

**`f32` storage costs accuracy as a function of how far the means sit from
zero**, measured over four leaves by 256 features:

| mean over separation | relative error in the difference |
|---|---|
| 0 | 1.1e-6 |
| 1e3 | 1.5e-6 |
| 1e5 | 3.2e-4 |
| 1e7 | 7.0e-2 |

Everything downstream consumes squared *differences* of means, so centred data
is fine and uncentred data at `1e7` is not.

**`MergeScratch` is six `p`-length arrays per candidate pair**, 48 KB at
`p = 2000` in `f32`. That is over any sensible GPU shared-memory budget, so a
cubecl merge kernel would have to fuse `prepare` into the solve and recompute
the separations in registers rather than materialising them.

### The kernel in isolation

The pruning recursion at 8192 cells by 2000 features:

| | time |
|---|---|
| numpy oracle, same equations | 468 ms |
| Rust, `f64` storage | 67 ms |
| Rust, `f32` storage | 40 ms |

Loglikelihoods agree with `reference/bonsai_ref.py` to twelve significant
figures. That oracle covers the pruning kernel alone, not the merge score or the
branch solve. The parallel axis is the feature axis, not the tree level, so it
is indifferent to tree shape: a ladder runs in 9.0 ms where level-parallelism
takes 67.7 ms. This is not an end-to-end claim; [Now](#now) is.

### Reproducing

```bash
cargo bench --bench pipeline     # end to end
cargo bench --bench steps        # per-step attribution
cargo bench --bench kernels      # single-threaded throughput per kernel
cargo bench --bench start_tree   # does step 2 earn its keep
cargo bench --bench merge_scan   # one round of candidate-pair scoring
```

Run on an idle machine and check `uptime` first. An implausible speedup is the
signature of a sweep that did nothing.
