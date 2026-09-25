# Performance

Where the time goes, the rules we learnt, and a log of what worked and what
didn't. Timings on a ten-core M1 Max unless stated. Log dates are first commit;
this file started 2026-09-12, so rows dated then were measured on or before it.

## Now

`BonsaiParams::default()` on Sanity-preprocessed Baron, 2026-09-25, load average
6 to 7:

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

On 2026-09-13 the same runs took 4.7 s, 166 s and 611 s. From 5,000 to 10,000
cells: total `n^1.58`, SPR `n^1.60`, NNI `n^1.53`. Two points over two gene
panels, so a slope, not a law.

- SPR is three fifths of the run, branch optimisation (4, 7, 8) over a quarter.
  Neither has a safe approximation left; the log has the ones that failed.
- Full-tree sweeps are small: prune plus up-sweep is 2.8 per cent of busy thread
  time at 5,000.
- A third of SPR's thread time is idle. Round one accepts a move every few
  candidates, so chunks sit at the floor of eight proposals on ten threads; a
  bigger floor was slower.
- Hottest code by self time: `edge_newton_simd`, `split_derivative`, `log`,
  all in the branch solve inside merges and placements.

Counts to tree end to end is in [comparison](COMPARISON.md).

## Rules

1. **Measure a component's share before optimising it.** The share caps the
   return. `BlockedState` was 6x on a prune that's 2.4 per cent of a run.
2. **Check the pipeline calls it.** `BlockedState` was only built by a
   benchmark; the kNN restriction and ellipsoid bounds sat unwired for a week
   while the search ran cubic.
3. **Pick kernels by call count, not looks.** `edge_newton` runs 35 to 45 times
   per pair and earned SIMD; `split_derivative` looks hot, runs 0.07 times, and
   measured flat.
4. **Measure exponents at the shipped parameters.** At 200 features polytomy and
   NNI went `n^2.83` and `n^2.60`; at 2,000, `n^1.08` and `n^0.97`.
5. **Expect the first diagnosis to be wrong.** Four for four on SPR and NNI; see
   [Diagnoses that were wrong](#diagnoses-that-were-wrong).
6. **Ablate at more than one noise level.** At low noise SPR costs 20x and
   changes nothing; at noise 1.6 it wins 27.5 of 4,090 splits and is 7.3x
   *faster* than skipping it.
7. **Justifications expire.** "X is fine because Y is small" needs rechecking
   when Y's denominator moves. The linkage made prune-based steps most of the
   run; the prune itself was still 2.8 per cent, but that had to be re-measured.
8. **Smaller isn't cheaper.** Collapsing zero-length edges before step 5 drops
   an eighth of the nodes and makes SPR 30 per cent slower: regrafts land next
   to bigger stars.
9. **A generator isn't the data.** Ward tied on synthetic and wins on real.
   Splice pairs cost 5 to 11 nats synthetic, 437 to 737 real.
10. **Sweep sizes, not just shapes.** Revisit radius 3 held on every shape and
    at 5,000 cells, then broke at 10,000.
11. **Watch core utilisation.** After the linkage change the pipeline got 1.26x
    from ten cores, invisible until a `RAYON_NUM_THREADS=1` run.
12. **Time on an idle machine, alternating against a kept baseline.** On
    2026-09-24 load wandered 2 to 27 and the same step 6 measured 98 s and
    156 s. Quality survives load; timings don't.
13. **One real-data run is one draw.** Equally valid searches spread 460 nats at
    5,000 and 1,700 at 10,000; see
    [How much one real-data run says](#how-much-one-real-data-run-says). Trust an
    approximation that reproduces the exact tree; treat a few hundred nats as
    noise unless it repeats across sizes.

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

The big wins all stop materialising things: lazy rows form only what a proposal
reads, the NNI filter tests the star result instead of building a tree, the slot
store writes only changed rows. None made a kernel faster.

The parallel changes are bit-identical on purpose: the branch solve writes one
slot per node, the NNI scan breaks ties on the lower node id.
`test_the_greedy_phase_is_deterministic_whatever_the_thread_count` pins it at 1,
3 and 8 threads.

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

- **SPR's `O(n p)` acceptance re-prune.** 0.03 per cent of the step: the split
  fingerprint discards 99.7 per cent of candidates first. The cost was
  *proposing*, five `O(n p)` sweeps per candidate to read a few dozen rows.
- **NNI's re-prune.** The cost was the filter, which built and walked a whole
  tree per candidate.
- **The placement beam behind SPR's `n^1.46`.** The beam is flat: 41 to 49 nodes
  over a sixteenfold growth in `n`.
- **SPR leaving work for NNI at 5k to 10k.** SPR was accepting rounding noise and
  cycling.

The acceptance path did come back later. Once proposals were cheap, the
`O(n p)` state copy per accepted move was 23 of 42 s in SPR's first round at
5,000 cells (2026-09-24), all sequential with nine threads idle. The slot store
removed it.

### Revisit radius

After the first SPR sweep, only subtrees within `r` edges of a clade the last
sweep created are proposed again: TSP's don't-look bits. Before it, later rounds
cost two thirds of step 5 for a quarter of the moves.

Measured 2026-09-24 over thirteen datasets: balanced, random-branch and
unbalanced trees at noise 0.4, 1.0, 1.6 at 4,096 by 1,000, plus the four Sanity
configurations. `r = 5` is within a few nats of exact on all. `r = 3` too,
except at 10,000 cells, where step 6 inherits what step 5 skipped. Table in
`DEFAULT_REVISIT_RADIUS`, `src/search/spr.rs`.

### Lazy NNI

As specified, the greedy phase rescores every edge for each single interchange:
1.1 s a round at 5,000 cells and 2.2 s at 10,000, for 27 and 44 moves.
`NniSearch::Approximate` caches each edge's gain under its leaf set, rescores
only edges within five of the clades a move created, and rechecks the leading
cached gain before taking it (Minoux 1978 lazy greedy). When nothing cached
improves, a full scan runs, so it stops exactly where the exact phase does.

2026-09-25, same thirteen datasets, steps 5 to 8: identical finished tree on
all, to the last printed digit. On step 6 alone, radius 2 and 3 drifted 0.01 and
7.5 nats at 5,000; radius 5 was identical at both sizes. What's left is the
per-round settle, about 0.28 s at 10,000, which a lazy row provider like SPR's
could remove.

### How much one real-data run says

Not much. Runs differing only in SPR's subtree order are equally valid and
spread widely (2026-09-25):

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

About 460 nats and recovery 0.58 to 0.67 at 5,000; 1,700 nats and 0.40 to 0.48
at 10,000. A few hundred nats on one run is noise. An approximation is only
proven harmless when it reproduces the exact result outright, as lazy NNI does.
The revisit radius does on the 512-cell and low-noise sets; at 5,000 and 10,000
it lands 3 nats below and 467 above exact, both inside the spread. Same caution
the other way: the loose branch tolerance's 1,281-nat, 0.38-recovery loss at
10,000 looks like a bad basin, not a measured cost.

Two basins, one with visibly worse recovery, and the loglikelihood separates
them every time. Best-of-several by loglikelihood is cheap now a run is a few
minutes. The specified longest-branch order isn't dominated: random did better
at 5,000 and worse at 10,000.

### Splice pairs through the regrafted subtree

A regraft onto an internal node leaves a four-member star, resolved by scoring
all six pairs. A pair and its complement give the same unrooted tree, so the
three pairs containing the regrafted subtree already cover every topology. Half
the splice for free? No. Against the radius-5 default, 2026-09-24:

| data | steps 5 to 8 | loglikelihood | Robinson-Foulds | recovery |
|---|---|---|---|---|
| 5,000 real, all six pairs | 84.8 s | -5,557,978 | 1,288 | 0.666 |
| 5,000 real, three pairs | 131.3 s | -5,558,415 | 1,310 | 0.545 |
| 10,000 real, all six pairs | 257.8 s | -11,252,721 | 2,624 | 0.476 |
| 10,000 real, three pairs | 566.0 s | -11,253,458 | 2,567 | 0.384 |

Worse branch lengths make SPR accept 5 to 20 per cent more, cheaper moves, and
NNI then runs two to four times longer. Step 7 doesn't repair the recovery loss:
the damage is in the topology.

### Feature subsampling

The merge gain sums over `p` features, so estimating on `p'` and rescaling looks
like a free 30x. Five checkpoints through a 128-member star at 2000 features,
three seeds, one shared column set per checkpoint:

| subsample | argmax agreed | mean relative error | mean nats lost |
|---|---|---|---|
| 32 | 1 / 15 | 9.4e-1 | 94.6 |
| 128 | 3 / 15 | 3.7e-1 | 26.1 |
| 512 | 4 / 15 | 1.2e-1 | 12.0 |

The estimator's SD is `p * sigma / sqrt(p')`, about 11 per cent of the gain at
`p' = 512`; competing merges differ by far less. A random *projection* is still
open, since it summarises every feature, but needs the precisions to be roughly
rank-1 in gene by cell. Unchecked.

### Reducibility

A merge changes the peeled remainder every other pair's score depends on, so it
can lift another pair above the merged pair's score. At 64 and 128 members: 10
to 32 inversions over 61 to 125 rounds, worst excess 3.6 to 13.2 nats, up to 60
per cent relative. Ward is reducible by construction, another reason to start
from it.

### Starting tree

`StartTree::Linkage` is the default; `StartTree::GreedyMerge` is SPEC 9.1. Use
the linkage for real work, the greedy merge to reproduce the paper.

2026-09-13, Sanity-preprocessed Baron, same genes, scored against the
generating tree:

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

Linkage wins loglikelihood and Robinson-Foulds at every size, recovery at three
of four, and is 3 to 5x faster. The exception, recovery at 512, flips on the
replicate.

The greedy criterion chains. Mean leaf depth after step 2:

| cells | greedy | linkage | log2(n) |
|---|---|---|---|
| 512 | 18.7 | 8.9 | 9.0 |
| 5,000 | 114.5 | 12.5 | 12.3 |
| 10,000 | 151.7 | 13.5 | 13.3 |

Steps 3 to 7 then spend their budget repairing it: at 10,000 the greedy start
leaves SPR 11,330 accepted moves over 32 rounds and NNI 457 over 458, against
4,755 and 47 from the linkage. A cluster's effective leaf carries `1/size` of
the noise, so at a few thousand features a big cluster sits closer to each
member than its own relatives do and swallows them one by one. Ward is biased
the other way: Lance-Williams distance grows with cluster size.

Rescheduling doesn't fix it. Merging every mutually-best pair per round made
step 2 7 to 8x faster with a *better* merge score and a worse tree:
Robinson-Foulds 175 to 186 at 512, 1,368 to 1,425 at 5,000, depth only down to
16.1 and 74.1. A big cluster still ends up every singleton's preferred partner;
those singletons are mutual with nothing, the round starves, and the chain
builds anyway.

### Earlier snapshots

2026-09-12, Ward start, 16384 synthetic leaves by 2000, before the slot store
and revisit radius: linkage 22 per cent, step 4 4, SPR 73, NNI 2. Exponents:
linkage 2.00, polytomy 0.98, branch 1.03, SPR 1.45, NNI 0.98, total 1.52.

At 512 by 2000, identical loglikelihoods throughout: exhaustive scan 1478.8 s,
kNN and ellipsoid bounds 37.4 s, lazy SPR and structural NNI 9.1 s.

## Reference

### Memory

**Dense `n x n` is dead above about 20k cells.** A Ward distance matrix is
2.1 GB at 16384 in `f64`, 4 TB at a million. A `k = 16` neighbour graph at a
million is 128 MB. That's why the linkage goes through `ann-search-rs`.

**`f32` accuracy depends on how far means sit from zero** (four leaves, 256
features):

| mean over separation | relative error in the difference |
|---|---|
| 0 | 1.1e-6 |
| 1e3 | 1.5e-6 |
| 1e5 | 3.2e-4 |
| 1e7 | 7.0e-2 |

Centred data is fine; uncentred at `1e7` isn't.

**`MergeScratch` is six `p`-length arrays per pair**, 48 KB at `p = 2000` in
`f32`. Over any sensible GPU shared-memory budget, so a cubecl merge kernel
would need to fuse `prepare` into the solve and recompute separations in
registers.

### The kernel in isolation

Pruning at 8192 cells by 2000 features:

| | time |
|---|---|
| numpy oracle, same equations | 468 ms |
| Rust, `f64` storage | 67 ms |
| Rust, `f32` storage | 40 ms |

Loglikelihoods match `reference/bonsai_ref.py` to twelve significant figures
(pruning only, not merge or branch solve). Parallel over features, so a ladder
runs in 9.0 ms where level-parallelism takes 67.7 ms. Kernel only; [Now](#now)
is the end-to-end number.

### Reproducing

```bash
cargo bench --bench pipeline     # end to end
cargo bench --bench steps        # per-step attribution
cargo bench --bench kernels      # single-threaded throughput per kernel
cargo bench --bench start_tree   # does step 2 earn its keep
cargo bench --bench merge_scan   # one round of candidate-pair scoring
```

Check `uptime` first. An implausible speedup means a sweep that did nothing.
