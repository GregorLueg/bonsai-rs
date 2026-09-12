# Performance: what worked, what did not, and why

A record of every optimisation attempted on this crate, kept because most of the
value is in the failures and in the reasoning that led to them. `CHANGELOG.md`
says what changed. This says what it cost, what it bought, and what the next
person should not repeat.

Every number here was measured on an M1 Max unless stated. Where a machine was
under load at the time, that is recorded, because it makes the seconds
pessimistic and the ratios trustworthy.

---

## The rules

Nine, all of them paid for.

### 1. Measure a component's share of the whole before optimising it

`BlockedState` was a feature-blocked parallel pruning sweep. At 8192 leaves by
2000 features it ran a ladder tree in 11.2 ms against level-parallelism's
67.7 ms. Real code, real tests, a real measurement.

It was deleted on 2026-09-06, because instrumenting the whole search rather than
the kernel showed **the prune is 2.1 to 2.4 per cent of a run and the up-sweep
another 0.5**, with the share flat in feature count. A perfect tenfold speedup
bought under 3 per cent of wall time. Cost: 324 lines and a doc comment that
misled every reader for a month.

### 2. Check that the pipeline calls it at all

Two modules on this project were built, tested, benchmarked and never wired in.
`BlockedState` was constructed in exactly two places, both inside a benchmark,
while this repo's `CLAUDE.md` called it "the production path" from the day it
was written.

The mirror image is worse. The kNN restriction and the ellipsoid bounds were
implemented and tested and *also* not wired in, and nobody noticed for a week
because the search still gave the right answer, just cubically. Wiring them up
was 1478.8 s to 37.4 s at 512 cells by 2000 features.

### 3. Pick the kernel by call count, not by how vectorisable it looks

`edge_newton` earned a SIMD tier at 35 to 45 calls per candidate pair.
`split_derivative` reads like the hot loop, runs **0.07 times per pair** because
the split solve almost always terminates at a bracket end, and measured flat.

### 4. An exponent measured at the wrong parameter value is worse than none

`benches/steps.rs` ran at 200 features and reported polytomy resolution at
`n^2.83` and the interchanges at `n^2.60`. Both were treated as the last
structural problems in the crate. Measured again at 2000 features, which is the
regime this crate is for, they are `n^1.08` and `n^0.97` and together are 3 per
cent of a run. Both exponents were round counts, and round counts collapse as
the feature axis grows because the likelihood landscape gets cleaner.

An hour of planning went into fixing two steps that did not need fixing.

### 5. Diagnose before fixing, and expect the first diagnosis to be wrong

Three for three so far.

- SPR was assumed to be paying for its `O(n p)` acceptance re-prune. That re-prune
  is **0.03 per cent** of the step, because the split fingerprint discards 99.7
  per cent of candidates before it runs. The cost was *proposing*: five `O(n p)`
  sweeps per candidate for rows of which a few dozen are read. 93.19 s to 6.71 s.
- NNI was assumed to be paying for its re-prune too. The cost was **the filter
  itself**: `interchange_at` built a whole tree and `split_fingerprint` walked
  it, `O(n)` each, per candidate, 1.47 s of 1.98 s at 4096 leaves. 1.98 s to
  0.75 s.
- SPR's remaining `n^1.46` was assumed to be the placement beam, and a
  FastTree-style tree-distance limit was planned for it. The beam is **flat**:
  41 to 49 nodes over a sixteenfold growth in `n`, `n^0.06`. The limit would
  have bought nothing.

### 6. Ablate the step, do not just optimise it

Asking "how do I make SPR faster" produced a plan. Asking "what happens if I
delete SPR" produced the answer, which is that at noise 0.3 it costs 20x the
wall clock and changes nothing at all.

### 7. Test the hard regime, not only the easy one

The same ablation, run at four noise levels instead of one, inverted. At noise
1.6 SPR wins 27.5 splits of 4090 and is 7.3x *faster* than not having it. Three
noise levels said delete it. The fourth said it is the thing that saves you.

### 8. A justification has a shelf life tied to the measurement behind it

"The tree sweeps are sequential and on measurement they do not need to be: the
whole prune is 2.4 per cent of a run" was correct when written. Then step 2 was
replaced by a linkage and the prune-based steps became a much larger share of
what is left. The sentence stayed true and stopped being a reason.

Any comment of the form "X is fine because Y is small" needs re-reading whenever
Y's denominator changes.

### 9. Look at core utilisation, not only wall time

The crate names three parallel axes and all three live in ingest or in search
step 2. Once step 2 was replaced, the pipeline got **1.26x out of ten cores**.
Nothing in the wall-clock numbers said so; it took a run at
`RAYON_NUM_THREADS=1` to see it.

---

## What worked

| change | effect | date |
|---|---|---|
| Wire in the kNN restriction and ellipsoid bounds | 1478.8 s to 37.4 s at 512 by 2000; `n^2.9` to `n^1.8` | 2026-09-01 |
| Lazy SPR proposal rows (`LazyRows`) | 93.19 s to 6.71 s at 2048 by 200; `n^1.98` to `n^1.47` | 2026-09-04 |
| Structural NNI filter over the star result | 1.98 s to 0.75 s at 2048 by 200; per round `n^1.53` to `n^1.02` | 2026-09-06 |
| `f64x4` tier on `edge_newton` | 0.95 to 0.71 ns per feature; 8 per cent of a whole run | 2026-09-12 |
| Adaptive ellipsoid sizing on redraw-versus-walk cost | 0.109 s against 0.170 and 0.175 for fixed schedules either side | 2026-08-31 |
| Ward linkage start replacing search step 2 | 66.08 s to 17.06 s at 2048 by 2000, identical tree | 2026-09-12 |
| Parallel edge solve in `optimise_branch_lengths` | 1.00x to 2.4x on ten cores, bit-identical | 2026-09-12 |
| Parallel edge scan in `nni_greedy` | 2.0x on top of the 3.8x it borrowed, bit-identical | 2026-09-12 |
| Edge solve started at `mean(d - s)`, stopped on the Newton correction | 22.4 to 4.6 passes per attachment; every `optimise_edge` caller | 2026-09-12 |
| Merge split by Illinois regula falsi instead of bisection | 58 to 20.5 derivative passes per pair | 2026-09-12 |
| SPR proposals in parallel chunks, restarted on acceptance, bit-identical | 200.04 s to 24.44 s at 16384 by 2000; `n^1.45` to `n^1.31` | 2026-09-12 |
| SPR acceptance on the candidate's own terms, rows assembled not swept | 2048 by 2000 at noise 1.6: 127.9 s to 19.1 s with the above | 2026-09-12 |
| Graph linkage by mutual-nearest-neighbour rounds instead of a chain | at 4096 by 2000, `k = 16`: depth 131 to 12, 248 splits from the dense tree to 0; `k = 8` now suffices where the chain needed 128 | 2026-09-12 |

Three notes on these.

**The two big rewrites were both about not materialising things.** `LazyRows`
forms the rows a proposal reads and no others; the NNI filter tests the star
result rather than building a tree and walking it. Neither made any kernel
faster.

**The SIMD tier is honest about its size.** 14 per cent of the merge scan and 8
per cent of a whole run at 2048 by 2000, `71.4 s` to `65.7 s`. That is worth
having and it is not a headline.

**The two parallel changes are bit-identical, deliberately.** The branch solve
writes one slot per node with no reduction. The NNI scan reduces to a running
best with ties broken on the lower node id, which is what the sequential scan
did implicitly, since the arena invariant makes ascending index order a
post-order. `test_the_greedy_phase_is_deterministic_whatever_the_thread_count`
pins it at 1, 3 and 8 threads.

---

## What did not work

| attempt | why not | date |
|---|---|---|
| `BlockedState`, feature-blocked parallel prune | 6x on a kernel that is 2.4 per cent of a run | 2026-09-06 |
| SIMD on `split_derivative` | runs 0.07 times per candidate pair; measured flat | 2026-09-12 |
| `edge_newton` with the division replaced by a multiply | 0.92 against 0.95 ns per feature, inside noise | 2026-09-12 |
| `edge_newton` with eight accumulator chains | 1.00 against 0.95, slower | 2026-09-12 |
| `edge_newton` with one division per four features | 1.23 against 0.95, much slower | 2026-09-12 |
| Ellipsoid sizing on walk depth | the walk is chunked, so every round reads shallow and the cap chose the answer | 2026-08-31 |
| Feature subsampling to rank merge candidates | argmax survives 4 of 15 checkpoints at a quarter of the features | 2026-09-12 |
| Chain or Boruvka agglomeration on the Bonsai merge gain | the score is not reducible: one round in five inverts, by up to 13 nats | 2026-09-12 |
| Tree-distance limit on SPR regrafts | the beam is `n^0.06`, so there is nothing to limit | 2026-09-12 |
| Warm-starting the beam's root find from the neighbour's optimum | 8.3 passes per solve against 5.8 from the mean start | 2026-09-12 |
| Counting sort in SPR's `assemble` | the height sort was never the cost; `Tree::from_parents` is | 2026-09-12 |
| Parallel pair scan on the four-member star inside the SPR proposal loop | 3.0 ms a resolution summed over threads against 0.7 sequential | 2026-09-12 |
| Redrawing the linkage graph more often to stop it chaining | fraction 0.9: depth 22 against 11; growth 4x: right tree at 12x the redraws; the chain was the cause, not the cadence | 2026-09-12 |

Two of these deserve their own paragraph.

**Feature subsampling.** The merge gain is a sum over `p` features, so estimating
it on `p'` of them and rescaling looks like a free 30x. It is not. Measured over
five checkpoints through a 128-member star at 2000 features, three seeds, with a
*shared* column set per checkpoint so the correlated part of the error is
already cancelled:

| subsample | argmax agreed | mean relative error | mean nats lost |
|---|---|---|---|
| 32 | 1 / 15 | 9.4e-1 | 94.6 |
| 128 | 3 / 15 | 3.7e-1 | 26.1 |
| 512 | 4 / 15 | 1.2e-1 | 12.0 |

The arithmetic says why. The estimator's standard deviation is
`p * sigma / sqrt(p')`, about 11 per cent of the gain at `p' = 512`, and
competing merges in a real round differ by far less than that. A random
*projection* is a different mechanism and is still open, since it summarises
every feature rather than discarding all but `p'`; it needs the precisions to be
approximately rank-1 in gene by cell, which has not been checked.

**Reducibility.** Merging changes the peeled remainder `R` that every other
pair's score depends on, so a merge can lift another pair above the score the
merged pair had. Measured at 64 and 128 members: 10 to 32 inversions over 61 to
125 rounds, worst excess 3.6 to 13.2 nats, up to 60 per cent relative. Ward has
no global remainder term and is reducible by construction, which is a second
reason to prefer it over a likelihood-driven linkage.

---

## Memory

**`f32` storage, `f64` accumulation, always.** The tree loglikelihood sums
thousands of features whose interesting differences are `O(1)` while the sum is
`O(p)`, so `f32` accumulation turns the convergence criterion into noise. `f32`
*storage* halves the working set the search streams and is the fastest path.

The cost of `f32` storage is a function of how far the means sit from zero,
measured 2026-08-31 over four leaves by 256 features:

| mean over separation | relative error in the difference |
|---|---|
| 0 | 1.1e-6 |
| 1e3 | 1.5e-6 |
| 1e5 | 3.2e-4 |
| 1e7 | 7.0e-2 |

Everything downstream consumes squared *differences* of means, so centred data
is fine and uncentred data at `1e7` is not.

**Dense `n x n` anything is dead above about 20k cells.** The Ward experiment's
distance matrix is 2.1 GB at 16384 in `f64` and 4 TB at 1M. A `k = 16` neighbour
graph at 1M is 128 MB. That, not the flop count, is what forces the linkage
through `ann-search-rs`.

**`MergeScratch` is six `p`-length arrays per candidate pair**, 48 KB at `p =
2000` in `f32`. That is over any sensible GPU shared-memory budget, so a cubecl
merge kernel has to fuse `prepare` into the solve and recompute the separations
from `M` and `W` in registers rather than materialising them.

**`NodeState::prune` asserts on the node count**, so it cannot be reused across
trees of different shape. Polytomy resolution therefore reallocates and refills
two `n x p` slabs on every sweep. That is invisible at 2000 features, where the
step is 0.1 per cent of a run, and would not be at 200.

---

## Where the time goes, 2026-09-12

Ward start, 2000 features, two seeds, Robinson-Foulds 0 at every size.

At 16384 leaves, before and after the two parallel changes above. Ten cores.

| step | before | after | gain |
|---|---|---|---|
| ward linkage | 60.94 | 59.49 | 1.0x |
| 3 polytomy | 0.32 | 0.30 | 1.1x |
| 4 branch | 25.06 | 10.26 | **2.4x** |
| 5 SPR | 200.81 | 200.04 | 1.0x |
| 6 NNI | 9.03 | 4.69 | **1.9x** |
| 7 branch | 1.40 | 0.65 | **2.2x** |
| total | 297.55 | 275.43 | 1.08x |

Nineteen seconds off 297, for free and bit-identical. The total barely moves
because SPR is 73 per cent of the run and untouched, which is rule 1 pointing at
itself: the two steps that were easy to parallelise were the two that did not
matter. What the change does buy is that branch optimisation and the
interchanges are no longer on the critical path at any size.

Exponents over the full sweep, unchanged by threading: linkage 2.00, polytomy
0.98, branch 1.03, SPR 1.45, NNI 0.98, total 1.52. Two things set the asymptote
and nothing else does:

1. The linkage at `n^2.00`, which is a dense distance matrix and has to become a
   neighbour graph.
2. SPR at `n^1.45`, which is `O(n)` candidates each paying an `O(n)` arena
   rebuild and an `O(n)` fingerprint, so `Theta(n^2)` of pure bookkeeping, and
   which runs at 1.18x on ten cores.

## Against the reference implementation

`/Users/gregorlueg/repos/others/bonsai-comparison`, run 2026-09-06. Matched
inputs, both implementations scored the same way.

| config | ours | reference | speedup | RF ours vs reference |
|---|---|---|---|---|
| 256 by 1000 | 1.87 s | 148.22 s | 79x | 0 |
| 1024 by 200 | 5.64 s | 241.96 s | 43x | 2 |
| 2048 by 200 | 17.22 s | 491.77 s | 29x | 2 |

Distance recovery matches to four decimal places throughout. Two deltas matter
and they want opposite things: the speed delta should widen, and
`rf_ours_vs_reference` should not. Rerun this after any change to the search.

`reference/bonsai_ref.py` is a different thing and covers the pruning kernel
alone, agreeing to twelve significant figures. It has no oracle for the merge
score or the branch solve, which is where most changes land.

## SPR, 2026-09-12: the fourth wrong diagnosis and what the profile said

Rule 5 held. The brief for this pass carried three beliefs about where step 5
spends its time, and the profile agreed with one of them.

- **The beam's root find dominates.** Half true. At 2048 by 2000 from a Ward
  start, noise 0.3, `place` was 64 per cent of the step: 48 in the solve and 16
  in forming the effective leaves. The solve was 22.4 Newton passes per
  attachment, and half of those were wasted (below).
- **The solve does not use the SIMD tier.** Wrong. `model::branch` imports
  `edge_newton_simd as edge_newton`; every caller had it.
- **The acceptance path is 0.03 per cent, do not touch it.** True at noise 0.3,
  where nothing is accepted, which is where it was measured. At noise 1.6,
  where SPR is the step that saves the run (rule 7), it was **33 to 41 per
  cent**: one candidate in twenty passes the split filter, each paid a fresh
  `O(n p)` prune, and each accepted one paid a second to settle the rows.

Two costs nobody had named at all:

- **The four-member star.** 77 per cent of regrafts land on an internal node,
  so the polytomy resolution of SPEC.md section 7.3 runs after nearly every
  proposal. Its six pair scores were 1.8 ms single-threaded, and 77 per cent of
  that was `MergeScratch::optimise_split`: a bisection to `1e-8` on the
  analytic derivative, 29 passes per sweep, 58 per pair. The kernel rule 3
  records as running 0.07 times per pair on the merge scan runs 58 times per
  pair here. Same kernel, different caller, different answer.
- **The step had no parallelism of its own.** The 1.18x on ten cores was the
  inner pair scan of that star, six pairs spread over ten threads.

### What changed

All measured on an M1 Max at load 4 to 9 throughout (another agent was
building in the same tree), 2000 features, Ward start, seed 31. Seconds are
pessimistic; ratios are interleaved and trustworthy.

| change | measure | before | after |
|---|---|---|---|
| Edge solve starts at `mean(d - s)` and stops on the Newton correction | Newton passes per attachment | 22.4 | 4.6 |
| Split solve by Illinois regula falsi, tolerance `1e-8` to `1e-10` | derivative passes per pair | 58 | 20.5 |
| Candidates proposed in parallel chunks, decided in order, chunk restarted on acceptance | 2048 at noise 0.3, ten threads | 12.40 s | 1.84 s |
| Acceptance on the candidate's own terms, rows assembled rather than swept | 2048 at noise 1.6, ten threads | 127.9 s | 19.1 s |
| Split fingerprint as a multiset hash instead of a sorted vector | fingerprint per candidate at 4096 | 59 us | 26 us |

The termination fix is worth its own sentence because it was a real defect,
not a tuning. Near the root the Newton correction points exactly at a bracket
end; the safeguard refuses it as not strictly inside and bisects a bracket
already narrower than the tolerance, one full pass per halving, until the
midpoint happens to satisfy the step test. Seven passes of fifteen on a
typical solve, on every `optimise_edge` call in the crate.

The parallel round is exact. A chunk is proposed against one tree and decided
in order; the first acceptance throws the rest of the chunk away and proposes
it again against the new tree, so every candidate that is decided was proposed
against the tree it is decided against, which is the sequential sweep's
invariant. The tree, the branches, the loglikelihood and the move list are the
same at 1, 3 and 8 threads and at every chunk floor tried
(`test_the_sweep_is_deterministic_whatever_the_thread_count`). The cost is the
discarded work, which is why the chunk floor is 8 and not 32: at 2048 by 2000,
noise 1.6, 641 of 27987 candidates accepted, 49614 proposals were made at a
floor of 32 and 37846 at 8, 24.1 s against 19.1 s.

Two things had to change elsewhere for the incremental acceptance to be cheap:

- `polytomy::rebuild` numbered nodes in a post-order that visited siblings
  last-first, so every splice reversed the order of equal-height siblings and
  `LazyRows` could not recognise an untouched subtree by its children. Fixed
  by pushing children in reverse; the incremental score went from 8.5 ms to
  0.7 ms per candidate.
- The star primitive's pair scan now runs on the calling thread below 64 pairs
  (`PAR_PAIRS_MIN`). Inside the proposal loop, a six-pair `par_iter` hands
  half its pairs to a worker that is busy with a whole other proposal and
  waits for it.

### Where the step's time goes now

Single thread, 2048 by 2000, noise 0.3, 10.24 s over 4093 candidates:

| part | share | note |
|---|---|---|
| beam: solve | 28 | 17.6 us per attachment, 39.5 attachments per candidate |
| beam: effective leaves | 17 | 10.7 us per attachment, scalar, division-bound |
| star resolution | 29 | of which split 15, prepare 4, root branch 3, total 2 |
| arena rebuilds and rows | 21 | `Tree::from_parents` is 58 per cent of the rebuilds, two per candidate |
| fingerprint | 1 | |

The rebuilds-and-rows line is the exponent. It is `O(n)` per candidate with
a large constant, `1.0 ms` of `2.9 ms` at 4096 and roughly half of the step at
16384. Its counting-sort replacement for `assemble`'s height sort measured
nothing, because the sort was not the cost; the passes and the allocations
are, and most of them are inside `Tree::from_parents`.

### What did not work

| attempt | why not |
|---|---|
| Warm-start the beam's root find from the neighbour's optimum | 8.3 passes per solve against 5.8 from the mean start; adjacent attachment points want different edge lengths |
| Counting sort in `assemble` | no measurable change at 4096; the sort was never the cost |
| Keeping the four-member star's pair scan parallel | 3.0 ms per resolution summed over threads against 0.7 sequential, inside the proposal loop |

### The tables

Robinson-Foulds gate, `benches/start_tree.rs` `spr` block, two seeds, ten
threads, load 7 to 18. The with-SPR arm's seconds include steps 6 and 7.

| leaves, noise 0.3 | with SPR s | RF | without s | RF | beam nodes |
|---|---|---|---|---|---|
| 1024 | 0.84 | 0.0 | 0.19 | 0.0 | 41.0 |
| 2048 | 1.80 | 0.0 | 0.37 | 0.0 | 42.9 |
| 4096 | 4.12 | 0.0 | 0.75 | 0.0 | 44.9 |
| 8192 | 10.18 | 0.0 | 1.49 | 0.0 | 47.0 |
| 16384 | 27.32 | 0.0 | 2.94 | 0.0 | 49.0 |

| 2048, noise | with SPR s | RF | without s | RF | splits won |
|---|---|---|---|---|---|
| 0.3 | 1.81 | 0.0 | 0.37 | 0.0 | 0.0 |
| 0.6 | 1.97 | 0.0 | 0.44 | 0.0 | 0.0 |
| 1.0 | 5.90 | 5.0 | 18.06 | 3.0 | -2.0 |
| 1.6 | 19.83 | 370.5 | 219.04 | 397.0 | 26.5 |

At noise 1.6 SPR still wins its 26.5 splits of 4090 (27.5 before) and is now
11x faster than not having it (7.3x before). The two splits it loses at noise
1.0 are inside the seed-to-seed spread of three to five.

Per step, `steps` block, ten threads, **load 18 to 24** while it ran, so the
seconds are pessimistic against the 2026-09-12 table above and the ratios are
what to read:

| step | before | after | gain |
|---|---|---|---|
| ward linkage | 59.49 | 60.47 | 1.0x |
| 3 polytomy | 0.30 | 0.32 | 0.9x |
| 4 branch | 10.26 | 8.85 | 1.2x |
| 5 SPR | 200.04 | 24.44 | **8.2x** |
| 6 NNI | 4.69 | 2.33 | 2.0x |
| 7 branch | 0.65 | 0.59 | 1.1x |
| total | 275.43 | 97.00 | 2.8x |

Steps 4, 6 and 7 gained from the two solver changes alone, since they share
`optimise_edge` and the merge split. SPR's exponent over 1024 to 16384 is
1.31, from 1.45, and it is still rising with size: 1.15, 1.21, 1.38, 1.49 per
doubling, which is the `O(n)` bookkeeping per candidate taking over from the
`O(p)` beam as `n` grows. Single-threaded, 2048 by 2000 at noise 0.3 runs in
10.24 s against 12.40 s on ten threads before, so the step is now 5.6x faster
on ten threads than on one, from 1.18x.

### What next

The exponent. Every candidate materialises two arenas and one candidate tree,
and none of them is needed until a move is accepted. A pruned *view* over the
current tree (the cut, the suppressed node, the two branch sums) answers
every question the beam asks in `O(1)`; the attachment star can be read off
that view's rows plus the pruned subtree's own row, since the up row at the
attachment point does not change when something is hung below it; the
candidate's terms are the view's terms plus the recomputed chain above the
attachment; and the fingerprint is a sum, so the splits the move changes can
be subtracted and added in `O(depth)`. That makes a proposal
`O(depth * p + beam)` and leaves `O(n)` for accepted moves only, which are one
candidate in forty at the hard noise and none at the easy one. It is the third
rewrite of the module and the first one that changes its complexity class.

Below that: the effective-leaf formation (17 per cent, four scalar divisions a
feature) and `split_derivative` (15 per cent, four more) are the two remaining
division-bound scalar loops, and `f64x4` division is the obvious tier for
both, subject to rule 3's measurement.

## The graph linkage, 2026-09-12: it chained, and cadence was not why

Rule 5 again, five for five. The brief said the neighbour-graph Ward linkage
chains because the graph goes stale between rebuilds, and asked for a cadence
sweep first. The sweep was run; the cadence was not the cause.

Exhaustive backend, 2000 features, two seeds, the drift block of
`benches/start_tree.rs`. `dense d` is the depth of the dense Ward tree, which
recovers the balanced generator exactly, so it is also `log2(n)`.

**Every seconds column in this section was taken on a machine at load 13 to
65** (another agent was running Sanity preprocessing on eight threads).
Depth, Robinson-Foulds and the trace are exact whatever the load; the seconds
are order-of-magnitude sanity checks and nothing else. The quiet re-runs are
listed at the end.

### The chain, as it was

| leaves | k | build s | to dense | depth | dense d |
|---|---|---|---|---|---|
| 512 | 16 | 0.09 | 10 | 13.5 | 9 |
| 1024 | 16 | 0.19 | 44 | 29.5 | 10 |
| 2048 | 16 | 0.46 | 104 | 56.5 | 11 |
| 4096 | 16 | 1.32 | 248 | 131.0 | 12 |

The cadence sweep at `k = 16`, with the halving fraction raised and with a new
trigger that redraws once any cluster has grown to a multiple of its size when
its list was drawn:

| leaves | fraction | growth | build s | to dense | depth |
|---|---|---|---|---|---|
| 2048 | 0.50 | off | 0.45 | 104 | 56.5 |
| 2048 | 0.90 | off | 0.55 | 30 | 22.0 |
| 2048 | 0.50 | 4 | 2.63 | 0 | 11.0 |
| 2048 | 0.50 | 2 | 28.82 | 0 | 11.0 |

Raising the fraction helps and does not fix it. The growth trigger fixes it at
every size from 512 to 2048, at six times the build cost, because it fires 161
times at 2048 where the halving fires 13. So the depth was a staleness symptom
and the trigger was the right instinct, but paying for it by redrawing the
graph was the wrong price. What the trace said:

```
REBUILD live=2017 deadend  min=1 med=1 max=32
REBUILD live=1889 deadend  min=1 med=1 max=160
MERGE   live=1595 sa=160 sb=16 deg_a=18 deg_b=1
MERGE   live=1594 sa=176 sb=16 deg_a=17 deg_b=1
```

**The chain is depth-first.** After 31 merges one cluster is size 32 while
every other cluster is a singleton. A centroid of `s` cells carries `1/s` of
the noise, and at 2000 features that noise term is 360 in squared distance
between two leaves against 333 per level of signal, so a big centroid sits
closer to any leaf than that leaf's own third cousins do and takes a slot in
every list at the next redraw. Once a block of 16 has merged its listed
relatives, the big cluster is the only edge it has left (`deg_b=1`), the mutual
nearest-neighbour test passes trivially, and the block is absorbed. One block
per merge: a caterpillar.

### The fix

Stop being depth-first. Each round finds every live cluster's nearest listed
neighbour in parallel, merges every mutually nearest pair, then updates. Ward
is reducible, so each such pair is one the sequential linkage would merge and
the dendrogram is identical (the complete-graph test in `tree::linkage` pins
that). Sizes stay level-synchronous, so no cluster becomes the attractor, and
the union lists are only ever one level stale.

| leaves | k | build s | to dense | depth | dense d |
|---|---|---|---|---|---|
| 512 | 8 to 128 | 0.06 to 0.18 | 0 | 9 | 9 |
| 1024 | 8 to 128 | 0.12 to 0.28 | 0 | 10 | 10 |
| 2048 | 8 to 128 | 0.29 to 0.72 | 0 | 11 | 11 |
| 4096 | 8 to 128 | 0.63 to 1.19 | 0 | 12 | 12 |

Every `k` from 8, every cadence including never redrawing on the count. The
chain needed `k = 128` for the same row at 4096, and its `k` grew as
`1.5 sqrt(n)`; the rounds need 8 and it does not grow. The build seconds were
taken at load 37 to 50 and are pessimistic; the previous table's were at load
4 to 13.

The hard regime, 2048 leaves, `k = 16`:

| tree | noise | to dense | to truth | depth | dense d | dense to truth |
|---|---|---|---|---|---|---|
| balanced | 0.3 | 0 | 0 | 11 | 11 | 0 |
| balanced | 1.0 | 0 | 112 | 12 | 12 | 112 |
| balanced | 1.6 | 137 | 1383 | 15 | 14 | 1317 |
| unbalanced | 0.3 | 0 | 1021 | 14.5 | 14.5 | 1021 |
| unbalanced | 1.0 | 146 | 1474 | 15 | 14.5 | 1464 |

Identical to dense wherever dense is exact; where they part, the same distance
to the truth and a depth within one level. The unbalanced generator is where
Ward itself is far from the truth, which is the refinement's job, not the
start's.

### The backend crossover, not yet placed

`resolve_backend` carried placeholder thresholds that sent everything above
4096 cells to kmknn. The `backend` block ran the three backends interleaved
per seed, at load 50 to 65:

| leaves | exhaustive s | kmknn s | NN-descent s | trees |
|---|---|---|---|---|
| 4096 | 0.78 | 2.98 | 1.33 | identical |
| 8192 | 1.44 | 8.32 | 2.39 | identical |
| 16384 | 5.20 | 30.32 | 4.48 | identical |

The one load-independent fact in that table is the last column: NN-descent's
graph builds the identical tree at every size, so switching to it loses
nothing the linkage can see. The seconds are unusable for placing the
crossover and were not used for it. kmknn is out of the automatic path on
the earlier quiet-machine number, 43.82 s at 8192 against 1.34 s for
exhaustive at 4096; the threshold stays at 4096 with NN-descent above it, and
is marked provisional on the constant.

### The end-to-end gate, provisional

`steps`, both starts interleaved per seed, load 44 to 57. Sanity check only.

| leaves | start | linkage | 3 poly | 4 branch | 5 spr | 6 nni | 7 branch | total | RF |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | dense ward | 26.39 | 0.15 | 5.49 | 15.06 | 1.99 | 0.36 | 49.44 | 0 |
| 8192 | graph | 1.38 | 0.13 | 5.32 | 13.66 | 1.73 | 0.38 | 22.60 | 0 |
| 16384 | dense ward | 100.90 | 0.45 | 10.55 | 40.81 | 3.90 | 0.68 | 157.29 | 0 |
| 16384 | graph | 3.70 | 0.31 | 9.88 | 35.97 | 3.27 | 0.67 | 53.80 | 0 |

Consistent with SPR at parity between the two starts (the chain start was 3
to 7x the dense one) and with the linkage no longer setting the asymptote,
but the machine was at load 50 and the gate is not passed until it is re-run
quiet.

### Queued for a quiet machine

1. `cargo bench --bench start_tree -- steps`, the gate.
2. `cargo bench --bench start_tree -- backend`, to place `EXHAUSTIVE_MAX_CELLS`.
3. `cargo bench --bench start_tree -- drift`, for the build-seconds columns
   and the cost of the halving redraw against the dry-list one alone.

### What did not work

| attempt | why not |
|---|---|
| Raising the halving fraction to 0.9 | depth 22 against 11 at 2048; the graph was not stale, the chain was lopsided |
| Redrawing once a cluster has grown 4x since its list was drawn | correct tree, 161 redraws for 13, six times the build |
| Redrawing at 2x growth | correct tree, sixty times the build |
