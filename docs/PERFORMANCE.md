# Performance

What worked, what did not, and the rules that came out of it. `docs/DESIGN.md`
says how the crate is built; this says how it got fast and what it cost.

All timings are from one M1 Max unless stated. They are here so that a constant's
doc comment has somewhere to point, and so that the next person does not repeat
a failure that has already been paid for.

## The rules

**Measure a component's share of the whole before optimising it.**
`BlockedState` was a feature-blocked parallel prune. At 8192 leaves by 2000
features it ran a ladder in 11.2 ms against level-parallelism's 67.7 ms. Real
code, real tests, a real measurement. It was deleted once the whole search was
instrumented rather than the kernel: the prune is 2.1 to 2.4 per cent of a run
and the up-sweep another 0.5, with the share flat in feature count. A perfect
tenfold speedup there buys under 3 per cent of wall time. Cost: 324 lines and a
doc comment that misled every reader for a month.

**Check that the pipeline calls it at all.** `BlockedState` was constructed in
exactly two places, both inside a benchmark, while `CLAUDE.md` called it the
production path. The mirror image is worse: the kNN restriction and the
ellipsoid bounds were implemented, tested, and also not wired in, and nobody
noticed for a week because the search still gave the right answer, just
cubically. Wiring them up was 1478.8 s to 37.4 s at 512 cells by 2000 features.

**Pick the kernel by call count, not by how vectorisable it looks.**
`edge_newton` earned a SIMD tier at 35 to 45 calls per candidate pair.
`split_derivative` reads like the hot loop, runs 0.07 times per pair because the
split solve almost always terminates at a bracket end, and measured flat.

**An exponent measured at the wrong parameter value is worse than none.**
`benches/steps.rs` at 200 features reported polytomy resolution at `n^2.83` and
the interchanges at `n^2.60`, and both were treated as the crate's last
structural problems. At 2000 features, which is the regime this crate is for,
they are `n^1.08` and `n^0.97` and together are 3 per cent of a run. Both
exponents were round counts, and round counts collapse as the feature axis grows
because the likelihood landscape gets cleaner.

**Diagnose before fixing, and expect the first diagnosis to be wrong.** Four for
four.

- SPR was assumed to be paying for its `O(n p)` acceptance re-prune. That
  re-prune is 0.03 per cent of the step, because the split fingerprint discards
  99.7 per cent of candidates before it runs. The cost was *proposing*: five
  `O(n p)` sweeps per candidate for rows of which a few dozen are read.
- NNI was assumed to be paying for its re-prune too. The cost was the filter
  itself, which built a whole tree and walked it, `O(n)` each, per candidate.
- SPR's remaining `n^1.46` was assumed to be the placement beam. The beam is
  flat: 41 to 49 nodes over a sixteenfold growth in `n`, `n^0.06`.
- The 5k-to-10k slowdown was assumed to be SPR leaving work for NNI. It was SPR
  accepting rounding noise and cycling.

**Ablate the step, do not just optimise it.** Asking how to make SPR faster
produced a plan. Asking what happens if SPR is deleted produced the answer.

**Test the hard regime, not only the easy one.** That same ablation, run at four
noise levels instead of one, inverted. At low noise SPR costs 20x the wall clock
and changes nothing. At noise 1.6 it wins 27.5 splits of 4090 and is 7.3x
*faster* than not having it. Three noise levels said delete it. The fourth said
it is the thing that saves you.

**A justification expires with the measurement behind it.** "The tree sweeps are
sequential and do not need to be, because the prune is 2.4 per cent of a run"
was true when written. Then step 2 was replaced by a linkage and the prune-based
steps became a much larger share of what was left. Any comment of the form "X is
fine because Y is small" wants re-reading whenever Y's denominator changes.

**A smaller input is not automatically a cheaper one.** Collapsing the
zero-length internal edges before step 5 takes the 10k tree from 19,998 nodes to
17,747, and SPR then runs 589 s against 451 on 2,251 fewer nodes. A chain of
zero-length binary nodes and a single high-degree polytomy are the same tree to
the likelihood but not to the search: collapsing the chain hands every regraft
landing nearby a bigger star to resolve.

**Look at core utilisation, not only wall time.** The crate has three parallel
axes and all three lived in ingest or in search step 2. Once step 2 was replaced
by a linkage, the pipeline got 1.26x out of ten cores. Nothing in the wall-clock
numbers said so; it took a run at `RAYON_NUM_THREADS=1` to see it.

## What worked

| change | effect |
|---|---|
| Wire in the kNN restriction and ellipsoid bounds | 1478.8 s to 37.4 s at 512 by 2000; `n^2.9` to `n^1.8` |
| Lazy SPR proposal rows | 93.19 s to 6.71 s at 2048 by 200; `n^1.98` to `n^1.47` |
| Structural NNI filter over the star result | 1.98 s to 0.75 s at 2048 by 200; per round `n^1.53` to `n^1.02` |
| Ward linkage start replacing search step 2 | 66.08 s to 17.06 s at 2048 by 2000, identical tree |
| SPR proposals in parallel chunks, restarted on acceptance | 200.04 s to 24.44 s at 16384 by 2000; `n^1.45` to `n^1.31` |
| SPR acceptance on the candidate's own terms, rows assembled not swept | 127.9 s to 19.1 s at 2048 by 2000, noise 1.6, with the above |
| Scale-relative SPR acceptance floor | 2383.75 s to 616.26 s at 10,000 by 2,767, and a better tree |
| Graph linkage by mutual-nearest rounds instead of a chain | depth 131 to 12 at 4096 by 2000, 248 splits from the dense tree to 0 |
| Edge solve started at `mean(d - s)`, stopped on the Newton correction | 22.4 to 4.6 passes per attachment |
| Merge split by Illinois regula falsi instead of bisection | 58 to 20.5 derivative passes per pair |
| Parallel edge solve in `optimise_branch_lengths` | 2.4x on ten cores, bit-identical |
| Parallel edge scan in `nni_greedy` | 2.0x on top of the 3.8x it borrowed, bit-identical |
| Adaptive ellipsoid sizing on redraw-versus-walk cost | 0.109 s against 0.170 and 0.175 for fixed schedules either side |
| `f64x4` tier on `edge_newton` | 0.95 to 0.71 ns per feature; 8 per cent of a whole run |

Three notes.

**The two big rewrites were both about not materialising things.** The lazy rows
form what a proposal reads and no more; the NNI filter tests the star result
rather than building a tree and walking it. Neither made any kernel faster.

**The SIMD tier is honest about its size.** 14 per cent of the merge scan and 8
per cent of a whole run, 71.4 s to 65.7 s at 2048 by 2000. Worth having, not a
headline.

**The parallel changes are bit-identical on purpose.** The branch solve writes
one slot per node with no reduction. The NNI scan reduces to a running best with
ties broken on the lower node id, which is what the sequential scan did
implicitly, since the arena invariant makes ascending index order a post-order.
`test_the_greedy_phase_is_deterministic_whatever_the_thread_count` pins it at 1,
3 and 8 threads.

## What did not work

| attempt | why not |
|---|---|
| `BlockedState`, feature-blocked parallel prune | 6x on a kernel that is 2.4 per cent of a run |
| SIMD on `split_derivative` | runs 0.07 times per candidate pair; measured flat |
| `edge_newton` with the division replaced by a multiply | 0.92 against 0.95 ns per feature, inside noise |
| `edge_newton` with eight accumulator chains | 1.00 against 0.95, slower |
| `edge_newton` with one division per four features | 1.23 against 0.95, much slower |
| Ellipsoid sizing on walk depth | the walk is chunked, so every round reads shallow and the cap chose the answer |
| Feature subsampling to rank merge candidates | argmax survives 4 of 15 checkpoints at a quarter of the features |
| Chain or Boruvka agglomeration on the Bonsai merge gain | the score is not reducible: one round in five inverts, by up to 13 nats |
| Tree-distance limit on SPR regrafts | the beam is `n^0.06`, so there is nothing to limit |
| Warm-starting the beam's root find from the neighbour's optimum | 8.3 passes per solve against 5.8 from the mean start |
| Counting sort in SPR's `assemble` | the height sort was never the cost; `Tree::from_parents` is |
| Parallel pair scan on the four-member star inside the SPR proposal loop | 3.0 ms a resolution summed over threads against 0.7 sequential |
| Redrawing the linkage graph more often to stop it chaining | the chain was the cause, not the cadence |
| Collapsing zero-length internal edges before step 5 | 589 s against 451 in SPR, on an eighth fewer nodes |

Two deserve a paragraph.

**Feature subsampling.** The merge gain is a sum over `p` features, so estimating
it on `p'` of them and rescaling looks like a free 30x. Measured over five
checkpoints through a 128-member star at 2000 features, three seeds, with a
shared column set per checkpoint so the correlated part of the error is already
cancelled:

| subsample | argmax agreed | mean relative error | mean nats lost |
|---|---|---|---|
| 32 | 1 / 15 | 9.4e-1 | 94.6 |
| 128 | 3 / 15 | 3.7e-1 | 26.1 |
| 512 | 4 / 15 | 1.2e-1 | 12.0 |

The arithmetic says why. The estimator's standard deviation is
`p * sigma / sqrt(p')`, about 11 per cent of the gain at `p' = 512`, and
competing merges in a real round differ by far less than that. A random
*projection* is a different mechanism and is still open, since it summarises
every feature rather than discarding all but `p'`. It needs the precisions to be
approximately rank-1 in gene by cell, which has not been checked.

**Reducibility.** Merging changes the peeled remainder that every other pair's
score depends on, so a merge can lift another pair above the score the merged
pair had. Measured at 64 and 128 members: 10 to 32 inversions over 61 to 125
rounds, worst excess 3.6 to 13.2 nats, up to 60 per cent relative. Ward has no
global remainder term and is reducible by construction, which is a second reason
to prefer it for the starting tree.

## Memory

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

## Where the time goes

Ward start, 2000 features, 16384 leaves, ten cores, two seeds, Robinson-Foulds 0
at every size.

| step | seconds | share |
|---|---|---|
| linkage | 59.49 | 22% |
| 3 polytomy | 0.30 | 0% |
| 4 branch | 10.26 | 4% |
| 5 SPR | 200.04 | 73% |
| 6 NNI | 4.69 | 2% |
| 7 branch | 0.65 | 0% |

Exponents over the full sweep, unchanged by threading: linkage 2.00, polytomy
0.98, branch 1.03, SPR 1.45, NNI 0.98, total 1.52. Two things set the asymptote
and nothing else does.

1. The linkage at `n^2.00`, which is a dense distance matrix and wants to become
   a neighbour graph at every size rather than above a threshold.
2. SPR at `n^1.45`, which is `O(n)` candidates each paying an `O(n)` arena
   rebuild and an `O(n)` fingerprint, so `Theta(n^2)` of pure bookkeeping, and
   which runs at 1.18x on ten cores.

## The arc

Same data, same trees, identical loglikelihoods at every stage, 512 cells by
2000 features:

| | seconds |
|---|---|
| exhaustive candidate scan | 1478.8 |
| with the kNN restriction and ellipsoid bounds | 37.4 |
| with the lazy SPR proposal and structural NNI filter | 9.1 |

None of that traded accuracy. Every step is exact and returns byte-identical
trees, which is what the correctness gates on SPEC sections 10 and 11 exist to
guarantee.

## The kernel in isolation

The pruning recursion at 8192 cells by 2000 features:

| | time |
|---|---|
| numpy oracle, same equations | 468 ms |
| Rust, `f64` storage | 67 ms |
| Rust, `f32` storage | 40 ms |

Loglikelihoods agree with `reference/bonsai_ref.py` to twelve significant
figures. That oracle covers the pruning kernel alone; it has no answer for the
merge score or the branch solve, which is where most changes land.

The parallel axis is the feature axis, not the tree level, because the model
factorises over features. That makes it indifferent to tree shape: a pathological
ladder runs in 9.0 ms where level-parallelism takes 67.7 ms.

**This is not an end-to-end claim.** The search has costs the kernel benchmark
never touches, which is why "where the time goes" above is the honest table.

## Reproducing

```bash
cargo bench --bench pipeline     # end to end
cargo bench --bench steps        # per-step attribution
cargo bench --bench kernels      # single-threaded throughput per kernel
cargo bench --bench start_tree   # does step 2 earn its keep
cargo bench --bench merge_scan   # one round of candidate-pair scoring
```

Run on an idle machine and check `uptime` first. An implausible speedup is the
signature of a sweep that did nothing.
