# The 5k to 10k slowdown on realistic data

Interim, 2026-09-13 13:00. Diagnosis only; nothing in the crate was changed.
Every section says whether a number was measured or inferred. Seconds carry the
load average they were taken at; counts, gains, node counts and depths are
load-independent.

Model running this investigation: Fable 5.1.

## Headline

**SPR at 10k is not slow, it is stuck.** From round 10 or so it accepts moves
whose "gain" is the rounding noise of an `f64` sum of magnitude `1.1e7`,
because the acceptance floor `StarParams::min_gain = 1e-9` was measured for a
merge score of magnitude `O(p)` and is applied unchanged to a whole-tree
loglikelihood of magnitude `O(n p)`. Resumed from the cached post-round-100 tree,
each round accepts exactly one move of gain `5.6e-8`, the tree cycles with
period two, and the fresh loglikelihood does not change. With the floor raised
to `1e-4` the same tree accepts nothing. **The queued uncapped 10k run
(`SPR_MAX_ROUNDS=100000`) will therefore not converge; at 21 s a round it runs
for about 24 days and blocks the two 5k replicates behind it. Kill it.**

Ninety-one of the hundred rounds at 10k did nothing, which is roughly 1900 s of
the 2129 s. With a sane floor SPR at 10k should be on the order of nine rounds,
about 200 s at the measured 21 s a round, and the pipeline about 3x the 5k time
for 2x the cells rather than 14.5x. That is an inference from the per-round
cost; the whole-pipeline number at a raised floor is being measured now.

The linkage start is **not** degrading: mean leaf depth is `log2(n) + 0.24` at
both 5k and 10k, and Robinson-Foulds to the truth is 40.1 and 42.3 per cent of
the splits. NNI's extra moves are real gains, not noise, and their count is
still unexplained; the subsample ladder that answers it is running.

## What was measured

All on the harness's cached Sanity-preprocessed data in
`/Users/gregorlueg/repos/others/bonsai-comparison/work_real/`, seed 31, 2701
features at 5k and 2767 at 10k. Code is the immutable snapshot
`/tmp/bonsai-snap-7c37de1`, whose `spr.rs`, `nni.rs` and `linkage.rs` checksum
identical to this worktree's HEAD, and which is what the harness's 5k and 10k
numbers came from.

The driver is a scratchpad crate, not in the tree, because everything it
needed is already public: `spr_round` gives per-round move lists with gains,
`nni_greedy` with `max_rounds: 1` steps one interchange at a time, and
`Tree::parent` gives depth. Source:
`/private/tmp/claude-501/-Users-gregorlueg-repos-shared-bonsai-rs/e844eef2-43f9-4bd6-a9c6-3d0db6c72a46/scratchpad/slowdown/src/main.rs`.
It should become a bench block once the other agent's uncommitted `Cargo.toml`
edit lands; see recommendations.

### The cached stage trees, scored (load 2.4)

`harness score-tree` on the trees the 5k and 10k runs wrote at each stage
boundary. RF is to the generating tree; 9994 and 19994 splits respectively.

| size | stage | loglik | RF | RF share | nodes | polytomies | zero branches | mean leaf depth | log2 n |
|---|---|---|---|---|---|---|---|---|---|
| 5k | 2 linkage | -10909072 | 4010 | 0.401 | 9998 | 0 | 0 | 12.53 | 12.29 |
| 5k | 5 spr | -5562956 | 1294 | 0.129 | 9980 | 17 | 303 | 16.45 | |
| 5k | 6 nni | -5562911 | 1295 | 0.130 | 9981 | 16 | 304 | 16.44 | |
| 5k | 7 branch | -5557976 | 1295 | 0.130 | 9981 | 16 | 56 | 16.44 | |
| 5k | truth | | 0 | | 9999 | 0 | 0 | 15.90 | |
| 10k | 2 linkage | -22460257 | 8460 | 0.423 | 19998 | 0 | 0 | 13.53 | 13.29 |
| 10k | 5 spr | -11264121 | 2661 | 0.133 | 19959 | 35 | 684 | 17.28 | |
| 10k | 6 nni | -11263728 | 2661 | 0.133 | 19963 | 31 | 677 | 17.41 | |
| 10k | 7 branch | -11253231 | 2661 | 0.133 | 19963 | 31 | 150 | 17.41 | |
| 10k | truth | | 0 | | 19999 | 0 | 0 | 17.79 | |

Max leaf depth: linkage 21 at 5k and 35 at 10k against the truth's 29 and 32;
the generator is unbalanced, so max depth is not the diagnostic, mean depth is.

### SPR resumed from the cached post-SPR trees

`spr_round` called exactly as `spr` composes it (seed advanced per round), from
`ours_stages/5_spr.nwk`, which Newick round-trips bit-exactly. `drift` is
`loglik_before + sum(gains) - fresh_prune_after`, so a round whose gains are
real has drift near zero and a round whose gains are rounding has drift equal
to its gains.

| tree | floor | round | moves | sum gain | drift | nodes | RF to previous | RF to two back | s | load |
|---|---|---|---|---|---|---|---|---|---|---|
| 5k, 9 rounds converged | 1e-9 | 1 | 0 | 0 | 0 | 9980 | | | 8.6 | 5.7 |
| 10k, 100 rounds truncated | 1e-9 | 1 | 1 | 5.588e-8 | 5.588e-8 | 19958 | | | 20.8 | 12.8 |
| | | 2 | 1 | 5.588e-8 | 5.774e-8 | 19957 | 2 | | 20.9 | 14.6 |
| | | 3 | 1 | 5.960e-8 | 5.774e-8 | 19958 | 2 | 3 | 20.8 | 15.2 |
| | | 4 | 1 | 5.588e-8 | 5.588e-8 | 19959 | 2 | 1 | 21.0 | |
| | | 5 | 1 | 5.588e-8 | 5.588e-8 | 19958 | 2 | 1 | 20.8 | |
| | | 6 | 1 | 5.588e-8 | 5.774e-8 | 19957 | 2 | 1 | 21.0 | 19.3 |
| 10k, same tree | 1e-4 | 1 | 0 | 0 | 0 | 19959 | | | 20.7 | 18.3 |

Fresh loglikelihood after six rounds at the default floor: `-11264121.000`,
the starting value. `5.96e-8` is `2^-24`; the ulp of `1.1e7` is `2^-29`, so
the accepted gain is about 32 ulps of the sum.

### NNI resumed from the cached post-SPR trees

`nni_greedy` with `max_rounds: 1` in a loop; gain is the fresh loglikelihood
difference. 5k reproduces the harness exactly: 20 moves, final loglik
`-5562911.237`, 1.17 s a round at load 17.

| size | moves | first five gains, nats | last five gains, nats | s per round | load |
|---|---|---|---|---|---|
| 5k | 20 | 11.3, 5.1, 6.1, 4.7, 3.6 | 0.13, 3.4, 0.08, 0.23, 0.18 | 1.17 | 17 |
| 10k | in progress, 26 done at write time | 26.4, 15.1, 12.3, 10.7, 10.1 | round 19: 3.0 | 2.44 | 15 |

Every NNI gain is O(0.1) to O(10) nats. None is rounding.

## What the numbers say

### 1. SPR rounds: settled, measured

The mechanism, from the code and the resume table:

- `spr_round` accepts a candidate when `proposal.loglik > best + min_gain`,
  where both sides are sums over all internal nodes of a tree of 20k nodes,
  magnitude `1.1e7` at 10k. `min_gain` is `1e-9`. `DEFAULT_MIN_GAIN`'s doc
  comment says it was measured on `score_merge`, a sum of magnitude `O(p)`
  whose rounding floor is `1e-12`; the floor of the whole-tree sum is four to
  five orders larger, and the observed accepted gain is `5.6e-8`.
- The candidates that pass are likelihood-neutral topology changes. The star
  primitive refuses a merge below `min_gain` and leaves a polytomy (the node
  count drops by one per accepted move, then recovers), so a regraft onto a
  zero-length region produces a tree with different splits and the same
  likelihood. The fingerprint filter only removes identical topologies. Which
  of these neutral candidates rounds positive is arbitrary, so every round
  finds one and the tree cycles with period two (RF 2 to the previous round,
  RF 1 to two rounds back).
- The 10k tree has 684 zero-length branches after SPR and 35 polytomies; 5k
  has 303 and 17. That is the pool of neutral moves, and it is proportional
  to `n`, so this is not a cliff at 10k. 5k converged at round 9 because its
  round 9 happened to round negative on every neutral candidate; the resumed
  5k round also found nothing. That is luck, not a margin.
- Raising the floor to `1e-4` stops it dead on the same tree.

What this does **not** yet say, and the ladder will: how many of the 4843 moves
at 10k were real and how the moves-per-round profile looked in rounds 1 to 9.
The "238 to 48 per round" collapse in the brief is an average over a run whose
last 91 rounds were noise; the real per-round profile at 10k has not been seen.
Inferred: rounds 1 to 9 at 10k looked like 5k's (about 4300 moves) and the
remaining 91 rounds accepted about 540, six a round, tailing to one. That is a
guess and is marked as such.

### 2. NNI move count: partly settled

Measured: the moves are real, gains of 0.08 to 26 nats, so this is not the SPR
mechanism. Measured: per-round cost is 1.17 s at 5k and 2.44 s at 10k, `n^1.06`,
so the 8.9x is entirely the move count. The first gains at 10k are about twice
the first gains at 5k (26 against 11).

Not settled: whether the count grows because SPR left more behind at 10k or
because the interchange neighbourhood grows. The obvious reading, that the
truncated SPR left work for NNI, is **not** supported: by round 10 SPR at 10k
was in the noise regime and was not going to find those interchanges in
another thousand rounds; every one of the 89 NNI moves is a move SPR's beam
does not propose. The subsample ladder (1250, 2500, 5000 cells drawn from the
10k Sanity output, same genes, same noise) gives NNI moves against `n` with
the preprocessing held fixed, and is running.

### 3. Linkage start: settled, does not degrade

Measured, load-independent: mean leaf depth of the linkage tree is 12.53 at 5k
(`log2 = 12.29`) and 13.53 at 10k (`log2 = 13.29`), both `log2(n) + 0.24`. No
polytomies, no zero branches. RF to the truth is 40.1 per cent of splits at 5k
and 42.3 at 10k, and after SPR it is 12.9 and 13.3 per cent, so the relative
quality of both the start and the refined tree is flat in `n` to within two
points. The `3.5x` in linkage seconds is real but the stage is 0.2 per cent of
the run. This does not explain 1 or 2.

### 4. Exponents: not settled

Rungs in hand on realistic data: 512 (two seeds, greedy start, pre-branch
code), 5000 and 10000 (linkage start, this branch). The 10k SPR and NNI
seconds are not usable for an exponent because SPR was 91 rounds of noise and
NNI's move count is unexplained. The subsample ladder gives 1250, 2500 and
5000 from one Sanity run, which is the cleanest way to hold the preprocessing
fixed; its seconds are at load 15 to 20 and its counts are exact. Native 2.5k
data would need a Sanity run and was not generated.

Per-round costs, which are the only exponent numbers safe to quote now:
SPR 8.6 s at 5k and 20.8 s at 10k, `n^1.27` (load 6 and 13 to 19, so
pessimistic at 10k); NNI 1.17 and 2.44 s, `n^1.06` (load 17 and 15).

## Recommendations, ranked by expected effect

1. **Kill the queued uncapped 10k run now.** Measured: one neutral move per
   round on the tree it starts from, with no floor change. It will run to the
   cap.
2. **Scale SPR's acceptance floor to the loglikelihood magnitude, or give
   `spr` a progress-based stop.** Measured: `1e-4` stops the 10k tail in one
   round; the noise is `5.6e-8` at `|L| = 1.1e7`, i.e. about `5e-15 |L|`.
   A floor of `1e-12 |L|` (the tolerance
   `test_the_incremental_loglik_matches_a_fresh_prune` already uses) is
   `1.1e-5` at 10k, `5.6e-6` at 5k and `5e-7` at 512, and clears the observed
   noise by 200x while staying far below any move that carries information.
   Alternatively stop when a round's summed gain is below that floor, which
   also survives many small neutral moves in one round. Either way the
   `DEFAULT_MIN_GAIN` doc comment has to say it is a merge-score floor, not a
   tree-loglikelihood floor, and SPR needs its own named constant with this
   measurement in the comment. Expected effect at 10k: SPR from 2129 s to the
   order of 9 rounds, about 200 s. Inferred from the per-round cost, being
   measured in the ladder's raised-floor rung.
3. **Fix `DEFAULT_MAX_ROUNDS`'s justification** while there: it is a runaway
   guard whose comment argues from 16 to 64 leaf fixtures, and at 10k it was
   the only thing that ended the run. Once 2 lands it can stay at 100 with a
   comment that says why it never binds.
4. **Do not touch the linkage or NNI's per-round cost.** Both measured flat.
5. Move the scratchpad driver into `benches/` as a `slowdown` block once the
   pending `Cargo.toml` edit from the NNI agent is committed, so the fix for 2
   can be verified on the same per-round trace.

## What I could not settle

- The real per-round profile of the 10k SPR run in rounds 1 to 9, and how many
  of its 4843 moves were noise. Ladder in progress.
- Why NNI's move count grows superlinearly. Ladder in progress; the candidate
  explanations left are a larger interchange neighbourhood per leaf and a
  start that is relatively worse in a way RF does not show.
- Any exponent on native realistic data beyond the two rungs; the 10k rung is
  contaminated by the SPR tail.
- Whether the 91 noise rounds changed the 10k tree's quality. RF after SPR is
  13.3 per cent of splits against 12.9 at 5k, so probably not, but the
  cycling moved nodes into and out of polytomies (19959 to 19957 and back) and
  the uncapped run would have shown whether that drifts.
- The `1.07x` between the 5k native run's NNI seconds (22.45 s at load 4) and
  the resumed one (26.7 s at load 17) is load, not code; both did 20 moves to
  the same loglikelihood.
