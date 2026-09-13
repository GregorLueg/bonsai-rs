# Overnight log, 2026-09-12 into 09-13

Newest entry at the top. Each entry: headline, old against new, key message, table.

Priority for the night, as set: **match their quality on realistic data first.
Optimisation only after that. Nothing already optimised gets thrown away.**

---

## Entry 4, 07:45. Branch plus linkage at 5k: 30x faster than the reference and better on every metric

### Headline

**5,000 realistic cells in 164 s against the reference's 4,880 s, with a better
tree on all three quality measures.** Distance recovery 0.666 against their
0.496, where the generating tree itself scores 0.679. We reach 98 per cent of
what the data allows; they reach 73.

### Old against new code

Branch `7c37de1` with `StartTree::Linkage`, built from an immutable `git
archive` snapshot, against the main checkout `1c3765e` with the greedy merge,
and against the reference. Same data, same gene set, same scorer.

### Key message

Every piece of yesterday's work compounds here, and the two that matter most
were built for different reasons than the one they served.

The linkage was built to be cheaper than the greedy merge and turned out to
produce a better tree. The SPR rewrite was built for speed and its benefit at
5k is amplified because a better start leaves it less to undo. Together they
take our 5k run from an hour and three quarters to under three minutes, and
past the reference on quality rather than merely level with it.

The interchange stage, which I spent half the night treating as the crisis,
contributes 44 nats of 5.35 million. It is still doing nothing and it still
does not matter at this size.

### Tables

5,000 cells, 2,701 genes, one seed. **Load 4.09 for our new run and 5.37 for
the reference, so this comparison is fair.** The old main-checkout run was at
load 51 and is not comparable on time; it is shown for quality only.

| | distance recovery | RF to truth | loglik | seconds |
|---|---|---|---|---|
| truth | 0.6789 | 0 | -10,901,098 | NA |
| **branch, linkage** | **0.6658** | **1,295** | **-5,557,976** | **164.4** |
| main, greedy | 0.5303 | 1,332 | -5,560,651 | 6,449 (load 51) |
| reference | 0.4958 | 1,937 | -5,563,770 | 4,880 |

Against the reference: **642 splits better, 5,794 nats better, 0.170 better on
recovery, 29.7x faster.**

Stage split, our two configurations:

| step | main, greedy | branch, linkage |
|---|---|---|
| 1-2 start | 316.0 | **1.6** |
| 3 polytomy | 11.1 | 0.1 |
| 4 branch | 30.5 | 8.8 |
| 5 SPR | 5,778.8 | **124.0** |
| 6 NNI | 295.4 | 22.5 |
| 7 branch | 17.3 | 7.3 |
| total | 6,449.3 | **164.4** |

### Caveats

- **One seed.** Everything above rests on a single 5k draw.
- **The old-against-new speedup is not measurable from this table.** The old run
  was at load 51 and its SPR was single-threaded, so it was starved. Do not
  quote 39x. The defensible number is the 29.7x against the reference at
  comparable load.
- At 512 the reference still wins on two seeds. The crossover is real and
  somewhere between 512 and 5,000.
- The branch is not answer-preserving from step 3; see entry 2.

---

## Entry 3, 02:30. At 5k cells we beat the reference on every metric. The 512 gap reverses.

### Headline

**The quality deficit is a small-`n` phenomenon and it inverts by 5,000 cells.**
At 512 they win by 43 splits. At 5,000 we win by **605 splits**, 3,118 nats and
0.034 of distance recovery. Every one of the three metrics agrees.

This is the exact thing the owner warned about before the harness existed: a
problem at low `n` that is not the problem at high `n` is not the thing to fix.

### Old against new code

Both sides are the **main checkout `1c3765e`**, so this is yesterday's code
before any of the branch work. The branch's 5k runs died with the agents and are
queued. That matters: our 5k time here is dominated by an SPR the branch makes
roughly eight times faster, and it does not include the linkage start.

### Key message

We were about to spend the night closing a quality gap that does not exist at
the size the project is actually aimed at. Two seeds at 512 say they are better
there; one seed at 5,000 says we are substantially better there. The crossover
is real and it is between those two sizes.

The mechanism is visible in our own stage table. At 5k our SPR gains **344,706
nats** and takes 89.6 per cent of the run. Their advantage at 512 came from
their interchange stage; ours at 5k comes from SPR continuing to find work where
theirs appears to stop. Whether their search budget stops scaling with `n` is
their business and not something we can or should determine.

**So the priority reverts to speed**, with one caveat below.

### Tables

Realistic Sanity-preprocessed data, main checkout both sides.

| size | | distance recovery | RF to truth | loglik | seconds |
|---|---|---|---|---|---|
| 512, seed 31 | ours | 0.5876 | 189 | -527,761 | 209.8 |
| | reference | **0.6330** | **146** | **-527,255** | 553.6 |
| 512, seed 32 | ours | 0.2210 | 327 | -514,991 | 63.4 |
| | reference | **0.3031** | **240** | **-513,553** | 371.0 |
| **5,000** | **ours** | **0.5303** | **1,332** | **-5,560,651** | 6,449.3 |
| | reference | 0.4958 | 1,937 | -5,563,770 | 4,879.8 |

Out of 9,994 splits at 5k, we get 13.3 per cent wrong and they get 19.4 per cent.

Our 5k stages:

| step | seconds | share | gain, nats |
|---|---|---|---|
| 2 merge | 312.8 | 4.8% | 1,933,614 |
| 3 polytomy | 11.1 | 0.2% | 1,233 |
| 4 branch | 30.5 | 0.5% | 45,867 |
| **5 SPR** | **5,778.8** | **89.6%** | **344,706** |
| 6 NNI | 295.4 | 4.6% | 827 |
| 7 branch | 17.3 | 0.3% | 10,595 |

### Caveats, and they are not small

- **One seed at 5k.** Two at 512. The inversion rests on a single 5k draw and
  needs at least two more before it is a fact.
- Our 5k run is the old SPR. The branch should cut 5,778 s hard without changing
  the answer much, but that is unmeasured on realistic data at this size.
- The linkage start is untested at 5k. At 512 it was the best tree we produced.

---

## Entry 2, 01:20. The linkage start nearly closes the quality gap, and is 34x faster

### Headline

The graph linkage that landed last night, built to be *cheaper* than the greedy
merge, turns out to be **better** on realistic data. It takes 512 realistic
cells from 43 Robinson-Foulds splits behind the reference to **2 splits behind**,
and from 506 nats behind to 244, while running 34 times faster than our own
greedy path and 89 times faster than theirs.

### Old against new code

Main checkout `1c3765e` with `StartTree::GreedyMerge`, against branch `7c37de1`
with `StartTree::Linkage`. Reference is the published implementation on the same
data and gene set. Load 11 to 12, so seconds are indicative, quality is exact.

### Key message

The diagnosis last night localised the gap to our interchanges, and that reading
was too narrow. The interchanges are inert either way, performing **zero** moves
from the linkage start. What actually happened is that the greedy merge was
handing refinement a bad tree on realistic data, SPR was spending 115 s digging
it out, and it never quite got there. Give SPR a good start and it finishes the
job: 129 accepted moves over 5 rounds, and the tree lands 2 splits off the
reference.

So hypothesis 6 was right and hypothesis 2 was a symptom. This does not make the
interchange work pointless, since 2 splits and 244 nats remain and the stage is
still doing nothing, but it is no longer the headline.

### Table

512 realistic cells, 2382 genes, one seed.

| configuration | loglik | RF to truth | seconds |
|---|---|---|---|
| main, greedy merge | -527,761 | 189 | 209.8 |
| **branch, graph linkage** | **-527,499** | **148** | **6.2** |
| reference | -527,255 | 146 | 553.6 |
| truth | -968,550 | 0 | NA |

Gap to the reference: **506 nats and 43 splits, down to 244 nats and 2 splits.**

### Where the splits are won and lost

Per-stage Robinson-Foulds, same data. This is the most informative table of the
night.

| stage | reference | ours, greedy | ours, linkage |
|---|---|---|---|
| after step 2 | 370 | **564** | **316** |
| after SPR | 182 | 183 | 148 |
| after interchanges | **146** | 175 | 148 |

Read it in two halves.

**Our step 2 is the weakest link on realistic data.** It produces RF 564 where
theirs produces 370, and our own new linkage produces 316, better than either
merge. That is the entry deficit SPR then spends 115 s paying back.

**Their interchange stage is the strongest.** From a nearly identical post-SPR
position, 182 against our 183, it wins 36 splits and ours wins 8. The linkage
route reaches 148 through SPR alone with the interchanges contributing nothing
at all. So if that stage worked as well as theirs, 148 should go materially
below their 146, and the target stops being parity and starts being better.

### Caveats

One seed, one size. Two rows are provisional: the harness was building from a
worktree while another agent had uncommitted changes in `src/search/nni.rs`, so
the branch-greedy interchange numbers may be measuring a half-finished
experiment. Flagged and being re-run from a clean tree. Seeds 32 and 33 and the
5k run are queued.

### Correction to entry 1 and to what I told the owner

I said every change on the branch was answer-preserving. **That is not true from
step 3 onward.** Step 2 is bit-identical, `-541,581.13` on both, but polytomy
resolution diverges by 151 nats and the difference persists to the end, leaving
the branch 243 nats and 14 splits *better* than main.

The mechanism is almost certainly benign and was predictable. Polytomy
resolution keys on branches being **exactly** zero, which `docs/SPEC.md` section
9.2 is explicit about. The branch tightened the merge split solver's tolerance
from `1e-8` to `1e-10` and swapped bisection for Illinois regula falsi, which
changes which splits land exactly at a bracket end, which changes which branches
come out at exactly zero, which changes the polytomies. A tie flips.

Benign or not, the claim of answer-preservation was verified on synthetic data
at 200 features and does not hold here, and I should not have stated it without
this test existing. Worth a fixture that pins it.

---

## Entry 1, 00:05. Baseline on realistic data at 512 cells

### Headline

First like-for-like comparison on Sanity-preprocessed data rather than the
synthetic generator. **We are faster and worse.** The whole gap is one stage:
their nearest-neighbour reordering is their dominant cost and does real work,
ours takes 31 s and gains 9 nats.

### Old against new code

Neither. This is the *current main checkout*, `1c3765e`, which is the code
before today's branch. Today's ten commits are not in this measurement. The
comparison is bonsai-rs against the reference implementation, not against
ourselves.

### Key message

On the synthetic generator we matched or beat them at every size, and that is
what every performance decision this week was validated against. On realistic
data at the same feature count we are 43 Robinson-Foulds splits further from the
truth and 506 nats behind. The synthetic fixture was hiding it. Speed work is
paused until this closes.

The stage table localises it completely. Our interchanges gain **9.3 nats** for
31 s of work. Theirs are 283.5 s of 550, over half their runtime. We put our
effort into SPR and they put theirs into reordering, and they finish ahead.

### Tables

512 cells, 2382 genes after selection, one seed. `truth` is the generating tree.

| | distance recovery | RF to truth | loglik | seconds |
|---|---|---|---|---|
| bonsai-rs | 0.587594 | 189 | -527761.08 | 209.77 |
| reference | 0.632982 | **146** | **-527255.03** | 553.58 |
| truth | 0.733166 | 0 | -968549.92 | NA |

RF between the two trees is 113 of 1018 splits, so these are genuinely
different topologies, not a tie broken differently.

**Calibration, because "we are worse" overstates it.** The reference misses 146
splits of 1018 itself, so realistic data at this size is hard for both. Our
excess over them is 43 splits, **4.2 per cent** of the total. On the other
metric: refinement, steps 3 to 7, gained us 13,820 nats and we finish 506 behind
them, so they extracted **3.7 per cent** more from the same stages. Two
independent measures agreeing at about four per cent is a consistent, closeable
gap rather than a broken search. It is still the first thing to fix, because a
four per cent quality deficit is not something to ship while claiming parity.

Our stages:

| step | seconds | share | gain, nats |
|---|---|---|---|
| 1 star | 0.24 | 0.1% | 0 |
| 2 merge | 54.47 | 26% | 157788 |
| 3 polytomy | 2.35 | 1% | 2236 |
| 4 branch | 3.08 | 1% | 2915 |
| 5 SPR | 114.96 | 55% | 8060 |
| **6 NNI** | **30.85** | **15%** | **9.3** |
| 7 branch | 3.83 | 2% | 600 |

Their stages, seconds only, from their own log:

| stage | seconds |
|---|---|
| first greedy maximisation | 246.1 |
| redoing starry nodes | 0.7 |
| optimisation of diffusion times | 5.2 |
| SPR moves | 13.0 |
| **reordering next-to-nearest neighbours** | **283.5** |
| optimisation of diffusion times | 0.7 |

### Hypotheses on the table

Ranked by how much they would explain, not by how likely I think they are.
`PERFORMANCE.md` rule 5 says the first diagnosis has been wrong four times out
of four on this project, so none of these is being treated as established.

| # | hypothesis | test | who |
|---|---|---|---|
| 1 | We enter refinement already behind, so step 6 is a symptom | score their per-stage trees under our scorer | harness |
| 2 | Our interchanges reach a genuine local optimum and stop; theirs escape | instrument rounds, filter rejections, acceptances | clean-room |
| 3 | The specified escape hatch is off, and inert even when on | `n_random` above zero, then a temperature | clean-room |
| 4 | One linear pass through steps 5 and 6 leaves value behind | alternate to a joint fixed point | clean-room |
| 5 | The structural filter is over-broad and rejects real moves | count rejections against acceptances | clean-room |
| 6 | The greedy merge itself is the weak start on realistic data | run the branch's graph linkage on realistic data | harness |

Hypothesis 1 outranks the rest: if it holds, everything aimed at step 6 is
aimed at the wrong step.

### What is running overnight

1. 5k realistic, both sides, to confirm the direction holds at scale.
2. NNI knob variants at 512: random phase on, higher round cap, lower gain
   floor. All are parameters, no crate changes.
3. Extra seeds at 512, because one draw is not a fact.
4. A clean-room investigation of why our interchanges are inert, from
   `docs/SPEC.md` section 9.4 only.
5. Queued for a quiet machine: the linkage end-to-end gate, the backend
   crossover, the build-seconds columns.
