# Handover, 2026-09-26

For the next agent working on bonsai-rs performance. Read `CLAUDE.md` first;
its licence rule binds you and any subagent you start.

## Where things stand

0.2.0, crate and Python bindings. The full search, linkage start to step 8, on
a ten-core M1 Max:

| cells | features | seconds | was, 2026-09-25 |
|---|---|---|---|
| 5,000 | 2,701 | 53 | 166 |
| 10,000 | 2,767 | 144 | 211 |
| 25,000 | 2,846 | 453 | 1,009 |

Roughly `n^1.3` to `n^1.4` over these three points. Per step, 10k to 25k:
SPR `n^1.25` and 58 per cent of the run, NNI `n^1.5`, branch solves `n^1.0`,
step 8 `n^2.2` (small in absolute terms), linkage `n^1.5`.

Against the published implementation (`docs/COMPARISON.md`): 25 to 70 times
faster than its fastest route and ahead on Robinson-Foulds and on the refit
loglikelihood from 5,000 cells up.

## What this session did

Commits `6e2b21a` to `fcdfbb0` on `worktree-declarative-exploring-corbato`.

| change | result | trees |
|---|---|---|
| NNI keeps rows across moves (`RowStore` + `LazyRows`), no settle per round | step 6: 295 s to 18.5 s at 25k | identical |
| Polytomy resolution does the same across sweeps | step 3: 69 s to 3.5 s at 25k | identical |
| SPR decides "no split changed" from the resolution star before splicing | step 5: -8 % at 25k | identical |
| SPR masked views (`search::masked`): no-move proposals decided without building the pruned or regrafted tree; moves reuse the view's placement | step 5: -6 % at 10k, -17 to -23 % at 25k | identical |
| SPR recheck (`SprApprox::recheck`, default on): after an acceptance, re-apply the chunk's remaining moves on the new tree instead of re-proposing | full search -6 to -22 % | differ, within spread |
| Backbone mode built, tuned, measured, then removed (`feat!`, 0.2.0) | never beat the full search up to 25k | - |
| `BonsaiResult.plot()` in Python, matplotlib as the `plot` extra | - | - |
| Harness: `sanity-rs`, `refine-tree`, `rf`, `score-tree` with `loglik_refit` | - | - |

Every "identical" row was gated on bit-identical output: the test suite, debug
cross-checks against the built path on every proposal (masked views), and real
10k and 25k runs compared against the previous build to every printed digit.

Why backbone went: with the Ward linkage start (3 per cent of a run) there is
no expensive start for it to save, and it keeps the whole-tree refinement that
is the rest. `docs/SPEC.md` section 15 has the numbers. In the published
implementation it does pay (2 to 4.5 times its own standard run), because that
run is slow.

## Open questions, in order

1. **Incremental arena updates** (proposal below). The biggest remaining
   structural lever; SCALING.md section 1 names it.
2. **Recheck at 10k.** One run gave recovery 0.386 against 0.476 without it
   (loglik within 1k nats). PERFORMANCE.md records two basins at 10k from SPR
   order alone (0.40 to 0.48). Settle with a seed spread: `SprParams::seed`
   with `PruneOrder::Random`, recheck on and off, 5 seeds each, about 30 min.
   If recheck lands in the low basin more often, turn it off by default.
3. **SPR worker utilisation.** Measured at 10 threads: 69 to 73 per cent
   inside the parallel phase (stragglers at the chunk barrier). The fix is a
   copy-on-write `RowStore` so proposing continues while the decider accepts.
   Ceiling about 1.3x on SPR, unmeasured.
4. **Step 8 scales `n^2.2`.** `resolve_polytomies` rescans from the top after
   every resolution and rebuilds the arena each time; capped at 64 sweeps
   (`MAX_SWEEPS`), which binds on real 25k data (137 polytomies, 64 resolved).
   Lifting the cap made it quadratic (8,382 resolutions, 432 s), so it needs
   the incremental arena first.
5. `docs/PERFORMANCE.md` doesn't yet record this session's work.

## Proposal: incremental arena updates

### The problem

The crate's `Tree` is an arena whose invariant is that ascending index is a
post-order with levels contiguous (CLAUDE.md, "Arena"). Every topology change
therefore rebuilds and relabels the whole arena, and everything keyed by node
index is remapped. That is `O(n)` per change, and the number of changes grows
with `n`, so each search step carries an `n^2` term. Where it sits today:

| where | per | `O(n)` passes |
|---|---|---|
| SPR accepted move | 11,382 at 25k | `RowStore::accept` (inherit map + slot remap), `leaf_words`, `leaves_below`, fingerprint, `word_index`, `mark_near_new_clades` |
| SPR move proposal (24 % of proposals) | ~65,000 at 25k | `cut` array copy, `regraft` assemble, `LazyRows::new` on the attached tree, splice rebuild, candidate `to_old` and `LazyRows::new` + `loglik` |
| NNI accepted move | 602 at 25k | `splice_star_mapped` rebuild, `leaves_below`, `leaf_words` x2, `LazyRows::new`, `RowStore::accept` |
| polytomy resolution | up to 64 per call | zero-edge collapse rebuild, splice rebuild, `LazyRows::new`, `RowStore::accept` |

The arithmetic a move changes is `O(depth p)`: only rows on the paths above
the cut and the attachment move. Everything else is bookkeeping.

### The design

A mutable search topology with stable node ids, used inside steps 3, 5, 6
and 8, and converted to and from the arena `Tree` at the step boundaries (one
`O(n)` build each):

- **Stable ids.** Nodes keep their id for their lifetime; a removed node's id
  goes on a free list. Parent pointer, child list, branch per id.
- **Rows keyed by id.** `RowStore` already maps node to slot; with stable ids
  the map becomes the identity plus a free list, and `accept` writes only the
  rows on the changed paths.
- **Path-local derived state.** Leaf words, leaf counts and the split
  fingerprint change only along the two paths a move touches: update them
  there (the masked views in `search::masked` already compute exactly these
  deltas, see `attach`). The word-to-node map gets removals and insertions for
  those nodes only.
- **Candidate scoring without a candidate tree.** Loglikelihood after a move =
  current total - old contributions + new contributions of the dirty nodes,
  `O(depth p)`. Build nothing unless the move is accepted.
- **The arena stays** for everything that sweeps the whole tree (settle,
  global branch solve, posteriors). Those run between steps, not per move.

### What will bite

1. **Bits depend on child order.** `prune_general` sums children in arena
   order and `prune_binary` is not symmetric in its arguments. Today's rows
   are the bits of the arena's order: leaves by index, then internal nodes by
   (height, original index). A stable-id topology must either reproduce that
   order for recomputed rows (the masked views do this, `Shape::settle`) or
   accept rounding-level differences. The first keeps the "identical trees"
   gate; the second needs the quality sweep instead. Decide up front.
2. **SPR's candidate order** is by leaf word; unaffected. The chunk logic and
   determinism at any thread count must stay.
3. **Root handling.** The cut can suppress a degree-two root and reroot
   (`cut`, the `(None, [a, b])` arm). Stable ids make this a pointer change,
   but the arena conversion at the step boundary must reproduce the rooting.
4. **Several modules read `Tree` directly** (`place`, `centre_star`,
   `LazyRows`, `NodeState::prune`). The `Walk` trait in `model::place` is the
   pattern: put a narrow trait in front of what the search reads.

### How to measure it first

Time the accept path at 25k and at a larger synthetic set before building:
`harness gen` makes balanced sets at powers of two, e.g. 131,072 by 1,000.
Wrap the SPR accept block in `sweep` in timers (temporary, uncommitted; this
session did the same for the funnel counts) and compare its share at 25k and
131k. If the accept path's share grows with `n`, as the step 8 exponent
suggests, the proposal pays; if it stays under 10 per cent, do open question 3
first.

## How to reproduce

Data (outside this repo): `~/repos/others/bonsai-comparison/work_real/`.
`n512`, `n512_s32`, `n5000`, `n10000` are original-Sanity sets with the
published implementation's trees alongside. `n25000rs`, `n512rs`,
`syn8k_lo`, `syn8k_hi` were added by this session; `handover/README.md` there
says what they are.

Scripts: `~/repos/others/bonsai-comparison/handover/`: `ab_refine.sh` (A/B
timing of steps 3 to 8 between two harness builds), `quality_sweep.sh` (full
search on all five sets, scored), `truth_csv.py` (for `sanity-rs` preps).

Harness (`reference/comparison`, `cargo build --release`):

```sh
H=reference/comparison/target/release/harness
START=linkage OURS_TAG=x $H ours <dir>           # full search, per-step timings
$H refine-tree <dir> <dir>/ours_stages/2_merge.nwk  # steps 3 to 8 from a tree
$H score-tree <dir> <tree.nwk>                    # loglik, loglik_refit, RF, recovery
$H rf <dir> a.nwk b.nwk                           # RF between two trees
$H sanity-rs <sim_dir> <out_dir>                  # counts to prepped CSV, Rust Sanity
```

Env knobs: `SPR_RECHECK=0/1`, `SPR_MAX_ROUNDS`, `SPR_MIN_GAIN`,
`NNI_RANDOM`, `NNI_SEED`, `NNI_MAX_ROUNDS`, `NNI_MIN_GAIN`, `REFINE_OUT`
(write the refined tree).

The published backbone: `~/repos/others/bonsai-comparison/ref_backbone.sh`,
single process. Its final tree is `results/final_bonsai_*/tree.nwk` at depth
two; an earlier version of the script looked at depth one and silently lost
every tree.

## Traps

- **One timed job at a time.** Copy the baseline binary aside before changing
  code, alternate base and new at least twice, print `uptime`. Load on this
  machine wandered from 3 to 20 over the session and moved runs by 10 per cent.
- **Debug cross-checks on real data:** build the harness with
  `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true CARGO_TARGET_DIR=<elsewhere>`;
  the masked views then check every proposal against the built path.
- **Check the `sanity` feature too:** `cargo check --features sanity`. It adds
  a second `From` impl to `BonsaiErrors`, and a closure whose error type
  compiled without it failed with it.
- **`cargo fmt` touches `src/bonsai.rs`** (one long line in a test, pre-existing
  drift). Commit that separately or revert it; don't mix it into other work.
- **The worktree guard** refuses commands with shell variables, `cd` before
  `git`, or heredocs it can't parse. Write a script to a file and run that, or
  use `git -C <path>`.
- **Verify what subagents hand back.** A subagent's script reported success
  while missing every output file.
- **Licence:** of the published repository, only `README_bonsai.md` and
  `backbone_based_bonsai_parsing.py` (copied into bonsai-comparison by the
  user) have been read, and only for how to invoke it. Nothing else, ever, and
  nothing from them into this crate.
