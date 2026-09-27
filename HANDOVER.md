# Handover, 2026-09-27

For the next agent working on bonsai-rs performance. Read `CLAUDE.md` first;
its licence rule binds you and any subagent you start.

## Where things stand

0.2.0, unreleased; 0.1.1 is the last release. The full search, linkage start
to step 8, on a ten-core M1 Max:

| cells | features | 0.2.0 | 0.1.1 |
|---|---|---|---|
| 512 | 2,382 | 2.9 s | |
| 5,000 | 2,701 | 31.7 s | |
| 10,000 | 2,767 | 72.4 s | 202.9 s |
| 25,000 | 2,846 | 208.3 s | 1,028.7 s |
| 65,536 synthetic | 1,000 | 226.5 s | |

From 10k to 25k: total `n^1.15`, SPR `n^1.10` and 60 per cent of the run,
NNI `n^1.54`, linkage `n^1.63`, step 8 `n^1.64`. SPR from 65k to 131k
synthetic: `n^1.08`. Per-step tables in `docs/PERFORMANCE.md`, "Now".

Against the published implementation (`docs/COMPARISON.md`): 50 to 80 times
faster than its backbone mode at 5,000 and 10,000 cells, 150 to 220 times
faster than its standard run, and ahead on Robinson-Foulds and recovery from
5,000 cells up.

## What changed since 0.1.1

`docs/DESIGN.md`, "Search topology", has the design. In short:

- SPR and NNI's lazy phase run on `search::live::LiveTree`: stable ids, arena
  order kept per level, one conversion each way per step.
- SPR scores a move on `search::masked` views with a fixed-point total, so
  nothing is built until a move is accepted, and applies it to the changed
  paths only.
- New nodes are numbered by assembly, not traversal. This changed trees within
  the seed spread at 5,000 cells; every other change was gated on bit-identical
  trees against the previous build on up to twelve sets.
- Prune and up sweep are level-parallel: steps 4 and 7 3.4 to 4.1x.
- Analytics and test-only public API removed (`CHANGELOG.md`, 0.2.0).

## Open questions, in order

1. **Recheck at 10k.** Recovery 0.409 with `SprApprox::recheck` on (the
   default) against 0.476 for 0.1.1, which had no recheck; loglikelihood 1,015
   nats lower. Two basins at 10k from SPR order alone are known (0.40 to 0.48,
   PERFORMANCE.md). Settle with a seed spread: `SPR_SEED=1..5`, `SPR_RECHECK=0`
   and `1`, about 15 minutes now. If recheck lands in the low basin more
   often, turn it off by default.
2. **NNI grows `n^1.5`.** Per-move bookkeeping is gone; the three full edge
   scans per step are the suspect. Time the split before touching it.
3. **Step 8's polytomy loop** rescans and rebuilds the arena after every
   resolution, capped at 64 sweeps (`MAX_SWEEPS`) for that reason. Move it onto
   the live tree the way NNI went, then see whether the cap can go.
4. **Linkage is 13 per cent at 25k and `n^1.6`.** `mutual_pairs` recomputes
   every live cluster's nearest neighbour each round; SCALING.md section 1 has
   the RAC idea.
5. **Memory.** The global branch solve holds four `n x p` blocks; about 86 GB
   at a million cells by 2,700 features in `f32`. Nothing here touches it.

Also unmeasured since the live tree: SPR's thread utilisation (about half idle
at the chunk barrier in a sampled 25k profile before it), and a pure ladder
through the level-parallel prune. The COMPARISON.md figure sections date from
the 0.1.0 trees.

## How to reproduce

Data (outside this repo): `~/repos/others/bonsai-comparison/work_real/`.
`n512`, `n512_s32`, `n5000`, `n10000` are original-Sanity sets with the
published implementation's trees alongside; `n25000rs` is Rust Sanity.
Synthetic: `syn8k_lo`, `syn8k_hi`, `syn16k_unbal`, `syn16k_rand`,
`syn32k_hi`, `syn65k_hi`, `syn131k_hi`.

Harness (`reference/comparison`, `cargo build --release`):

```sh
H=reference/comparison/target/release/harness
START=linkage OURS_TAG=x $H ours <dir>           # full search, per-step timings
$H refine-tree <dir> <dir>/ours_stages/2_merge.nwk  # steps 3 to 8 from a tree
$H score-tree <dir> <tree.nwk>                    # loglik, loglik_refit, RF, recovery
$H rf <dir> a.nwk b.nwk                           # RF between two trees
$H gen <dir> <n> <p> <noise_sd> <seed> [balanced|random|unbalanced]
$H sanity-rs <sim_dir> <out_dir>                  # counts to prepped CSV, Rust Sanity
```

Env knobs: `SEARCH=exact`, `SPR_RECHECK=0/1`, `SPR_SEED` (random SPR order),
`SPR_MAX_ROUNDS`, `SPR_MIN_GAIN`, `NNI_RANDOM`, `NNI_SEED`, `NNI_MAX_ROUNDS`,
`NNI_MIN_GAIN`, `REFINE_OUT` (write the refined tree).

The published backbone: `~/repos/others/bonsai-comparison/ref_backbone.sh`,
single process. Its final tree is `results/final_bonsai_*/tree.nwk` at depth
two; an earlier version of the script looked at depth one and silently lost
every tree.

## Traps

- **One timed job at a time.** Copy the baseline binary aside before changing
  code, alternate base and new at least twice, print `uptime`. Load on this
  machine wanders from 2 to 17.
- **Debug cross-checks on real data:** build the harness with
  `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true CARGO_TARGET_DIR=<elsewhere>`.
  SPR then checks every proposal and every accepted move against the built
  path, NNI every interchange. Run it on `n5000` after touching `live`,
  `masked`, `spr` or `nni`.
- **Arena order is part of the answer.** Beam start points are ranks and child
  order sets a row's bits. A change that renumbers anything is a quality-sweep
  change, not a tree-diff one.
- **Check the `sanity` feature too:** `cargo check --features sanity`.
- **`cargo fmt` touches `src/bonsai.rs`** (pre-existing drift). Commit that
  separately or revert it.
- **The worktree guard** refuses commands with shell variables, `cd` before
  `git`, or heredocs it can't parse. Write a script to a file and run that, or
  use `git -C <path>`.
- **Verify what subagents hand back.** A subagent's script once reported
  success while missing every output file.
- **Licence:** of the published repository, only `README_bonsai.md` and
  `backbone_based_bonsai_parsing.py` (copied into bonsai-comparison by the
  user) have been read, and only for how to invoke it. Nothing else, ever, and
  nothing from them into this crate.
