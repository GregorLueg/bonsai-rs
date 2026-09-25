# Backbone mode is slower than the full search

`backbone::backbone` is meant to make 50k to 100k cells workable. Right now it
loses to the plain search at every size tried. This note is the brief for
fixing it.

## Measured, 2026-09-25

Ten-core M1 Max, `BonsaiParams::default()`, main as of this date. Harness
subcommands `ours` (with `START=linkage`) and `backbone`, see below.

| cells | features | run | seconds | loglik | RF to truth | recovery |
|---|---|---|---|---|---|---|
| 512 | 2,382 | full search | 3.62 | -527,499.0 | | |
| 512 | 2,382 | backbone, `backbone_cells >= n` | 3.56 | -527,499.0 | | |
| 512 | 2,382 | backbone, 128 | 16.17 | -527,641.7 | | |
| 10,000 | 2,767 | full search | 200.65 | -11,252,721.0 | 2,624 | 0.476 |
| 10,000 | 2,767 | backbone, 2,048 (default) | killed after more than 390 | | | |

- The `backbone_cells >= n` row short-circuits to `bonsai_prepared` and gives
  the same loglik as `ours`, so the harness wiring is right.
- At 10k it ran on about two of ten cores (`ps` showed 197 per cent CPU).
- The 128-cell backbone at 512 cells placed 384 cells with 6 reoptimisations:
  4.5x slower than searching all 512 directly, and 143 nats worse.

## Where the time goes

One 5 s `sample` of the 10k process, placement phase (no `refine`, SPR or NNI
frames yet). This is one snapshot, not a timed phase split:

- Most samples are in `model::global::collapse_onto_every_node`, called from
  `Growing::place_cell` (`src/backbone.rs`). It recomputes the effective leaf
  at every node of the whole tree for every placed cell: `O(n p)` per cell,
  `O(n^2 p)` over the growth phase.
- `model::place::place`, the beam search itself, barely registers.
- `optimise_branch_lengths` from `Growing::reoptimise` is a distant second.
  Each call builds a fresh `NodeState` and optimises the whole tree, every time
  the tree grows by `regrow_fraction` (0.25).
- The low core use suggests the collapse runs mostly sequential.

Backbone inherits the SPR and NNI approximations already: the seed search is
`bonsai_prepared(..., params.bonsai)` and the final pass is
`refine(..., params.bonsai)`, both with `SprSearch::Approximate` and
`NniSearch::Approximate` by default. The two pieces of its own code, placement
and growth reoptimisation, never saw the September tuning.

## What to fix

1. Get a timed phase split first: seed search, placement, reoptimise, final
   refine. `backbone` has no timers; add them (or make the phases callable) so
   the harness can report each. The final refine is whole-tree SPR plus NNI
   and may cost as much as the full search on its own; unmeasured.
2. Placement must not collapse the whole tree per cell. Options: update the
   node posteriors only along the path the attachment changes, or place cells
   in batches against one frozen collapse and re-collapse between batches.
   Batch placement is also where the parallelism is.
3. Then check whether growth reoptimisation earns its cost at all, against
   leaving branch lengths to the final refine.

Gate: backbone at 10k must beat 200.65 s, and its loglik, RF and recovery must
stay in the run-to-run spread of the full search (a few hundred nats at 10k,
see "How much one real-data run says" in `docs/PERFORMANCE.md`). Then 25k
(`work_real/sim/n25000` is simulated, needs Sanity) and 50k.

## Reproducing

```sh
cd reference/comparison && cargo build --release
H=target/release/harness
D=~/repos/others/bonsai-comparison/work_real/n10000   # or a copy: ours writes ours.nwk into it
START=linkage $H ours $D
$H backbone $D [backbone_cells]     # BACKBONE_TAG=x suffixes the outputs
$H score-tree $D $D/backbone.nwk
```

`backbone` writes `backbone.nwk`, `backbone_seconds.txt` and
`backbone_steps.tsv` (refine per-step loglik plus the growth report).
