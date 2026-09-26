# CLAUDE.md

Guidance for Claude Code in this repository.

## Crate

`bonsai-rs`: tree representations of high-dimensional data under Brownian motion. Clean-room Rust implementation of Bonsai (de Groot, Morillo Leonardo, Pachkov, van Nimwegen, *Nature Biotechnology* 2026, doi 10.1038/s41587-026-03220-2). Library only. Input is a means matrix *and* a per-cell per-feature SD matrix; output is a tree, not coordinates. Python bindings in `python/`.

## Licence rules, binding

**Never open, clone, read or grep `dhdegroot/Bonsai-data-representation` or its Zenodo drop.** Not here, not in a sibling worktree, not through a subagent. It's CC-BY-NC-4.0; a port would be Adapted Material under section 1(a), incompatible with MIT.

Implementation reads [docs/SPEC.md](docs/SPEC.md) and nothing else. It's transcribed from the CC-BY-4.0 paper and SI, which are free to implement. Every kernel cites its SI equation.

Don't carry over their tuned constants. Every threshold is a named `const` whose doc comment gives its source: an SI equation, or our measurement with the date. [PROVENANCE.md](PROVENANCE.md) has the full position.

## Commands

```bash
cargo build --release
cargo test --release

# Single test
cargo test --release -- model::branch::tests::test_single_feature_optimum --exact --nocapture

# Benches: pipeline, steps, kernels, start_tree, merge_scan, prune_sweep, ...
cargo bench --bench steps

# numpy oracle for the pruning kernel
uv run --with numpy reference/bonsai_ref.py --leaves 8192 --features 2000

cargo doc --no-deps --open
```

Release profile: `opt-level = 3`, `lto = "thin"`, `codegen-units = 4`.

## Layout

```
src/
  lib.rs          # crate policy header, #![warn(missing_docs)]
  errors.rs       # single BonsaiErrors enum, sectioned by subsystem
  bonsai.rs       # the pipeline: ingest, steps 1 to 8
  ingest.rs       # scale transform, feature selection, Sanity handover (SPEC 3)
  model/          # likelihood, branch, merge, place, global
  search/         # star, polytomy, spr, nni, candidates, bounds
  tree/           # arena, linkage, newick, layout, distance, cluster, simulate
  utils/          # BonsaiFloat, scalar kernels, simd.rs (only file naming `wide`)
benches/          # plain `main`, harness = false
reference/        # numpy oracle, written from SPEC.md
python/           # PyO3 bindings, versioned separately
docs/             # SPEC (the only implementation source), DESIGN, PERFORMANCE, COMPARISON
```

## Invariants

**Arena.** Leaves are `0..n_leaves`, internal nodes follow ordered by height. So ascending index is a post-order, a parent row sits above its children (`split_at_mut`, no aliasing), and each level is contiguous. Violations break silently. `Tree::from_parents` relabels to enforce it.

**One layout.** `NodeState`, row-major `[node][feature]`, sequential. A feature-blocked parallel `BlockedState` was 6x faster on the kernel, called "the production path" for a month, and only ever built by a benchmark. Prune is 2.1 to 2.4 per cent of a run. Deleted 2026-09-06. **Measure a component's share of the whole, and check the pipeline calls it, before optimising it.**

**Numerics.** Storage generic over `BonsaiFloat`, every reduction in `f64`. Don't "simplify" back:

- Effective means as a convex combination, `m_k + (m_l - m_k) * wd_l / w_a`, not `(wd_k*m_k + wd_l*m_l)/w_a`.
- The quadratic term via the pairwise identity (S33), never `sum wd*m^2 - w*m^2`.

Means are judged on absolute error, loglikelihoods on relative.

**Units.** Input divided by `sqrt(v_g)` at ingest; `2*pi` and `v_g` terms dropped. Loglikelihoods are up to a constant. **This crate works in `L`, not `2L`**: halve any threshold from the paper.

**Determinism.** Same input, same tree, any thread count. Parallel reductions sum per-unit contributions in a fixed order. Never `.sum()` on a `ParallelIterator`.

## Conventions

- British English (`normalise`, `optimise`, `neighbours`, `centred`).
- `#![warn(missing_docs)]`; every item gets `### Params` / `### Returns`.
- One `thiserror` enum in `src/errors.rs`; add to the matching section. Caller-reachable failures return `Err`; panics are for broken invariants.
- `wide` types only in `src/utils/simd.rs`.
- Benches need `harness = false` in `Cargo.toml` and must check their output before reporting a timing. An implausible speedup means a sweep that did nothing.
- No SIMD tier without a measurement in its doc comment. Pick kernels by call count: `edge_newton` runs 35 to 45 times per pair and earned one; `split_derivative` looks hot, runs 0.07 times, measured flat.

## Elsewhere

[docs/SPEC.md](docs/SPEC.md) the algorithm, [docs/DESIGN.md](docs/DESIGN.md) the build, [docs/PERFORMANCE.md](docs/PERFORMANCE.md) what was optimised and what failed, [docs/COMPARISON.md](docs/COMPARISON.md) the black-box comparison and what may be said, [PROVENANCE.md](PROVENANCE.md) the licence position, [CHANGELOG.md](CHANGELOG.md) history.
