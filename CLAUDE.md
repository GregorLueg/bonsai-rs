# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate

`bonsai-rs`: tree representations of high-dimensional data under Brownian motion. A clean-room Rust implementation of the Bonsai algorithm (de Groot, Morillo Leonardo, Pachkov, van Nimwegen, *Nature Biotechnology* 2026, doi 10.1038/s41587-026-03220-2). Library crate only, no binaries. Input is a means matrix *and* a per-cell per-feature standard deviations matrix; output is a tree, not coordinates.

## Licence rules, which are binding

**Never open, clone, read or grep `dhdegroot/Bonsai-data-representation` or its Zenodo drop.** Not here, not in a sibling worktree, not through a subagent. That code is CC-BY-NC-4.0, and a port would be Adapted Material under section 1(a) of that licence, which is incompatible with the MIT licence on this crate.

Implementation reads `docs/SPEC.md` and nothing else. The spec is transcribed from the CC-BY-4.0 paper and Supplementary Information, which are separately licensed from the code and are free to implement. Every kernel cites the SI equation number it implements.

Do not carry over their tuned constants. Every threshold here is a named `const` whose doc comment says where the number came from: an SI equation, or our own measurement with the date. `PROVENANCE.md` records the full position and the disclosure.

## Commands

```bash
cargo build --release
cargo test --release

# Single test
cargo test --release -- model::branch::tests::test_single_feature_optimum --exact --nocapture

# Pruning sweep timings across tree shapes, precisions and sizes
cargo bench --bench prune_sweep

# The numpy oracle the Rust kernels are checked against, and the Python
# baseline the timings are quoted relative to
uv run --with numpy reference/bonsai_ref.py --leaves 8192 --features 2000

cargo doc --no-deps --open
```

Release profile is `opt-level = 3`, `lto = "thin"`, `codegen-units = 4`.

## Architecture

### Layout

```
src/
  lib.rs              # crate policy header, #![warn(missing_docs)], module list
  errors.rs           # single BonsaiErrors enum, thiserror-backed, sectioned
  tree/
    mod.rs            # the flat arena, level ordering, fixture builders
  model/
    likelihood.rs     # row-major pruning recursion            (SPEC 4, 5)
    branch.rs         # branch-length root find                (SPEC 6)
  utils/
    traits.rs         # BonsaiFloat
    kernels.rs        # scalar feature-axis kernels
    simd.rs           # SIMD tiers; the only file naming a `wide` type
benches/              # plain `main`, harness = false
reference/            # numpy oracle, written from SPEC.md
docs/SPEC.md          # the clean-room specification: the only implementation source
```

### The arena invariant

Leaves occupy `0..n_leaves`. Internal nodes follow, ordered by height above the leaves. Three things depend on this and will break silently if it is violated:

- Ascending index order is a valid post-order, so the sweep is a linear scan.
- A parent row lies above all of its children's rows, so writing a parent while reading its children is a `split_at_mut`, not an unsafe alias.
- Each level is a contiguous index range.

`Tree::from_parents` relabels internal nodes to enforce it. That is free for callers because internal rows are always computed, never supplied, so nothing caller-held is indexed by an internal node.

### One layout

`NodeState` is row-major `[node][feature]` and sequential. Every prune in `bonsai`, `backbone`, `spr`, `nni` and `polytomy` goes through it.

There used to be a second, `BlockedState`, laid out `[block][node][feature]` and parallel over the feature axis. It was deleted on 2026-09-06 and the history is worth keeping, because the mistake is repeatable. It was real code with real tests and a real measurement behind it: at 8192 leaves by 2000 features it ran a ladder tree in 11.2 ms against level-parallelism's 67.7 ms. This file called it "the production path" from the day it was written. It never was. It was constructed in exactly two places, both inside a benchmark.

What settled it was instrumenting the whole search rather than arguing from the kernel timing: **prune is 2.1 to 2.4 per cent of a run and the up-sweep another 0.5**, with the share flat in feature count, so 512 by 2000 gives the same 2.3 per cent as 512 by 200. A perfect tenfold speedup there buys under 3 per cent of wall time. The cost is in the merge pair scan and in SPR's proposal generation, neither of which is a tree sweep.

So: **measure a component's share of the whole before optimising it, and check that the pipeline calls it before doing either.** Two modules on this project were built, tested, benchmarked and never wired in. The kNN and ellipsoid restrictions mattered and the search was cubic for a week without them; this one did not matter, and the only cost was the 324 lines and a doc that misled every reader for a month.

### Numeric policy

Storage is generic over `BonsaiFloat`; every reduction accumulates in `f64` regardless. The tree loglikelihood sums thousands of features whose interesting differences are `O(1)` while the sum is `O(p)`, so `f32` accumulation would turn the convergence criterion into noise. `f32` *storage* is worth having and is the fastest path.

Two conditioning choices that matter and should not be "simplified" back:

- Effective means are formed as a convex combination (`m_k + (m_l - m_k) * wd_l / w_a`), not as `(wd_k*m_k + wd_l*m_l)/w_a`. The result is pinned between the two child means and cannot cancel. It feeds straight back into the recursion, so error here compounds up the tree.
- The quadratic term uses the pairwise identity (S33) for two children, never `sum wd*m^2 - w*m^2`.

Accuracy is judged on absolute error for means and relative for loglikelihoods. Everything downstream consumes squared *differences* of means, so a mean near zero carrying large relative error is fine and a tolerance that fails on it is the wrong tolerance.

### Units

Input is divided by `sqrt(v_g)` at ingest (SPEC 3.1), which removes the diffusion scale from every kernel. The `2*pi` and `v_g` terms are dropped because their count is independent of topology. So loglikelihoods are meaningful only up to an additive constant, and are not comparable to the reference implementation's.

**This crate works in the loglikelihood, not twice it.** Every acceptance threshold in the paper is stated in `2L`, so anything transcribed from it must be halved.

### Determinism

Same input, same tree, whatever the thread count. Floating-point addition is not associative, so parallel reductions collect per-unit contributions and sum them in a fixed order rather than reducing in rayon's split order. Do not replace those with `.sum()` on a `ParallelIterator`.

## Conventions and gotchas

- British English throughout (`normalise`, `optimise`, `neighbours`, `centred`).
- `#![warn(missing_docs)]` is on. Every item gets `### Params` / `### Returns`.
- One `thiserror` enum in `src/errors.rs`, grouped by subsystem. Add to the matching section, not the bottom. Anything a caller could hit returns `Err`; panics are for broken invariants the type system cannot state.
- `wide` types appear only in `src/utils/simd.rs`.
- Benches need a `harness = false` entry in `Cargo.toml` and must check their output before reporting a timing. An implausible speedup is the signature of a sweep that did nothing.
- Do not add a SIMD tier without a measurement in its doc comment, and pick the kernel by call count rather than by how vectorisable it looks. `edge_newton` earned one at 35 to 45 calls per candidate pair; `split_derivative` reads like the hot loop, runs 0.07 times per pair, and measured flat.

## What's tracked outside this file

- The algorithm: `docs/SPEC.md`.
- How the crate is built and why: `docs/DESIGN.md`.
- What was optimised, what failed, and the rules: `docs/PERFORMANCE.md`.
- The black-box comparison and what may be said about it: `docs/COMPARISON.md`.
- The licence position and disclosure: `PROVENANCE.md`.
- Change history: `CHANGELOG.md`.
