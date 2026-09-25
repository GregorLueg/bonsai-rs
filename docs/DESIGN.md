# Design

How it's built and why. Equations are in [specs](SPEC.md), tuning history in
[performance](PERFORMANCE.md).

## Model

Continuous-trait Felsenstein pruning: each node is one mean and one precision
per feature, parents are formed from children, the loglikelihood falls out of
the sweep. `model::likelihood` does the recursion (SPEC 4, 5), `model::branch`
one edge length (SPEC 6), `model::merge` a candidate merge (SPEC 8),
`model::place` an attachment (SPEC 7).

## Arena

One flat `Tree`: a parent index and a branch length per node. Leaves are
`0..n_leaves`, internal nodes follow ordered by height. Break this and three
things fail silently:

- Ascending index is a post-order, so a sweep is a linear scan.
- A parent row sits above its children, so writing it while reading them is a
  `split_at_mut`, not an unsafe alias.
- Each level is a contiguous range.

`Tree::from_parents` relabels internal nodes to enforce it. Free for callers:
internal rows are computed, never supplied.

## State layout

`NodeState`, row-major `[node][feature]`, sequential. Every prune goes through
it. A second, feature-blocked parallel layout was faster on the kernel, only
ever called by a benchmark, and deleted; [performance](PERFORMANCE.md) has the story.

Where there's parallelism over a sweep it's over features, since the model
factorises there. A ladder costs what a balanced tree costs.

## Numerics

Storage generic over `BonsaiFloat`, every reduction in `f64`. The loglikelihood
sums thousands of features with `O(1)` differences on an `O(p)` total; `f32`
accumulation would turn convergence into noise. `f32` *storage* halves the
working set and is the fastest path.

Don't "simplify" these back:

- Effective means as a convex combination, `m_k + (m_l - m_k) * wd_l / w_a`,
  not `(wd_k*m_k + wd_l*m_l)/w_a`. Pinned between the child means, can't cancel,
  and errors here compound up the tree.
- The quadratic term via the pairwise identity (S33), never
  `sum wd*m^2 - w*m^2`.

Judge means on absolute error and loglikelihoods on relative. Everything
downstream uses squared *differences* of means, so relative error on a mean near
zero doesn't matter.

## Units

Input is divided by `sqrt(v_g)` at ingest (SPEC 3.1). The `2*pi` and `v_g`
terms are dropped since their count doesn't depend on topology, so
loglikelihoods are only meaningful up to a constant and not comparable across
implementations.

**This crate works in `L`, not `2L`.** The paper states thresholds in `2L`;
halve anything transcribed.

## Determinism

Same input, same tree, any thread count. Parallel reductions collect per-unit
contributions and sum in a fixed order. The NNI scan keeps a running best with
ties to the lower node id, which is what a sequential scan does anyway. Tests
pin this at 1, 3 and 8 threads. No `.sum()` on a `ParallelIterator`.

## Pipeline

`bonsai()` runs ingest, SPEC section 9's seven steps, then one of ours:

1. Star, its one branch optimised.
2. Greedy agglomeration, or Ward linkage instead.
3. Resolve polytomies.
4. Global branch lengths.
5. SPR.
6. NNI, generalised to polytomies.
7. Global branch lengths again.
8. Collapse internal zero-length edges left by 4 to 7, reoptimise locally.

The order matters in three places the spec doesn't mention:

- **5 and 6 after 4.** Both reject proposals with an unchanged topology
  fingerprint so they terminate. Before branch lengths are optimised, that
  filter rejects the only improvements on offer.
- **3 separate from 2.** A polytomy made in step 2 is often no longer optimal
  once the centre moves.
- **8 last.** Only step 3 collapses, and it runs before 4, so later zero-length
  edges would survive. Collapsing earlier recovers slightly more topology at
  100x the cost, because every nearby regraft then faces a bigger star. After
  the search it can't change what the search finds, so it's safe.

Step 8 leaves zero-length *leaf* edges alone. Those are a SPEC 6 boundary
optimum: the model has no evidence to separate that cell from its parent. A
statement about noise, not topology.

## Tractability

Two restrictions from the spec; without them the search was cubic for a week.

- **Candidates** (SPEC 11): only near neighbours in the transformed means get
  scored, via an ANN index rebuilt on a cadence.
- **Upper bounds** (SPEC 10): the gain is linearised and bounded over an
  ellipsoid; the scan stops once the best exact score beats every remaining
  bound.

Neither is exact: a best pair can sit outside every neighbour list, and the
10.2 bound isn't strict. Both have correctness gates against the exhaustive
scan.

## Knobs

Every threshold is a named `const` citing an SI equation or our measurement.
Nothing comes from the published source (see [provenance](../PROVENANCE.md)). Ours: neighbour
count `k`, kNN rebuild cadence, placement tolerance, ellipsoid `nsteps`
schedule, backbone size, SPR revisit radius, NNI rescore radius.

`SprSearch` and `NniSearch` follow `StartTree`: `Exact` is SPEC 9.3 and 9.4,
`Approximate` is the default with our shortcuts, each measured across shapes and
sizes first.

## Errors

One `thiserror` enum in `src/errors.rs`, by subsystem. Caller-reachable failures
return `Err`; panics are for broken invariants. The asserts in `NodeState::prune`
are real checks, not `debug_assert`s: a state built for one tree indexes in
bounds against a smaller one and would return a well-formed wrong answer.
