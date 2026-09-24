# Design

How this implementation is built and why. The equations are in `docs/SPEC.md`,
which is the transcription of the paper's Supplementary Information and the only
thing the code is written from. The tuning history is in `docs/PERFORMANCE.md`.

## The model, briefly

Every node in the tree carries a latent position in feature space. An edge of
length `t` says the child is drawn from a Gaussian centred on the parent with
variance `t` per feature. Each observed cell contributes a Gaussian measurement
likelihood with its own error bars. The objective is the marginal likelihood
with every latent internal position integrated out, which stays tractable
because it is all Gaussian and factorises over features.

That integration is the continuous-trait form of Felsenstein's pruning
algorithm. A node is summarised by one mean and one precision per feature, a
parent is formed from its children, and the tree loglikelihood falls out of the
sweep. `model::likelihood` implements it (SPEC 4 and 5), `model::branch` solves
for a single edge length (SPEC 6), `model::merge` scores a candidate merge
(SPEC 8) and `model::place` attaches a node to an existing tree (SPEC 7).

## The arena

One flat `Tree`: a parent index and a branch length per node, nothing else.
Leaves occupy `0..n_leaves`, internal nodes follow, ordered by height above the
leaves. Three things depend on that ordering and break silently without it.

- Ascending index order is a valid post-order, so a sweep is a linear scan.
- A parent row lies above all of its children's rows, so writing a parent while
  reading its children is a `split_at_mut` rather than an unsafe alias.
- Each level is a contiguous index range.

`Tree::from_parents` relabels internal nodes to enforce it. That is free for
callers, because internal rows are always computed and never supplied, so
nothing caller-held is indexed by an internal node.

## One state layout

`NodeState` is row-major `[node][feature]` and sequential. Every prune in
`bonsai`, `backbone`, `spr`, `nni` and `polytomy` goes through it.

There used to be a second layout, blocked `[block][node][feature]` and parallel
over the feature axis. It was faster on the kernel and never called by anything
outside a benchmark, and it was deleted. `docs/PERFORMANCE.md` keeps the story
because the mistake is a repeatable one.

The parallel axis in this crate is the feature axis where it is the feature
axis at all, because the model factorises over features. That makes the kernels
indifferent to tree shape: a pathological ladder costs what a balanced tree
costs.

## Numeric policy

Storage is generic over `BonsaiFloat`. Every reduction accumulates in `f64`
regardless. The tree loglikelihood sums thousands of features whose interesting
differences are `O(1)` while the sum is `O(p)`, so `f32` accumulation would turn
the convergence criterion into noise. `f32` *storage* is worth having, halves
the working set the search streams, and is the fastest path.

Two conditioning choices matter and should not be simplified back.

- Effective means are formed as a convex combination,
  `m_k + (m_l - m_k) * wd_l / w_a`, not as `(wd_k*m_k + wd_l*m_l)/w_a`. The
  result is pinned between the two child means and cannot cancel. It feeds
  straight back into the recursion, so error here compounds up the tree.
- The quadratic term uses the pairwise identity (S33) for two children, never
  `sum wd*m^2 - w*m^2`.

Accuracy is judged on absolute error for means and relative for loglikelihoods.
Everything downstream consumes squared *differences* of means, so a mean near
zero carrying large relative error is fine, and a tolerance that fails on it is
the wrong tolerance.

## Units

Input is divided by `sqrt(v_g)` at ingest (SPEC 3.1), which removes the
diffusion scale from every kernel. The `2*pi` and `v_g` terms are dropped
because their count is independent of topology. So loglikelihoods are meaningful
only up to an additive constant, and are not comparable across implementations.

**This crate works in `L`, not `2L`.** Every acceptance threshold in the paper
is stated in twice the loglikelihood, so anything transcribed from it is halved.

## Determinism

Same input, same tree, whatever the thread count. Floating-point addition is not
associative, so parallel reductions collect per-unit contributions and sum them
in a fixed order rather than reducing in rayon's split order. The NNI scan
reduces to a running best with ties broken on the lower node id, which is what a
sequential scan does implicitly given the arena invariant. Tests pin this at 1,
3 and 8 threads. Do not replace those reductions with `.sum()` on a
`ParallelIterator`.

## The pipeline

`bonsai()` runs ingest, then the seven search steps of SPEC section 9, then an
eighth of our own.

1. Star, with every leaf on one branch, and that branch optimised.
2. Greedy likelihood-driven agglomeration, or a Ward linkage in its place.
3. Resolve the polytomies a zero-length branch stands for.
4. Global branch-length optimisation.
5. Subtree pruning and regrafting.
6. Nearest-neighbour interchange, generalised to polytomies.
7. Global branch-length optimisation again.
8. Collapse the internal zero-length edges steps 4 to 7 left behind, and
   reoptimise what the collapse changed.

Three constraints on that order, none of them in the specification.

**Steps 5 and 6 must follow step 4.** Both reject a proposal whose topology
fingerprint is unchanged, because otherwise they accept moves that only
reoptimise branch lengths and never terminate on topology. Run before the branch
lengths are optimised, that filter rejects the only improvements available and
topology recovery gets worse.

**Step 3 must follow step 2 rather than being folded into it.** Polytomies are
created by step 2 when an optimal branch length comes out at zero, and the
configuration that was optimal when it was created often is not once the centre
has moved.

**Step 8 runs last.** Step 3 is the only specified step that collapses, and it
runs before step 4, so every zero-length edge the later branch solves create
outlives the only pass that would remove it. Collapsing earlier recovers
slightly more topology and costs two orders of magnitude more time, because a
collapsed tree hands every nearby regraft a higher-degree star to resolve.
Running after the search cannot change what the search finds, which is the
property that makes step 8 safe to do unconditionally.

What step 8 cannot touch is a zero-length edge at a *leaf*. Those are not
internal edges: they are the model declining to separate two cells it has no
evidence to separate, they are a SPEC 6 boundary optimum, and they are a
statement about noise in the input rather than about topology.

## What makes the search tractable

Two restrictions, both from the specification, and the crate was cubic for a
week without them.

- **Candidate restriction** (SPEC 11): only pairs that are near neighbours in
  the transformed means are scored, through an ANN index rebuilt on a cadence.
- **Upper bounds on merge scores** (SPEC 10): the gain is linearised and bounded
  over an ellipsoid, and the primitive walks the bound-ordered pairs and stops
  when the best exact score beats every remaining bound.

Neither is exact, and both are very good in practice. `search::candidates` can
miss a best pair that is in no neighbour list; the section 10.2 bound is not
strict. Their own modules say so, and both carry correctness gates against the
exhaustive scan.

## Tuning knobs

Every threshold is a named `const` whose doc comment says where the number came
from: an SI equation, or ours by measurement. No constant is carried over from
the published implementation, whose source is off limits under the licence
position in `PROVENANCE.md`. Explicitly ours to determine: the neighbour count
`k`, the kNN rebuild cadence, the placement-search tolerance, the ellipsoid
`nsteps` schedule, the default backbone size and the SPR revisit radius.

SPR has an exact and an approximate mode, `SprSearch`, in the same spirit as
`StartTree`: `Exact` is the search SPEC section 9.3 specifies, `Approximate` is
the default and carries this crate's shortcuts, each measured across tree shapes
and sizes before it was switched on.

## Errors

One `thiserror` enum in `src/errors.rs`, grouped by subsystem. Anything a caller
could hit returns `Err`. Panics are for broken invariants the type system cannot
state, and the assertions in `NodeState::prune` are real checks rather than
`debug_assert`s because a state built for one tree indexes entirely within
bounds against a smaller one and would otherwise return a well-formed answer
computed from the wrong rows.
