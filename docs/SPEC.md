# Bonsai: implementation specification

Transcribed from the CC-BY-4.0 paper and Supplementary Information of de Groot,
Morillo Leonardo, Pachkov and van Nimwegen, *Nature Biotechnology* 2026,
doi 10.1038/s41587-026-03220-2. See [provenance](../PROVENANCE.md).

**This file is the only implementation source.** Code cites the `(Sxx)` numbers
below. Where this document restructures an SI expression for numerical or
computational reasons, that is flagged as a *deviation* and carries a test that
pins it to the SI form.

Maths is written ASCII-flavoured so it can be grepped from doc comments.
`sum_g` is a sum over features, `sum_{i in C(a)}` over the children of `a`.

---

## 1. Notation

| symbol | meaning |
|---|---|
| `n` | number of cells (leaves) |
| `p` | number of features (genes) |
| `mu[g,i]` | measured mean of feature `g` in cell `i` |
| `sig[g,i]` | measured standard deviation on that mean |
| `v[g]` | total variance of feature `g` across the dataset |
| `t[i]` | length of the branch upstream of node `i` (a diffusion time) |
| `x[g,i]` | unknown true position of node `i` in feature space |
| `w[g,i]` | precision of a leaf, `1/sig[g,i]^2` |
| `W[g,a]` | effective precision of the subtree below `a` (SI writes `wbar`) |
| `M[g,a]` | effective mean of the subtree below `a` (SI writes `mubar`) |
| `Wd[g,a]` | effective precision corrected for diffusion along `t[a]` (SI writes `wbar*`) |
| `C(a)` | children of `a` given the current root |

Effective-leaf quantities depend on which node is the root, because that decides
what is downstream. Where it matters, say so explicitly.

## 2. The model

Two kinds of factor (SI.B.1). A Gaussian measurement likelihood per leaf (S2):

```
P(cell_i | x_i) = prod_g N(mu[g,i] ; x[g,i], sig[g,i]^2)
```

and a Brownian-motion transition along each branch (S12):

```
P(x_i | x_parent, t_i) = prod_g N(x[g,i] ; x[g,parent], v[g] * t_i)
```

The tree likelihood (S1, S13) marginalises every latent node position with a
uniform prior on the root. Because every factor is Gaussian and factorises over
features, the integrals are analytic.

The likelihood is **independent of the choice of root** (S14). Rerooting is free
and is used throughout.

## 3. Ingest

### 3.1 The scale transform (S21)

Divide out `v[g]` once, at ingest:

```
mu_t[g,i]  = mu[g,i]  / sqrt(v[g])
sig_t[g,i] = sig[g,i] / sqrt(v[g])
```

After this the diffusion prior has unit variance per feature and `v[g]`
disappears from every kernel. It contributes only a term
`-(|C(r)| - 1)/2 * sum_g log v[g]` to the loglikelihood, which is
topology-independent (SI.B.2.3) and is dropped.

**Everything downstream works in transformed units.** Node positions reported to
the user are multiplied back by `sqrt(v[g])`.

### 3.2 Dropping the 2*pi terms

The count of `-(1/2) log(2*pi)` factors is exactly `p * (n - 1)`, independent of
topology (SI.B.2.3 "Note on the factors of 2*pi"). Dropped.

**Consequence:** loglikelihoods reported by this crate are up to an additive
constant. Only differences are meaningful. See section 12.

### 3.3 Feature selection by signal-to-noise (SI.B.1.3)

Signal-to-noise for feature `g` (S6):

```
S[g] = (1/n) * sum_i delta[g,i]^2 / eps[g,i]^2
```

where `delta` is the posterior estimate of the deviation from the feature mean
and `eps` its error bar. When the input is a mean/SD pair rather than Sanity
output, estimate them (S7-S11):

1. Maximum-likelihood feature mean, given `v[g]` (S8):
   `mubar[g] = (sum_i mu[g,i]/(v[g]+sig[g,i]^2)) / (sum_i 1/(v[g]+sig[g,i]^2))`
2. Substitute `mubar[g]` for the true mean and solve for `v[g]` (S9):
   `sum_i (1/(v[g]+sig[g,i]^2)) * ((mu[g,i]-mubar[g])^2/(v[g]+sig[g,i]^2) - 1) = 0`
   One-dimensional, monotone in the useful range; bracket and solve.
   Note this has the same shape as the branch-length root find in section 6.
3. Then (S10, S11):
   `delta[g,i] = v[g]/(v[g]+sig[g,i]^2) * (mu[g,i]-mubar[g])`
   `eps[g,i]^2 = v[g]*sig[g,i]^2/(v[g]+sig[g,i]^2)`

Keep features above a threshold on `S[g]`. Their default is 1; ours is a
documented `const` we set ourselves.

`v[g]` from step 2 is also what section 3.1 divides by, when Sanity has not
supplied one.

### 3.4 From Sanity posteriors to likelihood parameters (S5)

Bonsai needs likelihood means and error bars, Sanity reports posteriors. Given
posterior mean `xstar[g,i]`, posterior SD `eps[g,i]` and Sanity's `v[g]`:

```
mu[g,i]      = xstar[g,i] * v[g] / (v[g] - eps[g,i]^2)
sig[g,i]^2   = eps[g,i]^2 * v[g] / (v[g] - eps[g,i]^2)
```

This is ill-conditioned as `eps^2 -> v`, which is why they added a Sanity mode
returning the posterior-maximising `v[g]` rather than its expectation. Guard the
denominator and report how many features were affected.

## 4. Effective leaves and the pruning recursion (SI.B.2.2, S19)

The whole subtree below `a` is summarised as a single leaf:

```
W[g,a]  = sum_{k in C(a)} Wd[g,k]
M[g,a]  = (sum_{k in C(a)} Wd[g,k] * M[g,k]) / W[g,a]
Wd[g,a] = W[g,a] / (1 + t[a] * W[g,a])
```

A precision-weighted mean; the branch length adds to the variance. For a leaf
that is an observed cell, `W[g,i] = 1/sig_t[g,i]^2` and `M[g,i] = mu_t[g,i]`.

Applied recursively in post-order, this reduces any tree to a star around the
root. That is the continuous-trait Felsenstein pruning algorithm.

## 5. Tree loglikelihood (SI.B.2.3, S20)

Per node `a` with children `C(a)`, dropping the constants of section 3:

```
L(a) = sum_{k in C(a)} L(k)
     + (1/2) * sum_g [ sum_{k in C(a)} log Wd[g,k]
                       - log( sum_{k in C(a)} Wd[g,k] )
                       - sum_{k in C(a)} Wd[g,k] * (M[g,a] - M[g,k])^2 ]
```

The total is `L(root)`. Evaluated in post-order over the arena, one linear scan.

**Deviation (numerical).** The quadratic term is better computed with the
three-point identity (S33), which avoids forming `M[g,a]`:

```
sum_i w_i * (Mbar - M_i)^2 = (1/sum_i w_i) * sum_{i<j} w_i w_j (M_i - M_j)^2
```

Use the pairwise form for small child sets (2 or 3, the common case) and the
direct form otherwise. Both must agree to 1e-12 relative in tests.

## 6. Branch lengths (SI.B.3)

Collapse everything except one edge `k-l` into two effective leaves, one on each
side. Then (S24), per feature, with

```
s[g] = 1/W[g,k] + 1/W[g,l]
d[g] = (M[g,k] - M[g,l])^2
```

the loglikelihood contribution of that edge is

```
L(t) = -(1/2) * sum_g [ log(s[g] + t) + d[g] / (s[g] + t) ]
```

and its derivative (S25) is

```
L'(t) = -(1/2) * sum_g (1/(s[g]+t)) * (1 - d[g]/(s[g]+t))
```

**This is the crate's hottest kernel.** Precompute `s` and `d` once per edge;
each Newton step is then one reciprocal, one FMA and one multiply per element,
with no transcendental. `sum_g log(s[g]+t)` is needed only for the final value,
never inside the iteration.

Single-edge optimisation is the root of `L'(t) = 0`. The bracket is
`[0, max_g(d[g] - s[g])]`: beyond the upper end every term of `f` is positive, so
the root cannot lie further out, and `prep_edge` returns it for free on the pass
that already touches both arrays. `f(0) >= 0` means the optimum is the boundary
`t = 0`, which is how the search manufactures the polytomies section 9.2 then
resolves.

**Deviation.** Solved by safeguarded Newton in `t` rather than by Newton in
`log t`. The bracket already guarantees positivity, so the change of variable
buys nothing and costs an exp and a log per iteration. The Newton step is taken
when it stays inside the bracket and at least halves the previous step, and a
bisection step is taken otherwise; `f` is not globally monotone, so an
unsafeguarded Newton can leave the bracket.

Global optimisation is quasi-Newton over all branch lengths using the same
expression for every edge, with the two-sided effective leaves obtained from one
upward and one downward sweep over the arena.

## 7. Attaching a node to the tree (SI.B.4)

### 7.1 The attachment score (S27)

Attaching node `q` (precision `W_q`, position `M_q`) below node `a` with branch
length `t`, where `W_a`, `M_a` summarise the whole existing tree seen from `a`:

```
dL(t) = -(1/2) * sum_g [ log(t + 1/W[g,a] + 1/W[g,q])
                         + (M[g,a]-M[g,q])^2 / (t + 1/W[g,a] + 1/W[g,q]) ]
```

Identical in form to section 6, so the same kernel and the same root find serve.

### 7.2 Beam search over attachment points (SI.B.4.1)

1. Pick start points spread over the tree. They use `log(n)` centres from the
   distance-based clustering of the Methods section. Our count and our centre
   selection are ours to choose and to document.
2. From each start point: score the current node, score all its neighbours, and
   recurse into every neighbour whose score exceeds `best_seen - Delta`.
3. Take the best attachment point over all start points.

`Delta` is a tolerance we set, not theirs. `Delta = 0` is greedy hill-climbing;
`Delta = inf` is exhaustive. Both are useful test modes.

### 7.3 Attaching to an edge (SI.B.4.2)

Attaching to a node creates a polytomy there, since the node already had three
neighbours. Rather than special-casing edge attachment, always follow an
attachment with the polytomy resolution of section 9.2. That covers edge
attachment as a special case.

## 8. The merge score (SI.C.2.1, S31, S34)

The core of the search. Given a star around root `r`, what is the loglikelihood
gain from inserting an ancestor `a` above two of its children, `k` and `l`?

### 8.1 Peeling the rest of the star

Let `R` be every child of `r` except `k` and `l`. Peel it off in `O(p)`:

```
WR[g] = W[g,r] - Wd[g,k] - Wd[g,l]
MR[g] = (M[g,r]*W[g,r] - Wd[g,k]*M[g,k] - Wd[g,l]*M[g,l]) / WR[g]
```

This is why the merge score does not cost `O(n*p)` per pair.

### 8.2 Before and after

Both trees are three-leaf stars over `{k, l, R}`. Write

```
diffusion(W, t) = 1 / (t + 1/W)
```

Before the merge, the three precisions at `r` are

```
O1 = diffusion(W[g,k], t_rk)     O2 = diffusion(W[g,l], t_rl)     O3 = WR[g]
```

with `t_rk`, `t_rl` the existing branch lengths from the root. After the merge,
the star is centred on `a` and the `R` side is diffused by the new branch `t_ar`:

```
A1 = diffusion(W[g,k], t_ak)     A2 = diffusion(W[g,l], t_al)
A3 = diffusion(WR[g],  t_ar)
```

### 8.3 The score

Precompute the three squared separations once per candidate pair. They do not
depend on any branch length:

```
d_kl[g] = (M[g,k] - M[g,l])^2
d_kR[g] = (M[g,k] - MR[g])^2
d_lR[g] = (M[g,l] - MR[g])^2
```

Then with `SA = A1+A2+A3` and `SO = O1+O2+O3` (S34):

```
dL = (1/2) * sum_g [   log(A1*A2*A3) - log(SA) - log(O1*O2*O3) + log(SO)
                     - (A1*A2*d_kl + A1*A3*d_kR + A2*A3*d_lR) / SA
                     + (O1*O2*d_kl + O1*O3*d_kR + O2*O3*d_lR) / SO ]
```

The `O` half is fixed for a given pair and current root, so it is computed once
and reused across every step of the branch-length optimisation below.

### 8.4 Optimising the three new branch lengths (SI.C.2)

A tree-wide reoptimisation after every proposed merge is infeasible, so only
`t_ak`, `t_al`, `t_ar` are optimised, in two stages:

**Stage 1.** Optimise `T = t_ak + t_al` with `a` detached from the root. With `a`
detached the tree is just `k - a - l`, so this is exactly section 6 with
`s[g] = 1/W[g,k] + 1/W[g,l]` and `d[g] = d_kl[g]`. One bounded Newton root find.
`T` fixes the `k`-to-`l` distance and is not revisited later in the iteration.

**Stage 2.** Optimise all three, constrained to `t_ak + t_al = T`. Two free
parameters: `u = t_ak` in `[0, T]` and `t_ar >= 0`. Analytic gradient via

```
d(diffusion(W,t))/dt = -diffusion(W,t)^2
```

so every partial derivative of `dL` falls out of section 8.3 by the chain rule.

**Deviation.** Coordinate descent over the two freedoms rather than a
two-dimensional Newton, because each half is then a bracketed one-dimensional
solve reusing machinery that already exists, and no Hessian is needed. Holding
`u` fixed collapses `k` and `l` into one effective leaf at the ancestor, so
`t_ar` is an ordinary edge solve (section 6). Holding `t_ar` fixed leaves a
bracketed solve on `[0, T]` for the split, driven by the sign of `d(dL)/du`.

Two sweeps is the default. One sweep is measurably short; two agree with twenty
to the bit on almost every fixture, but the coordinate descent converges slowly
where the split and the root branch are strongly coupled, and the shortfall
there reaches `5e-3` relative in the gain. The numbers are on `MergeParams`.
Stage 2 is not cosmetic: it roughly triples the gain over the unrefined
half-and-half split.

The split's bracket is closed, not open: an optimum at an end has to come back
as exactly `0` or exactly `T`, because section 9.2 keys polytomy resolution on
zero-length branches.

**Cost note.** Stage 2 is the dominant term in a merge scan, so the solve that
drives it must not compute anything it does not read. In particular the split
search needs only the sign of `d(dL)/du`, never the gain, and the gain is where
the logarithms are.

The rationale for the two stages is theirs and worth keeping: `t_ar` is very
likely to be reoptimised later, either because another ancestor lands above `a`
or because the root moves. The `k`-to-`l` separation is not.

## 9. The search (SI.C)

Seven steps:

1. Star tree, branch lengths optimised.
2. Iteratively add the most likely internal node.
3. Resolve polytomies.
4. Global branch-length optimisation.
5. SPR moves.
6. Nearest-neighbour interchanges.
7. Final global branch-length optimisation.

**Deviation: an eighth step, 2026-09-13.** Step 3 is the only step that
collapses zero-length edges, and it runs before step 4. Section 6 is explicit
that a branch solve landing on `t = 0` is normal, so every zero-length edge
steps 4 to 7 create outlives the only pass that would remove one. This crate
therefore runs the collapse of section 9.2 once more after step 7, followed by
a branch reoptimisation.

Measured on Sanity-preprocessed data at 10,000 cells by 2,767 genes: 32
internal zero-length edges survive step 7, and removing them takes
Robinson-Foulds from 2659 to 2632 and the loglikelihood from -11253193.9 to
-11253188.3, for 2.75 seconds.

It runs last rather than before step 5, which was the other candidate.
Collapsing early reaches 2614, eighteen splits better again, but costs 247
seconds because a collapsed tree hands every nearby regraft a higher-degree
star, and it moves distance recovery the wrong way on the one seed measured.
Running after the search cannot change what the search finds, which is what
makes it safe unconditionally, and the collapse only ever removes structure so
it cannot invent a split the data does not support.

What it does not touch is a zero-length edge at a **leaf**, which is not an
internal edge. Those are the model declining to separate two cells it has no
evidence to separate, 129 of them at that size, and they are the answer rather
than a defect.

### 9.1 The star primitive (SI.C.2)

**One routine drives steps 2, 3 and 6.** Given a node `X` and the star of its
children: score every candidate pair by section 8, merge the best, summarise the
new ancestor as an effective leaf (section 4), repeat. Stop when `X` has three
children left or no pair gives a positive gain.

Step 2 is this with `X = root`.

### 9.2 Polytomy resolution (SI.C.3)

Merging can produce zero-length branches, which collapse into polytomies. A
zero-length edge does not change the likelihood, so the configuration was optimal
when it was created, but the root moves as the search proceeds and it often stops
being optimal. So: walk every node with more than two children and run the star
primitive on it.

**The collapse has to be done, not assumed.** A zero-length branch leaves two
separate nodes in the arena, so a degree test alone finds nothing on a tree the
merge scan has left structurally binary, which is the ordinary case: step 3 then
does nothing at all. `search::polytomy` deletes every internal node whose
upstream branch is exactly zero and reattaches its children to the node above,
which leaves the loglikelihood untouched, before it counts degrees. The test is
exact rather than a tolerance, which is why section 8.4's split bracket has to
be closed. It runs at the top of every sweep, since a resolution can itself
place an ancestor at zero distance from its centre.

### 9.3 SPR (SI.C.5)

Prune a node and its subtree, then regraft. Two selection strategies: random, or
ordered by upstream branch length descending. The second is their default and the
reasoning is sound: a long upstream branch means the subtree is dissimilar to its
parent, so it is the least confidently placed. Regrafting summarises the pruned
subtree as an effective leaf and runs the beam search of section 7.2, followed by
polytomy resolution.

**Deviation: the default revisits only what changed, 2026-09-24.** Sweeps repeat
until one accepts nothing, and each proposes every subtree. This crate's default
(`SprSearch::Approximate`) proposes every subtree in the first sweep only; later
sweeps propose only subtrees within five edges of a clade the previous sweep's
moves created. `SprSearch::Exact` is the search as specified. Measured over
three tree shapes at three noise levels and four real-data configurations up to
10,000 cells, the approximation lands within a few nats of the exact search and
roughly halves steps 5 to 8 at 5,000 and 10,000 cells; [performance](PERFORMANCE.md) has
the numbers. `test_a_revisit_radius_wider_than_the_tree_changes_nothing` pins
the approximate path to the exact one when the radius covers the tree.

### 9.4 NNI, generalised to polytomies (SI.C.6)

The textbook NNI reconnects four subtrees around an internal edge. Bonsai's trees
have polytomies, so:

1. Pick an internal edge `k-l`, both ends internal.
2. Delete `k`, moving all of its subtrees onto `l`.
3. Run the star primitive on `l`.

For four subtrees this recovers exactly the classical NNI. Two phases:

- **Random.** Inside the star primitive, sample the pair to merge with
  probability proportional to the resulting tree likelihood,
  `p(q,r) = P(T_merge[q,r]) / sum_{i<j} P(T_merge[i,j])`. Escapes local optima.
  Note this is a softmax over *loglikelihoods*: subtract the maximum before
  exponentiating.
- **Greedy.** Score an NNI for every edge, perform the best, repeat until no move
  improves the likelihood.

**Deviation: the greedy stopping rule needs a topology filter.** Taken literally,
"repeat until no move improves the likelihood" does not terminate on topology at
all. Collapsing an edge and re-resolving the star regroups the same subtrees but
reoptimises three branch lengths at the centre, so it reports a gain while
changing nothing about the topology. Started from the generating tree itself,
the greedy phase ran 104 to 239 rounds with Robinson-Foulds pinned at zero
throughout: it was doing branch-length descent wearing a topology search's
clothes. Any proposal whose split fingerprint matches the current tree is
therefore discarded. Ladder recovery went from 115 rounds to 23. Branch lengths
are steps 4 and 7 and should not be smuggled in here.

The corollary is that **NNI must run after step 4**, which is the order the SI
gives anyway. On a ladder with unoptimised branch lengths the filter makes
recovery worse (RF 44 to 24 rather than to 0), because the landscape is then
dominated by wrong branch lengths and the only improvements available are the
ones the filter rejects.

**Deviation: the default greedy phase rescores lazily, 2026-09-25.** Taken
literally the greedy phase scores every edge every round and performs one move,
so it costs a full scan per move. This crate's default (`NniSearch::Approximate`)
caches each edge's gain, rescores after a move only the edges within five edges
of the clades the move created, and rescores the leading cached gain on the
current tree before taking it (lazy greedy evaluation, Minoux 1978). A full scan
runs whenever nothing cached improves, so the phase still stops only where no
edge improves. On thirteen datasets across three tree shapes, three noise levels
and four real-data configurations up to 10,000 cells, the finished tree matched
`NniSearch::Exact` on every one, with step 6 five to eight times faster where
it had work to do.

**The random phase is close to inert at realistic feature counts.** This is a
property of the method as published, not of this implementation. The softmax is
over loglikelihoods whose gaps scale as `O(p)`, so it concentrates on the greedy
pick as `p` grows. Measured at 32 leaves over 8 seeds, counting how many moved
the topology off its starting point at all:

| features | seeds that moved |
|---|---|
| 8 | 8/8 |
| 32 | 4/8 |
| 128 | 2/8 |
| 512 | 3/8 |

At 256 features it left the topology untouched for every seed tried and only
reoptimised branch lengths. Since the paper works at thousands of features, the
phase it describes as escaping local optima will rarely move anything. A
temperature would fix that, but the SI specifies none and inventing one is a
change to the method rather than to its implementation, so this crate ships the
phase off by default with the measurement recorded against it.

## 10. Upper bounds on merge scores (SI.D.1)

**Not an optimisation. Without this the algorithm does not run.** Naive scaling
is `O(n^3 p)` (SI.D). Section 11 removes one factor of `n`; this section removes
most of another.

### 10.1 The problem

After a merge, the root moves. Section 8.3 shows every other pair's `dL` depends
on the root only through `MR[g]` and `WR[g]`, which in turn depend on `M[g,r]`
and `W[g,r]`. So in principle all `dL` must be recomputed every round.

### 10.2 Linearise, then bound over an ellipsoid

Take a first-order expansion of `dL` in the root's movement `(xi_mu, xi_w)`
(S37), then maximise it over an ellipsoid of plausible movements. This is not a
strict mathematical bound: the linear approximation can underestimate. The SI
argues, correctly, that the underestimate is tiny for small ellipsoids and is
swamped by the overestimate from assuming the root travels to exactly the worst
point on a high-dimensional ellipsoid boundary.

**Deviation (implementation).** The SI's flattened `d(dL)/dW[g,r]` (S36) is a
seven-term expression and is the single most likely thing in this document to be
transcribed wrong. Compute it instead by the chain rule through the peeling of
section 8.1, holding the pair's own quantities fixed:

```
d(MR)/d(M_r) = W_r / WR
d(MR)/d(W_r) = (M_r - MR) / WR
d(WR)/d(W_r) = 1
d(WR)/d(M_r) = 0
```

with `d(dL)/d(MR)` and `d(dL)/d(WR)` read straight off section 8.3. The result
must match S35 and S36; a central-difference test pins it.

### 10.3 Ellipsoid size (S38-S43)

The root is a precision-weighted mean over `nc` children. Removing two and adding
one gives, per merge step, movement scaling as

```
xi_mu ~ 1/sqrt(nc * W[g,r])        xi_w ~ W[g,r]/nc
```

Over `nsteps` merges the position random-walks while the precision only
decreases, so

```
xi_mu ~ sqrt(nsteps / (nc * W[g,r]))       xi_w ~ nsteps * W[g,r] / nc
```

Ellipsoids (S41):

```
E_mu = { xi_mu : sum_g (nc * W[g,r] / nsteps) * xi_mu[g]^2 < 1 }
E_w  = { xi_w  : sum_g (nc^2 / (W[g,r]^2 * nsteps^2)) * xi_w[g]^2 < 1 }
```

Rescale to unit balls (S42), at which point the linearised change is a dot
product and the maximum over the ball is the vector's norm (S45):

```
dL_UB = sum_g |d(dL)/d(M[g,r])| * sqrt(nsteps/(nc*W[g,r]))
      + sum_g |d(dL)/d(W[g,r])| * W[g,r] * nsteps / nc
```

Larger `nsteps` means looser bounds that stay valid longer. It is a pure
speed/tightness knob and does not affect the answer.

### 10.4 Using the bounds (SI.D.1, "Using the upper bound information")

1. In the first round compute true `dL` and an upper bound for every candidate
   pair. Keep a list sorted by upper bound, descending, tagged with the root
   state `(M_r0, W_r0)` at which the bounds were computed.
2. In later rounds, walk the list from the top computing true `dL` at the current
   root. Stop as soon as the best true value seen exceeds the next entry's upper
   bound: that pair is provably the best.
3. Merge it. Recompute the root. If the root has left the ellipsoid, recompute
   all bounds next round.
4. Either way, the new ancestor needs true `dL` and bounds against every
   candidate partner; insert those into the list.

### 10.5 Sizing `nsteps` online (SI.D.1.1)

Track how deep into the sorted list each round had to go. Deep means the bounds
are too loose, so shrink the ellipsoid. Shallow means we can afford looser bounds
and fewer recomputations, so grow it. **The schedule and its clamps are ours, and
must be tuned and dated, not inherited.**

**Deviation: depth alone is the wrong signal.** The walk over the sorted list is
chunked, because scoring pairs one at a time wastes the parallel scan. It
therefore cannot stop part way through a chunk, so once the bounds are working
at all every round reports the same minimum depth, every round reads as shallow,
and the ellipsoid grows until it meets whatever cap it was given. Measured, the
cap rather than the schedule was choosing the answer. What is tracked instead is
a running average of redraw cost against walk cost, hill-climbing on the
difference, which for a trade-off of this shape settles at the bottom of the
bowl. The value is robustness rather than the last few per cent: fixed schedules
either side of the optimum cost 0.175 and 0.170 against the adaptive 0.109.

### 10.6 Two corrections to this section, 2026-08-31

Both were found by measurement while implementing, and both are errors in this
document rather than in the SI.

**S45 is a norm, not a sum of absolute values.** The equation as transcribed sums
`|dL/dM| * scale` over features. That is the maximum of the linear form over the
circumscribing *box*, not over the ellipsoid, and the two differ by up to
`sqrt(p)`. At two thousand features the box form inflates every bound by about
45 times, at which point nothing is ever pruned and the whole section is dead
weight. The prose either side of S45 says "the vector's norm", which is correct;
take the Euclidean norm.

**S41's half-axes need the feature count.** The ellipsoid is a norm over all `p`
features at once, so a movement of one merge's size in *every* feature sits at
radius `sqrt(p)`, not at radius 1. Both scales therefore pick up a factor of
`sqrt(p)`. Without it, `nsteps` is wrong by a factor of `p`, and worse, a default
measured at one feature count means nothing at another, which is the kind of
constant that looks tuned and is actually arbitrary.

**Restructuring worth keeping.** The maximum of a linear form over a ball of
radius `R` is `R` times its maximum over the unit ball. So rather than adding a
fixed worst-case slack to every bound, store the ellipsoid as a metric,
accumulate the distance the centre has travelled in that metric round by round,
and let a pair's bound be `gain + distance * unit_slack`. Strictly tighter, and
it makes a pair scored during the current round need no special case: its
distance is zero, so its bound is exactly its own gain.

## 11. Candidate restriction (SI.D.2)

Bounds alone still require a full `O(n*k*p)` rescore whenever the root leaves the
ellipsoid. Cut the candidate set too.

Only the single best pair per round is needed, never a full ranking. Nodes close
together in plain Euclidean distance are the plausible candidates; a distant pair
is never the best merge. So:

1. Before the first round, compute a `k`-nearest-neighbour graph over the root's
   children in Euclidean distance on the transformed means.
2. Score only each node against its neighbours: `n*k` pairs, not `n^2`.
3. A new ancestor **inherits the union** of its two children's neighbour lists.
4. Rebuild the graph after a set number of new ancestors.

They report `k` in the range 5 to 10 and a rebuild every `0.1*n` ancestors. **Our
`k` and our cadence are ours to determine and to date.** Both are pure
speed/accuracy knobs; neither changes the model.

**Deviation: the union must be symmetric.** Rule 3 taken literally gives the
ancestor its partners' lists but leaves no existing member pointing *at* the
ancestor. A second ancestor then inherits lists that never mention the first, no
pair of ancestors is ever offered as a candidate, and the star stalls several
merges early. So the graph is symmetrised at rebuild and that symmetry is
maintained on merge: both children are replaced by the ancestor everywhere they
appear, not only in the ancestor's own list. Caught by the `k >= n - 1` equality
gate of section 13.2, which is the test that exists for exactly this.

**Correction, 2026-08-28.** An earlier draft of this section claimed lists grow
under union inheritance and that this is why rule 4 exists. That was inference,
not something the SI says, and measurement contradicts it: under the symmetric
union the graph *thins*, because a merge removes two members' edges and adds back
one deduplicated union. The widest round is always the first. Measured at
`n = 128`, total pairs scored per star ran from 79 495 at cadence 1 down to
59 957 with no rebuild at all, against 349 500 exhaustive. A rebuild therefore
replenishes the candidate set rather than trimming it.

Rule 4 is still worth following, but for the other reason: merging moves the
points. An ancestor's effective mean is not either child's, so a graph built on
the old positions goes stale as a description of the current geometry, and
inheritance only propagates the old adjacency. Rebuilding re-derives the graph
from where the members actually are. Recovery is flat in the cadence over the
sizes tested, so this costs accuracy only at the extreme of never rebuilding.

Approximation, honestly: the best pair in the whole dataset might not be in any
neighbour list. The SI's argument is that with `n*k` candidates the probability
of missing it is very close to one; that is an empirical claim, and the exactness
test in section 13 is how we check it on our own fixtures.

Measured on our own fixtures, 2026-08-28, 24 replicates at 128 leaves by 200
features across three tree shapes, Robinson-Foulds to the generating tree out of
125 splits, exhaustive baseline 3.67:

| `k` | mean RF | excess over exhaustive | replicates identical to exhaustive |
|---|---|---|---|
| 4 | 7.62 | +3.96 | 0/24 |
| 6 | 4.25 | +0.58 | 5/24 |
| 8 | 4.08 | +0.42 | 12/24 |
| 12 | 3.67 | 0 | 21/24 |
| 16 | 3.67 | 0 | 24/24 |

The knee sits in the same place at 64 and 256 leaves, so the required `k` does
not appear to scale with `n` over the range that can be checked against an
exhaustive baseline. Their 5 to 10 is not free but is not far off: `k = 8` costs
0.42 splits of 125 and reproduces the exhaustive topology in half the replicates.
This crate ships 16, where both measures stop moving.

Note what that table cannot say. An exhaustive baseline costs `O(n^3 p)`, so
nothing above a few hundred members is checkable this way, and the claim that
`k = 16` remains sufficient at `10^4` upwards is extrapolation from three sizes.

## 12. Units and conventions

- **Loglikelihood, not twice it.** The reference works throughout in `2*L`, and
  every acceptance threshold in the paper is stated in those units. This crate
  works in `L`. Any threshold transcribed from the paper must be halved. This is
  asserted in `model::likelihood`'s
  `test_the_loglikelihood_is_l_and_not_twice_it`, which transcribes S20 with its
  factor of one half and checks the recursion against it.
- **Additive constant.** Sections 3.1 and 3.2 drop terms that are constant across
  topologies. Absolute loglikelihoods are therefore not comparable to the
  reference implementation's; differences are.
- **Transformed units.** All internal state is in `mu/sqrt(v)` units. Convert on
  the way out only.

## 13. Fixtures and tests

### 13.1 Simulation (SI.E.2)

Generate ground truth on a known tree, then add realistic noise.

**Binary tree, constant branch lengths (SI.E.2.1).** Root at the origin in `p`
dimensions. For each child, add a per-feature Gaussian step of variance
`t * v[g]` with `t = 1`. Ten generations gives 1024 leaves. Keep only the last
generation. Then per feature: centre across cells, rescale to variance `v[g]`,
add the target mean `mu[g]`.

**Random branch lengths (SI.E.2.2).** As above but `log(t)` uniform on
`[log 0.5, log 2]`, drawn independently per child.

**Unbalanced (SI.E.2.3).** Maintain a list of leaves, initially just the root.
Repeatedly pick a random leaf, give it two children, replace it in the list with
them. 1023 repetitions gives 1024 leaves.

**Counts (SI.E.1).** Sample a per-cell total `N_c`, then draw
`count[g,c] ~ Poisson(N_c * exp(x[g,c]))`. Only needed once a Sanity-style
preprocessing path exists; the tree tests can consume LTQs and error bars
directly.

`v[g]` is drawn from an exponential distribution with mean 2, which is what they
observed in real data.

**Direct-coordinate fixtures (SI.F.1)** are simpler and better for the kernel
tests: draw `v[g]` exponential with mean 2, draw a per-cell `t_c` log-uniform on
`[0.1, 10]` rescaled to mean 1, then `x[g,i] ~ N(0, t_i * v[g])`.

### 13.2 What must be tested

- **Numerical agreement** with the independent numpy reference to 1e-9 relative
  on the tree loglikelihood, at several `(n, p)`.
- **Analytic gradients**: section 6's `L'(t)`, and both partials of section 10.2,
  against central differences.
- **Root independence** (S14): reroot and rescore, expect equality. Asserted in
  `model::likelihood`'s `test_the_loglikelihood_is_the_same_at_every_rooting`,
  over every edge of four topologies.
- **Effective-leaf equivalence** (S19): scoring a tree with a subtree summarised
  equals scoring it unsummarised.
- **Monotonicity**: loglikelihood never decreases across steps 1 to 7.
- **Dropped-constant claims**: the `2*pi` count really is `p*(n-1)` and the
  `v[g]` term really is topology-independent, checked on two topologies over the
  same data.
- **Identity S33**: pairwise and direct forms of the quadratic term agree.
- **Bound soundness**: on small fixtures, no pair's true `dL` ever exceeds its
  recorded upper bound.
- **Acceleration equivalence**: sections 10 and 11 enabled must return the
  identical tree to the unaccelerated search on fixtures small enough to run
  both. Divergence is a bug in the bounds, not a tolerable approximation.
- **Structural recovery**: Robinson-Foulds against the simulated ground truth,
  and the linear relation between tree path distance and true squared Euclidean
  distance (Fig. S8).

## 14. Layout (SI.G)

Deferred; not needed before the tree exists.

Three layouts. The **ladderised dendrogram** sorts branches at each node by the
number of leaves below them; only horizontal distances carry meaning. The
**equal-angle** and **equal-daylight** circular layouts are Felsenstein,
*Inferring Phylogenies*, pp. 578-584, which the SI defers to; equal-angle is
`O(n)` and never crosses edges, equal-daylight is iterative and is worth gating
behind a node-count threshold.

The **hyperbolic disk** projection is specified outright: translate by an origin
and scale by a zoom, convert to polar, leave the angle alone and map the radius

```
r -> r / (1 + sqrt(1 + r^2))
```

which sends 0 to 0 and infinity to 1.

Distances are always sums of branch lengths along the tree path, never 2D
distances in the picture.

## 15. Backbone mode (Methods, not the SI)

For atlas scale. Four steps: preprocess everything; run the standard algorithm on
a random subset to get a backbone; place the remaining cells one at a time with
the beam search of section 7.2; then use the result as the initial tree for a
final standard run, which in practice is mostly SPR, NNI and branch-length
optimisation because few polytomies survive.

Reoptimise branch lengths and recompute the beam search start points after the
backbone has grown by a set fraction. Their default backbone is 10k cells; ours
is ours. Growing in more than one round costs time but can find a better optimum.

## 16. Open questions

- Whether the softmax in section 9.4 is over `L` or `2*L` in their
  implementation. It changes the temperature of the random NNI phase. We pick `L`
  and document it; if random NNI underperforms, this is the first thing to check.
- The clustering used for beam-search start points (section 7.2) and for root
  selection is described in the Methods as iterative branch cutting minimising
  the summed pairwise leaf distances. Cheap enough, but the cost of recomputing
  it during backbone growth needs measuring.
- `1/W` appears throughout. Nodes with a very small effective precision want a
  guard. Decide whether to clamp or to carry variance rather than precision in
  the hot path.
