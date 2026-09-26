# Scaling to a million cells

A proposal, 2026-09-26. Nothing here has been built or run. Every number is
either measured elsewhere in this repository (and cited), taken from the paper,
or arithmetic flagged as such. The effect of every proposed change is
**unmeasured**; section 8 lists the experiments that would settle each one.

## The short version

The backbone is not the only route, and for this crate it's the wrong one to
lean on. It speeds up the *start tree*, which is 3 per cent of a run here. It
then hands the full-size tree to the standard refinement, and the refinement is
where both walls sit at a million cells: SPR bookkeeping that is `O(n)` per
proposal, so `O(n^2)` per sweep, and a node state of about 43 GB.

The proposal is **partitioned refinement with exact boundary leaves**. Cut the
tree into pieces of at most `B` cells (a few thousand). Refine every piece with
the existing pipeline, `refine()`, where each edge leaving the piece is
replaced by one pseudo-leaf: the rest of the tree on that side, collapsed by the
pruning recursion. Under Brownian motion with Gaussian noise that collapse is
**exact** (SPEC 4, S19), so a move scored inside a piece is scored against the
true full-tree likelihood. Then re-cut along shifted boundaries and repeat until
few cells move.

This is uDance (Balaban et al. 2024) and the DCM iteration (Huson, Nettles,
Warnow 1999) with one decisive difference. DNA methods have to approximate the
outside with outgroups and stitch the pieces back with supertree methods. Here
the outside is a sufficient statistic of size `2p`, and the stitch is the
identity.

What it buys, if it works:

- Every subproblem stays inside the size range every constant in this crate was
  measured at (512 to 10,000 cells). Rule 10 of [performance](PERFORMANCE.md)
  says constants break when sizes move; this keeps them where they were tuned.
- Work linear in `n` for a fixed `B`: `n/B` pieces at `O(B^1.6)` each.
- Memory of one piece per worker, not one tree.
- Coarse-grained parallelism: one piece per thread, no SPR chunk restarts.
- No new kernel. The pseudo-leaf is a row of `PreparedData`.

## 1. Where a million cells breaks the current pipeline

### SPR is quadratic at scale, and it's bookkeeping

Each SPR proposal (`propose`, `src/search/spr.rs:1458`) runs, before and after
any scoring:

- `prune_subtree` copies the parent and branch arrays and calls `assemble`,
  a full relabel of the arena;
- `regraft` assembles the attached tree again;
- `leaf_words` hashes a leaf set for every node;
- `split_fingerprint_with` walks the tree again;
- the `to_old` map is another pass over every node.

That is about five `O(n)` passes over the arena per proposal. A sweep proposes
every prunable node, about `2n` of them. The scoring side is flat: the beam
reads 41 to 49 nodes whatever the tree size ([performance](PERFORMANCE.md#diagnoses-that-were-wrong)),
so a proposal's `O(beam * p)` arithmetic does not grow with `n`.

Arithmetic, not measured: at 10,000 cells the arena passes are roughly
`5 * 2e4 = 1e5` touches against `45 * 2,767 * ~5 Newton passes = 6e5` flops of
scoring, so scoring still dominates. At 100,000 they're level. At a million the
bookkeeping is 15 times the scoring and the sweep is `2e6 * 1e7 = 2e13` memory
touches. That's the `n^1.6` slope turning into `n^2`.

This is our implementation, not the method. The paper's final refinement fits
`C^1.07` (SI Fig. S18B). It's fixable in place with an incremental arena
(section 7), and should be fixed whatever else happens.

### NNI settles globally per accepted move

The lazy NNI still pays a per-round settle, about 0.28 s at 10,000 cells, and
that's `O(n p)` ([performance](PERFORMANCE.md#lazy-nni)). Accepted moves grow
with `n`, so the step has the same shape: linear work per move, a linear number
of moves.

### Memory

Arithmetic, 1M cells by 2,700 features, `f32`:

| what | size |
|---|---|
| input means and SDs | 21.6 GB |
| `NodeState`, means and precisions, 2M nodes | 43.2 GB |
| `UpState` for the global branch solve | another 43.2 GB |

On a 64 GB machine the global branch solve (steps 4 and 7) can't allocate. The
paper's own peak-memory fit for the main process is `0.0003 * C^1.00` GB (SI
Fig. S19), about 300 GB at 1M in their Python representation. Their atlas run
took a week on 10 CPUs.

### The start tree is probably fine, but untimed at scale

Linkage went 1.7 s to 5.9 s from 5,000 to 10,000 cells, `n^1.8` on two points.
Extrapolated to 1M that's hours, which is a reason to measure, not a verdict.
`mutual_pairs` recomputes every live cluster's nearest neighbour every round;
RAC (Sumengen et al. 2021) recomputes only clusters adjacent to a merge. That is
the obvious first fix if the linkage turns out to be the slow part.

## 2. What the backbone does and doesn't solve

From the Methods (paper summary, not the published source): build on a random
10,000-cell subset, place every remaining cell with the beam search of SPEC
7.2, optionally in growth steps (10k to 40k to 250k), then run the standard
steps 3 to 7 on the full tree.

It replaces the start tree. Here the start is a Ward linkage over an NN-descent
graph that already costs 3 per cent at 10,000 cells, and on real data Ward beat
the paper's greedy start on loglikelihood and Robinson-Foulds at every size
measured ([performance](PERFORMANCE.md#starting-tree)). So the backbone
replaces the part of our pipeline that already works.

The last step of the backbone is the full-size refinement. That's the part with
the quadratic SPR and the 43 GB state. The backbone doesn't touch it.

It also places cells onto a tree built from 1 per cent of the data. A rare
population absent from the subset has nowhere good to land, and the placement
beam starts from `log(n)` spread-out points on a tree that doesn't represent
it. uDance picks diverse representatives for exactly this reason. That's a
quality risk we'd have to measure, on top of the speed.

**The two compose.** If the Ward start turns out slow or poor at 1M, the backbone
is a legitimate way to *produce the start tree* for the scheme below. They're not
competitors; they answer different questions.

Measured since this was written: an implementation of it lost to the full
search on every dataset and seed up to 25k cells, and was removed in 0.2.0; SPEC
section 15 has the numbers.

## 3. The property everything rests on

A subtree collapses to one effective leaf `(M, W)` per feature, exactly (SPEC
4, S19). So does the complement of a subtree, via the up sweep that
`collapse_onto_every_node` already does. Neither is an approximation.

Take a piece `P` of the tree: a connected set of nodes. Every edge leaving `P`
has a far side that is a subtree (when rooted at the edge). Replace each far
side by its effective leaf, hung on the boundary edge with that edge's length.
Call these pseudo-leaves. The result is a tree of `|P| + (number of boundary
edges)` leaves.

**Claim.** For any change to the topology or branch lengths inside `P`,
including moving a pseudo-leaf, the change in that small tree's loglikelihood
equals the change in the full tree's.

Sketch: root the full tree inside `P`. Every node outside `P` then has only
outside descendants, so its S20 term depends only on the outside and doesn't
move. Each boundary edge feeds `P` exactly the pseudo-leaf's `(M, W)`, diffused
along the edge (S19). The remaining terms are the terms of the small tree. Root
independence (S14) covers any rooting. The test for this is cheap and should
come first: score an SPR move inside a piece both ways and assert agreement to
`1e-9` relative.

Consequences:

- A pseudo-leaf is a leaf. `PreparedData` with `transformed_means` and
  `transformed_precisions` for the piece's cells plus one row per boundary,
  handed to `refine()`, runs steps 3 to 8 on the piece unchanged. Skip `prepare`:
  features and scaling come from the parent run.
- Moving a pseudo-leaf is a real move: it re-attaches the rest of the world, or a
  cut-off clade, somewhere else inside `P`. Nothing needs to forbid it.
- The piece is solved by the method *as specified*, with the whole dataset's
  evidence. No outgroup choice, no supertree merge, no ASTRAL.

A pseudo-leaf's precision is the sum of up to a million diffused precisions. It's
bounded by the diffusion `1 / (t + 1/W)`, so a branch of any positive length caps
it. `f32` storage should hold it, but that's one of the open items in SPEC 16
("`1/W` wants a guard") that this scheme makes real.

## 4. The algorithm

1. **Ingest** once, as now.
2. **Start tree.** Ward linkage over the whole set, as now. The backbone is the
   fallback if that's too slow at 1M (section 2).
3. **Cut.** A post-order walk accumulating leaf counts. Cut the edge above a node
   when its uncut count reaches `B`. Pieces are connected, have one boundary
   upwards and a handful downwards, and hold at most `B` cells.
4. **Summarise.** One down sweep and one up sweep over the tree to get each
   boundary's pseudo-leaf. `O(n p)` time. Memory can stay at one row per
   boundary plus a piece's worth of rows: the sweeps go in post-order, and a row
   can be dropped once its parent has consumed it.
5. **Refine** every piece in parallel with `refine()`. One piece per thread,
   `RAYON_NUM_THREADS=1` inside, so there are no nested pools and no SPR chunk
   restarts.
6. **Write back.** Splice each refined piece into the arena by boundary; pseudo-
   leaves map back to the edges they stand for. One `Tree::from_parents`.
7. **Verify.** Recompute the full loglikelihood (piece by piece, `O(n p)`). See
   section 5 for why this can't be assumed.
8. **Shift and repeat.** Re-cut with a different offset (for instance cut at `B/2`
   first, or start the walk from a different root) so that old boundaries fall
   inside new pieces. Stop when the fraction of cells whose parent split changed
   in a pass falls below `delta`. That's the NN-descent stopping rule, applied to
   partitions.
9. **Finish.** The last pass leaves every edge optimised inside some piece. No
   global branch solve is needed, which is what keeps `UpState` off the heap.

### Cost, extrapolated

Steps 3 to 8 of the current pipeline took 68.7 s at 5,000 cells and 204 s at
10,000 on ten cores ([performance](PERFORMANCE.md#now)). Two points, one gene
panel.

| `B` | pieces at 1M | one pass, pieces at those times |
|---|---|---|
| 5,000 | 200 | about 3.8 h |
| 10,000 | 100 | about 5.7 h |

Per pass, and with the per-piece times measured with ten threads per piece.
Running ten pieces single-threaded side by side could do better, because a third
of SPR's thread time is idle ([performance](PERFORMANCE.md#now)), but that's
unmeasured. Smaller `B` is cheaper per pass (`n * B^0.6`) and leaves more
boundary, so it needs more passes. The optimum is an experiment.

Two or three passes would put a 1M-cell run at hours rather than the paper's
week. **That's a prediction from extrapolated per-piece times, not a result.**

## 5. Where it can go wrong

### Parallel pieces see stale boundaries

Pieces refined together each see the others as they were. Write the full
loglikelihood as `f_A(A) + f_B(B) + g(s_A, s_B)`, where `s_X` is piece `X`'s
summary as seen from the rest. Refining `A` maximises `f_A + g(., s_B)` with
`s_B` frozen, and likewise for `B`. The combined gain is the sum of the two
predicted gains plus

```
g(s_A', s_B') - g(s_A', s_B) - g(s_A, s_B') + g(s_A, s_B)
```

a mixed second difference. It's zero when the pieces don't interact, and small
when they're far apart in the tree, because each piece enters the other's
boundary through a precision-weighted average over everything between them. Not
zero, though, and not bounded by anything I can prove.

So step 7 is not optional. If the full loglikelihood fell, fall back to a
two-colour schedule: refine alternate pieces, re-summarise, refine the rest.
That's Gauss-Seidel instead of Jacobi, monotone by construction, at twice the
wall time. The size of the coupling term is the first thing to measure.

### Mistakes that straddle a boundary

A cell Ward put in the wrong piece can only move within that piece in one pass.
The shifted cut is the standard answer (DCM3, uDance's iterations). Two cheaper
aids, both unmeasured:

- **kNN discordance.** A cell whose feature-space neighbours mostly sit in other
  pieces is probably misplaced. Count it for every cell from the graph we already
  build; put the high-discordance cells on the boundary side of the next cut, or
  give their neighbours' piece a copy of them as candidates.
- **Deterministic cut offsets.** Cut positions depend on the tree and a pass
  counter only, so the determinism rule holds at any thread count.

### Moves longer than a piece

3 per cent of accepted SPR moves travel far and carry real gain (the failed
"start the beam at the origin only" attempt, [performance](PERFORMANCE.md#what-did-not-work)).
A move from one end of the tree to the other can't happen inside any piece. The
shifted passes only reach it if an intermediate position also improves. How
much this costs at 10,000 cells, where the monolithic run gives the answer, is
experiment E3.

## 6. Why not the other routes

| route | what it does | why not first |
|---|---|---|
| Backbone (the paper) | subset, place, full refinement | refinement unchanged; start already cheap; rare cells |
| Prefix-doubling placement (HNSW, ParlayANN) | place batches of doubling size against a frozen tree, repair locally | a better backbone; same full-size refinement at the end |
| matOptimize-style parallel SPR (Ye et al. 2022) | score all moves against one state, commit a non-conflicting batch | still needs the whole tree's state in memory; plan B if section 5's coupling term is bad |
| Feature split (ExaML) | shard features across machines, all-reduce scalars | exact, fixes memory, not work; every Newton step becomes a latency-bound reduction |
| Likelihood agglomeration with a lazy heap (Bateni et al. 2024, FastTree top-hits) | replace Ward by a scalable merge-gain start | Ward already wins on real data, and the gain chains |
| Threshold rounds (SCC, RAC, TeraHAC) | parallel agglomeration by rounds | guarantees need a reducible linkage; the merge gain isn't; only helps the start |
| Gradient or hyperbolic relaxations (HypHC, GradME, Dodonaphy) | continuous surrogate, decode to a tree | shown at `10^1` to `10^4` leaves, optimise a surrogate, not our likelihood |
| Feature subsampling | score on `p' << p` | measured and failed, argmax survives 4 of 15 checkpoints at `p' = 512` |

The matOptimize route deserves one more line. It's the right way to parallelise
SPR *within* a piece if pieces ever get big, and "score against a frozen state,
commit a deterministic non-conflicting batch" is also a cleaner fix for SPR's
idle thread time than the chunk restart. It's a candidate for section 7, not an
alternative to the decomposition.

## 7. The graph-index intuition, honestly

Trees are sparse graphs, and several ANN tricks carry over. Some are already here
under other names.

- **NN-descent's new/old flags.** Only re-examine pairs with a new end. That's
  the SPR revisit radius and the lazy NNI: TSP's don't-look bits, which is where
  both came from. In the scheme above it becomes "only re-cut around pieces that
  changed".
- **NN-descent's convergence counter.** Stop when fewer than `delta * n` updates
  happen. Fits the outer loop of section 4 directly. It doesn't fit SPR's inner
  loop: SPR already stops on no improving move, and the revisit set already does
  the counting.
- **"A neighbour of a neighbour is a neighbour".** Seeding the placement beam at
  the tree positions of the pruned subtree's feature-space nearest neighbours,
  instead of at `start_points`, which spreads eight starts over the node index
  and is marked as a placeholder in `src/model/place.rs`. This is HNSW's upper
  layers replaced by the kNN graph, and SCAMPP's restriction (Wedell et al. 2022).
  The beam is flat already, so it wouldn't change the scaling; it might remove the
  ladder cliff that needs eight starts.
- **HNSW hierarchical insertion.** That's the backbone: a coarse structure,
  then greedy descent. Section 2.
- **Vamana's robust prune on merge**, `prune(N(u) + N(v))` with union-find
  redirects (Bateni et al. 2024). The right way to keep a kNN graph over live
  clusters without rebuilds. The linkage's symmetric union does the same job;
  robust prune would stop the lists thinning. Start-tree only.

What doesn't carry: anything whose correctness rests on the metric being
reducible. The merge gain isn't (up to 13 nats of inversion at 128 members,
[performance](PERFORMANCE.md#reducibility)), so RAC, ParHAC and TeraHAC's
guarantees go. They still work as heuristics for Ward, which is reducible, and
that's where Ward is used.

## 8. Experiments, in order

Each one either kills the proposal or earns the next. Nothing past E1 should
start until the machine is idle, per rule 12.

1. **E0, confirm the diagnosis.** Profile SPR at 10,000 and 40,000 synthetic
   cells (the simulator already exists). Split each proposal's time into arena
   passes and scoring. Expected: scoring dominates at 10k, bookkeeping at 40k. If
   bookkeeping doesn't grow, section 1 is wrong and the case for decomposition
   rests on memory alone.
2. **E1, the exactness gate.** On a small fixture, cut a piece, build its
   `PreparedData` with pseudo-leaves, score a move inside it, and compare with the
   full-tree delta. `1e-9` relative, like every other gate in SPEC 13.2. If this
   fails, stop.
3. **E2, one pass at 10,000 cells.** `B` in {1,000, 2,500, 5,000}, real Baron
   data, one pass without shifting. Compare loglikelihood, Robinson-Foulds,
   recovery and wall time to the monolithic run. The monolithic spread is about
   1,700 nats at this size ([performance](PERFORMANCE.md#how-much-one-real-data-run-says)),
   so anything inside that is a tie, not a loss. Also log the section 5 coupling
   term: predicted gains summed against the realised full gain.
4. **E3, shifted passes.** Same sizes, two and three passes. How many passes to
   match the monolithic tree, and how many far-travelling moves are lost.
5. **E4, scale.** 50,000 and 100,000 cells, synthetic and a real atlas subset.
   Exponent of the whole run in `n`, and peak memory. Here the monolithic run
   stops being a reference and the loglikelihood is all there is.

Separately, and worth doing regardless: an incremental arena for SPR, so a
proposal touches the nodes it changes and not every node. Link-cut or Euler-tour
trees are the textbook answer; a lighter one is to stop relabelling on proposal
and relabel once on acceptance. It helps every route above, the backbone
included.

## References

Verified by search during this review unless marked (mem), which means from
memory and not rechecked.

- de Groot, Morillo Leonardo, Pachkov, van Nimwegen (2026). Bonsai. *Nat
  Biotechnol*. doi:10.1038/s41587-026-03220-2. Backbone: Methods; scaling fits:
  SI Figs. S16 to S19.
- Balaban et al. (2024). uDance. *Nat Biotechnol*. doi:10.1038/s41587-023-01868-8
- Huson, Nettles, Warnow (1999). Disk-covering methods. *J Comput Biol* (mem);
  Roshan et al. (2004). Rec-I-DCM3. *CSB* (mem)
- Wedell, Cai, Warnow (2022, 2023). SCAMPP and BSCAMPP. *IEEE/ACM TCBB*; WABI.
- Turakhia et al. (2021). UShER. *Nat Genet* 53:809.
- Ye et al. (2022). matOptimize. *Bioinformatics* 38:3734; Thornlow et al.
  (2023). Online phylogenetics. *Syst Biol* 72:1039.
- De Maio et al. (2023). MAPLE. *Nat Genet* 55:746; Ly-Trong et al. (2024).
  CMAPLE. *MBE* 41:msae134.
- Price, Dehal, Arkin (2009, 2010). FastTree and FastTree 2. *MBE*; *PLoS ONE* (mem
  for FastTree 2 details)
- Piñeiro, Pichel et al. (2024). VeryFastTree 4. *GigaScience* 13:giae055.
- Kozlov, Aberer, Stamatakis (2015). ExaML. *Bioinformatics* (mem)
- Kurt, Bouchard-Côté, Lagergren (2024). Sparse neighbour joining.
  *Bioinformatics* 40:btae701.
- Ho, Ané (2014). Linear-time Gaussian trait likelihood. *Syst Biol* 63:397;
  Bastide et al. (2021). *Ann Appl Stat* 15:971.
- Zhang, Zhang, Gao, Wu (2025). ScisTree2. *Genome Res* 35:2781.
- Sumengen et al. (2021). RAC. arXiv:2105.11653.
- Dhulipala et al. (2021, 2022, 2024). Graph HAC, ParHAC, TeraHAC. ICML;
  NeurIPS; SIGMOD.
- Bateni et al. (2024). Efficient centroid-linkage clustering. NeurIPS.
  arXiv:2406.05066.
- Monath et al. (2019, 2021). Grinch; SCC. KDD.
- Kobren et al. (2017). PERCH. KDD.
- Dong, Charikar, Li (2011). NN-descent. WWW (mem)
- Malkov, Yashunin (2020). HNSW. *IEEE TPAMI*.
- Chami et al. (2020). HypHC. NeurIPS (mem); Penn et al. (2023). GradME. *GBE*;
  Macaulay, Fourment (2024). Dodonaphy. *Bioinf Adv*.
