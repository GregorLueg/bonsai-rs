# Provenance

What was read, by whom, under which licence. This is the basis for the MIT
licence on this crate.

## Sources used

| source | licence | used for |
|---|---|---|
| de Groot, Morillo Leonardo, Pachkov, van Nimwegen. *Bonsai reconstructs tree representations for distortion-free visualization and exploration of high-dimensional data.* Nature Biotechnology (2026). doi 10.1038/s41587-026-03220-2 | CC-BY-4.0 (open access) | model, search outline, backbone mode, root selection |
| Supplementary Information to the above, sections SI.B, SI.C, SI.D, SI.E, SI.G | CC-BY-4.0 | every equation in `docs/SPEC.md` |
| Felsenstein, *Inferring Phylogenies*, pp. 578-584 | textbook | equal-angle and equal-daylight layouts, which the SI defers to |

## Sources not used

`dhdegroot/Bonsai-data-representation` (Zenodo doi 10.5281/zenodo.20370956) is
**CC-BY-NC-4.0**, repo-wide. Section 1(a) defines Adapted Material as material
"translated, altered, arranged, transformed, or otherwise modified in a manner
requiring permission". A Python-to-Rust port is a translation, inherits
NonCommercial, and can't be MIT.

Copyright protects expression, not algorithms: 17 USC 102(b) and EU Software
Directive Art. 1(2) both exclude methods and mathematical concepts. The maths is
in the CC-BY-4.0 SI. Most of it isn't theirs anyway: continuous-trait pruning
under Brownian motion is Felsenstein, Gaussian message passing on a tree is
textbook.

**Their source is off limits.** No session on this repository opens, clones,
reads or greps it, its Zenodo drop or any mirror. Not directly, not in a sibling
worktree, not through a subagent.

## Disclosure

A proposal session in August 2026, before this repository existed, read the
reference source while assessing feasibility. That proposal
(`bixverse-project/proposals/bonsai-rs.md`) is not an implementation source.
What it carried across was SI.B maths or criticism of their engineering.
Every design decision here comes from `docs/SPEC.md`.

## Comparison harness, 2026-09-06

A black-box comparison was run as a two-team clean room.

A separate agent got read access to a local clone of the reference **for
interface only**: how to invoke it, what input it expects, what it emits. It
could not read the source to learn the method, could not write inside this
repository, and reported results only. Excluded from its report: algorithm
descriptions, code structure, numerical methods, and performance explanations
drawn from the source.

No session that writes this crate's code has read that clone. The harness lives
at `~/repos/others/bonsai-comparison`, outside the crate.

The first comparison used this crate's simulated data (SPEC 13.1, the SI.E.2
recipe). Later runs used Baron pancreas (GSE84133, public GEO),
Sanity-preprocessed once and handed identically to both. None of the
reference's example datasets are used, so no NonCommercial data is involved and
both are scored against a ground truth neither produced.

Running the reference to observe its output is black-box observation under EU
Software Directive Art. 5(3) and creates no Adapted Material. It ran on the
author's private machine, so NonCommercial isn't engaged.

## What ships, 2026-09-13

`docs/COMPARISON.md` and `docs/figures/` carry the harness's tables and figures.

**Measurements are facts.** Wall time, peak RSS, Robinson-Foulds, distance
recovery. Facts about a program's behaviour aren't copyrightable.

**The figures are ours.** Our code draws them from Newick and CSV. Program
output isn't a derivative work unless the program injects its own expression,
and a Newick tree computed from the user's input doesn't; same reason a
compiler's licence doesn't reach the binary. The live clause is NonCommercial,
and the runs were private.

Not carried across: the scripts that invoke their CLI and shape its input. Both
encode their configuration semantics, and no session writing crate code should
have that in front of it.

## Harness published, 2026-09-25

`reference/comparison/` is the harness minus everything that touches the
reference. A Claude Code subagent, with no access to the clone (a sibling
directory, never entered), audited the harness and staged the copy. It removed:
the scripts that invoke the reference, a data export in the reference's input
layout, a compatibility step for the reference's file naming, a parser for the
reference's log format, a scorer loop over the reference's intermediate outputs,
and comments naming its CLI flags or defaults. The session writing this crate
then reviewed the staged copy, which by then contained none of that. The data
generator follows SI section E and Sanity's documented interface; the author
states the harness contains no reference code, and the audit found no imports,
paths or copied content from the clone.

## Stating differences

This never changes with time. The line is where a claim's evidence comes from.

**Allowed:** anything measured from inputs and outputs, and any difference
against the paper or SI. "Thirty times faster at 5,000 cells." "Their tree has
335 polytomies against our 31." "The SI specifies X and we do Y."

**Never allowed:** a claim whose evidence is their source. Descriptions of their
code or decomposition, or explanations of *why* their implementation behaves as
it does beyond what the paper says. Writing one makes this crate Adapted
Material.

## Rules

1. Implementation reads `docs/SPEC.md`, not the PDFs, not their code. Every
   kernel cites its SI equation.
2. No mirroring of their module structure, decomposition or naming.
3. No carried-over tuned constants. Every threshold is a named `const` whose doc
   comment gives its source: an SI equation, or our measurement in
   `docs/PERFORMANCE.md`. Ours to determine: neighbour count `k`, kNN rebuild
   cadence, placement-search tolerance, ellipsoid `n_steps` schedule, default
   backbone size.
4. Output-level parity testing is fine (black-box, Art. 5(3)). Reference trees
   are generated on demand, never committed, never on commercial hardware; the
   constraint there is NonCommercial, not copyright.
5. The paper is cited in the crate docs and the README.

None of this is legal advice.
