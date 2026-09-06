# Provenance

`bonsai-rs` is a clean-room implementation of the Bonsai algorithm. This file
records what was read, by whom, and under which licence, so the basis for the
MIT licence on this crate is auditable.

## Sources used

| source | licence | used for |
|---|---|---|
| de Groot, Morillo Leonardo, Pachkov, van Nimwegen. *Bonsai reconstructs tree representations for distortion-free visualization and exploration of high-dimensional data.* Nature Biotechnology (2026). doi 10.1038/s41587-026-03220-2 | CC-BY-4.0 (open access) | model, search outline, backbone mode, root selection |
| Supplementary Information to the above, sections SI.B, SI.C, SI.D, SI.E, SI.G | CC-BY-4.0 | every equation in `docs/SPEC.md` |
| Felsenstein, *Inferring Phylogenies*, pp. 578-584 | textbook | equal-angle and equal-daylight layouts, which the SI defers to |

## Sources deliberately not used

`dhdegroot/Bonsai-data-representation` (Zenodo doi 10.5281/zenodo.20370956) is
licensed **CC-BY-NC-4.0**, repo-wide, with no software-specific terms. Section
1(a) of that licence defines Adapted Material as material "translated, altered,
arranged, transformed, or otherwise modified in a manner requiring permission".
A Python-to-Rust port is a translation and would inherit NonCommercial, which is
incompatible with the MIT licence on this crate.

Copyright protects expression, not the algorithm. 17 USC 102(b) and EU Software
Directive Art. 1(2) both exclude procedures, methods of operation and
mathematical concepts. The mathematics is in the CC-BY-4.0 Supplementary
Information and is free to implement. Most of it is not novel to them in any
case: continuous-trait pruning under Brownian motion is Felsenstein, and Gaussian
message passing on a tree is textbook.

**Their source is therefore off limits.** No session working on this repository
opens, clones, reads or greps that repository, its Zenodo drop, or any mirror.
Not directly, not in a sibling worktree, not through a subagent.

## Disclosure

A proposal session in August 2026, predating this repository, did read the
reference source while assessing feasibility. That proposal
(`bixverse-project/proposals/bonsai-rs.md`) is not used as an implementation
source. What it carried across was either mathematics that appears in SI.B
regardless of where it was read, or criticism of their engineering, which is not
the kind of thing anyone copies. Every design decision in this crate is taken
from `docs/SPEC.md`.

## The comparison harness, 2026-09-06

A black-box comparison against the reference implementation was run. The
arrangement is the two-team clean room, and it is recorded here because the
whole licence position rests on it.

A separate agent was given read access to a local clone of the reference, **for
interface only**: how to invoke it, what input files it expects, what output it
emits. It was forbidden from reading the source to learn how the method works,
forbidden from writing any file inside this repository, and required to report
results only. Explicitly excluded from its report: algorithm descriptions, code
structure, numerical methods, and any performance explanation derived from
reading source.

No session that writes this crate's code has read that clone. The harness lives
outside this repository, at `~/repos/others/bonsai-comparison`, and is not part
of the crate.

The comparison uses this crate's own simulated data (SPEC.md section 13.1, the
paper's SI.E.2 recipe), not the reference's example datasets, so that both
implementations are measured against a ground truth neither produced and no
NonCommercial data file is involved.

Running the reference to observe its output is black-box observation, protected
under EU Software Directive Art. 5(3), and does not create Adapted Material. It
was run on the author's private machine, so the NonCommercial clause is not
engaged.

## Rules this crate follows

1. Implementation reads `docs/SPEC.md`, not the PDFs and not their code. Every
   kernel cites the SI equation number it implements.
2. No mirroring of their module structure, function decomposition or naming.
3. No carrying over of their tuned constants. Every threshold is a named `const`
   with a doc comment saying where the number came from: an SI equation, or our
   own tuning with the date it was measured. Explicitly ours to determine, not
   theirs to donate: the neighbour count `k`, the kNN rebuild cadence, the
   placement-search tolerance, the ellipsoid `n_steps` schedule, the default
   backbone size.
4. Output-level parity testing against a reference run is permitted. Comparing
   outputs is black-box observation, protected under EU Software Directive
   Art. 5(3), and does not create Adapted Material. Reference trees are generated
   on demand, never committed, and never generated on commercial hardware: the
   live constraint there is the NonCommercial clause, not copyright.
5. The paper is cited in the crate documentation and in the README.

None of this is legal advice.
