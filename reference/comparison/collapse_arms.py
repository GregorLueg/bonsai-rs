"""Does the localised "arm" survive collapse the same way regardless of
when/how collapse runs?

Three trees, same data and seed, at work_real/n10000/:
  - ours_relfloor.nwk      : no collapse.                          RF 2659, 129 leaf /  32 internal
  - ours_late_collapse.nwk : resolve_polytomies on the finished     RF 2632, 129 leaf /   0 internal
                              tree, then the branch-length
                              reoptimisation that always follows
                              step 3 (scoring an unsettled tree
                              against settled ones would be unfair).
  - ours_step45.nwk        : collapse before step 5.                RF 2614, 117 leaf /  25 internal

Two counting traps in this comparison, both real, neither a contradiction:

1. Late-collapse has *fewer* internal nodes (19942 vs 19967) but *more*
   polytomies (40 vs 30): collapsing a zero-length internal edge merges two
   nodes into one of higher degree, so removing degenerate structure
   creates polytomies under the degree definition.
2. The flood-filled internal_degenerate_leaf_labels() total grows
   (300 -> 431) purely because the largest polytomy's degree jumps 4 -> 15
   on collapse (several edges merge into one node), so its subtree flood
   sweeps in leaves that are not themselves degenerate (own branch length
   is normal). structural_leaf_zero_labels() is immune to this and is the
   fair cross-arm metric: it is bit-identical, same 129 cells, between
   no-collapse and late-collapse.

Usage:
    .venv/bin/python collapse_arms.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import treeviz as tv
from degeneracy import draw_highlighted_panel
from figures import DPI, N_GROUPS, WORK_DIR, OUT_DIR, clade_colours

CONFIG = "n10000"
ARMS = [
    ("no collapse (ours_relfloor.nwk)", "ours_relfloor.nwk"),
    ("late collapse (resolve_polytomies on finished tree)", "ours_late_collapse.nwk"),
    ("early collapse (ours_step45.nwk)", "ours_step45.nwk"),
]


def main() -> None:
    cdir = WORK_DIR / CONFIG
    truth = tv.parse_newick((cdir / "truth.nwk").read_text())
    label_to_group = tv.cut_into_groups(truth, target_groups=N_GROUPS)
    colours = clade_colours(N_GROUPS)

    present = []
    for label, filename in ARMS:
        path = cdir / filename
        if path.exists():
            present.append((label, filename, tv.parse_newick(path.read_text())))
        else:
            print(f"MISSING: {path} -- not on disk, skipping this arm")

    if not present:
        print("no arms available, nothing to draw")
        return

    print(f"\n=== {CONFIG}: collapse-timing comparison, {len(present)}/3 arms available ===")
    for label, filename, tree in present:
        rep = tv.degeneracy_report(tree)
        internal = tv.internal_degenerate_leaf_labels(tree)
        structural = tv.structural_leaf_zero_labels(tree)
        win_i, tot_i = tv.densest_degenerate_window(tree, internal)
        win_s, tot_s = tv.densest_degenerate_window(tree, structural)
        n_internal_nodes = tree.n_nodes - rep["n_leaves"]
        conc_i = f"{100*win_i/tot_i:.0f}%" if tot_i else "n/a (none)"
        conc_s = f"{100*win_s/tot_s:.0f}%" if tot_s else "n/a (none)"
        print(
            f"  {label:52s} [{filename:24s}] zero_leaf={rep['n_zero_length_leaf']:4d} "
            f"zero_internal={rep['n_zero_length_internal']:3d} polytomies={rep['n_polytomies']:3d} "
            f"internal_nodes={n_internal_nodes:5d}"
        )
        print(
            f"  {'':52s}  densest-300-window: internal-assoc(flood)={conc_i} "
            f"[n={tot_i}]  structural-leaf-zero-only={conc_s} [n={tot_s}]"
        )
    # The flood-filled internal-associated set is sensitive to polytomy degree
    # (collapsing a zero-length internal edge merges two nodes into one of
    # higher degree, so its subtree flood can sweep in leaves that are not
    # themselves degenerate). structural-leaf-zero-only is immune to that and
    # is the fairer cross-arm comparison for "does the arm hold".

    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 5.5), squeeze=False)
    for ax, (label, filename, tree) in zip(axes[0], present):
        draw_highlighted_panel(ax, tree, label_to_group, colours, label)
    fig.suptitle(
        f"{CONFIG}: does the internal-associated arm survive collapse timing?",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = OUT_DIR / CONFIG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "collapse_arms.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"\nwrote {out_path} ({len(present)}/3 arms)")
    if len(present) < len(ARMS):
        print(
            "Re-run once the late-collapse tree exists at "
            f"{cdir / 'ours_late_collapse.nwk'} (or edit ARMS in this file to match "
            "whatever name it is written under) to complete the three-arm panel."
        )


if __name__ == "__main__":
    main()
