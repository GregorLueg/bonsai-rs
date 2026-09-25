"""Before/after test of the SPR acceptance-floor fix (relative floor:
max(star.min_gain, 1e-12 * max(|L|, 1)), replacing a fixed constant).

Compares, per config:
  - post-SPR: ours_stages/5_spr.nwk (baseline) vs ours_stages_relfloor/5_spr.nwk (fixed)
  - final:    ours.nwk (baseline)              vs ours_relfloor.nwk (fixed)

This is the acceptance test for the hypothesis that SPR churn on
likelihood-neutral regrafts created the degenerate region found by
degeneracy.py. If the fix collapses both the post-SPR and final counts, the
mechanism is confirmed. If the final-tree polytomy count survives largely
intact even though the post-SPR count falls, resolution (SPEC 9.2 step 3)
is not finishing and that is a separate, unaddressed defect.

Usage:
    .venv/bin/python before_after.py n10000 [n5000 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import treeviz as tv
from degeneracy import draw_highlighted_panel
from figures import DPI, N_GROUPS, WORK_DIR, OUT_DIR, clade_colours


def _load(path: Path) -> tv.ParsedTree | None:
    return tv.parse_newick(path.read_text()) if path.exists() else None


def _row(label: str, rep: dict) -> str:
    return (
        f"  {label:34s} zero_len={rep['n_zero_length']:5d} "
        f"[leaf={rep['n_zero_length_leaf']:5d} internal={rep['n_zero_length_internal']:4d}] "
        f"polytomies={rep['n_polytomies']:4d} degenerate_leaves={rep['n_degenerate_leaves']:5d} "
        f"({rep['pct_degenerate_leaves']:5.2f}%)"
    )


def compare_config(config: str) -> None:
    cdir = WORK_DIR / config
    fixed_final_path = cdir / "ours_relfloor.nwk"
    if not fixed_final_path.exists():
        print(f"{config}: {fixed_final_path} not present yet, skipping")
        return

    baseline_spr = _load(cdir / "ours_stages" / "5_spr.nwk")
    baseline_final = _load(cdir / "ours.nwk")
    fixed_spr = _load(cdir / "ours_stages_relfloor" / "5_spr.nwk")
    fixed_final = _load(fixed_final_path)
    assert baseline_final is not None and fixed_final is not None

    print(f"\n=== {config}: SPR acceptance-floor fix (relative floor), before vs after ===")
    reports: dict[str, dict] = {}
    for label, tree in [
        ("baseline post-SPR (ours_stages/5_spr.nwk)", baseline_spr),
        ("baseline final (ours.nwk)", baseline_final),
        ("fixed post-SPR (ours_stages_relfloor/5_spr.nwk)", fixed_spr),
        ("fixed final (ours_relfloor.nwk)", fixed_final),
    ]:
        if tree is None:
            print(f"  {label:34s} -- file not found, skipped")
            continue
        rep = tv.degeneracy_report(tree)
        reports[label] = rep
        print(_row(label, rep))

    # Polytomy survival: what fraction of post-SPR polytomies are still there in the final tree.
    for name, spr_rep, final_rep in [
        ("baseline", reports.get("baseline post-SPR (ours_stages/5_spr.nwk)"), reports.get("baseline final (ours.nwk)")),
        ("fixed", reports.get("fixed post-SPR (ours_stages_relfloor/5_spr.nwk)"), reports.get("fixed final (ours_relfloor.nwk)")),
    ]:
        if spr_rep and final_rep and spr_rep["n_polytomies"]:
            survival = 100.0 * final_rep["n_polytomies"] / spr_rep["n_polytomies"]
            print(f"  {name}: polytomy survival to final tree = {final_rep['n_polytomies']}/{spr_rep['n_polytomies']} ({survival:.0f}%)")

    # Concentration check on the internal-associated subset specifically (the
    # "arm" is a candidate search/resolution defect; structural leaf-zero
    # leaves are SPEC 6 boundary optima scattered by definition of noise, not
    # of interest here).
    truth = _load(cdir / "truth.nwk")
    label_to_group = tv.cut_into_groups(truth, target_groups=N_GROUPS) if truth is not None else {}
    colours = clade_colours(N_GROUPS)

    internal_before = tv.internal_degenerate_leaf_labels(baseline_final)
    internal_after = tv.internal_degenerate_leaf_labels(fixed_final)
    win_before, tot_before = tv.densest_degenerate_window(baseline_final, internal_before)
    win_after, tot_after = tv.densest_degenerate_window(fixed_final, internal_after)
    print(
        f"  concentration (final, internal-associated only): baseline densest 300-window holds "
        f"{win_before}/{tot_before} ({100*win_before/tot_before:.0f}%)"
        if tot_before else "  concentration (final, internal-associated only): baseline has none"
    )
    if tot_after:
        print(
            f"  concentration (final, internal-associated only): fixed densest 300-window holds "
            f"{win_after}/{tot_after} ({100*win_after/tot_after:.0f}%)"
        )
    else:
        print("  concentration (final, internal-associated only): fixed tree has none")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    draw_highlighted_panel(axes[0], baseline_final, label_to_group, colours, "baseline (ours.nwk)")
    draw_highlighted_panel(axes[1], fixed_final, label_to_group, colours, "fixed (ours_relfloor.nwk)")
    fig.suptitle(f"{config}: SPR acceptance-floor fix, degenerate leaves before vs after", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = OUT_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "floor_fix_before_after.png", dpi=DPI)
    plt.close(fig)

    br = reports.get("baseline final (ours.nwk)")
    fr = reports.get("fixed final (ours_relfloor.nwk)")
    # Leaf-zero branches are a SPEC 6 boundary optimum (t=0), not a search
    # defect, and no floor change can remove them -- judge the fix only on
    # internal-zero branches and polytomies, the part a resolve pass owns.
    zi0, zi1 = br["n_zero_length_internal"], fr["n_zero_length_internal"]
    zl0, zl1 = br["n_zero_length_leaf"], fr["n_zero_length_leaf"]
    p0, p1 = br["n_polytomies"], fr["n_polytomies"]
    if p1 == 0 and zi1 == 0:
        verdict = "internal pathology GONE in the final tree: zero internal zero-length branches and zero polytomies"
    elif p1 < p0 * 0.3 and zi1 < zi0 * 0.3:
        verdict = f"internal pathology substantially REDUCED: internal zero-length {zi0}->{zi1}, polytomies {p0}->{p1}"
    elif p1 < p0 and zi1 < zi0:
        verdict = f"internal pathology reduced but not eliminated: internal zero-length {zi0}->{zi1}, polytomies {p0}->{p1}"
    else:
        verdict = f"internal pathology UNCHANGED (or worse): internal zero-length {zi0}->{zi1}, polytomies {p0}->{p1}"
    print(f"  verdict (internal edges + polytomies, the part a resolve pass owns): {verdict}")
    print(f"  leaf-zero edges (SPEC 6 boundary optimum, not fixable by floor changes): {zl0}->{zl1}")


def main() -> None:
    configs = sys.argv[1:] or ["n10000", "n5000"]
    for config in configs:
        compare_config(config)


if __name__ == "__main__":
    main()
