"""Distance recovery globally, and restricted to the cells each implementation
declines to resolve. Same convention as figures.py: Pearson r between true
squared Euclidean distance and tree path distance."""
import sys
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import treeviz as tv

rng = np.random.default_rng(20260914)
d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work_real" / "n10000"

coords = pl.read_csv(d / "ours" / "truth.csv", has_header=False).to_numpy().astype(np.float64)
print(f"coords {coords.shape}")

trees, meta = {}, {}
for name, f in [("ours", "ours.nwk"), ("theirs", "theirs.nwk")]:
    t = tv.parse_newick((d / f).read_text())
    depth, cum = tv.depth_and_cumlen(t)
    up = tv.build_lca_tables(t, depth)
    trees[name] = t
    meta[name] = (depth, cum, up)

# label -> cell index, and label -> node id per tree
def leaf_nodes(t):
    return {t.label[i]: i for i in range(t.n_nodes) if t.is_leaf[i]}

nodes = {k: leaf_nodes(v) for k, v in trees.items()}
labels = sorted(nodes["ours"], key=lambda s: int(s.replace("cell", "")))
idx_of = {lab: i for i, lab in enumerate(labels)}

def recovery(name, subset_labels, n_pairs=200000):
    t = trees[name]
    depth, cum, up = meta[name]
    # Sorted, not set order: Python randomises string hashing per process, so
    # iterating the set directly makes the sampled pairs differ between runs.
    subs = sorted((l for l in subset_labels if l in nodes[name]),
                  key=lambda s: int(s.replace("cell", "")))
    m = len(subs)
    if m < 3:
        return float("nan"), 0
    if m * (m - 1) // 2 <= n_pairs:
        # Small enough to take every pair, which removes the sampling error.
        a, b = np.triu_indices(m, k=1)
    else:
        a = rng.integers(0, m, n_pairs)
        b = rng.integers(0, m, n_pairs)
        keep = a != b
        a, b = a[keep], b[keep]
    la = [subs[i] for i in a]
    lb = [subs[i] for i in b]
    u = np.array([nodes[name][x] for x in la])
    v = np.array([nodes[name][x] for x in lb])
    pd = tv.path_distances(up, depth, cum, u, v)
    ca = coords[[idx_of[x] for x in la]]
    cb = coords[[idx_of[x] for x in lb]]
    sq = ((ca - cb) ** 2).sum(axis=1)
    return float(np.corrcoef(sq, pd)[0, 1]), m

# The set ours strings into a ladder: top 2.5% by radius.
t = trees["ours"]; depth, cum, _ = meta["ours"]
lv = np.array([i for i in range(t.n_nodes) if t.is_leaf[i]])
rad = cum[lv]
arm = {t.label[i] for i in lv[rad >= np.quantile(rad, 0.975)]}

# The set theirs fans into its largest multifurcation.
tt = trees["theirs"]
order = tv.post_order(tt)
counts = tv.subtree_leaf_counts(tt, order)
poly = tv.polytomy_nodes(tt)
biggest = max(poly, key=lambda n: counts[n])
fan = tv._flood_leaves(tt, [biggest])

allc = set(labels)
print(f"\n{'set':28} {'n':>6} {'ours r':>9} {'theirs r':>9}")
for nm, s in [("all cells", allc), ("our ladder (arm)", arm), ("their largest fan", fan)]:
    ro, m = recovery("ours", s)
    rt, _ = recovery("theirs", s)
    print(f"{nm:28} {m:>6} {ro:>9.3f} {rt:>9.3f}")
