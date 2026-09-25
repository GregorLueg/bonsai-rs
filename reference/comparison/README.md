# Comparison harness

The data generator, scorer and figure scripts behind [comparison](../../docs/COMPARISON.md).
Not part of the published crate. Nothing here runs or knows about the published
implementation: to compare against it, or anything else, drop its tree into a
configuration as `theirs.nwk` and score.

## Build

```bash
cargo build --release      # target/release/harness, against ../..
```

Python scripts need `click numpy polars beartype matplotlib scikit-learn`, e.g.
`uv run --with click,numpy,polars,beartype prep.py ...`.

## Simulated Baron counts (SI section E)

1. Download the Baron et al. human pancreas counts, GEO GSE84133
   (`GSE84133_RAW.tar`), and untar into `data/baron/`.
2. Fit the simulator's inputs: per-gene mean LTQ and per-cell UMI totals.

   ```bash
   python prep.py baron --baron-dir data/baron --out data/baron/params.npz
   ```

3. Simulate: random binary tree, Brownian LTQs, Poisson counts, 17,499 genes.
   `512 s32` in the docs is `--seed 32`.

   ```bash
   python prep.py simulate --params data/baron/params.npz --out work_real/sim_n5000 --n-cells 5000
   ```

   Writes `counts.mtx` (genes by cells), `genes.tsv`, `cells.tsv`,
   `truth_ltq.npy` and `truth.nwk`.

4. Run Sanity (the original binary, `-v_m MAP`, extended output) and keep
   genes at `S >= 1`:

   ```bash
   python prep.py sanity --sanity-bin <sanity> --sim work_real/sim_n5000 --out work_real/sanity_n5000 --threads 10
   python prep.py select --sim work_real/sim_n5000 --sanity-dir work_real/sanity_n5000 --out work_real/n5000
   ```

5. Convert, search and score:

   ```bash
   ./target/release/harness prep  work_real/n5000
   START=linkage ./target/release/harness ours work_real/n5000
   ./target/release/harness score work_real/n5000
   ```

   `ours` runs the eight steps one at a time with default parameters and
   writes `ours.nwk`, `steps.tsv` and stage trees. Without `START=linkage` it
   uses the greedy merge of SPEC 9.1. `score` writes `metrics.tsv`: distance
   recovery, Robinson-Foulds to the truth, loglikelihood, and RF between
   `ours.nwk` and `theirs.nwk` if both exist.

The purely synthetic track skips steps 1 to 4:
`harness gen <dir> <n_leaves> <n_features> <noise_sd> <seed>`, then `ours` and
`score`.

## Figures and tables

| script | makes |
|---|---|
| `figures.py [--set e2e] [n5000 ...]` | radial layouts, recovery scatters, clade fragmentation |
| `faithfulness.py <dir>` | recovery over all cells and over the largest ladder and multifurcation |
| `degeneracy.py [n5000 ...]` | zero-length edge and polytomy counts, degenerate-region plots |
| `before_after.py` | SPR with the fixed against the scale-relative acceptance floor |
| `collapse_arms.py` | the arm with no collapse, collapse after the search, and collapse before step 5 |
| `score_intermediates.sh <dir> [x.nwk ...]` | stage-by-stage scores |
| `nni_knobs.sh <dir>` | the NNI knob attribution |
| `summarise.sh`, `summarise_real.sh`, `rescore.sh` | rebuild the results tables |

Figure scripts read `work_real/<config>/`; `figures.py --set e2e` expects the
counts-to-tree trees under the names in its `TREE_SETS`.

## Not here

- The Rust counts-to-tree route of [comparison](../../docs/COMPARISON.md#counts-to-tree):
  `sanity-sc-rs` on CPU or GPU, then `from_sanity_output` and `bonsai()`. The
  README's "From raw counts" section is that code.
- Anything that invokes the published implementation. See [provenance](../../PROVENANCE.md).
