"""Realistic data for bonsai-rs, built the way the paper's Methods describe.

Four stages, each a subcommand, each cached on disk so that the expensive
ones (simulation, Sanity) happen once per size:

* ``baron``    derive the simulator's empirical inputs from the Baron et al.
               2016 human pancreas UMI counts (GEO GSE84133): per-gene mean
               log transcription quotients and per-cell total UMI counts.
               SI section E.1 and E.2 take both from this dataset.
* ``simulate`` the SI section E count simulator: log transcription quotients
               diffused along a random binary tree, then Poisson UMI counts.
               Writes a Matrix Market count matrix, the generating tree and
               the noise-free true LTQs.
* ``sanity``   run Sanity over the counts, the real preprocessing step.
* ``select``   keep the genes with signal-to-noise at or above a threshold
               (SI equation S6) and write the matching noise-free truth for
               scoring.

Run with ``uv run --with click,numpy,polars,beartype prep.py <command> ...``.
"""

from __future__ import annotations

import gzip
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import click
import numpy as np
import polars as pl
from beartype import beartype

###########
# Globals #
###########

LOG = logging.getLogger("prep")

# SI section E.2: every simulated dataset has 17 499 genes before selection.
N_GENES_SIMULATED: int = 17_499

# SI section E.2: per-gene LTQ variances are exponential with mean 2, which is
# what Breda et al. measured on the Baron dataset.
GENE_VARIANCE_MEAN: float = 2.0

# SI section E.2.1: every branch of the generating tree has diffusion time 1.
BRANCH_LENGTH: float = 1.0

# Genes per block of the two passes over the LTQ matrix. Sets peak memory at
# roughly (2n - 1) * BLOCK * 8 bytes for the node coordinates, so 800 MB at
# 100k cells. Does not change the output.
GENE_BLOCK: int = 512

# Paper Methods, "Processing of simulated datasets": the default signal-to-noise
# threshold of 1.0. Applied here so that both implementations receive the same
# gene set and neither is asked to select.
DEFAULT_SNR_THRESHOLD: float = 1.0

# Sanity's extended-output files, plus the per-gene vectors the S5 posterior-
# to-likelihood conversion needs. Matrices are genes by cells; vectors one
# value per gene.
SANITY_MATRICES: tuple[str, ...] = (
    "delta.txt",
    "d_delta.txt",
    "log_transcription_quotients.txt",
    "ltq_error_bars.txt",
)
SANITY_VECTORS: tuple[str, ...] = ("mu.txt", "d_mu.txt", "variance.txt")


#########
# Baron #
#########


@beartype
def _read_baron_counts(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read one Baron ``*_umifm_counts.csv.gz``: cells as rows, three leading
    non-count columns (index, barcode, assigned_cluster).

    Returns:
        Per-gene totals, per-cell totals and the gene names.
    """
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        genes = header[3:]
        gene_totals = np.zeros(len(genes), dtype=np.float64)
        cell_totals: list[float] = []
        for line in fh:
            fields = line.rstrip("\n").split(",")
            counts = np.asarray(fields[3:], dtype=np.float64)
            gene_totals += counts
            cell_totals.append(float(counts.sum()))
    return gene_totals, np.asarray(cell_totals), genes


@click.command()
@click.option("--baron-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@beartype
def baron(baron_dir: Path, out: Path) -> None:
    """Pool the four human donors and write ``params.npz``.

    ``mu`` is the per-gene mean LTQ estimated as ``log(N_g / sum_g' N_g')`` over
    genes with at least one count (SI E.2). ``cell_totals`` is every cell's
    total UMI count (SI E.1 samples ``N_c`` from these).
    """
    files = sorted(baron_dir.glob("*human*_umifm_counts.csv.gz"))
    if not files:
        raise click.ClickException(f"no human count files under {baron_dir}")
    gene_totals: np.ndarray | None = None
    cell_totals: list[np.ndarray] = []
    genes: list[str] = []
    for f in files:
        g, c, names = _read_baron_counts(f)
        if gene_totals is None:
            gene_totals, genes = g, names
        else:
            if names != genes:
                raise click.ClickException(f"{f}: gene columns differ from {files[0]}")
            gene_totals += g
        cell_totals.append(c)
        LOG.info("%s: %d cells", f.name, len(c))
    assert gene_totals is not None
    totals = np.concatenate(cell_totals)
    expressed = gene_totals > 0
    mu = np.log(gene_totals[expressed] / gene_totals[expressed].sum())
    np.savez(out, mu=mu, cell_totals=totals)
    LOG.info(
        "%d cells, %d expressed genes of %d; mean total UMI %.0f, range %.0f to %.0f",
        len(totals),
        int(expressed.sum()),
        len(genes),
        totals.mean(),
        totals.min(),
        totals.max(),
    )


############
# Simulate #
############


@beartype
def _random_binary_tree(n_leaves: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """SI E.2.3: split a uniformly chosen current leaf until there are
    ``n_leaves`` leaves.

    Nodes are numbered in creation order, so every parent index is below its
    children's. Returns the parent array (root has parent -1) and the depth
    array; leaves are the nodes nobody names as a parent.
    """
    n_nodes = 2 * n_leaves - 1
    parent = np.full(n_nodes, -1, dtype=np.int64)
    depth = np.zeros(n_nodes, dtype=np.int64)
    leaves = [0]
    next_node = 1
    for _ in range(n_leaves - 1):
        pick = rng.integers(len(leaves))
        node = leaves[pick]
        for child in (next_node, next_node + 1):
            parent[child] = node
            depth[child] = depth[node] + 1
        leaves[pick] = next_node
        leaves.append(next_node + 1)
        next_node += 2
    return parent, depth


@beartype
def _newick(parent: np.ndarray, leaf_of_node: np.ndarray) -> str:
    """Newick for the generating tree, leaves labelled ``cell<i>`` through
    ``leaf_of_node``, every branch at ``BRANCH_LENGTH``."""
    n_nodes = len(parent)
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    for node in range(1, n_nodes):
        children[parent[node]].append(node)
    # Post-order without recursion: children are always above their parent in
    # index order, so a reverse index sweep has every child's string ready.
    text: list[str] = [""] * n_nodes
    for node in range(n_nodes - 1, -1, -1):
        if children[node]:
            inner = ",".join(text[c] for c in children[node])
            text[node] = f"({inner})"
        else:
            text[node] = f"cell{leaf_of_node[node]}"
        if node > 0:
            text[node] += f":{BRANCH_LENGTH}"
    return text[0] + ";\n"


@beartype
def _write_mtx_header(path: Path, n_genes: int, n_cells: int, nnz: int, body: Path) -> None:
    """Prepend the Matrix Market header to a body of ``row col value`` lines."""
    with path.open("wb") as out:
        out.write(b"%%MatrixMarket matrix coordinate integer general\n")
        out.write(f"{n_genes} {n_cells} {nnz}\n".encode())
        with body.open("rb") as src:
            while chunk := src.read(1 << 24):
                out.write(chunk)
    body.unlink()


@click.command()
@click.option("--params", type=click.Path(path_type=Path), required=True, help="params.npz from `baron`")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--n-cells", type=int, required=True)
@click.option("--n-genes", type=int, default=N_GENES_SIMULATED, show_default=True)
@click.option("--seed", type=int, default=31, show_default=True)
@beartype
def simulate(params: Path, out: Path, n_cells: int, n_genes: int, seed: int) -> None:
    """SI section E: LTQs diffused along a random binary tree, then Poisson
    UMI counts at a realistic sequencing depth.

    Writes under ``out``:

    * ``counts.mtx`` genes by cells, sorted by gene, plus ``genes.tsv`` and
      ``cells.tsv`` for Sanity's ``-mtx_genes`` / ``-mtx_cells``
    * ``truth_ltq.npy`` cells by genes, float32, the noise-free LTQs
    * ``truth.nwk`` the generating tree
    * ``sim_params.npz`` the per-gene ``mu`` and ``v`` drawn for this run
    """
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    empirical = np.load(params)
    baron_mu: np.ndarray = empirical["mu"]
    baron_totals: np.ndarray = empirical["cell_totals"]

    # SI E.2: one Baron gene's mean per simulated gene, exponential variances.
    mu = baron_mu[rng.integers(len(baron_mu), size=n_genes)]
    v = rng.exponential(GENE_VARIANCE_MEAN, size=n_genes)
    # SI E.1: total count per cell sampled from the real dataset.
    n_c = baron_totals[rng.integers(len(baron_totals), size=n_cells)]

    parent, depth = _random_binary_tree(n_cells, rng)
    n_nodes = len(parent)
    is_leaf = np.ones(n_nodes, dtype=bool)
    is_leaf[parent[1:]] = False
    leaf_nodes = np.flatnonzero(is_leaf)
    assert len(leaf_nodes) == n_cells
    leaf_of_node = np.full(n_nodes, -1, dtype=np.int64)
    leaf_of_node[leaf_nodes] = np.arange(n_cells)
    levels = [np.flatnonzero(depth == d) for d in range(1, int(depth.max()) + 1)]
    LOG.info("tree: %d leaves, depth %d, %.1f s", n_cells, depth.max(), time.time() - t0)

    truth = np.lib.format.open_memmap(
        out / "truth_ltq.npy", mode="w+", dtype=np.float32, shape=(n_cells, n_genes)
    )
    # Pass 1: diffuse, standardise per gene (SI E.2.1 step 4), store y[cell, gene]
    # and accumulate sum_g exp(y) per cell for the LTQ constraint of step 5.
    cell_z = np.zeros(n_cells, dtype=np.float64)
    y_nodes = np.empty((n_nodes, GENE_BLOCK), dtype=np.float64)
    for lo in range(0, n_genes, GENE_BLOCK):
        hi = min(lo + GENE_BLOCK, n_genes)
        b = hi - lo
        sd = np.sqrt(BRANCH_LENGTH * v[lo:hi])
        y_nodes[0, :b] = 0.0
        for nodes in levels:
            y_nodes[nodes, :b] = y_nodes[parent[nodes], :b] + rng.normal(size=(len(nodes), b)) * sd
        leaves = y_nodes[leaf_nodes, :b]
        leaves -= leaves.mean(axis=0)
        leaves *= np.sqrt(v[lo:hi] / leaves.var(axis=0))
        leaves += mu[lo:hi]
        cell_z += np.exp(leaves).sum(axis=1)
        truth[:, lo:hi] = leaves
    log_z = np.log(cell_z)
    LOG.info("pass 1 (diffusion) done, %.1f s", time.time() - t0)

    # Pass 2: LTQs are y minus the per-cell normaliser (E.2.1 step 5); counts
    # are Poisson with mean N_c exp(x) (E.1).
    body = out / "counts.body"
    nnz = 0
    cols = np.arange(1, n_cells + 1)
    with body.open("w") as fh:
        for lo in range(0, n_genes, GENE_BLOCK):
            hi = min(lo + GENE_BLOCK, n_genes)
            x = truth[:, lo:hi].astype(np.float64) - log_z[:, None]
            truth[:, lo:hi] = x
            counts = rng.poisson(n_c[:, None] * np.exp(x))
            for j in range(hi - lo):
                col = counts[:, j]
                nz = np.flatnonzero(col)
                if len(nz) == 0:
                    continue
                nnz += len(nz)
                fh.write(
                    "\n".join(f"{lo + j + 1} {c} {k}" for c, k in zip(cols[nz], col[nz], strict=True))
                )
                fh.write("\n")
    truth.flush()
    _write_mtx_header(out / "counts.mtx", n_genes, n_cells, nnz, body)
    (out / "genes.tsv").write_text("".join(f"gene{g}\n" for g in range(n_genes)))
    (out / "cells.tsv").write_text("".join(f"cell{c}\n" for c in range(n_cells)))
    (out / "truth.nwk").write_text(_newick(parent, leaf_of_node))
    np.savez(out / "sim_params.npz", mu=mu, v=v, n_c=n_c, seed=seed)
    density = nnz / (n_cells * n_genes)
    (out / "meta.tsv").write_text(
        f"n_cells\t{n_cells}\nn_genes\t{n_genes}\nseed\t{seed}\nnnz\t{nnz}\n"
        f"density\t{density:.4f}\nmean_total_umi\t{n_c.mean():.0f}\ntree_depth\t{depth.max()}\n"
        f"simulate_seconds\t{time.time() - t0:.1f}\n"
    )
    LOG.info(
        "counts: %d nonzero of %d (density %.3f), mean depth %.0f UMI, %.1f s total",
        nnz, n_cells * n_genes, density, n_c.mean(), time.time() - t0,
    )


##########
# Sanity #
##########


@click.command()
@click.option("--sanity-bin", type=click.Path(path_type=Path), required=True)
@click.option("--sim", type=click.Path(path_type=Path), required=True, help="output folder of `simulate`")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--threads", type=int, default=8, show_default=True)
@beartype
def sanity(sanity_bin: Path, sim: Path, out: Path, threads: int) -> None:
    """Run Sanity with extended output and the posterior-maximising gene
    variance (``-v_m MAP``, Sanity 2.0's default).

    Wall time and the load average go to ``out/sanity_timing.tsv``.
    """
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(sanity_bin),
        "-f", str(sim / "counts.mtx"),
        "-mtx_genes", str(sim / "genes.tsv"),
        "-mtx_cells", str(sim / "cells.tsv"),
        "-d", str(out),
        "-n", str(threads),
        "-e", "1",
        "-v_m", "MAP",
    ]
    load_before = _load_average()
    t0 = time.time()
    with (out / "sanity.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    (out / "sanity_timing.tsv").write_text(
        f"seconds\t{secs:.1f}\nthreads\t{threads}\nload_before\t{load_before}\nload_after\t{_load_average()}\n"
    )
    LOG.info("Sanity: %.1f s on %d threads (load before %s)", secs, threads, load_before)


@beartype
def _load_average() -> str:
    """The one-minute load average, for the timing records."""
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.split("load averages:")[-1].strip()


##########
# Select #
##########


@beartype
def _row_batches(path: Path, n_cells: int) -> Iterator[np.ndarray]:
    """Stream a Sanity genes-by-cells matrix (tab separated, no header, no
    index) in blocks of rows as float64 arrays. Streaming, because at 100k
    cells one matrix is 14 GB in memory."""
    lazy = pl.scan_csv(path, separator="\t", has_header=False, infer_schema_length=1000)
    for batch in lazy.collect_batches(chunk_size=GENE_BLOCK):
        # Sanity ends every line with a tab, which reads as one empty column.
        if batch.width not in (n_cells, n_cells + 1):
            raise click.ClickException(f"{path}: {batch.width} columns for {n_cells} cells")
        yield batch.select(batch.columns[:n_cells]).select(pl.all().cast(pl.Float64)).to_numpy()


@beartype
def _filter_lines(src: Path, dst: Path, keep: np.ndarray) -> None:
    """Copy the lines of ``src`` whose zero-based index is in ``keep``, in
    order, without parsing them."""
    wanted = set(int(k) for k in keep)
    with src.open("rb") as fin, dst.open("wb") as fout:
        for i, line in enumerate(fin):
            if i in wanted:
                fout.write(line)


@click.command()
@click.option("--sim", type=click.Path(path_type=Path), required=True)
@click.option("--sanity-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True, help="the configuration directory")
@click.option("--snr", type=float, default=DEFAULT_SNR_THRESHOLD, show_default=True)
@beartype
def select(sim: Path, sanity_dir: Path, out: Path, snr: float) -> None:
    """Keep genes with ``S_g >= snr`` (SI equation S6, computed on Sanity's
    posterior log fold changes and their error bars) and write:

    * ``out/sanity_sel/`` every Sanity output file subset to those genes, in
      Sanity's own layout, for ``prep`` (the Rust harness) to convert with
      its own ingest
    * ``out/ours/truth.csv`` the noise-free LTQs of those genes, cells by
      genes, scaled by ``1/sqrt(v_g)`` with Sanity's ``v_g``, the unit the
      tree is fit in
    * ``out/truth.nwk`` copied from the simulation
    * ``out/selection.tsv`` counts and the threshold
    """
    t0 = time.time()
    genes = (sanity_dir / "geneID.txt").read_text().split()
    cells = (sanity_dir / "cellID.txt").read_text().split()
    n_genes, n_cells = len(genes), len(cells)
    # Sanity drops genes with no counts at all, so its rows are a subset of the
    # simulated genes; the names carry the simulated index.
    sim_index = np.asarray([int(g.removeprefix("gene")) for g in genes], dtype=np.int64)
    variance = np.loadtxt(sanity_dir / "variance.txt", dtype=np.float64)
    # S6: mean over cells of squared deviation over squared error bar. delta is
    # already centred on Sanity's mu_g.
    s_g = np.concatenate(
        [
            np.mean(d**2 / e**2, axis=1)
            for d, e in zip(
                _row_batches(sanity_dir / "delta.txt", n_cells),
                _row_batches(sanity_dir / "d_delta.txt", n_cells),
                strict=True,
            )
        ]
    )
    if len(s_g) != n_genes:
        raise click.ClickException(f"delta.txt has {len(s_g)} rows for {n_genes} genes")
    keep = np.flatnonzero(s_g >= snr)
    if len(keep) == 0:
        raise click.ClickException(f"no gene reaches S_g >= {snr}")
    LOG.info("kept %d of %d genes at S_g >= %g, %.0f s", len(keep), n_genes, snr, time.time() - t0)

    sel = out / "sanity_sel"
    sel.mkdir(parents=True, exist_ok=True)
    (sel / "cellID.txt").write_text("".join(f"{c}\n" for c in cells))
    for name in ("geneID.txt", *SANITY_MATRICES, *SANITY_VECTORS):
        _filter_lines(sanity_dir / name, sel / name, keep)
    LOG.info("subset written, %.0f s", time.time() - t0)

    truth = np.load(sim / "truth_ltq.npy", mmap_mode="r")
    if truth.shape[0] != n_cells or truth.shape[1] <= sim_index.max():
        raise click.ClickException(f"truth_ltq.npy is {truth.shape} for {n_cells} cells and gene index {sim_index.max()}")
    ours = out / "ours"
    ours.mkdir(parents=True, exist_ok=True)
    scaled = truth[:, sim_index[keep]].astype(np.float64) / np.sqrt(variance[keep])[None, :]
    pl.DataFrame(scaled).write_csv(ours / "truth.csv", include_header=False, float_precision=7)
    (out / "truth.nwk").write_text((sim / "truth.nwk").read_text())
    sim_meta = (sim / "meta.tsv").read_text()
    (out / "selection.tsv").write_text(
        f"snr_threshold\t{snr}\nn_genes_in\t{n_genes}\nn_genes_kept\t{len(keep)}\nn_cells\t{n_cells}\n"
        f"select_seconds\t{time.time() - t0:.1f}\n"
    )
    (out / "meta.tsv").write_text(sim_meta + f"n_features\t{len(keep)}\nsnr_threshold\t{snr}\n")


#######
# CLI #
#######


@click.group()
def cli() -> None:
    """Realistic data for the comparison: Baron parameters, the SI count
    simulator, Sanity, and gene selection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)


cli.add_command(baron)
cli.add_command(simulate)
cli.add_command(sanity)
cli.add_command(select)

if __name__ == "__main__":
    cli()
