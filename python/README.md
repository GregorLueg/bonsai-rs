[![PyPI](https://img.shields.io/pypi/v/bonsai-rs.svg)](https://pypi.org/project/bonsai-rs/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://gregorlueg.github.io/bonsai-rs/)
[![CI](https://github.com/GregorLueg/bonsai-rs/actions/workflows/python-test.yml/badge.svg)](https://github.com/GregorLueg/bonsai-rs/actions/workflows/python-test.yml)

# bonsai-rs

Tree representations of single-cell data under Brownian motion. Python
bindings for the [bonsai-rs](https://github.com/GregorLueg/bonsai-rs) crate, a
clean-room implementation of Bonsai (de Groot et al., *Nature Biotechnology*
2026, [doi 10.1038/s41587-026-03220-2](https://doi.org/10.1038/s41587-026-03220-2)).

Raw UMI counts in, tree out. Sanity runs first, also in Rust, so the error bars
Bonsai needs come from the counts rather than from nowhere.

## Install

```bash
uv pip install bonsai-rs            # numpy only; dense counts
uv pip install "bonsai-rs[sparse]"  # adds scipy, for sparse counts
```

Wheels for Python 3.10 and up on Linux x86_64 and macOS.

## Use

```python
import bonsai_rs as bs

res = bs.bonsai_from_counts(counts, cell_totals=totals)  # cells x genes

res.tree  # parent array, branch lengths, leaf count
res.node_means  # posterior position of every node, ancestors too
xy = bs.layout(res.tree)  # (n_nodes, 2), edges run node -> parent
newick = bs.to_newick(res.tree, labels=cell_names)
```

`cell_totals` is each cell's UMI total over **all** genes. Leave it out and the
row sums are used, which is only right when `counts` holds every gene.

Already have means and standard deviations? `bs.bonsai(means, sds)`.

## Docs

[gregorlueg.github.io/bonsai-rs](https://gregorlueg.github.io/bonsai-rs/):
quickstart, the Sanity handover and its sharp edges, and the API reference.

## Development

```bash
uv sync
uv run maturin develop --release
uv run pytest
uv run --group docs mkdocs serve
```

## Licence

MIT. Built from the paper and its CC-BY-4.0 Supplementary Information only;
see [provenance](https://github.com/GregorLueg/bonsai-rs/blob/main/PROVENANCE.md) in the repository. Cite the paper.
