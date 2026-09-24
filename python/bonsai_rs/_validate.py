"""Array checks at the FFI boundary."""

from typing import Any

import numpy as np
from beartype import beartype

###########
# Globals #
###########

#: Largest count the Rust core stores, ``u32::MAX``.
_MAX_COUNT = np.iinfo(np.uint32).max


@beartype
def check_pair(
    means: np.ndarray, sds: np.ndarray, *, names: tuple[str, str] = ("means", "sds")
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a means/SD pair into what the core borrows.

    ``float32`` and ``float64`` pass through; anything else numeric becomes
    ``float64``. If the two differ, both go to the wider type. Both come back
    C-contiguous.

    Args:
        means: ``(n_cells, n_features)``.
        sds: Same shape.
        names: Argument names for error messages.

    Returns:
        The two arrays, same dtype, C-contiguous.

    Raises:
        TypeError: If either does not hold numbers.
        ValueError: If either is not 2-D or is empty, or the shapes differ.
    """
    out = []
    for arr, name in zip((means, sds), names, strict=True):
        if arr.dtype.kind not in "fiub":
            raise TypeError(f"{name} must hold numbers, got dtype {arr.dtype}")
        if arr.ndim != 2 or 0 in arr.shape:
            raise ValueError(f"{name} must be a non-empty 2-D array, got {arr.shape}")
        out.append(arr)
    if out[0].shape != out[1].shape:
        raise ValueError(
            f"{names[0]} is {out[0].shape} but {names[1]} is {out[1].shape}"
        )
    dtype = np.result_type(out[0], out[1])
    if dtype not in (np.float32, np.float64):
        dtype = np.dtype(np.float64)
    return (
        np.ascontiguousarray(out[0], dtype=dtype),
        np.ascontiguousarray(out[1], dtype=dtype),
    )


@beartype
def check_vector(x: np.ndarray, n: int, name: str) -> np.ndarray:
    """Coerce a per-feature or per-cell vector to contiguous ``float64``.

    Args:
        x: 1-D array.
        n: Required length.
        name: Argument name for error messages.

    Returns:
        The vector as contiguous ``float64``.

    Raises:
        ValueError: If it is not 1-D of length ``n``.
    """
    if x.ndim != 1 or len(x) != n:
        raise ValueError(f"{name} must be 1-D of length {n}, got shape {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float64)


@beartype
def check_values(values: np.ndarray) -> np.ndarray:
    """Check stored counts are non-negative integers and cast to ``uint32``.

    Args:
        values: The stored (non-zero) counts, any numeric dtype.

    Returns:
        Contiguous ``uint32`` counts.

    Raises:
        ValueError: If any count is negative, fractional or above ``u32::MAX``.
            Log-normalised input lands here, which is the point: Sanity models
            the Poisson sampling in raw counts.
    """
    if values.size and (
        values.min() < 0
        or values.max() > _MAX_COUNT
        or not np.array_equal(values, np.round(values))
    ):
        raise ValueError(
            "counts must be raw non-negative integers; "
            "log-normalised or scaled input is the wrong input for Sanity"
        )
    return np.ascontiguousarray(values, dtype=np.uint32)


# `counts` is a numpy array or any scipy sparse matrix/array. scipy is an
# optional dependency, so its types cannot appear in the annotation.
@beartype
def gene_major(counts: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Split a cells x genes count matrix into gene-major CSR parts.

    Gene-major CSR over cells x genes is CSC, which is what scipy hands back.

    Args:
        counts: ``(n_cells, n_genes)`` integer counts, dense or scipy sparse.

    Returns:
        ``(indices, values, indptr, n_cells, n_genes)``: cell index of each
        stored count gene by gene (``uint32``), the counts (``uint32``), gene
        offsets (``int64``, ``n_genes + 1`` long), and the two dimensions.

    Raises:
        TypeError: If ``counts`` is neither a numpy array nor scipy sparse.
        ValueError: If the counts are not raw non-negative integers.
    """
    if isinstance(counts, np.ndarray):
        if counts.ndim != 2 or 0 in counts.shape:
            raise ValueError(f"counts must be non-empty 2-D, got {counts.shape}")
        n_cells, n_genes = counts.shape
        gene, cell = np.nonzero(counts.T)
        values = counts.T[gene, cell]
        indptr = np.zeros(n_genes + 1, dtype=np.int64)
        np.cumsum(np.bincount(gene, minlength=n_genes), out=indptr[1:])
        indices = cell
    elif hasattr(counts, "tocsc"):
        csc = counts.tocsc(copy=True)
        csc.sum_duplicates()
        csc.eliminate_zeros()
        n_cells, n_genes = csc.shape
        indices, values, indptr = csc.indices, csc.data, csc.indptr
    else:
        raise TypeError(
            f"counts must be a numpy array or scipy sparse, got {type(counts)}"
        )
    return (
        np.ascontiguousarray(indices, dtype=np.uint32),
        check_values(values),
        np.ascontiguousarray(indptr, dtype=np.int64),
        int(n_cells),
        int(n_genes),
    )


@beartype
def cell_totals_of(
    counts: Any, cell_totals: np.ndarray | None, n_cells: int
) -> np.ndarray:
    """Resolve ``cell_totals``, defaulting to the row sums of ``counts``.

    Args:
        counts: The count matrix, dense or scipy sparse.
        cell_totals: Caller-supplied totals, or ``None``.
        n_cells: Number of cells.

    Returns:
        Contiguous ``float64`` totals, one per cell.
    """
    if cell_totals is None:
        cell_totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    return check_vector(cell_totals, n_cells, "cell_totals")
