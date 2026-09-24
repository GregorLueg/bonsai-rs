"""Type stubs for the compiled core.

Every entry point returns a plain dict of numpy arrays; the typed surface is
the dataclasses in `_types.py`.
"""

from typing import Any

import numpy as np

__version__: str
__core_version__: str

class BonsaiError(Exception):
    """The data was well formed but the method could not use it."""

def gpu_available() -> bool: ...
def sanity(
    indices: np.ndarray,
    values: np.ndarray,
    indptr: np.ndarray,
    n_cells: int,
    cell_totals: np.ndarray,
    rule: str,
    fixed_variance: float | None,
    double: bool,
    gpu: bool,
    /,
) -> dict[str, Any]: ...
def from_sanity(
    posterior_means: np.ndarray,
    posterior_sds: np.ndarray,
    variances: np.ndarray,
    max_amp: float | None,
    /,
) -> dict[str, Any]: ...
def bonsai(
    means: np.ndarray,
    sds: np.ndarray,
    variances: np.ndarray | None,
    start: str,
    search: str,
    min_snr: float | None,
    reroot: bool,
    /,
) -> dict[str, Any]: ...
def bonsai_from_counts(
    indices: np.ndarray,
    values: np.ndarray,
    indptr: np.ndarray,
    n_cells: int,
    cell_totals: np.ndarray,
    rule: str,
    fixed_variance: float | None,
    double: bool,
    gpu: bool,
    start: str,
    search: str,
    min_snr: float | None,
    max_amp: float | None,
    reroot: bool,
    /,
) -> dict[str, Any]: ...
def backbone(
    means: np.ndarray,
    sds: np.ndarray,
    variances: np.ndarray | None,
    start: str,
    search: str,
    min_snr: float | None,
    reroot: bool,
    backbone_cells: int | None,
    seed: int,
    /,
) -> dict[str, Any]: ...
def simulate(
    kind: str,
    n_leaves: int,
    n_features: int,
    branch_length: float,
    noise_sd: float,
    noise_spread: float,
    seed: int,
    /,
) -> dict[str, Any]: ...
def to_newick(
    parent: np.ndarray, branch: np.ndarray, n_leaves: int, labels: list[str], /
) -> str: ...
def read_newick(text: str, /) -> dict[str, Any]: ...
def layout(
    parent: np.ndarray,
    branch: np.ndarray,
    n_leaves: int,
    kind: str,
    hyperbolic: bool,
    /,
) -> tuple[np.ndarray, np.ndarray]: ...
def cluster(
    parent: np.ndarray, branch: np.ndarray, n_leaves: int, n_clusters: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def tree_distances(
    parent: np.ndarray,
    branch: np.ndarray,
    n_leaves: int,
    left: np.ndarray,
    right: np.ndarray,
    /,
) -> np.ndarray: ...
def robinson_foulds(
    left: tuple[np.ndarray, np.ndarray, int],
    right: tuple[np.ndarray, np.ndarray, int],
    /,
) -> int: ...
