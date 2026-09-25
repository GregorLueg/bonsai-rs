"""Reconstruction: Sanity, the S5 conversion, Bonsai and backbone mode."""

from typing import Any, Literal

import numpy as np
from beartype import beartype

from . import _bonsai_rs as _core
from ._types import BonsaiResult, Likelihood, SanityResult, Step, Tree
from ._validate import cell_totals_of, check_pair, check_vector, gene_major

###########
# Globals #
###########

Start = Literal["linkage", "greedy"]
Search = Literal["approximate", "exact"]
VarianceRule = Literal["marginalise", "posterior_mean", "max_posterior", "fixed"]


@beartype
def _result(d: dict) -> BonsaiResult:
    """Wrap the dict the core returns into a `BonsaiResult`."""
    return BonsaiResult(
        tree=Tree(d["parent"], d["branch"], d["n_leaves"]),
        loglik=d["loglik"],
        features=d["features"],
        dropped=d["dropped"],
        node_means=d["node_means"],
        node_sds=d["node_sds"],
        steps=[Step(*s) for s in d["steps"]],
    )


@beartype
def _variances(variances: np.ndarray | None, n_features: int) -> np.ndarray | None:
    """Check optional per-feature variances."""
    if variances is None:
        return None
    return check_vector(variances, n_features, "variances")


##########
# Sanity #
##########


# `counts` is dense numpy or scipy sparse; scipy is optional, so it is `Any`.
@beartype
def sanity(
    counts: Any,
    *,
    cell_totals: np.ndarray | None = None,
    variance_rule: VarianceRule = "marginalise",
    fixed_variance: float | None = None,
    dtype: type[np.float32] | type[np.float64] = np.float32,
    gpu: bool = False,
) -> SanityResult:
    """Posterior log expression and error bars from raw UMI counts.

    Runs Sanity (Breda et al., Nat. Biotechnol. 2021) via ``sanity-sc-rs``.

    Args:
        counts: ``(n_cells, n_genes)`` raw UMI counts, a numpy integer array
            or any scipy sparse matrix. Log-normalised input is rejected.
        cell_totals: Total UMIs of each cell over *all* genes. Defaults to the
            row sums of ``counts``, which is only right if ``counts`` holds
            every gene. Subset genes before calling and you must pass this.
        variance_rule: How the per-gene variance enters the estimates.
            ``"marginalise"`` integrates over its posterior and is the one to
            quote; the others are faster approximations.
        fixed_variance: The variance used for every gene when
            ``variance_rule="fixed"``.
        dtype: Storage type of the output. Reductions are ``float64`` either
            way.
        gpu: Run Sanity on the GPU through wgpu. The device path is
            ``float32`` whatever ``dtype`` says, since wgpu has no ``float64``.
            Check `gpu_available` first; asking for it where that is ``False``
            raises rather than falling back to the CPU.

    Returns:
        The posteriors, cells x genes.

    Raises:
        ValueError: On malformed or non-integer counts.
        BonsaiError: If ``gpu=True`` and no GPU can be reached.
    """
    indices, values, indptr, n_cells, _ = gene_major(counts)
    totals = cell_totals_of(counts, cell_totals, n_cells)
    d = _core.sanity(
        indices,
        values,
        indptr,
        n_cells,
        totals,
        variance_rule,
        fixed_variance,
        dtype is np.float64,
        gpu,
    )
    return SanityResult(
        log_fold_changes=d["log_fold_changes"].T,
        error_bars=d["error_bars"].T,
        mean_log_quotient=d["mean_log_quotient"],
        mean_log_quotient_error=d["mean_log_quotient_error"],
        variance=d["variance"],
    )


@beartype
def from_sanity(
    posterior_means: np.ndarray,
    posterior_sds: np.ndarray,
    variances: np.ndarray,
    *,
    max_amplification: float | None = None,
) -> Likelihood:
    """Recover likelihood means and SDs from Sanity posteriors (SI eq. S5).

    Sanity shrinks each cell towards the gene mean under a ``N(0, v)`` prior.
    Bonsai wants the measurement before that shrinkage, so this undoes it:
    ``mu = x * v / (v - eps^2)`` and ``sig^2 = eps^2 * v / (v - eps^2)``.

    Pass the zero-centred **log fold changes**, not the log transcription
    quotients. With the gene mean added, the per-cell amplification scales it
    differently in every cell and invents structure. On simulated counts that
    took Robinson-Foulds from 0 to 100 of a possible 122.

    Args:
        posterior_means: Log fold changes, ``(n_cells, n_genes)``.
        posterior_sds: Their error bars, same shape.
        variances: Sanity's per-gene variance ``v``.
        max_amplification: Largest ``v / (v - eps^2)`` accepted. A gene with
            any cell above it is dropped, not clamped. ``None`` for the
            default of 1000.

    Returns:
        Means, SDs and variances for the kept genes, plus which genes were kept
        and dropped.

    Raises:
        BonsaiError: If every gene is ill-conditioned.
    """
    m, s = check_pair(
        posterior_means, posterior_sds, names=("posterior_means", "posterior_sds")
    )
    v = check_vector(variances, m.shape[1], "variances")
    d = _core.from_sanity(m, s, v, max_amplification)
    return Likelihood(
        means=d["means"],
        sds=d["sds"],
        variances=d["variances"],
        features=d["features"],
        dropped=d["dropped"],
    )


##########
# Bonsai #
##########


@beartype
def bonsai(
    means: np.ndarray,
    sds: np.ndarray,
    *,
    variances: np.ndarray | None = None,
    start: Start = "linkage",
    search: Search = "approximate",
    min_signal_to_noise: float | None = None,
    reroot: bool = True,
) -> BonsaiResult:
    """Reconstruct a tree from per-cell means and error bars.

    Storage follows the input: ``float32`` in, ``float32`` all the way down.
    Same input gives the same tree whatever the thread count.

    Args:
        means: ``(n_cells, n_features)``.
        sds: Standard deviations on ``means``, same shape, strictly positive.
        variances: Per-feature variance. ``None`` estimates it from the data
            (SI eq. S9). From Sanity, pass ``Likelihood.variances``.
        start: ``"linkage"`` (Ward over a neighbour graph, the default) or
            ``"greedy"`` (the paper's greedy merge, for like-for-like
            reproduction).
        search: ``"approximate"`` (the default) or ``"exact"``. Exact runs
            SPR and NNI as the paper specifies them; the approximate search
            revisits only what the last moves touched and landed within a few
            nats of the exact one on every dataset measured, several times
            faster. Worth an exact run to check on data of your own.
        min_signal_to_noise: Features below this signal-to-noise are dropped
            before the search. ``None`` for the default of 1, the paper's.
        reroot: Reroot for display once the search is done. Changes the
            picture, not the likelihood.

    Returns:
        The tree, its loglikelihood and the posterior over every node.

    Raises:
        ValueError: On shape mismatches, non-positive SDs or non-finite means.
        BonsaiError: If no feature survives selection.
    """
    m, s = check_pair(means, sds)
    v = _variances(variances, m.shape[1])
    return _result(_core.bonsai(m, s, v, start, search, min_signal_to_noise, reroot))


# `counts` is dense numpy or scipy sparse; scipy is optional, so it is `Any`.
@beartype
def bonsai_from_counts(
    counts: Any,
    *,
    cell_totals: np.ndarray | None = None,
    variance_rule: VarianceRule = "marginalise",
    fixed_variance: float | None = None,
    dtype: type[np.float32] | type[np.float64] = np.float32,
    gpu: bool = False,
    start: Start = "linkage",
    search: Search = "approximate",
    min_signal_to_noise: float | None = None,
    max_amplification: float | None = None,
    reroot: bool = True,
) -> BonsaiResult:
    """Raw counts to tree: `sanity`, `from_sanity`, then `bonsai`.

    The whole chain runs in Rust, so the dense posteriors never cross into
    Python.

    Args:
        counts: As `sanity`.
        cell_totals: As `sanity`.
        variance_rule: As `sanity`.
        fixed_variance: As `sanity`.
        dtype: As `sanity`.
        gpu: As `sanity`. Only Sanity runs on the GPU; the tree search is
            CPU either way.
        start: As `bonsai`.
        search: As `bonsai`.
        min_signal_to_noise: As `bonsai`.
        max_amplification: As `from_sanity`.
        reroot: As `bonsai`.

    Returns:
        As `bonsai`, with ``features`` and ``dropped`` indexing the genes of
        ``counts``.
    """
    indices, values, indptr, n_cells, _ = gene_major(counts)
    totals = cell_totals_of(counts, cell_totals, n_cells)
    return _result(
        _core.bonsai_from_counts(
            indices,
            values,
            indptr,
            n_cells,
            totals,
            variance_rule,
            fixed_variance,
            dtype is np.float64,
            gpu,
            start,
            search,
            min_signal_to_noise,
            max_amplification,
            reroot,
        )
    )


@beartype
def backbone(
    means: np.ndarray,
    sds: np.ndarray,
    *,
    variances: np.ndarray | None = None,
    backbone_cells: int | None = None,
    seed: int = 0,
    start: Start = "linkage",
    search: Search = "approximate",
    min_signal_to_noise: float | None = None,
    reroot: bool = True,
) -> BonsaiResult:
    """Backbone mode for datasets too large to search directly.

    Reconstructs on a random subset, places every other cell onto it one at a
    time, then refines the whole tree. This is the paper's route to large
    datasets.

    Args:
        means: As `bonsai`.
        sds: As `bonsai`.
        variances: As `bonsai`.
        backbone_cells: Cells in the initial backbone. ``None`` for the default
            of 2048.
        seed: Seed for choosing the backbone subset.
        start: As `bonsai`, for the backbone.
        search: As `bonsai`.
        min_signal_to_noise: As `bonsai`.
        reroot: As `bonsai`.

    Returns:
        As `bonsai`.
    """
    m, s = check_pair(means, sds)
    v = _variances(variances, m.shape[1])
    return _result(
        _core.backbone(
            m, s, v, start, search, min_signal_to_noise, reroot, backbone_cells, seed
        )
    )
