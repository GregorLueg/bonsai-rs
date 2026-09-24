//! The reconstruction entry points: Sanity, the S5 conversion, Bonsai, the
//! chain from counts, backbone mode and the simulator.
//!
//! Every numeric entry point dispatches on the input's element type and runs
//! generic over it, so `float32` in means `float32` storage all the way down.
//! Reductions are `f64` either way; that is the core's policy, not ours.

use bonsai_rs::backbone::{BackboneParams, backbone as backbone_run};
use bonsai_rs::bonsai::{BonsaiParams, StartTree, bonsai as bonsai_run};
use bonsai_rs::ingest::{IngestParams, from_sanity as s5, from_sanity_output, prepare};
use bonsai_rs::tree::simulate::{
    SimulationParams, simulate_binary, simulate_binary_random_branches, simulate_unbalanced,
};
use bonsai_rs::utils::traits::BonsaiFloat;
use numpy::{Element, IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sanity_sc_rs::config::{SanityParams, VarianceRule};
use sanity_sc_rs::float::SanityFloat;
use sanity_sc_rs::input::CountMatrix;

use crate::convert::{flat, result_out, slice, tree_out};
use crate::error::BErr;
use crate::gpu::{check, run_sanity};

/// What a float has to be to go through every path in this file.
trait Float: BonsaiFloat + SanityFloat + Element {}
impl<T: BonsaiFloat + SanityFloat + Element> Float for T {}

/////////////
// Helpers //
/////////////

/// Two same-typed matrices, borrowed from numpy.
enum Pair<'py> {
    /// Both `float32`.
    F32(PyReadonlyArray2<'py, f32>, PyReadonlyArray2<'py, f32>),
    /// Both `float64`.
    F64(PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, f64>),
}

/// Borrow two arrays that must share a float type.
///
/// ### Params
///
/// * `a`, `b` - The arrays
///
/// ### Returns
///
/// Both, typed, or a `TypeError` if they are not both `float32` or both
/// `float64`. The Python layer has already cast, so this is a backstop.
fn pair<'py>(a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Pair<'py>> {
    if let (Ok(x), Ok(y)) = (a.extract(), b.extract()) {
        return Ok(Pair::F32(x, y));
    }
    if let (Ok(x), Ok(y)) = (a.extract(), b.extract()) {
        return Ok(Pair::F64(x, y));
    }
    Err(PyTypeError::new_err(
        "means and sds must both be float32 or both float64",
    ))
}

/// Resolve the exposed knobs into a full parameter set.
///
/// ### Params
///
/// * `start` - `"linkage"` or `"greedy"`
/// * `min_snr` - Signal-to-noise floor, `None` for the default
/// * `max_amp` - S5 amplification cap, `None` for the default
/// * `reroot` - Reroot for display once the search is done
///
/// ### Returns
///
/// The parameters, or a `ValueError` on an unknown start.
fn params(
    start: &str,
    min_snr: Option<f64>,
    max_amp: Option<f64>,
    reroot: bool,
) -> PyResult<BonsaiParams> {
    let mut p = BonsaiParams::default();
    p.start = match start {
        "linkage" => StartTree::Linkage,
        "greedy" => StartTree::GreedyMerge,
        s => return Err(PyValueError::new_err(format!("unknown start '{s}'"))),
    };
    p.ingest = IngestParams {
        min_signal_to_noise: min_snr.unwrap_or(p.ingest.min_signal_to_noise),
        max_sanity_amplification: max_amp.unwrap_or(p.ingest.max_sanity_amplification),
    };
    p.reroot = reroot;
    Ok(p)
}

/// Build a Sanity parameter set from the rule name.
///
/// ### Params
///
/// * `rule` - `"marginalise"`, `"posterior_mean"`, `"max_posterior"` or
///   `"fixed"`
/// * `fixed_variance` - The variance for `"fixed"`, ignored otherwise
///
/// ### Returns
///
/// The parameters, or a `ValueError`.
fn sanity_params(rule: &str, fixed_variance: Option<f64>) -> PyResult<SanityParams> {
    let variance_rule = match (rule, fixed_variance) {
        ("marginalise", _) => VarianceRule::Marginalise,
        ("posterior_mean", _) => VarianceRule::PosteriorMean,
        ("max_posterior", _) => VarianceRule::MaxPosterior,
        ("fixed", Some(v)) => VarianceRule::Fixed(v),
        ("fixed", None) => {
            return Err(PyValueError::new_err(
                "variance_rule 'fixed' needs fixed_variance",
            ));
        }
        (r, _) => {
            return Err(PyValueError::new_err(format!(
                "unknown variance_rule '{r}'"
            )));
        }
    };
    Ok(SanityParams {
        variance_rule,
        ..SanityParams::default()
    })
}

/// Assemble the gene-major count matrix Sanity reads.
///
/// ### Params
///
/// * `indices` - Cell index of each stored count, gene by gene
/// * `values` - The counts
/// * `indptr` - Gene offsets into `indices`, `n_genes + 1` long
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The matrix, or a `ValueError` naming the structural fault.
fn counts_in(
    indices: &PyReadonlyArray1<'_, u32>,
    values: &PyReadonlyArray1<'_, u32>,
    indptr: &PyReadonlyArray1<'_, i64>,
    n_cells: usize,
) -> PyResult<CountMatrix> {
    let indptr: Vec<usize> = slice(indptr)?.iter().map(|&x| x as usize).collect();
    Ok(CountMatrix::new(
        slice(indices)?.to_vec(),
        slice(values)?.to_vec(),
        indptr,
        n_cells,
    )
    .map_err(BErr::from)?)
}

////////////
// Sanity //
////////////

/// Run Sanity over gene-major sparse counts.
///
/// ### Params
///
/// * `indices`, `values`, `indptr`, `n_cells` - The counts, as [`counts_in`]
/// * `cell_totals` - Total UMIs per cell over all genes
/// * `rule`, `fixed_variance` - As [`sanity_params`]
/// * `double` - `float64` storage instead of `float32`
/// * `gpu` - Run Sanity on the GPU, see [`crate::gpu`]
///
/// ### Returns
///
/// A dict of gene-major `(n_genes, n_cells)` matrices and per-gene vectors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn sanity<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, u32>,
    values: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, i64>,
    n_cells: usize,
    cell_totals: PyReadonlyArray1<'py, f64>,
    rule: &str,
    fixed_variance: Option<f64>,
    double: bool,
    gpu: bool,
) -> PyResult<Bound<'py, PyDict>> {
    check(gpu)?;
    let counts = counts_in(&indices, &values, &indptr, n_cells)?;
    let totals = slice(&cell_totals)?;
    let sp = sanity_params(rule, fixed_variance)?;
    if double {
        sanity_out::<f64>(py, &counts, totals, sp, gpu)
    } else {
        sanity_out::<f32>(py, &counts, totals, sp, gpu)
    }
}

/// [`sanity`] at one float type.
///
/// ### Params
///
/// * `py` - Attached interpreter token
/// * `counts` - The counts
/// * `totals` - Total UMIs per cell
/// * `sp` - Sanity parameters
/// * `gpu` - Run Sanity on the GPU
///
/// ### Returns
///
/// As [`sanity`].
fn sanity_out<'py, T: Float>(
    py: Python<'py>,
    counts: &CountMatrix,
    totals: &[f64],
    sp: SanityParams,
    gpu: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = py.detach(|| run_sanity::<T>(counts, totals, sp, gpu))?;
    let (g, c) = (out.n_genes, out.n_cells);
    let d = PyDict::new(py);
    d.set_item(
        "log_fold_changes",
        out.log_fold_changes.into_pyarray(py).reshape([g, c])?,
    )?;
    d.set_item(
        "error_bars",
        out.error_bars.into_pyarray(py).reshape([g, c])?,
    )?;
    d.set_item("mean_log_quotient", out.mean_log_quotient.into_pyarray(py))?;
    d.set_item(
        "mean_log_quotient_error",
        out.mean_log_quotient_error.into_pyarray(py),
    )?;
    d.set_item("variance", out.variance.into_pyarray(py))?;
    Ok(d)
}

/// Recover likelihood means and SDs from Sanity posteriors (S5).
///
/// ### Params
///
/// * `posterior_means` - Log fold changes, `(n_cells, n_genes)`
/// * `posterior_sds` - Their error bars, same shape and type
/// * `variances` - Sanity's per-gene variance
/// * `max_amp` - Amplification cap, `None` for the default
///
/// ### Returns
///
/// A dict with `means`, `sds`, `variances`, `features` and `dropped`.
#[pyfunction]
pub fn from_sanity<'py>(
    py: Python<'py>,
    posterior_means: &Bound<'py, PyAny>,
    posterior_sds: &Bound<'py, PyAny>,
    variances: PyReadonlyArray1<'py, f64>,
    max_amp: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let v = slice(&variances)?;
    match pair(posterior_means, posterior_sds)? {
        Pair::F32(m, s) => s5_out(py, &m, &s, v, max_amp),
        Pair::F64(m, s) => s5_out(py, &m, &s, v, max_amp),
    }
}

/// [`from_sanity`] at one float type.
///
/// ### Params
///
/// As [`from_sanity`], typed.
///
/// ### Returns
///
/// As [`from_sanity`].
fn s5_out<'py, T: Float>(
    py: Python<'py>,
    m: &PyReadonlyArray2<'py, T>,
    s: &PyReadonlyArray2<'py, T>,
    v: &[f64],
    max_amp: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (mean, n, p) = flat(m)?;
    let (sd, _, _) = flat(s)?;
    let ip = params("linkage", None, max_amp, true)?.ingest;
    let lik = py
        .detach(|| s5(mean, sd, n, p, v, Some(ip)))
        .map_err(BErr)?;
    let k = lik.features.len();
    let d = PyDict::new(py);
    d.set_item("means", lik.means.into_pyarray(py).reshape([n, k])?)?;
    d.set_item("sds", lik.sds.into_pyarray(py).reshape([n, k])?)?;
    d.set_item("variances", lik.variances.into_pyarray(py))?;
    d.set_item(
        "features",
        lik.features
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "dropped",
        lik.dropped
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

////////////
// Bonsai //
////////////

/// Reconstruct a tree from means and standard deviations.
///
/// ### Params
///
/// * `means`, `sds` - `(n_cells, n_features)`, both `float32` or both `float64`
/// * `variances` - Per-feature variance, `None` to estimate
/// * `start`, `min_snr`, `reroot` - As [`params`]
///
/// ### Returns
///
/// The dict `convert::result_out` builds.
#[pyfunction]
pub fn bonsai<'py>(
    py: Python<'py>,
    means: &Bound<'py, PyAny>,
    sds: &Bound<'py, PyAny>,
    variances: Option<PyReadonlyArray1<'py, f64>>,
    start: &str,
    min_snr: Option<f64>,
    reroot: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let v = variances.as_ref().map(slice).transpose()?;
    let bp = params(start, min_snr, None, reroot)?;
    match pair(means, sds)? {
        Pair::F32(m, s) => bonsai_out(py, &m, &s, v, bp),
        Pair::F64(m, s) => bonsai_out(py, &m, &s, v, bp),
    }
}

/// [`bonsai`] at one float type.
///
/// ### Params
///
/// As [`bonsai`], typed and resolved.
///
/// ### Returns
///
/// As [`bonsai`].
fn bonsai_out<'py, T: Float>(
    py: Python<'py>,
    m: &PyReadonlyArray2<'py, T>,
    s: &PyReadonlyArray2<'py, T>,
    v: Option<&[f64]>,
    bp: BonsaiParams,
) -> PyResult<Bound<'py, PyDict>> {
    let (mean, n, p) = flat(m)?;
    let (sd, _, _) = flat(s)?;
    let res = py
        .detach(|| bonsai_run(mean, sd, n, p, v, Some(bp)))
        .map_err(BErr)?;
    let features = res.features.clone();
    result_out(py, res, features, Vec::new())
}

/// Counts to tree: Sanity, S5, then Bonsai, without leaving Rust.
///
/// ### Params
///
/// * `indices`, `values`, `indptr`, `n_cells` - The counts, as [`counts_in`]
/// * `cell_totals` - Total UMIs per cell over all genes
/// * `rule`, `fixed_variance` - As [`sanity_params`]
/// * `double` - `float64` storage instead of `float32`
/// * `gpu` - Run Sanity on the GPU, see [`crate::gpu`]
/// * `start`, `min_snr`, `max_amp`, `reroot` - As [`params`]
///
/// ### Returns
///
/// The dict `convert::result_out` builds, with `features` and `dropped`
/// indexing the genes of the count matrix.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bonsai_from_counts<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, u32>,
    values: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, i64>,
    n_cells: usize,
    cell_totals: PyReadonlyArray1<'py, f64>,
    rule: &str,
    fixed_variance: Option<f64>,
    double: bool,
    gpu: bool,
    start: &str,
    min_snr: Option<f64>,
    max_amp: Option<f64>,
    reroot: bool,
) -> PyResult<Bound<'py, PyDict>> {
    check(gpu)?;
    let counts = counts_in(&indices, &values, &indptr, n_cells)?;
    let totals = slice(&cell_totals)?;
    let sp = sanity_params(rule, fixed_variance)?;
    let bp = params(start, min_snr, max_amp, reroot)?;
    if double {
        chain::<f64>(py, &counts, totals, sp, bp, gpu)
    } else {
        chain::<f32>(py, &counts, totals, sp, bp, gpu)
    }
}

/// [`bonsai_from_counts`] at one float type.
///
/// ### Params
///
/// * `py` - Attached interpreter token
/// * `counts` - The counts
/// * `totals` - Total UMIs per cell
/// * `sp`, `bp` - Sanity and Bonsai parameters
/// * `gpu` - Run Sanity on the GPU
///
/// ### Returns
///
/// As [`bonsai_from_counts`].
fn chain<'py, T: Float>(
    py: Python<'py>,
    counts: &CountMatrix,
    totals: &[f64],
    sp: SanityParams,
    bp: BonsaiParams,
    gpu: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (res, genes, dropped) = py.detach(|| -> Result<_, BErr> {
        let out = run_sanity::<T>(counts, totals, sp, gpu)?;
        let lik = from_sanity_output(&out, Some(bp.ingest))?;
        let k = lik.features.len();
        let res = bonsai_run(
            &lik.means,
            &lik.sds,
            lik.n_cells,
            k,
            Some(&lik.variances),
            Some(bp),
        )?;
        Ok((res, lik.features, lik.dropped))
    })?;
    // `res.features` indexes the S5 survivors; map back to the count matrix.
    let features = res.features.iter().map(|&k| genes[k]).collect();
    result_out(py, res, features, dropped)
}

/// Backbone mode (SPEC 15): build on a subset, place the rest, refine.
///
/// ### Params
///
/// * `means`, `sds`, `variances`, `start`, `min_snr`, `reroot` - As [`bonsai`]
/// * `backbone_cells` - Cells in the backbone, `None` for the default
/// * `seed` - Seed for choosing the backbone
///
/// ### Returns
///
/// As [`bonsai`].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn backbone<'py>(
    py: Python<'py>,
    means: &Bound<'py, PyAny>,
    sds: &Bound<'py, PyAny>,
    variances: Option<PyReadonlyArray1<'py, f64>>,
    start: &str,
    min_snr: Option<f64>,
    reroot: bool,
    backbone_cells: Option<usize>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let v = variances.as_ref().map(slice).transpose()?;
    let mut bb = BackboneParams {
        bonsai: params(start, min_snr, None, reroot)?,
        seed,
        ..BackboneParams::default()
    };
    if let Some(c) = backbone_cells {
        bb.backbone_cells = c;
    }
    match pair(means, sds)? {
        Pair::F32(m, s) => backbone_out(py, &m, &s, v, bb),
        Pair::F64(m, s) => backbone_out(py, &m, &s, v, bb),
    }
}

/// [`backbone`] at one float type.
///
/// ### Params
///
/// As [`backbone`], typed and resolved.
///
/// ### Returns
///
/// As [`backbone`].
fn backbone_out<'py, T: Float>(
    py: Python<'py>,
    m: &PyReadonlyArray2<'py, T>,
    s: &PyReadonlyArray2<'py, T>,
    v: Option<&[f64]>,
    bb: BackboneParams,
) -> PyResult<Bound<'py, PyDict>> {
    let (mean, n, p) = flat(m)?;
    let (sd, _, _) = flat(s)?;
    let (res, _) = py.detach(|| -> Result<_, BErr> {
        let data = prepare(mean, sd, n, p, v, Some(bb.bonsai.ingest))?;
        Ok(backbone_run(&data, Some(bb))?)
    })?;
    let features = res.features.clone();
    result_out(py, res, features, Vec::new())
}

//////////////
// Simulate //
//////////////

/// Data drawn by Brownian motion on a known tree (SPEC 13.1).
///
/// ### Params
///
/// * `kind` - `"binary"`, `"random_branches"` or `"unbalanced"`
/// * `n_leaves` - Cells; a power of two for the balanced kinds
/// * `n_features` - Features
/// * `branch_length` - Branch length, ignored by `"random_branches"`
/// * `noise_sd` - Error-bar scale relative to the data spread
/// * `noise_spread` - Log-uniform spread of the error bars about `noise_sd`
/// * `seed` - Seed
///
/// ### Returns
///
/// A dict with the tree, `truth`, `means`, `sds` and `variances`. Everything
/// but `variances` is in transformed units.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simulate<'py>(
    py: Python<'py>,
    kind: &str,
    n_leaves: usize,
    n_features: usize,
    branch_length: f64,
    noise_sd: f64,
    noise_spread: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = Some(SimulationParams {
        n_leaves,
        n_features,
        branch_length,
        noise_sd,
        noise_spread,
        seed,
        ..SimulationParams::default()
    });
    let sim = match kind {
        "binary" => simulate_binary::<f64>(sp),
        "random_branches" => simulate_binary_random_branches::<f64>(sp),
        "unbalanced" => simulate_unbalanced::<f64>(sp),
        k => return Err(PyValueError::new_err(format!("unknown simulation '{k}'"))),
    }
    .map_err(BErr)?;
    let (n, p) = (sim.n_leaves, sim.n_features);
    let d = PyDict::new(py);
    tree_out(&d, &sim.tree)?;
    d.set_item("truth", sim.truth.into_pyarray(py).reshape([n, p])?)?;
    d.set_item("means", sim.means.into_pyarray(py).reshape([n, p])?)?;
    d.set_item("sds", sim.sds.into_pyarray(py).reshape([n, p])?)?;
    d.set_item("variances", sim.variances.into_pyarray(py))?;
    Ok(d)
}
