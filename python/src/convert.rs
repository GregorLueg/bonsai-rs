//! numpy in, numpy out, and the tree across the boundary.
//!
//! A tree crosses as three things: a parent array with `-1` for the root, the
//! branch length above each node, and the leaf count. That is exactly what
//! [`Tree::from_parents`] takes, so nothing but the root sentinel is
//! translated.

use bonsai_rs::bonsai::BonsaiResult;
use bonsai_rs::tree::{NO_NODE, Tree};
use numpy::{
    Element, IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::BErr;

////////////
// Inputs //
////////////

/// Borrow a C-contiguous 2-D array as `(data, n_rows, n_cols)`.
///
/// The slice borrows from `a`, so `a` must outlive any `Python::detach` it is
/// passed into. Keep this call outside the closure.
///
/// ### Params
///
/// * `a` - Read-only view of a 2-D numpy array
///
/// ### Returns
///
/// `(data, n_rows, n_cols)`, row-major. Errors if the array is not
/// C-contiguous; the Python layer runs `np.ascontiguousarray` first, so this is
/// a backstop.
pub(crate) fn flat<'a, T: Element>(
    a: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = a.shape();
    let data = a.as_slice().map_err(|_| {
        PyValueError::new_err("array must be C-contiguous; use np.ascontiguousarray")
    })?;
    Ok((data, shape[0], shape[1]))
}

/// Borrow a contiguous 1-D array as a slice.
///
/// ### Params
///
/// * `a` - Read-only view of a 1-D numpy array
///
/// ### Returns
///
/// The slice, or a `ValueError` if the array is not contiguous.
pub(crate) fn slice<'a, T: Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous; use np.ascontiguousarray"))
}

/// Rebuild a tree from its parent array, branch lengths and leaf count.
///
/// ### Params
///
/// * `parent` - Parent per node, `-1` for the root
/// * `branch` - Branch length above each node
/// * `n_leaves` - Leaves, which occupy `0..n_leaves`
///
/// ### Returns
///
/// The tree, or a `ValueError` if the arrays do not describe one.
pub(crate) fn tree_in(
    parent: &PyReadonlyArray1<'_, i64>,
    branch: &PyReadonlyArray1<'_, f64>,
    n_leaves: usize,
) -> PyResult<Tree> {
    let parent: Vec<u32> = slice(parent)?
        .iter()
        .map(|&p| match p {
            -1 => Ok(NO_NODE),
            p if p >= 0 && p < NO_NODE as i64 => Ok(p as u32),
            p => Err(PyValueError::new_err(format!(
                "parent index {p} is neither -1 nor a node index"
            ))),
        })
        .collect::<PyResult<_>>()?;
    Ok(Tree::from_parents(parent, slice(branch)?.to_vec(), n_leaves).map_err(BErr)?)
}

/////////////
// Outputs //
/////////////

/// Write a tree into `out` as `parent`, `branch` and `n_leaves`.
///
/// ### Params
///
/// * `out` - Dict to fill
/// * `tree` - The tree
///
/// ### Returns
///
/// Nothing, or the first insertion error.
pub(crate) fn tree_out(out: &Bound<'_, PyDict>, tree: &Tree) -> PyResult<()> {
    let py = out.py();
    let parent: Vec<i64> = (0..tree.n_nodes() as u32)
        .map(|i| tree.parent(i).map_or(-1, i64::from))
        .collect();
    out.set_item("parent", parent.into_pyarray(py))?;
    out.set_item("branch", tree.branches().to_vec().into_pyarray(py))?;
    out.set_item("n_leaves", tree.n_leaves())?;
    Ok(())
}

/// Pack a finished reconstruction into a dict of numpy arrays.
///
/// ### Params
///
/// * `py` - Attached interpreter token
/// * `res` - The reconstruction
/// * `features` - Original feature index of each retained column; `res.features`
///   unless the caller's feature axis was itself a subset
/// * `dropped` - Features dropped before selection, empty if not applicable
///
/// ### Returns
///
/// The dict the Python layer turns into a `BonsaiResult`.
pub(crate) fn result_out<'py, T: Element>(
    py: Python<'py>,
    res: BonsaiResult<T>,
    features: Vec<usize>,
    dropped: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    tree_out(&out, &res.tree)?;
    let (n_nodes, k) = (res.tree.n_nodes(), features.len());
    let as_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<_>>();
    out.set_item("loglik", res.loglik)?;
    out.set_item("features", as_i64(features).into_pyarray(py))?;
    out.set_item("dropped", as_i64(dropped).into_pyarray(py))?;
    out.set_item(
        "node_means",
        res.node_means.into_pyarray(py).reshape([n_nodes, k])?,
    )?;
    out.set_item(
        "node_sds",
        res.node_sds.into_pyarray(py).reshape([n_nodes, k])?,
    )?;
    let steps: Vec<(&str, f64, f64)> = res
        .steps
        .iter()
        .map(|s| (s.step, s.loglik, s.gain))
        .collect();
    out.set_item("steps", steps)?;
    Ok(out)
}
