//! Everything that takes a finished tree: Newick, layouts, clustering and
//! distances.

use bonsai_rs::tree::cluster::cluster as cluster_tree;
use bonsai_rs::tree::distance::tree_distances as distances;
use bonsai_rs::tree::layout::{dendrogram, equal_angle, equal_daylight};
use bonsai_rs::tree::newick::{parse_newick, write_newick};
use bonsai_rs::tree::simulate::robinson_foulds as rf;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{slice, tree_in, tree_out};
use crate::error::BErr;

/// Arrays of a tree as they cross the boundary: parent, branch, leaf count.
type TreeArgs<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
    usize,
);

/// A 1-D numpy array handed back to Python.
type Arr<'py, T> = Bound<'py, PyArray1<T>>;

/// Serialise a tree to Newick.
///
/// ### Params
///
/// * `parent` - Parent per node, `-1` for the root
/// * `branch` - Branch length above each node
/// * `n_leaves` - Number of leaves
/// * `labels` - One label per leaf
///
/// ### Returns
///
/// The Newick string, semicolon terminated.
#[pyfunction]
pub fn to_newick(
    parent: PyReadonlyArray1<'_, i64>,
    branch: PyReadonlyArray1<'_, f64>,
    n_leaves: usize,
    labels: Vec<String>,
) -> PyResult<String> {
    let tree = tree_in(&parent, &branch, n_leaves)?;
    Ok(write_newick(&tree, &labels).map_err(BErr)?)
}

/// Parse a Newick string.
///
/// ### Params
///
/// * `text` - The Newick string
///
/// ### Returns
///
/// A dict with `parent`, `branch`, `n_leaves` and the leaf `labels`.
#[pyfunction]
pub fn read_newick<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    let (tree, labels) = parse_newick(text).map_err(BErr)?;
    let out = PyDict::new(py);
    tree_out(&out, &tree)?;
    out.set_item("labels", labels)?;
    Ok(out)
}

/// Node coordinates for drawing.
///
/// ### Params
///
/// * `parent`, `branch`, `n_leaves` - The tree
/// * `kind` - `"daylight"`, `"angle"` or `"dendrogram"`
/// * `hyperbolic` - Project onto the Poincare disk afterwards
///
/// ### Returns
///
/// `(x, y)`, one entry per node.
#[pyfunction]
pub fn layout<'py>(
    py: Python<'py>,
    parent: PyReadonlyArray1<'py, i64>,
    branch: PyReadonlyArray1<'py, f64>,
    n_leaves: usize,
    kind: &str,
    hyperbolic: bool,
) -> PyResult<(Arr<'py, f64>, Arr<'py, f64>)> {
    let tree = tree_in(&parent, &branch, n_leaves)?;
    let lay = py
        .detach(|| match kind {
            "daylight" => Ok(equal_daylight(&tree, None).map(|(l, _)| l)),
            "angle" => Ok(equal_angle(&tree, None)),
            "dendrogram" => Ok(dendrogram(&tree, None)),
            _ => Err(kind.to_string()),
        })
        .map_err(|k| PyValueError::new_err(format!("unknown layout '{k}'")))?
        .map_err(BErr)?;
    let lay = if hyperbolic {
        lay.hyperbolic(None)
    } else {
        lay
    };
    Ok((lay.x.into_pyarray(py), lay.y.into_pyarray(py)))
}

/// Cut the tree into clusters minimising summed within-cluster distance.
///
/// ### Params
///
/// * `parent`, `branch`, `n_leaves` - The tree
/// * `n_clusters` - Requested cluster count, clamped into `1..=n_leaves`
///
/// ### Returns
///
/// `(leaf_cluster, centres, sizes)`.
#[pyfunction]
pub fn cluster<'py>(
    py: Python<'py>,
    parent: PyReadonlyArray1<'py, i64>,
    branch: PyReadonlyArray1<'py, f64>,
    n_leaves: usize,
    n_clusters: usize,
) -> PyResult<(Arr<'py, i64>, Arr<'py, i64>, Arr<'py, i64>)> {
    let tree = tree_in(&parent, &branch, n_leaves)?;
    let c = py.detach(|| cluster_tree(&tree, n_clusters));
    let wide = |v: Vec<i64>| v.into_pyarray(py);
    Ok((
        wide(c.leaf_cluster.iter().map(|&x| x as i64).collect()),
        wide(c.centres.iter().map(|&x| x as i64).collect()),
        wide(c.sizes.iter().map(|&x| x as i64).collect()),
    ))
}

/// Path distance along the tree for each leaf pair `(left[i], right[i])`.
///
/// ### Params
///
/// * `parent`, `branch`, `n_leaves` - The tree
/// * `left`, `right` - Leaf indices, same length
///
/// ### Returns
///
/// One distance per pair.
#[pyfunction]
pub fn tree_distances<'py>(
    py: Python<'py>,
    parent: PyReadonlyArray1<'py, i64>,
    branch: PyReadonlyArray1<'py, f64>,
    n_leaves: usize,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
) -> PyResult<Arr<'py, f64>> {
    let tree = tree_in(&parent, &branch, n_leaves)?;
    let (l, r) = (slice(&left)?, slice(&right)?);
    if l.len() != r.len() {
        return Err(PyValueError::new_err("left and right differ in length"));
    }
    let in_range = |x: i64| x >= 0 && (x as usize) < n_leaves;
    let pairs: Vec<(usize, usize)> = l
        .iter()
        .zip(r)
        .map(|(&a, &b)| {
            if in_range(a) && in_range(b) {
                Ok((a as usize, b as usize))
            } else {
                Err(PyValueError::new_err(format!(
                    "leaf pair ({a}, {b}) is out of range for {n_leaves} leaves"
                )))
            }
        })
        .collect::<PyResult<_>>()?;
    Ok(py.detach(|| distances(&tree, &pairs)).into_pyarray(py))
}

/// Robinson-Foulds distance between two trees over the same leaves.
///
/// ### Params
///
/// * `left`, `right` - The two trees, as `(parent, branch, n_leaves)`
///
/// ### Returns
///
/// The number of splits in one tree but not the other.
#[pyfunction]
pub fn robinson_foulds(left: TreeArgs<'_>, right: TreeArgs<'_>) -> PyResult<usize> {
    let a = tree_in(&left.0, &left.1, left.2)?;
    let b = tree_in(&right.0, &right.1, right.2)?;
    Ok(rf(&a, &b).map_err(BErr)?)
}
