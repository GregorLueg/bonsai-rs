//! Python bindings for `bonsai-rs`.
//!
//! Functions only, no handles. Every entry point returns a dict of numpy arrays
//! that the Python layer turns into frozen dataclasses.

use pyo3::prelude::*;

mod convert;
mod error;
mod gpu;
mod run;
mod tree;

////////////
// Module //
////////////

/// Assemble the extension module.
///
/// ### Params
///
/// * `m` - The module being initialised
///
/// ### Returns
///
/// Nothing, or the first registration error.
#[pymodule]
fn _bonsai_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bindings and core crate are versioned separately.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", bonsai_rs::VERSION)?;
    m.add("BonsaiError", m.py().get_type::<error::BonsaiError>())?;

    m.add_function(wrap_pyfunction!(gpu::gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(run::sanity, m)?)?;
    m.add_function(wrap_pyfunction!(run::from_sanity, m)?)?;
    m.add_function(wrap_pyfunction!(run::bonsai, m)?)?;
    m.add_function(wrap_pyfunction!(run::bonsai_from_counts, m)?)?;
    m.add_function(wrap_pyfunction!(run::simulate, m)?)?;

    m.add_function(wrap_pyfunction!(tree::to_newick, m)?)?;
    m.add_function(wrap_pyfunction!(tree::read_newick, m)?)?;
    m.add_function(wrap_pyfunction!(tree::layout, m)?)?;
    m.add_function(wrap_pyfunction!(tree::cluster, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_distances, m)?)?;
    m.add_function(wrap_pyfunction!(tree::robinson_foulds, m)?)?;
    Ok(())
}
