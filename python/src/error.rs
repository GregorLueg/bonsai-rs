//! Mapping [`BonsaiErrors`] onto Python exceptions.
//!
//! No new error enum: the library already has one. `BonsaiErrors` and `PyErr`
//! are both foreign here, so the `From` impl needs a local newtype.

use bonsai_rs::errors::BonsaiErrors;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

////////////////
// Exceptions //
////////////////

create_exception!(
    _bonsai_rs,
    BonsaiError,
    PyException,
    "Base class for every error raised by the bonsai-rs Rust core that is not a bad argument."
);

/////////////
// Newtype //
/////////////

/// Carries a [`BonsaiErrors`] to the FFI boundary.
///
/// A tuple struct, so it doubles as the conversion function: `.map_err(BErr)?`
/// in anything returning [`PyResult`].
pub(crate) struct BErr(
    /// The error the library raised.
    pub BonsaiErrors,
);

impl From<BErr> for PyErr {
    fn from(e: BErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            // The caller handed us bad numbers or a bad knob.
            BonsaiErrors::ShapeMismatch { .. }
            | BonsaiErrors::EmptyInput { .. }
            | BonsaiErrors::NonPositiveSd { .. }
            | BonsaiErrors::NonFiniteMean { .. }
            | BonsaiErrors::NonPositiveVariance { .. }
            | BonsaiErrors::BadParameter { .. }
            | BonsaiErrors::NodeOutOfRange { .. }
            | BonsaiErrors::MalformedTree { .. }
            | BonsaiErrors::Sanity(_) => PyValueError::new_err(msg),

            // The data was well formed but the method could not use it.
            BonsaiErrors::NoFeaturesRetained { .. }
            | BonsaiErrors::IllConditionedConversion { .. }
            | BonsaiErrors::NeighbourGraph { .. }
            | BonsaiErrors::RootFindDiverged { .. } => BonsaiError::new_err(msg),
        }
    }
}

impl From<sanity_sc_rs::errors::SanityErrors> for BErr {
    fn from(e: sanity_sc_rs::errors::SanityErrors) -> Self {
        Self(e.into())
    }
}

impl From<BonsaiErrors> for BErr {
    fn from(e: BonsaiErrors) -> Self {
        Self(e)
    }
}
