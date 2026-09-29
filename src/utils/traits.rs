//! Shared traits and trait boundaries used across the crate.

use crate::utils::simd::BonsaiSimd;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Floating-point types that `bonsai-rs` can store node coordinates in.
///
/// Governs storage only: every reduction in this crate accumulates in `f64`
/// (see [`wide`] and [`narrow`]).
///
/// ### `f32` storage wants centred means
///
/// Kernels read means only through `(m_k - m_l)^2`, so `f32` loses precision
/// as `|mean| / separation` grows. Relative error in a loglikelihood
/// difference, 4 leaves by 256 features: 1.1e-6 at 0, 1.5e-6 at 1e3, 3.2e-4 at
/// 1e5, 7.0e-2 at 1e7. [`crate::ingest`] scales but does not centre, so
/// callers whose means sit far from zero relative to their spread should
/// centre them or use `f64`; subtracting a per-feature constant leaves the
/// loglikelihood unchanged.
pub trait BonsaiFloat:
    Float
    + BonsaiSimd
    + FromPrimitive
    + ToPrimitive
    + Send
    + Sync
    + Sum
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Debug
    + Default
    + 'static
{
}

impl<T> BonsaiFloat for T where
    T: Float
        + BonsaiSimd
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Sum
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Debug
        + Default
        + 'static
{
}

/// Widen a storage float to `f64` for accumulation.
///
/// `f32` and `f64` convert infallibly, so the `NAN` fallback is unreachable
/// for the types this crate is used with.
///
/// ### Params
///
/// * `x` - Value to widen
///
/// ### Returns
///
/// The value as an `f64`.
#[inline(always)]
pub fn wide<T: BonsaiFloat>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

/// Narrow an `f64` back to the storage float.
///
/// ### Params
///
/// * `x` - Value to narrow
///
/// ### Returns
///
/// The value as a `T`.
#[inline(always)]
pub fn narrow<T: BonsaiFloat>(x: f64) -> T {
    T::from_f64(x).unwrap_or_else(T::nan)
}
