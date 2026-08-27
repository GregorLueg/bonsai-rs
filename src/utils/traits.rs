//! Shared traits and trait boundaries used across the crate.

use crate::utils::simd::BonsaiSimd;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Floating-point types that `bonsai-rs` can store node coordinates in.
///
/// The bounds break down as follows. `Float`, `FromPrimitive` and
/// `ToPrimitive` give the arithmetic and the widening to `f64` that every
/// kernel performs before it accumulates (see the note on precision below).
/// `BonsaiSimd` carries the vectorised feature-axis kernels, so algorithm code
/// never names a lane width. `Send`/`Sync`/`'static` are needed because the
/// pair scan and the gene-blocked sweeps fan out under rayon. The `*Assign`
/// family keeps the in-place accumulation loops readable. `Debug` is there so
/// assertion failures in the tests print something useful.
///
/// ### Note on precision
///
/// This trait governs *storage*, not accumulation. Every reduction in this
/// crate accumulates in `f64` regardless of `T`, because the tree
/// loglikelihood is a sum over thousands of features whose interesting
/// differences are `O(1)` while the sum itself is `O(p)`. Storing in `f32`
/// halves memory traffic on the dominant access pattern; accumulating in `f32`
/// would make the convergence criterion noise.
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
/// Every kernel goes through this rather than calling `to_f64()` and unwrapping
/// at each site. `f32` and `f64` both convert infallibly, so the fallback is
/// unreachable for the types this crate is used with.
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
