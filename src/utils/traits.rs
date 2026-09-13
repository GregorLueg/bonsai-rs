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
/// merge-round pair scan, the bound scan of `search::bounds` and the ingest
/// feature pass all fan out under rayon. The `*Assign`
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
///
/// ### `f32` storage wants centred means
///
/// Every kernel reads the means only through `(m_k - m_l)^2`, so what has to
/// survive the narrowing is the *separation between cells*, not the position of
/// the feature. Storing an uncentred mean spends the mantissa on an offset that
/// then cancels: the loss is governed by `|mean| / separation`, and `f32` has
/// about seven digits to spend on it.
///
/// Measured on four leaves by 256 features, comparing the difference
/// of two topologies' loglikelihoods, which is the quantity a search decides
/// on, against the same computation in `f64`:
///
/// | `|mean| / separation` | relative error in the difference |
/// |---|---|
/// | 0 | 1.1e-6 |
/// | 1e3 | 1.5e-6 |
/// | 1e5 | 3.2e-4 |
/// | 1e7 | 7.0e-2 |
///
/// Up to about `1e3` the error is the ordinary `f32` floor. By `1e5` the
/// decision is wrong in its fourth digit and by `1e7` it is wrong in its first.
/// **Note that [`crate::ingest`] does not centre**: SPEC.md section 3.1's
/// transform is a scale, `mu / sqrt(v)`, and the feature mean of section 3.3 is
/// used for the signal-to-noise filter and then dropped. So a caller whose raw
/// means sit far from zero relative to their spread should centre them before
/// asking for `f32` storage, or use `f64`. Subtracting a per-feature constant
/// from every cell leaves the loglikelihood exactly unchanged, since only
/// differences enter.
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
