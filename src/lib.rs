//! Tree representations of high-dimensional data under Brownian motion.
//!
//! A clean-room implementation of the Bonsai algorithm (de Groot, Morillo
//! Leonardo, Pachkov and van Nimwegen, *Nature Biotechnology* 2026,
//! doi 10.1038/s41587-026-03220-2), built from the paper and its CC-BY-4.0
//! Supplementary Information. See `PROVENANCE.md` and `docs/SPEC.md`.
//!
//! ### Numeric policy
//!
//! Storage is generic over `BonsaiFloat`; every reduction accumulates in `f64`
//! regardless. The tree loglikelihood is a sum over thousands of features whose
//! interesting differences are `O(1)` while the sum is `O(p)`, so `f32`
//! accumulation would turn the convergence criterion into noise. `f32` storage
//! is still worth having: the kernels are memory bound.
//!
//! ### Units
//!
//! Input means and standard deviations are divided by the square root of the
//! per-feature variance at ingest, which removes the diffusion scale from every
//! kernel. Loglikelihoods drop the `2*pi` and variance terms, both of which are
//! independent of topology, so absolute values are meaningful only up to an
//! additive constant. Unlike the reference implementation, this crate works in
//! the loglikelihood and not twice it.
//!
//! ### Parallelism
//!
//! Rayon fans out over independent nodes within a tree level and over candidate
//! pairs within a merge round. The feature-axis kernels are sequential by
//! design, so nothing here nests.

#![warn(missing_docs)]
// Indexed loops are what the numeric kernels want. Rewriting them as zipped
// iterators to appease clippy costs readability and, where the compiler stops
// hoisting the bounds check, throughput.
#![allow(clippy::needless_range_loop)]

pub mod backbone;
pub mod bonsai;
pub mod errors;
pub mod ingest;
pub mod model;
pub mod search;
pub mod tree;
pub mod utils;
