//! Errors in `bonsai-rs`.
//!
//! One enum for the whole crate, grouped by subsystem. Add new variants to the
//! matching section rather than the bottom, so the file stays navigable as the
//! port grows. Anything a caller could plausibly hit returns `Err`; panics are
//! reserved for broken invariants the type system cannot state.

use thiserror::Error;

/// All error variants that can occur across `bonsai-rs` operations.
#[derive(Debug, Error)]
pub enum BonsaiErrors {
    // -- input validation --
    /// The means and standard deviations matrices disagree in shape.
    #[error(
        "Shape mismatch: means are {mean_cells} cells by {mean_features} features, standard deviations are {sd_cells} by {sd_features}."
    )]
    ShapeMismatch {
        /// Number of cells in the means matrix
        mean_cells: usize,
        /// Number of features in the means matrix
        mean_features: usize,
        /// Number of cells in the standard deviations matrix
        sd_cells: usize,
        /// Number of features in the standard deviations matrix
        sd_features: usize,
    },

    /// A dataset with no cells or no features cannot produce a tree.
    #[error("Empty input: {n_cells} cells by {n_features} features.")]
    EmptyInput {
        /// Number of cells supplied
        n_cells: usize,
        /// Number of features supplied
        n_features: usize,
    },

    /// Standard deviations must be strictly positive; a zero implies infinite
    /// precision and blows up the pruning recursion.
    #[error(
        "Non-positive standard deviation {value} at cell {cell}, feature {feature}. Standard deviations must be strictly positive."
    )]
    NonPositiveSd {
        /// Offending value
        value: f64,
        /// Cell index
        cell: usize,
        /// Feature index
        feature: usize,
    },

    /// A mean was not a finite number.
    #[error("Non-finite mean {value} at cell {cell}, feature {feature}.")]
    NonFiniteMean {
        /// Offending value
        value: f64,
        /// Cell index
        cell: usize,
        /// Feature index
        feature: usize,
    },

    /// Per-feature variances must be strictly positive; the ingest transform of
    /// SPEC.md section 3.1 divides by their square root.
    #[error("Non-positive variance {value} for feature {feature}.")]
    NonPositiveVariance {
        /// Offending value
        value: f64,
        /// Feature index
        feature: usize,
    },

    /// The signal-to-noise filter of SPEC.md section 3.3 kept nothing.
    #[error(
        "The signal-to-noise threshold {threshold} retained none of the {n_features} features. Lower the threshold."
    )]
    NoFeaturesRetained {
        /// Number of features considered
        n_features: usize,
        /// Threshold that was applied
        threshold: f64,
    },

    /// Recovering likelihood parameters from Sanity posteriors was
    /// ill-conditioned for every feature (SPEC.md section 3.4, S5).
    ///
    /// The conversion amplifies by `v / (v - eps^2)`, which blows up as the
    /// posterior error bar approaches the feature variance. Losing the whole
    /// panel to it usually means the posteriors came from a Sanity run
    /// reporting the *expectation* of `v`; the posterior-maximising mode is the
    /// one this model wants.
    #[error(
        "Recovering likelihood parameters from Sanity posteriors exceeded the amplification cap {cap} for all {n_features} features."
    )]
    IllConditionedConversion {
        /// Number of features considered
        n_features: usize,
        /// Amplification cap that was applied
        cap: f64,
    },

    /// A caller-supplied parameter was outside its valid range.
    ///
    /// Distinct from `MalformedTree`: the tree is fine, the knob is not.
    #[error("Parameter '{name}' has invalid value {value}. {expected}")]
    BadParameter {
        /// Field name, as the caller wrote it
        name: &'static str,
        /// Offending value
        value: f64,
        /// What would have been acceptable
        expected: &'static str,
    },

    // -- tree structure --
    /// A node index was outside the arena.
    #[error("Node index {index} is out of range for a tree with {n_nodes} nodes.")]
    NodeOutOfRange {
        /// Offending index
        index: usize,
        /// Number of nodes in the arena
        n_nodes: usize,
    },

    /// The parent array does not describe a single rooted tree.
    #[error("Malformed tree: {reason}")]
    MalformedTree {
        /// What is wrong with it
        reason: String,
    },

    // -- numerics --
    /// A bracketed root find failed to converge.
    #[error(
        "Root find for the branch length did not converge within {max_iter} iterations (last step {last_step:e})."
    )]
    RootFindDiverged {
        /// Iteration budget that was exhausted
        max_iter: usize,
        /// Size of the final step
        last_step: f64,
    },
}
