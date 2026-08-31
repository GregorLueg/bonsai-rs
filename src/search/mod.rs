//! The tree search of SPEC.md section 9.

pub mod candidates;
pub mod nni;
pub mod polytomy;
pub mod star;

/// The leaf data a search step scores its trees against.
///
/// Both blocks are row-major `[leaf][feature]` in the transformed units of
/// SPEC.md section 3.1, which is the layout
/// [`crate::model::likelihood::NodeState::new`] expects. Search steps rebuild
/// the tree, so they rebuild the node state with it and need the leaf rows
/// rather than a settled state.
#[derive(Clone, Copy, Debug)]
pub struct Leaves<'a, T> {
    /// Transformed means, `[leaf][feature]`.
    pub means: &'a [T],
    /// Transformed precisions, same layout.
    pub precisions: &'a [T],
    /// Number of features.
    pub n_features: usize,
}
