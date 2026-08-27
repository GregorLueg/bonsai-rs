//! The tree arena.
//!
//! Nodes live in flat `Vec`s addressed by `u32`. The layout carries one
//! invariant that the rest of the crate leans on hard:
//!
//! > Leaves occupy indices `0..n_leaves`. Internal nodes follow, ordered by
//! > height above the leaves, so every non-root node has a strictly larger
//! > parent index and each level is a contiguous index range.
//!
//! Three things fall out of that. Ascending index order is a valid post-order,
//! so the pruning sweep is a bare linear scan with no traversal and no
//! recursion. A parent row lies strictly above all of its children's rows, so
//! borrowing it for writing while reading them is a `split_at_mut` rather than
//! an unsafe alias. And a whole level is one contiguous block of rows, which is
//! what lets the sweep fan out over rayon and what keeps the reads local.
//!
//! [`Tree::from_parents`] enforces the ordering by relabelling internal nodes,
//! which is free for callers: internal rows are always computed by the sweep,
//! never supplied, so no caller-held data is indexed by an internal node.

use crate::errors::BonsaiErrors;

/// Sentinel parent index for the root.
pub const NO_NODE: u32 = u32::MAX;

/// An unrooted tree stored as a flat arena, held in a rooted representation.
///
/// The likelihood is independent of which node is designated the root
/// (SPEC.md section 2, S14). The root here is a bookkeeping choice that fixes
/// what "downstream" means for the effective-leaf recursion.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Parent of each node, `NO_NODE` for the root.
    parent: Vec<u32>,
    /// Length of the branch above each node. The root's entry is unused.
    branch: Vec<f64>,
    /// Number of leaves; nodes `0..n_leaves` are leaves.
    n_leaves: usize,
    /// Start offset into `children` for each node, length `n_nodes + 1`.
    child_ptr: Vec<u32>,
    /// Children of every node, grouped by parent, CSR style.
    children: Vec<u32>,
    /// Node index at which each level starts, length `n_levels + 1`. Level `L`
    /// covers nodes `level_ptr[L]..level_ptr[L + 1]`, all of whose children lie
    /// strictly below `level_ptr[L]`.
    level_ptr: Vec<u32>,
}

impl Tree {
    /// Build a tree from a parent array and its branch lengths.
    ///
    /// Internal nodes are relabelled into level order, so the caller does not
    /// have to supply them that way; only the requirement that every parent
    /// index exceeds its children's is checked rather than fixed, since a
    /// violation there means the input is not describing a tree the way this
    /// arena expects.
    ///
    /// ### Params
    ///
    /// * `parent` - Parent index per node, `NO_NODE` for the single root
    /// * `branch` - Length of the branch above each node; the root's entry is
    ///   ignored
    /// * `n_leaves` - Number of leaves, which must occupy indices `0..n_leaves`
    ///
    /// ### Returns
    ///
    /// The tree, or `MalformedTree` if the input is not a single rooted tree
    /// satisfying the ordering invariant described in the module docs.
    pub fn from_parents(
        parent: Vec<u32>,
        branch: Vec<f64>,
        n_leaves: usize,
    ) -> Result<Self, BonsaiErrors> {
        let n_nodes = parent.len();
        if branch.len() != n_nodes {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!(
                    "parent array has {n_nodes} entries but branch array has {}",
                    branch.len()
                ),
            });
        }
        if n_leaves == 0 || n_leaves > n_nodes {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("{n_leaves} leaves is not valid for {n_nodes} nodes"),
            });
        }

        let mut n_roots = 0usize;
        for (i, &par) in parent.iter().enumerate() {
            if par == NO_NODE {
                n_roots += 1;
                continue;
            }
            if par as usize >= n_nodes {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!("node {i} has parent {par}, out of range"),
                });
            }
            if (par as usize) <= i {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!(
                        "node {i} has parent {par}: parents must have larger indices so that \
                         ascending index order is a post-order"
                    ),
                });
            }
            if (par as usize) < n_leaves {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!("node {i} has leaf {par} as its parent"),
                });
            }
        }
        if n_roots != 1 {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("expected exactly one root, found {n_roots}"),
            });
        }

        // Height above the leaves. Ascending index order is already a valid
        // post-order at this point, so one forward scan settles every node.
        let mut height = vec![0u32; n_nodes];
        for i in 0..n_nodes {
            if let Some(par) = (parent[i] != NO_NODE).then(|| parent[i] as usize) {
                height[par] = height[par].max(height[i] + 1);
            }
        }

        // Relabel internal nodes into level order. Sorting by height alone would
        // be enough for correctness; the index tiebreak keeps the permutation
        // deterministic and keeps siblings adjacent, which is what makes a
        // parent's two child rows land near each other in memory.
        let mut order: Vec<u32> = (n_leaves..n_nodes).map(|i| i as u32).collect();
        order.sort_unstable_by_key(|&i| (height[i as usize], i));

        let mut relabel: Vec<u32> = (0..n_nodes as u32).collect();
        for (slot, &old) in order.iter().enumerate() {
            relabel[old as usize] = (n_leaves + slot) as u32;
        }

        let mut new_parent = vec![NO_NODE; n_nodes];
        let mut new_branch = vec![0.0f64; n_nodes];
        for old in 0..n_nodes {
            let new = relabel[old] as usize;
            new_parent[new] = match parent[old] {
                NO_NODE => NO_NODE,
                par => relabel[par as usize],
            };
            new_branch[new] = branch[old];
        }
        let parent = new_parent;
        let branch = new_branch;

        // Level boundaries in the relabelled indexing.
        let mut level_ptr = vec![n_leaves as u32];
        for w in order.windows(2) {
            if height[w[1] as usize] != height[w[0] as usize] {
                level_ptr.push(relabel[w[1] as usize]);
            }
        }
        level_ptr.push(n_nodes as u32);

        // The invariant guarantees acyclicity and connectedness: every non-root
        // node points strictly upwards in index, so following parents always
        // terminates, and it can only terminate at the single root.
        let mut counts = vec![0u32; n_nodes + 1];
        for &par in &parent {
            if par != NO_NODE {
                counts[par as usize + 1] += 1;
            }
        }
        for i in 0..n_nodes {
            counts[i + 1] += counts[i];
        }
        let child_ptr = counts.clone();
        let mut cursor = counts;
        let mut children = vec![0u32; n_nodes - 1];
        for (i, &par) in parent.iter().enumerate() {
            if par != NO_NODE {
                children[cursor[par as usize] as usize] = i as u32;
                cursor[par as usize] += 1;
            }
        }

        for i in n_leaves..n_nodes {
            let n_child = (child_ptr[i + 1] - child_ptr[i]) as usize;
            if n_child < 2 {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!("internal node {i} has {n_child} children, expected at least 2"),
                });
            }
        }
        for i in 0..n_leaves {
            if child_ptr[i + 1] != child_ptr[i] {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!("leaf {i} has children"),
                });
            }
        }

        Ok(Self {
            parent,
            branch,
            n_leaves,
            child_ptr,
            children,
            level_ptr,
        })
    }

    /// Number of levels above the leaves.
    ///
    /// ### Returns
    ///
    /// The level count.
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.level_ptr.len() - 1
    }

    /// The half-open node range covered by one level.
    ///
    /// Every node in the range has all of its children strictly below the
    /// range's start, so the nodes in a level can be processed concurrently.
    ///
    /// ### Params
    ///
    /// * `level` - Level index, counting up from the leaves
    ///
    /// ### Returns
    ///
    /// The `(start, end)` node indices of that level.
    #[inline]
    pub fn level(&self, level: usize) -> (usize, usize) {
        (
            self.level_ptr[level] as usize,
            self.level_ptr[level + 1] as usize,
        )
    }

    /// Total number of nodes, leaves included.
    ///
    /// ### Returns
    ///
    /// The node count.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.parent.len()
    }

    /// Number of leaves.
    ///
    /// ### Returns
    ///
    /// The leaf count.
    #[inline]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Index of the root, which is always the highest-numbered node.
    ///
    /// ### Returns
    ///
    /// The root's index.
    #[inline]
    pub fn root(&self) -> u32 {
        (self.n_nodes() - 1) as u32
    }

    /// Children of a node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose children are wanted
    ///
    /// ### Returns
    ///
    /// The children, empty for a leaf.
    #[inline]
    pub fn children(&self, node: u32) -> &[u32] {
        let lo = self.child_ptr[node as usize] as usize;
        let hi = self.child_ptr[node as usize + 1] as usize;
        &self.children[lo..hi]
    }

    /// Parent of a node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose parent is wanted
    ///
    /// ### Returns
    ///
    /// The parent, or `None` for the root.
    #[inline]
    pub fn parent(&self, node: u32) -> Option<u32> {
        match self.parent[node as usize] {
            NO_NODE => None,
            p => Some(p),
        }
    }

    /// Length of the branch above a node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose upstream branch is wanted
    ///
    /// ### Returns
    ///
    /// The branch length; meaningless for the root.
    #[inline]
    pub fn branch(&self, node: u32) -> f64 {
        self.branch[node as usize]
    }

    /// All branch lengths, indexed by node.
    ///
    /// ### Returns
    ///
    /// The branch lengths.
    #[inline]
    pub fn branches(&self) -> &[f64] {
        &self.branch
    }

    /// All branch lengths, mutably, for the optimiser.
    ///
    /// ### Returns
    ///
    /// The branch lengths.
    #[inline]
    pub fn branches_mut(&mut self) -> &mut [f64] {
        &mut self.branch
    }

    /// Internal nodes in post-order.
    ///
    /// By the arena invariant this is simply ascending index order, so it costs
    /// nothing to produce.
    ///
    /// ### Returns
    ///
    /// An iterator over the internal nodes, children before parents.
    #[inline]
    pub fn internal_postorder(&self) -> impl Iterator<Item = u32> + '_ {
        (self.n_leaves..self.n_nodes()).map(|i| i as u32)
    }

    /// Build a balanced binary tree over `n_leaves` leaves.
    ///
    /// `n_leaves` must be a power of two. Used by tests and fixtures; the
    /// resulting node numbering satisfies the arena invariant by construction.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, a power of two and at least two
    /// * `branch` - Branch length assigned to every non-root node
    ///
    /// ### Returns
    ///
    /// The tree.
    pub fn balanced_binary(n_leaves: usize, branch: f64) -> Result<Self, BonsaiErrors> {
        if n_leaves < 2 || !n_leaves.is_power_of_two() {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("{n_leaves} is not a power of two of at least two"),
            });
        }
        let n_nodes = 2 * n_leaves - 1;
        let mut parent = vec![NO_NODE; n_nodes];

        // Pair up each level in turn, allocating ancestors upwards.
        let mut level: Vec<u32> = (0..n_leaves as u32).collect();
        let mut next_free = n_leaves as u32;
        while level.len() > 1 {
            let mut up = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks_exact(2) {
                let a = next_free;
                next_free += 1;
                parent[pair[0] as usize] = a;
                parent[pair[1] as usize] = a;
                up.push(a);
            }
            level = up;
        }

        Self::from_parents(parent, vec![branch; n_nodes], n_leaves)
    }

    /// Build a ladder (caterpillar) tree over `n_leaves` leaves.
    ///
    /// Every internal node has one leaf child and one internal child, so the
    /// tree is maximally deep and every level holds exactly one node. This is
    /// the worst case for the level-parallel sweep, which is exactly why it is
    /// worth having: the Supplementary Information notes that biological trees
    /// can be deep and laddery, so it bounds what tree shape can cost.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, at least two
    /// * `branch` - Branch length assigned to every non-root node
    ///
    /// ### Returns
    ///
    /// The tree.
    pub fn ladder(n_leaves: usize, branch: f64) -> Result<Self, BonsaiErrors> {
        if n_leaves < 2 {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("{n_leaves} leaves is not enough for a ladder"),
            });
        }
        let n_nodes = 2 * n_leaves - 1;
        let mut parent = vec![NO_NODE; n_nodes];

        // Internal node `n_leaves + i` joins the previous rung to leaf `i + 2`,
        // except the first, which joins leaves 0 and 1.
        parent[0] = n_leaves as u32;
        parent[1] = n_leaves as u32;
        for i in 1..n_leaves - 1 {
            let rung = (n_leaves + i) as u32;
            parent[n_leaves + i - 1] = rung;
            parent[i + 1] = rung;
        }

        Self::from_parents(parent, vec![branch; n_nodes], n_leaves)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_binary_shape() {
        let tree = Tree::balanced_binary(8, 1.0).unwrap();
        assert_eq!(tree.n_leaves(), 8);
        assert_eq!(tree.n_nodes(), 15);
        assert_eq!(tree.root(), 14);
        assert!(tree.parent(tree.root()).is_none());
        for i in 0..8 {
            assert!(tree.children(i).is_empty());
        }
        for i in 8..15 {
            assert_eq!(tree.children(i).len(), 2);
        }
    }

    #[test]
    fn test_postorder_visits_children_first() {
        let tree = Tree::balanced_binary(16, 0.5).unwrap();
        let mut seen = vec![false; tree.n_nodes()];
        for i in 0..tree.n_leaves() {
            seen[i] = true;
        }
        for node in tree.internal_postorder() {
            for &c in tree.children(node) {
                assert!(seen[c as usize], "child {c} not visited before parent {node}");
            }
            seen[node as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_rejects_parent_with_smaller_index() {
        // Node 2 is the root, node 3 hangs below it: the parent index is
        // smaller than the child's, which breaks the post-order invariant.
        let parent = vec![2, 2, NO_NODE, 2];
        let err = Tree::from_parents(parent, vec![1.0; 4], 2);
        assert!(matches!(err, Err(BonsaiErrors::MalformedTree { .. })));
    }

    #[test]
    fn test_rejects_two_roots() {
        let parent = vec![2, 2, NO_NODE, NO_NODE];
        let err = Tree::from_parents(parent, vec![1.0; 4], 2);
        assert!(matches!(err, Err(BonsaiErrors::MalformedTree { .. })));
    }

    #[test]
    fn test_ladder_is_maximally_deep() {
        let tree = Tree::ladder(32, 1.0).unwrap();
        assert_eq!(tree.n_leaves(), 32);
        assert_eq!(tree.n_nodes(), 63);
        // One node per level, so no two internal nodes are independent.
        assert_eq!(tree.n_levels(), 31);
        for level in 0..tree.n_levels() {
            let (start, end) = tree.level(level);
            assert_eq!(end - start, 1);
        }
        for node in tree.internal_postorder() {
            assert_eq!(tree.children(node).len(), 2);
        }
    }

    #[test]
    fn test_star_tree_is_valid() {
        // Four leaves all hanging off a single root.
        let parent = vec![4, 4, 4, 4, NO_NODE];
        let tree = Tree::from_parents(parent, vec![1.0; 5], 4).unwrap();
        assert_eq!(tree.children(tree.root()).len(), 4);
        assert_eq!(tree.internal_postorder().count(), 1);
    }
}
