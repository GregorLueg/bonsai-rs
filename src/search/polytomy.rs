//! Polytomy resolution (SPEC.md section 9.2) and the splice primitive the rest
//! of the search is built on.
//!
//! Steps 3, 5 and 6 build a star around a node `X` ([`centre_star`]), resolve
//! it with [`crate::search::star::resolve_star`] and splice the result back
//! ([`splice_star`]). The star is `X`'s children plus its upstream side as one
//! more member, read off [`UpState`], so "stop at three members" means
//! "resolved" for internal nodes and the root alike.
//!
//! The upstream member is not a node: it stands for everything outside `X`'s
//! subtree and sits where `X`'s parent `P` sits. Anything attached to it
//! attaches to `P`; only ancestors strictly between `X` and the upstream member
//! become new nodes, on the edge above `X`. The path from the centre to the
//! upstream member is the only part of the star topology reversed on the way
//! back in.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::search::Leaves;
use crate::search::spr::{LazyRows, RowStore, lazy_centre_star};
use crate::search::star::{Star, StarParams, StarResult, resolve_star};
use crate::tree::{NO_NODE, Tree};
use crate::utils::traits::BonsaiFloat;

////////////////
// Parameters //
////////////////

/// Star size at which a centre is already resolved (SPEC.md section 9.1).
///
/// Counting the upstream side as a member makes one number serve root and
/// internal nodes. Read by [`crate::search::spr`] to detect regrafts that leave
/// a polytomy.
pub(crate) const RESOLVED_STAR_MEMBERS: usize = 3;

/// Runaway guard on the fixed-point loop; termination rests on the
/// loglikelihood, so this should never bind (zero-branch stars settle in a
/// handful of sweeps, measured).
///
/// [`resolve_polytomies`] rejects a non-positive `min_gain`; a floor merely too
/// small for the feature count cannot be detected.
const MAX_SWEEPS: usize = 64;

///////////////////
// Input, output //
///////////////////

/// The star around one node of a tree.
///
/// Row-major `[member][feature]`, precisions not diffusion corrected, matching
/// [`Star`]. The upstream member, if any, is last, with the centre's parent in
/// its `member_nodes` slot.
#[derive(Clone, Debug)]
pub struct CentreStar<T> {
    /// Node the star is centred on.
    pub centre: u32,
    /// Tree node each member stands for, upstream last.
    pub member_nodes: Vec<u32>,
    /// Whether the last member is the upstream side rather than a child.
    pub has_upstream: bool,
    /// Nodes the star swallowed, removed from the tree on splice (an
    /// interchange lists the collapsed end of its edge here).
    pub deleted: Vec<u32>,
    /// Effective means, `[member][feature]`, row-major.
    pub means: Vec<T>,
    /// Effective precisions, same layout.
    pub precisions: Vec<T>,
    /// Branch from each member to the centre.
    pub branch: Vec<f64>,
    /// Number of features.
    pub n_features: usize,
}

impl<T: BonsaiFloat> CentreStar<T> {
    /// Borrow the star in the form the primitive takes.
    ///
    /// ### Returns
    ///
    /// The star view.
    pub fn view(&self) -> Star<'_, T> {
        Star {
            means: &self.means,
            precisions: &self.precisions,
            branch: &self.branch,
            n_features: self.n_features,
        }
    }

    /// Whether resolving this star could change anything.
    ///
    /// ### Returns
    ///
    /// True when the centre carries more members than the primitive stops at.
    pub fn is_polytomy(&self) -> bool {
        self.member_nodes.len() > RESOLVED_STAR_MEMBERS
    }
}

/// What one splice did.
#[derive(Clone, Debug)]
pub struct Splice {
    /// The tree with the centre's star resolved and spliced back.
    pub tree: Tree,
    /// Total loglikelihood gain claimed by the merges, in nats.
    ///
    /// Exact against the tree the star was built from (SPEC.md section 4), not
    /// against the caller's original where it was edited first.
    pub gain: f64,
    /// Number of merges the primitive performed. Zero means the tree came back
    /// unchanged.
    pub n_merges: usize,
}

/// What a run of [`resolve_polytomies`] did.
#[derive(Clone, Debug)]
pub struct PolytomyResult {
    /// The tree with every polytomy resolved as far as it will go.
    pub tree: Tree,
    /// Number of polytomies the input tree had, once its zero-length internal
    /// edges were collapsed into their parents.
    pub n_polytomies: usize,
    /// Number of resolutions that changed the tree.
    ///
    /// Routinely larger than `n_polytomies`: a new ancestor left at zero
    /// distance from its centre is collapsed next sweep into a fresh polytomy.
    pub n_resolved: usize,
    /// Number of sweeps over the tree, the last of which found nothing.
    pub sweeps: usize,
}

///////////////////
// The primitive //
///////////////////

/// Where every node of a rebuilt tree came from.
///
/// Walks up from each leaf in the input parent array and in the rebuilt tree
/// in step; leaves keep their indices.
///
/// ### Params
///
/// * `parent` - The parent array [`rebuild`] was given
/// * `out` - The tree it built
/// * `n_old` - Node count of the tree the array was edited from; input indices
///   at or above it are nodes the edit created
///
/// ### Returns
///
/// Per node of `out`, its node in the original tree or [`NO_NODE`] for a
/// created one, or `MalformedTree` if the two walks disagree.
fn map_back(parent: &[u32], out: &Tree, n_old: usize) -> Result<Vec<u32>, BonsaiErrors> {
    let mut to_old = vec![NO_NODE; out.n_nodes()];
    for leaf in 0..out.n_leaves() as u32 {
        let (mut from, mut to) = (leaf, leaf);
        while to_old[to as usize] == NO_NODE {
            to_old[to as usize] = from;
            match (parent[from as usize], out.parent(to)) {
                (NO_NODE, None) => break,
                (up, Some(next)) if up != NO_NODE => (from, to) = (up, next),
                _ => {
                    return Err(BonsaiErrors::MalformedTree {
                        reason: format!("rebuild map lost step at leaf {leaf}"),
                    });
                }
            }
        }
    }
    for old in &mut to_old {
        if *old != NO_NODE && *old as usize >= n_old {
            *old = NO_NODE;
        }
    }
    Ok(to_old)
}

/// Renumber an arbitrary parent array into the arena invariant and build it.
///
/// [`Tree::from_parents`] checks but does not fix that a parent index exceeds
/// its children's, which a splice breaks, so nodes are renumbered in a
/// post-order from the root. Unreached nodes (deleted ones) are dropped.
///
/// ### Params
///
/// * `parent` - Parent index per node, [`NO_NODE`] where there is none
/// * `branch` - Branch above each node, same indexing
/// * `root` - Node to walk from
/// * `n_leaves` - Number of leaves, occupying indices `0..n_leaves`
///
/// ### Returns
///
/// The tree, or `MalformedTree` if the walk did not reach every leaf or the
/// arena rejected the result.
fn rebuild(
    parent: &[u32],
    branch: &[f64],
    root: u32,
    n_leaves: usize,
) -> Result<Tree, BonsaiErrors> {
    let n = parent.len();

    let mut ptr = vec![0u32; n + 1];
    for &par in parent {
        if par != NO_NODE {
            ptr[par as usize + 1] += 1;
        }
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let mut cursor = ptr.clone();
    let mut kids = vec![0u32; ptr[n] as usize];
    for (i, &par) in parent.iter().enumerate() {
        if par != NO_NODE {
            kids[cursor[par as usize] as usize] = i as u32;
            cursor[par as usize] += 1;
        }
    }

    let mut new_id = vec![NO_NODE; n];
    let mut next = n_leaves as u32;
    let mut n_seen_leaves = 0usize;
    let mut stack: Vec<(u32, bool)> = vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            if (node as usize) < n_leaves {
                new_id[node as usize] = node;
                n_seen_leaves += 1;
            } else {
                new_id[node as usize] = next;
                next += 1;
            }
            continue;
        }
        stack.push((node, true));
        let (lo, hi) = (ptr[node as usize] as usize, ptr[node as usize + 1] as usize);
        // Pushed last-first so siblings keep their arena order; `search::spr`
        // relies on it to recognise untouched subtrees.
        for &child in kids[lo..hi].iter().rev() {
            stack.push((child, false));
        }
    }
    if n_seen_leaves != n_leaves {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "the splice left {n_seen_leaves} of {n_leaves} leaves connected to the root"
            ),
        });
    }

    let n_new = next as usize;
    let mut new_parent = vec![NO_NODE; n_new];
    let mut new_branch = vec![0.0f64; n_new];
    for old in 0..n {
        let here = new_id[old];
        if here == NO_NODE {
            continue;
        }
        new_parent[here as usize] = match parent[old] {
            NO_NODE => NO_NODE,
            par => new_id[par as usize],
        };
        new_branch[here as usize] = branch[old];
    }
    Tree::from_parents(new_parent, new_branch, n_leaves)
}

/// Write a resolved star into a parent array, appending its new nodes.
///
/// [`splice_result`] without the rebuild, so non-overlapping stars can share
/// one rebuild.
///
/// ### Params
///
/// * `parent` - Parent per node, grown by the star's new internal nodes
/// * `branch` - Branch above each node, same indexing
/// * `star` - The star that was resolved, in the array's node ids
/// * `result` - What the primitive built
fn apply_splice<T: BonsaiFloat>(
    parent: &mut Vec<u32>,
    branch: &mut Vec<f64>,
    star: &CentreStar<T>,
    result: &StarResult<T>,
) {
    let old = parent.len();
    let n_new = result.parent.len() - star.member_nodes.len();
    parent.resize(old + n_new, NO_NODE);
    branch.resize(old + n_new, 0.0);

    // Deleted nodes keep no parent and gain no children, so the rebuild never
    // reaches them.
    for &node in &star.deleted {
        parent[node as usize] = NO_NODE;
    }
    for (node, up, t) in splice_edits(star, result, old as u32) {
        parent[node as usize] = up;
        branch[node as usize] = t;
    }
}

/// [`splice_result`], plus each new node's node in the original tree, via
/// [`map_back`].
///
/// ### Params
///
/// * `tree` - The tree the star was built from
/// * `star` - The star that was resolved
/// * `result` - What the primitive built
///
/// ### Returns
///
/// The spliced tree and, per node of it, its node in `tree` or [`NO_NODE`].
fn splice_result_mapped<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    result: &StarResult<T>,
) -> Result<(Tree, Vec<u32>), BonsaiErrors> {
    let mut parent: Vec<u32> = (0..tree.n_nodes())
        .map(|i| tree.parent(i as u32).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    apply_splice(&mut parent, &mut branch, star, result);
    let out = rebuild(&parent, &branch, tree.root(), tree.n_leaves())?;
    let to_old = map_back(&parent, &out, tree.n_nodes())?;
    Ok((out, to_old))
}

/// The parent and branch every node a resolved star touches ends up with.
///
/// New ancestors are numbered `first_new` upwards in creation order. Shared
/// with the masked views of [`crate::search::masked`].
///
/// ### Params
///
/// * `star` - The star that was resolved
/// * `result` - What the primitive built
/// * `first_new` - Id of the first new ancestor
///
/// ### Returns
///
/// One `(node, parent, branch)` per node whose parent or branch the splice
/// sets.
pub(crate) fn splice_edits<T: BonsaiFloat>(
    star: &CentreStar<T>,
    result: &StarResult<T>,
    first_new: u32,
) -> Vec<(u32, u32, f64)> {
    let n = star.member_nodes.len();
    let n_local = result.parent.len();
    let map = |i: usize| -> u32 {
        if i < n {
            star.member_nodes[i]
        } else {
            first_new + (i - n) as u32
        }
    };
    let mut edits = Vec::with_capacity(n_local + 1);

    // Ancestors between centre and upstream member, nearest the upstream
    // member first; their direction flips.
    let upstream = star.has_upstream.then(|| n - 1);
    let mut chain: Vec<u32> = Vec::new();
    let mut on_chain = vec![false; n_local];
    if let Some(u) = upstream {
        let mut cur = result.parent[u];
        while cur != NO_NODE {
            chain.push(cur);
            on_chain[cur as usize] = true;
            cur = result.parent[cur as usize];
        }
    }

    for i in 0..n_local {
        if Some(i) == upstream || on_chain[i] {
            continue;
        }
        let up = match result.parent[i] {
            NO_NODE => star.centre,
            up => map(up as usize),
        };
        edits.push((map(i), up, result.branch[i]));
    }

    if let Some(u) = upstream {
        let above = star.member_nodes[u];
        match chain.split_first() {
            None => edits.push((star.centre, above, result.branch[u])),
            Some((&top, _)) => {
                edits.push((map(top as usize), above, result.branch[u]));
                for j in 1..chain.len() {
                    edits.push((
                        map(chain[j] as usize),
                        map(chain[j - 1] as usize),
                        result.branch[chain[j - 1] as usize],
                    ));
                }
                let bottom = chain[chain.len() - 1] as usize;
                edits.push((star.centre, map(bottom), result.branch[bottom]));
            }
        }
    }
    edits
}

/// Map a resolved star back onto tree node ids and rebuild the arena.
///
/// Local index `i < n_members` is the member's own node (the upstream slot is
/// the centre's parent); `n_members + a` is a new internal node.
///
/// ### Params
///
/// * `tree` - The tree the star was built from
/// * `star` - The star that was resolved
/// * `result` - What the primitive built
///
/// ### Returns
///
/// The spliced tree, or the error the arena failed with.
pub(crate) fn splice_result<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    result: &StarResult<T>,
) -> Result<Tree, BonsaiErrors> {
    let mut parent: Vec<u32> = (0..tree.n_nodes())
        .map(|i| tree.parent(i as u32).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    apply_splice(&mut parent, &mut branch, star, result);
    rebuild(&parent, &branch, tree.root(), tree.n_leaves())
}

/// Build the star around a node: its children, plus its upstream side.
///
/// The upstream member is [`UpState`]'s row for the centre, with the centre's
/// own branch, so nothing is transformed on the way in.
///
/// ### Params
///
/// * `tree` - The tree
/// * `down` - Down rows, settled by [`NodeState::prune`] against this tree
/// * `up` - Up rows, settled by [`UpState::sweep`] against the same
/// * `centre` - Internal node to build the star around
///
/// ### Returns
///
/// The star, or `NodeOutOfRange` for an index outside the arena, or
/// `MalformedTree` if the centre is a leaf.
pub fn centre_star<T: BonsaiFloat>(
    tree: &Tree,
    down: &NodeState<T>,
    up: &UpState<T>,
    centre: u32,
) -> Result<CentreStar<T>, BonsaiErrors> {
    if centre as usize >= tree.n_nodes() {
        return Err(BonsaiErrors::NodeOutOfRange {
            index: centre as usize,
            n_nodes: tree.n_nodes(),
        });
    }
    let kids = tree.children(centre);
    if kids.is_empty() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("node {centre} is a leaf and has no star around it"),
        });
    }

    let p = down.n_features();
    let parent = tree.parent(centre);
    let n = kids.len() + usize::from(parent.is_some());
    let mut star = CentreStar {
        centre,
        member_nodes: Vec::with_capacity(n),
        has_upstream: parent.is_some(),
        deleted: Vec::new(),
        means: Vec::with_capacity(n * p),
        precisions: Vec::with_capacity(n * p),
        branch: Vec::with_capacity(n),
        n_features: p,
    };

    for &child in kids {
        star.member_nodes.push(child);
        star.means.extend_from_slice(down.means(child));
        star.precisions.extend_from_slice(down.precisions(child));
        star.branch.push(tree.branch(child));
    }
    if let Some(par) = parent {
        star.member_nodes.push(par);
        star.means.extend_from_slice(up.means(centre));
        star.precisions.extend_from_slice(up.precisions(centre));
        star.branch.push(tree.branch(centre));
    }
    Ok(star)
}

/// Resolve a star and splice the result back into the tree.
///
/// ### Params
///
/// * `tree` - The tree the star was built from
/// * `star` - The star, from [`centre_star`] or from an interchange's collapse
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The new tree and what the resolution gained, or the error the primitive or
/// the arena failed with.
pub fn splice_star<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    params: Option<StarParams>,
) -> Result<Splice, BonsaiErrors> {
    let result = resolve_star(star.view(), params)?;
    let gain = result.merges.iter().map(|x| x.gain).sum();
    let n_merges = result.merges.len();
    let tree = splice_result(tree, star, &result)?;
    Ok(Splice {
        tree,
        gain,
        n_merges,
    })
}

////////////////////////////
// Step 3: the polytomies //
////////////////////////////

/// Whether a node's star could change anything, read off the tree without
/// building it (same test as [`CentreStar::is_polytomy`]).
///
/// ### Params
///
/// * `tree` - The tree
/// * `node` - Internal node to test
///
/// ### Returns
///
/// True when the node carries more members than the primitive stops at.
fn is_polytomy(tree: &Tree, node: u32) -> bool {
    tree.children(node).len() + usize::from(tree.parent(node).is_some()) > RESOLVED_STAR_MEMBERS
}

/// Count the nodes whose star is bigger than the primitive stops at.
///
/// ### Params
///
/// * `tree` - The tree
///
/// ### Returns
///
/// The number of polytomies, the root's trifurcation not among them.
fn count_polytomies(tree: &Tree) -> usize {
    tree.internal_postorder()
        .filter(|&node| is_polytomy(tree, node))
        .count()
}

/// Collapse every zero-length internal edge into the node above it.
///
/// The merge scan leaves structurally binary nodes joined by exact-zero edges
/// ([`crate::model::branch::optimise_edge`] and the bracket ends in
/// [`crate::model::merge`] return a literal `0.0`, so the test is exact, not a
/// tolerance). Collapsing leaves the loglikelihood unchanged and exposes the
/// polytomies step 3 resolves.
///
/// ### Params
///
/// * `tree` - Tree to collapse; not modified
///
/// ### Returns
///
/// The collapsed tree and, per node of it, its node in `tree`; or `None` if
/// there was no zero-length internal edge, or the error the arena rejected the
/// rebuild with.
pub(crate) fn collapse_zero_edges(tree: &Tree) -> Result<Option<(Tree, Vec<u32>)>, BonsaiErrors> {
    let n = tree.n_nodes();
    let n_leaves = tree.n_leaves();
    let root = tree.root();

    // Leaves carry an observation and are never collapsed; the root has no
    // upstream branch.
    let drop: Vec<bool> = (0..n)
        .map(|i| i >= n_leaves && i as u32 != root && tree.branch(i as u32) == 0.0)
        .collect();
    if !drop.iter().any(|&d| d) {
        return Ok(None);
    }

    let mut parent = vec![NO_NODE; n];
    let mut branch = vec![0.0f64; n];
    for i in 0..n {
        if drop[i] {
            continue;
        }
        // Walk past every dropped ancestor: a chain collapses onto the node above it.
        let mut up = tree.parent(i as u32);
        while let Some(par) = up {
            if !drop[par as usize] {
                break;
            }
            up = tree.parent(par);
        }
        parent[i] = up.unwrap_or(NO_NODE);
        branch[i] = tree.branch(i as u32);
    }

    let out = rebuild(&parent, &branch, root, n_leaves)?;
    let to_old = map_back(&parent, &out, n)?;
    Ok(Some((out, to_old)))
}

/// Resolve every polytomy in a tree (SPEC.md section 9.2).
///
/// Each sweep collapses zero-length edges ([`collapse_zero_edges`]), then
/// resolves the first node (ascending index, a post-order) whose star exceeds
/// [`RESOLVED_STAR_MEMBERS`], and restarts because [`Tree::from_parents`]
/// renumbers internal nodes. It runs to a fixed point: termination rests on
/// every accepted resolution raising the loglikelihood by more than `min_gain`,
/// the collapse leaving it alone, and finitely many topologies. Collapsing every
/// sweep gave 3 to 4 times as many resolutions and closer recovery than
/// collapsing once on entry.
///
/// Down rows are settled once and kept in a [`RowStore`]; stars read up rows
/// through [`LazyRows`], so a sweep costs its stars and `O(depth p)` chains
/// rather than a full settle (69 s to 3.5 s on 25k cells, 2026-09-26, same
/// loglikelihood). One resolution per sweep, since each changes every up row.
///
/// ### Params
///
/// * `tree` - Tree to resolve; not modified
/// * `leaves` - The leaf data the tree is scored against
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The resolved tree and what it gained, or the error the primitive or the
/// arena failed with.
pub fn resolve_polytomies<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    params: Option<StarParams>,
) -> Result<PolytomyResult, BonsaiErrors> {
    // At zero the primitive accepts rounding-artefact merges, the next collapse
    // folds them back and the same merge recurs forever.
    let min_gain = params.unwrap_or_default().min_gain;
    if !min_gain.is_finite() || min_gain <= 0.0 {
        return Err(BonsaiErrors::BadParameter {
            name: "StarParams::min_gain",
            value: min_gain,
            expected: "a finite value strictly greater than zero when resolving polytomies; see \
                       DEFAULT_MIN_GAIN for the per-feature rounding floor it must clear",
        });
    }
    let mut tree = match collapse_zero_edges(tree)? {
        Some((collapsed, _)) => collapsed,
        None => tree.clone(),
    };
    let n_polytomies = count_polytomies(&tree);
    let mut n_resolved = 0usize;
    let mut sweeps = 0usize;

    let (down, _) = crate::search::settled_down(&tree, leaves)?;
    let mut store = RowStore::from_state(&down, tree.n_nodes());
    drop(down);

    loop {
        sweeps += 1;
        // No-op on the first sweep; later resolutions can create zero edges.
        if let Some((collapsed, to_old)) = collapse_zero_edges(&tree)? {
            store.accept(&collapsed, &to_old, &tree)?;
            tree = collapsed;
        }
        let identity: Vec<u32> = (0..tree.n_nodes() as u32).collect();
        let rows = LazyRows::new(&tree, &identity, &tree, &store)?;

        let mut accepted: Option<(Tree, Vec<u32>)> = None;
        for node in tree.internal_postorder() {
            // Degree test off the tree first: building a star copies `O(deg p)` rows.
            if !is_polytomy(&tree, node) {
                continue;
            }
            let star = lazy_centre_star(&tree, &rows, node)?;
            // Splicing is `O(n)`, so resolve first and splice only a kept result.
            let result = resolve_star(star.view(), params)?;
            if !result.merges.is_empty() {
                let (next, to_old) = splice_result_mapped(&tree, &star, &result)?;
                accepted = Some((next, to_old));
                break;
            }
        }
        drop(rows);
        match accepted {
            None => break,
            Some((next, to_old)) => {
                store.accept(&next, &to_old, &tree)?;
                n_resolved += 1;
                tree = next;
            }
        }
        if sweeps >= MAX_SWEEPS {
            break;
        }
    }

    Ok(PolytomyResult {
        tree,
        n_polytomies,
        n_resolved,
        sweeps,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::simulate::{
        SimulationParams, robinson_foulds, simulate_binary_random_branches, splits,
    };
    use approx::assert_relative_eq;

    /// Settle a tree's down and up rows against leaf data.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The settled down rows, the settled up rows and the tree loglikelihood.
    fn settle(tree: &Tree, leaves: Leaves<'_, f64>) -> (NodeState<f64>, UpState<f64>, f64) {
        crate::search::settle(tree, leaves).expect("state")
    }

    /// Loglikelihood of a tree, computed from nothing but the leaf data.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `leaves` - The leaf data
    ///
    /// ### Returns
    ///
    /// The tree loglikelihood.
    fn loglik(tree: &Tree, leaves: Leaves<'_, f64>) -> f64 {
        crate::search::tree_loglik(tree, leaves).expect("state")
    }

    /// Branch length below which the fixtures collapse an internal edge.
    ///
    /// The random-branch simulator draws `log(t)` uniformly on `[log 0.5, log 2]`,
    /// so this cuts roughly the shorter third of the internal edges and leaves a
    /// tree with several polytomies of a few members each.
    const COLLAPSE_BELOW: f64 = 0.8;

    /// A small simulated dataset and the tree it was generated on.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, a power of two
    /// * `n_features` - Number of features
    /// * `seed` - Simulation seed
    ///
    /// ### Returns
    ///
    /// The truth tree, the means and the precisions.
    fn dataset(n_leaves: usize, n_features: usize, seed: u64) -> (Tree, Vec<f64>, Vec<f64>) {
        let data = simulate_binary_random_branches::<f64>(Some(SimulationParams {
            n_leaves,
            n_features,
            seed,
            ..SimulationParams::default()
        }))
        .expect("simulate");
        let precisions = data.precisions();
        (data.tree, data.means, precisions)
    }

    /// A star tree over `n_leaves` leaves: one giant polytomy at the root.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves
    /// * `branch` - Branch length given to every leaf
    ///
    /// ### Returns
    ///
    /// The tree.
    fn star_shaped(n_leaves: usize, branch: f64) -> Tree {
        let mut parent = vec![n_leaves as u32; n_leaves + 1];
        parent[n_leaves] = NO_NODE;
        Tree::from_parents(parent, vec![branch; n_leaves + 1], n_leaves).expect("star tree")
    }

    /// Count the internal edges of exactly zero length.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    ///
    /// ### Returns
    ///
    /// The count, the root and the leaves excluded.
    fn zero_internal_edges(tree: &Tree) -> usize {
        (0..tree.n_nodes())
            .filter(|&i| {
                i >= tree.n_leaves()
                    && tree.parent(i as u32).is_some()
                    && tree.branch(i as u32) == 0.0
            })
            .count()
    }

    /// Collapse every internal edge shorter than a threshold, making
    /// polytomies.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `keep` - Only edges above this length survive
    ///
    /// ### Returns
    ///
    /// The collapsed tree.
    fn collapse_short_edges(tree: &Tree, keep: f64) -> Tree {
        let cut: Vec<bool> = (0..tree.n_nodes())
            .map(|i| {
                let node = i as u32;
                tree.parent(node).is_some()
                    && !tree.children(node).is_empty()
                    && tree.branch(node) < keep
            })
            .collect();

        let mut parent = vec![NO_NODE; tree.n_nodes()];
        let mut branch = vec![0.0f64; tree.n_nodes()];
        for i in 0..tree.n_nodes() {
            if cut[i] {
                continue;
            }
            // Walk up to the nearest ancestor that survives, absorbing the
            // lengths of the edges that were cut on the way.
            let mut here = tree.parent(i as u32);
            let mut length = tree.branch(i as u32);
            while let Some(node) = here {
                if !cut[node as usize] {
                    break;
                }
                length += tree.branch(node);
                here = tree.parent(node);
            }
            parent[i] = here.unwrap_or(NO_NODE);
            branch[i] = if here.is_some() { length } else { 0.0 };
        }
        rebuild(&parent, &branch, tree.root(), tree.n_leaves()).expect("collapsed tree")
    }

    #[test]
    fn test_a_resolution_with_nothing_to_gain_returns_the_same_tree() {
        // The reroot test. The upstream member is a stand-in for everything
        // above the centre and must never become a node; if it does, the splice
        // duplicates the centre's parent and the tree silently changes shape.
        // Nothing here is a polytomy, so every splice must be the identity.
        let (p, n) = (48usize, 16usize);
        let (tree, m, w) = dataset(n, p, 7);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let (down, up, before) = settle(&tree, leaves);
        let want = splits(&tree);

        for node in tree.internal_postorder() {
            let star = centre_star(&tree, &down, &up, node).expect("star");
            let spliced = splice_star(&tree, &star, None).expect("splice");
            assert_eq!(spliced.n_merges, 0, "node {node} merged a resolved star");
            assert_eq!(
                splits(&spliced.tree),
                want,
                "the splice at node {node} changed the topology"
            );
            assert_eq!(spliced.tree.n_nodes(), tree.n_nodes());
            assert_relative_eq!(loglik(&spliced.tree, leaves), before, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_an_ancestor_can_land_on_the_edge_above_the_centre() {
        // The other half of the reroot test, and the half a binary fixture
        // never reaches. Four leaves: A alone under the root, B, C and D in a
        // polytomy under X. A and B are the same point and C and D are a long
        // way off in opposite directions, so the merge the star wants is B with
        // the *upstream* member, which stands for A.
        //
        // Spliced back, that merge is a new node N on the edge above X: the
        // root keeps A and gains N in place of X, N holds B and X, and X keeps
        // C and D. Anything that treated the upstream member as a node of its
        // own would put a copy of the root below the root and lose A.
        let p = 24usize;
        let mut m: Vec<f64> = Vec::with_capacity(4 * p);
        for g in 0..p {
            m.push((g as f64 * 0.31).sin());
        }
        let a: Vec<f64> = m.clone();
        m.extend_from_slice(&a); // B, identical to A
        m.extend(a.iter().map(|x| x + 30.0)); // C, far one way
        m.extend(a.iter().map(|x| x - 30.0)); // D, far the other
        let w = vec![1.0f64; 4 * p];
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };

        // 0 = A, 1 = B, 2 = C, 3 = D; 4 = X over {1, 2, 3}; 5 = root over {0, 4}.
        let tree = Tree::from_parents(
            vec![5, 4, 4, 4, 5, NO_NODE],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            4,
        )
        .expect("start tree");
        let (down, up, before) = settle(&tree, leaves);

        let star = centre_star(&tree, &down, &up, 4).expect("star");
        assert!(star.has_upstream);
        assert_eq!(star.member_nodes, vec![1, 2, 3, 5]);

        let spliced = splice_star(&tree, &star, None).expect("splice");
        assert_eq!(spliced.n_merges, 1);
        assert_eq!(spliced.tree.n_nodes(), tree.n_nodes() + 1);
        assert_eq!(spliced.tree.n_leaves(), 4);

        let got = spliced.tree;
        let root = got.root();
        assert_eq!(got.children(root).len(), 2);
        // A is still a child of the root, which is the assertion a reroot fails.
        assert!(got.children(root).contains(&0));
        let sibling = got
            .children(root)
            .iter()
            .copied()
            .find(|&c| c != 0)
            .expect("the root has two children");
        assert!(sibling >= got.n_leaves() as u32, "A's sibling is a leaf");
        // The new node holds B and the old centre, which still holds C and D.
        assert!(got.children(sibling).contains(&1));
        let centre = got
            .children(sibling)
            .iter()
            .copied()
            .find(|&c| c != 1)
            .expect("the new node has two children");
        let mut below: Vec<u32> = got.children(centre).to_vec();
        below.sort_unstable();
        assert_eq!(below, vec![2, 3]);

        let after = loglik(&got, leaves);
        assert!(after > before);
        assert_relative_eq!(spliced.gain, after - before, max_relative = 1e-8);
    }

    #[test]
    fn test_a_splice_keeps_every_leaf_where_it_was() {
        // Leaf indices are the caller's cell indices and must survive the
        // renumbering untouched, or every downstream comparison against the
        // ground truth is silently wrong.
        let (p, n) = (32usize, 24usize);
        let (_, m, w) = dataset(32, p, 3);
        let (m, w) = (&m[..n * p], &w[..n * p]);
        let leaves = Leaves {
            means: m,
            precisions: w,
            n_features: p,
        };
        let tree = star_shaped(n, 0.5);
        let (down, up, _) = settle(&tree, leaves);

        let star = centre_star(&tree, &down, &up, tree.root()).expect("star");
        let spliced = splice_star(&tree, &star, None).expect("splice");
        assert!(spliced.n_merges > 0);
        assert_eq!(spliced.tree.n_leaves(), n);
        for leaf in 0..n as u32 {
            assert!(spliced.tree.children(leaf).is_empty());
        }
    }

    #[test]
    fn test_the_claimed_gain_is_the_real_gain() {
        // The splice's gain is the sum of the primitive's per-merge gains,
        // which are computed against a three-leaf star. The whole-tree pruning
        // recursion knows none of that, so this pins the upstream member's
        // effective leaf as much as it pins the splice.
        let (p, n) = (40usize, 32usize);
        let (_, m, w) = dataset(n, p, 11);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let tree = collapse_short_edges(&dataset(n, p, 11).0, COLLAPSE_BELOW);
        let (down, up, before) = settle(&tree, leaves);

        let mut checked = 0usize;
        for node in tree.internal_postorder() {
            let star = centre_star(&tree, &down, &up, node).expect("star");
            if !star.is_polytomy() {
                continue;
            }
            let spliced = splice_star(&tree, &star, None).expect("splice");
            if spliced.n_merges == 0 {
                continue;
            }
            let after = loglik(&spliced.tree, leaves);
            assert_relative_eq!(spliced.gain, after - before, max_relative = 1e-7);
            checked += 1;
        }
        assert!(checked > 0, "the fixture had no polytomy to resolve");
    }

    #[test]
    fn test_resolution_never_lowers_the_loglikelihood() {
        // Monotonicity, scored independently with `NodeState::prune` on the
        // tree that came back rather than on anything the primitive reported.
        for seed in [1u64, 2, 3] {
            let (p, n) = (48usize, 32usize);
            let (truth, m, w) = dataset(n, p, seed);
            let leaves = Leaves {
                means: &m,
                precisions: &w,
                n_features: p,
            };
            let start = collapse_short_edges(&truth, COLLAPSE_BELOW);
            let before = loglik(&start, leaves);
            let out = resolve_polytomies(&start, leaves, None).expect("resolve");
            let after = loglik(&out.tree, leaves);
            assert!(
                after >= before - 1e-9,
                "seed {seed}: {before} fell to {after}"
            );
        }
    }

    #[test]
    fn test_resolution_removes_the_polytomies() {
        let (p, n) = (48usize, 32usize);
        let (truth, m, w) = dataset(n, p, 5);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let start = collapse_short_edges(&truth, COLLAPSE_BELOW);
        assert!(count_polytomies(&start) > 0, "the fixture had no polytomy");

        let out = resolve_polytomies(&start, leaves, None).expect("resolve");
        println!(
            "{} polytomies in, {} resolutions over {} sweeps, {} left",
            out.n_polytomies,
            out.n_resolved,
            out.sweeps,
            count_polytomies(&out.tree)
        );
        // A residual polytomy is legitimate: it means no pair of members gained
        // anything, which is the primitive saying the data does not resolve
        // that node. What is not legitimate is the count going up.
        assert!(
            count_polytomies(&out.tree) <= count_polytomies(&start),
            "resolution created polytomies"
        );
        assert_eq!(out.n_polytomies, count_polytomies(&start));
        assert!(out.n_resolved > 0);
        // The last sweep is the one that found nothing.
        assert!(out.sweeps >= 2);
    }

    #[test]
    fn test_a_single_giant_polytomy_resolves_to_a_binary_tree() {
        let (p, n) = (40usize, 24usize);
        let (_, m, w) = dataset(32, p, 9);
        let (m, w) = (&m[..n * p], &w[..n * p]);
        let leaves = Leaves {
            means: m,
            precisions: w,
            n_features: p,
        };
        let start = star_shaped(n, 0.5);
        let out = resolve_polytomies(&start, leaves, None).expect("resolve");

        assert!(loglik(&out.tree, leaves) > loglik(&start, leaves));
        assert_eq!(out.n_polytomies, 1);
        assert_eq!(count_polytomies(&out.tree), 0);
        assert!(loglik(&out.tree, leaves) > loglik(&start, leaves));
    }

    #[test]
    fn test_a_tree_with_no_polytomies_is_left_alone() {
        let (p, n) = (32usize, 16usize);
        let (tree, m, w) = dataset(n, p, 13);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let out = resolve_polytomies(&tree, leaves, None).expect("resolve");
        assert_eq!(out.n_polytomies, 0);
        assert_eq!(out.n_resolved, 0);
        assert_eq!(out.sweeps, 1);
        assert_eq!(splits(&out.tree), splits(&tree));
    }

    #[test]
    fn test_collapsing_a_zero_length_edge_leaves_the_loglikelihood_alone() {
        // The collapse is only allowed to change the arena, never the model. A
        // zero-length edge puts its two ends at the same point, so deleting the
        // lower end and reattaching its children one level up has to leave the
        // loglikelihood where it was, which is also the proof that the rewiring
        // is structurally right.
        let (p, n) = (32usize, 32usize);
        let (mut tree, m, w) = dataset(n, p, 13);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };

        // A deep internal node, so the collapse has a real parent to fold into.
        let victim = tree
            .internal_postorder()
            .find(|&node| tree.parent(node).is_some_and(|par| par != tree.root()))
            .expect("no internal node with an internal parent");
        let parent = tree.parent(victim).expect("victim has a parent");
        let degree_before = tree.children(parent).len();
        let kids = tree.children(victim).len();
        tree.branches_mut()[victim as usize] = 0.0;
        let before = loglik(&tree, leaves);

        let collapsed = collapse_zero_edges(&tree)
            .expect("collapse")
            .expect("the zero-length edge was not found")
            .0;
        let after = loglik(&collapsed, leaves);

        assert_eq!(collapsed.n_nodes(), tree.n_nodes() - 1);
        assert_eq!(collapsed.n_leaves(), tree.n_leaves());
        assert_relative_eq!(after, before, max_relative = 1e-14);
        // The parent inherited the collapsed node's children in place of it.
        assert!(
            collapsed
                .internal_postorder()
                .any(|node| collapsed.children(node).len() == degree_before + kids - 1),
            "no node picked up the collapsed node's children"
        );
        assert!(count_polytomies(&collapsed) > count_polytomies(&tree));
    }

    #[test]
    fn test_the_merge_scans_zero_length_branches_are_the_polytomies_step_three_resolves() {
        // Step 2 leaves a structurally binary tree whose zero-length internal
        // edges are polytomies in everything
        // but the arena, and step 3 counted degrees only and so did nothing at
        // all on it. This is the pipeline's own step 2 followed by its step 3.
        use crate::search::star::{Star, star_tree};
        let (p, n) = (64usize, 128usize);
        let (_, m, w) = dataset(n, p, 21);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let (merged, _) = star_tree(
            Star {
                means: &m,
                precisions: &w,
                branch: &vec![1.0f64; n],
                n_features: p,
            },
            None,
        )
        .expect("star tree");

        let zeros = zero_internal_edges(&merged);
        assert!(
            zeros > 0,
            "the merge scan left no zero-length internal edge"
        );
        assert_eq!(
            count_polytomies(&merged),
            0,
            "the fixture is already a structural polytomy and proves nothing"
        );

        let out = resolve_polytomies(&merged, leaves, None).expect("resolve");
        assert!(
            out.n_polytomies > 0,
            "step 3 saw no polytomy in a tree with {zeros} zero-length internal edges"
        );
        assert!(
            out.n_resolved > 0 && loglik(&out.tree, leaves) > loglik(&merged, leaves),
            "step 3 gained nothing"
        );
        assert!(loglik(&out.tree, leaves) > loglik(&merged, leaves));

        // The last sweep collapses before it scans, so nothing that reads the
        // result can find a zero-length internal edge left in it.
        let left = zero_internal_edges(&out.tree);
        assert_eq!(left, 0, "{left} zero-length internal edges survived step 3");
    }

    #[test]
    fn test_resolution_is_a_fixed_point() {
        // A second call must find nothing, which is the statement that the
        // first one ran to a fixed point rather than to a pass limit.
        let (p, n) = (48usize, 32usize);
        let (truth, m, w) = dataset(n, p, 17);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let start = collapse_short_edges(&truth, COLLAPSE_BELOW);
        let once = resolve_polytomies(&start, leaves, None).expect("resolve");
        let twice = resolve_polytomies(&once.tree, leaves, None).expect("resolve");
        assert_eq!(twice.n_resolved, 0);
        assert_eq!(splits(&twice.tree), splits(&once.tree));
    }

    #[test]
    fn test_resolution_moves_a_collapsed_tree_back_towards_the_truth() {
        // Collapsing the short internal edges throws away real splits. Putting
        // them back is what step 3 is for, so the Robinson-Foulds distance to
        // the generating tree must fall.
        let (p, n) = (256usize, 32usize);
        let (truth, m, w) = dataset(n, p, 23);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let start = collapse_short_edges(&truth, COLLAPSE_BELOW);
        let before = robinson_foulds(&start, &truth).expect("rf");
        let out = resolve_polytomies(&start, leaves, None).expect("resolve");
        let after = robinson_foulds(&out.tree, &truth).expect("rf");
        assert!(
            after < before,
            "distance to truth went from {before} to {after}"
        );
    }

    /// A zero-branch star at `min_gain = 0` used to spin forever: the primitive
    /// accepted a rounding-artefact merge, the next sweep's collapse folded it
    /// back, and the same merge was found again, without ever terminating.
    #[test]
    fn test_a_non_positive_min_gain_is_rejected_rather_than_spun_on() {
        let (p, n) = (4usize, 8usize);
        let (_, m, w) = dataset(n, p, 3);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let star = star_shaped(n, 0.0);

        for bad in [0.0, -1e-9, f64::NAN, f64::INFINITY] {
            let params = StarParams {
                min_gain: bad,
                ..Default::default()
            };
            assert!(
                matches!(
                    resolve_polytomies(&star, leaves, Some(params)),
                    Err(BonsaiErrors::BadParameter { .. })
                ),
                "min_gain = {bad} should be rejected"
            );
        }

        // The floor that ships still resolves the same fixture.
        let out = resolve_polytomies(&star, leaves, None).expect("default min_gain");
        assert!(out.sweeps <= MAX_SWEEPS, "the guard should not bind here");
    }

    #[test]
    fn test_the_centre_star_rejects_a_leaf() {
        let (p, n) = (16usize, 8usize);
        let (tree, m, w) = dataset(n, p, 2);
        let leaves = Leaves {
            means: &m,
            precisions: &w,
            n_features: p,
        };
        let (down, up, _) = settle(&tree, leaves);
        assert!(matches!(
            centre_star(&tree, &down, &up, 0),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            centre_star(&tree, &down, &up, tree.n_nodes() as u32),
            Err(BonsaiErrors::NodeOutOfRange { .. })
        ));
    }
}
