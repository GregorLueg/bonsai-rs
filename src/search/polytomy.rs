//! Polytomy resolution (SPEC.md section 9.2), and the splice primitive the
//! rest of the search is built on.
//!
//! ### The primitive
//!
//! Steps 3, 5 and 6 of the search all do the same thing: take a node `X`, build
//! a star from the things attached to it, resolve that star with
//! [`crate::search::star::resolve_star`], and splice the result back into the
//! tree. That is [`centre_star`] followed by [`splice_star`], and it is written
//! once here.
//!
//! The star around `X` is its children **plus its upstream side as one more
//! member**, the latter read off [`UpState`]. Counting the upstream side is
//! what makes the primitive's "stop at three members" rule mean "resolved" for
//! an internal node as well as for the root, which has no upstream side and
//! whose members are just its children.
//!
//! ### The upstream member is not a node
//!
//! It is a stand-in for everything outside `X`'s subtree, and it sits exactly
//! where `X`'s parent `P` sits. So when the star is spliced back, anything the
//! resolution attached to the upstream member attaches to `P` itself; only the
//! ancestors that ended up strictly between `X` and the upstream member become
//! real new nodes, and they land on the edge above `X`. Treating the upstream
//! member as a node of its own would duplicate `P`, orphan `P`'s other
//! children and silently reroot the tree, which is what
//! `test_a_resolution_with_nothing_to_gain_returns_the_same_tree` exists to
//! catch.
//!
//! The star primitive returns its topology rooted at the centre, so the path
//! from the centre out to the upstream member is the one part of it that has to
//! be reversed on the way back in. Everything else keeps its star parent.

use crate::errors::BonsaiErrors;
use crate::model::global::UpState;
use crate::model::likelihood::NodeState;
use crate::search::Leaves;
use crate::search::star::{Star, StarParams, StarResult, resolve_star};
use crate::tree::{NO_NODE, Tree};
use crate::utils::traits::BonsaiFloat;
use rayon::prelude::*;

////////////////
// Parameters //
////////////////

/// Star size at which a centre is already resolved and nothing can be gained.
///
/// The same number the star primitive stops at (SPEC.md section 9.1): three
/// members around a centre is a degree-three node. Counting the upstream side
/// as a member is what makes one number serve both the root and an internal
/// node, so "more than two children" in SPEC.md section 9.2 becomes "more than
/// three members" here and the root's trifurcation is correctly left alone.
///
/// `pub(crate)` because [`crate::search::spr`] reads it to decide whether a
/// regraft left a polytomy behind that needs resolving.
pub(crate) const RESOLVED_STAR_MEMBERS: usize = 3;

/// Runaway guard on the fixed-point loop, not a working limit.
///
/// The loop's termination argument is the loglikelihood, not the sweep count,
/// so this should never bind: swept over zero-branch stars across four decades
/// of precision, the loop settles in a handful of sweeps every time. The same
/// role as `MAX_NEWTON_ITER` in [`crate::model::branch`].
///
/// It exists because that argument is only sound while `min_gain` clears the
/// per-feature rounding floor of a merge gain. [`resolve_polytomies`] now
/// rejects a non-positive `min_gain`, which is the case that actually span, but
/// a floor merely *too small* for the feature count is a caller error this
/// cannot detect, and an unbounded loop crossing an FFI boundary takes the
/// session with no interrupt point.
const MAX_SWEEPS: usize = 64;

///////////////////
// Input, output //
///////////////////

/// The star around one node of a tree, ready for the primitive and for the
/// splice back.
///
/// Means and precisions are row-major `[member][feature]` and precisions are
/// not diffusion corrected, matching [`Star`]. The upstream member, when there
/// is one, is always **last**; `member_nodes` holds the centre's parent in that
/// slot, which is where the upstream effective leaf sits.
#[derive(Clone, Debug)]
pub struct CentreStar<T> {
    /// Node the star is centred on.
    pub centre: u32,
    /// Tree node each member stands for, upstream last.
    pub member_nodes: Vec<u32>,
    /// Whether the last member is the upstream side rather than a child.
    pub has_upstream: bool,
    /// Nodes that the star swallowed and that must disappear from the tree.
    ///
    /// Empty for a plain polytomy resolution. An interchange collapses one end
    /// of an internal edge into the other and lists the deleted end here; its
    /// children are members of the star, so nothing else refers to it once the
    /// splice is done.
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
    /// Exact against the tree the star was built from, because the upstream
    /// member summarises the rest of that tree exactly (SPEC.md section 4). It
    /// is *not* a gain against the caller's original tree where the caller
    /// edited it first, as an interchange does.
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
    /// Total loglikelihood gain over the input tree, in nats.
    pub gain: f64,
    /// Number of polytomies the input tree had, once its zero-length internal
    /// edges were collapsed into their parents.
    pub n_polytomies: usize,
    /// Number of resolutions that changed the tree.
    ///
    /// Routinely larger than `n_polytomies`, and not a sign of anything wrong: a
    /// resolution that leaves its new ancestor at zero distance from its centre
    /// is collapsed at the top of the next sweep and makes a polytomy that was
    /// not in the entry count. Within one sweep no node becomes a polytomy that
    /// was not one already, since a resolution only lowers its own centre's
    /// degree and the ancestors it creates are binary.
    pub n_resolved: usize,
    /// Number of sweeps over the tree, the last of which found nothing.
    pub sweeps: usize,
}

///////////////////
// The primitive //
///////////////////

/// Build the star around a node: its children, plus its upstream side.
///
/// The upstream member's effective leaf is [`UpState`]'s row for the centre,
/// which sits at the centre's parent, and its branch to the centre is the
/// centre's own upstream branch. That is exactly the convention [`Star`] wants,
/// a precision that has not been diffusion corrected with the branch carried
/// separately, so nothing is transformed on the way in.
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

/// [`splice_star`], plus where every node of the new tree came from.
///
/// The same tree [`splice_star`] builds, node for node, so a caller that keeps
/// rows across moves sees exactly the arena the plain splice would have given
/// it. The map is recovered by walking up from each leaf in the spliced parent
/// array and in the rebuilt tree in step, which pairs every reachable node once.
///
/// ### Params
///
/// * `tree` - The tree the star was built from
/// * `star` - The star, from [`centre_star`] or from an interchange's collapse
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The new tree and, per node of it, its node in `tree` or [`NO_NODE`] for a
/// node the splice created; or the error the primitive or the arena failed
/// with.
pub(crate) fn splice_star_mapped<T: BonsaiFloat>(
    tree: &Tree,
    star: &CentreStar<T>,
    params: Option<StarParams>,
) -> Result<(Tree, Vec<u32>), BonsaiErrors> {
    let result = resolve_star(star.view(), params)?;
    let mut parent: Vec<u32> = (0..tree.n_nodes())
        .map(|i| tree.parent(i as u32).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    apply_splice(&mut parent, &mut branch, star, &result);
    let out = rebuild(&parent, &branch, tree.root(), tree.n_leaves())?;

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
                        reason: format!("splice map lost step at leaf {leaf}"),
                    });
                }
            }
        }
    }
    for old in &mut to_old {
        if *old != NO_NODE && *old as usize >= tree.n_nodes() {
            *old = NO_NODE;
        }
    }
    Ok((out, to_old))
}

/// Map a resolved star back onto tree node ids and rebuild the arena.
///
/// Local index `i < n_members` is the member's own node, except the upstream
/// slot which is the centre's parent; local index `n_members + a` is a new
/// internal node. The one part of the star's topology that is not carried over
/// as it stands is the path from the centre out to the upstream member, which
/// the star holds pointing down and the tree needs pointing up.
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
fn splice_result<T: BonsaiFloat>(
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

/// Write a resolved star into a parent array, appending its new nodes.
///
/// The body of [`splice_result`] without the rebuild, so that several stars
/// whose members do not overlap can be written into one array and rebuilt
/// once. See [`splice_result`] for the index mapping.
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
    let n = star.member_nodes.len();
    let old = parent.len();
    let n_local = result.parent.len();
    parent.resize(old + n_local - n, NO_NODE);
    branch.resize(old + n_local - n, 0.0);

    // A deleted node keeps no parent and, since all of its children are members
    // of the star, gains no children either, so the rebuild's walk never
    // reaches it and it drops out of the arena.
    for &node in &star.deleted {
        parent[node as usize] = NO_NODE;
    }

    let map = |i: usize| -> u32 {
        if i < n {
            star.member_nodes[i]
        } else {
            (old + i - n) as u32
        }
    };

    // The ancestors between the centre and the upstream member, nearest the
    // upstream member first. These are the nodes whose direction flips.
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
        let node = map(i) as usize;
        parent[node] = match result.parent[i] {
            NO_NODE => star.centre,
            up => map(up as usize),
        };
        branch[node] = result.branch[i];
    }

    if let Some(u) = upstream {
        // The upstream member is the centre's parent, and it keeps its own
        // place in the tree: only what hangs off it changes.
        let above = star.member_nodes[u] as usize;
        match chain.split_first() {
            None => {
                parent[star.centre as usize] = above as u32;
                branch[star.centre as usize] = result.branch[u];
            }
            Some((&top, _)) => {
                parent[map(top as usize) as usize] = above as u32;
                branch[map(top as usize) as usize] = result.branch[u];
                for j in 1..chain.len() {
                    let node = map(chain[j] as usize) as usize;
                    parent[node] = map(chain[j - 1] as usize);
                    branch[node] = result.branch[chain[j - 1] as usize];
                }
                let bottom = chain[chain.len() - 1] as usize;
                parent[star.centre as usize] = map(bottom);
                branch[star.centre as usize] = result.branch[bottom];
            }
        }
    }
}

/// Resolve the polytomies at a given set of centres in one pass.
///
/// One settle of the tree, then every centre's star is resolved against it in
/// parallel and all the splices are written into one parent array and rebuilt
/// once. That is an approximation [`resolve_polytomies`] refuses to make: a
/// splice moves the up rows the later centres were resolved against. It is
/// meant for backbone growth, whose output the full refinement then scores
/// exactly.
///
/// Two centres where one is the other's parent would both rewrite the lower
/// one's parent pointer, so the lower one is skipped and reported back; the
/// caller can pass the skipped centres again.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaves` - The leaf data the tree is scored against
/// * `centres` - Internal nodes to resolve; leaves, non-polytomies and
///   duplicates are ignored
/// * `params` - Star primitive knobs, or `None` for the defaults
///
/// ### Returns
///
/// The tree with the centres resolved, the summed claimed gain, and the number
/// of centres skipped because their parent was also a centre.
pub(crate) fn resolve_centres<T: BonsaiFloat>(
    tree: &Tree,
    leaves: Leaves<'_, T>,
    centres: &[u32],
    params: Option<StarParams>,
) -> Result<(Tree, f64, usize), BonsaiErrors> {
    let n_leaves = tree.n_leaves();
    let mut chosen = vec![false; tree.n_nodes()];
    for &c in centres {
        if (c as usize) >= n_leaves && is_polytomy(tree, c) {
            chosen[c as usize] = true;
        }
    }
    let mut todo: Vec<u32> = Vec::new();
    let mut skipped = 0usize;
    for c in 0..tree.n_nodes() as u32 {
        if !chosen[c as usize] {
            continue;
        }
        if tree.parent(c).is_some_and(|up| chosen[up as usize]) {
            skipped += 1;
        } else {
            todo.push(c);
        }
    }
    if todo.is_empty() {
        return Ok((tree.clone(), 0.0, skipped));
    }

    let (down, up, _) = crate::search::settle(tree, leaves)?;
    let resolved: Vec<(CentreStar<T>, StarResult<T>)> = todo
        .par_iter()
        .map(|&c| {
            let star = centre_star(tree, &down, &up, c)?;
            let result = resolve_star(star.view(), params)?;
            Ok((star, result))
        })
        .collect::<Result<_, BonsaiErrors>>()?;

    let mut parent: Vec<u32> = (0..tree.n_nodes())
        .map(|i| tree.parent(i as u32).unwrap_or(NO_NODE))
        .collect();
    let mut branch = tree.branches().to_vec();
    let mut gain = 0.0f64;
    for (star, result) in &resolved {
        if result.merges.is_empty() {
            continue;
        }
        gain += result.merges.iter().map(|x| x.gain).sum::<f64>();
        apply_splice(&mut parent, &mut branch, star, result);
    }
    Ok((
        rebuild(&parent, &branch, tree.root(), n_leaves)?,
        gain,
        skipped,
    ))
}

/// Renumber an arbitrary parent array into the arena invariant and build it.
///
/// [`Tree::from_parents`] relabels internal nodes for itself but *checks*
/// rather than fixes the requirement that a parent index exceed its children's,
/// and a splice breaks that as soon as it puts a new node above the centre. So
/// the nodes are numbered here in a post-order from the root, which gives the
/// requirement by construction. Anything the walk does not reach, which is what
/// a deleted node becomes, is dropped.
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
        // Pushed last-first so the post-order visits siblings in their arena
        // order. Siblings of equal height then keep their relative order
        // through the relabelling, which is what lets `search::spr` recognise
        // the subtrees a splice left alone by their children alone.
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

////////////////////////////
// Step 3: the polytomies //
////////////////////////////

/// Collapse every zero-length internal edge into the node above it.
///
/// The polytomies SPEC.md section 9.2 goes looking for are not structural when
/// they are made: the merge scan solves a branch length to zero and the arena
/// still holds two separate nodes joined by an edge of length zero. A
/// zero-length edge puts its two ends at the same point, so deleting the lower
/// one and hanging its children off the upper one leaves the loglikelihood
/// exactly unchanged and gives the upper node the degree the model says it
/// already has. Without this, [`count_polytomies`] finds nothing on a tree the
/// greedy merge has left structurally binary, which is the ordinary case, and
/// step 3 of the search does nothing at all.
///
/// ### Exactly zero, not a tolerance
///
/// The two places a branch reaches zero are the early return in
/// [`crate::model::branch::optimise_edge`], which fires when the two effective
/// leaves already sit closer than their own error bars allow, and the ends of
/// the split bracket in [`crate::model::merge`]. Both return a literal `0.0`,
/// precisely so that this test can be exact. A tolerance would need a scale to
/// be relative to, and nothing in
/// SPEC.md fixes one: a branch that is merely short is a claim the model is
/// entitled to make, and collapsing it would be editing the answer.
///
/// ### Params
///
/// * `tree` - Tree to collapse; not modified
///
/// ### Returns
///
/// The collapsed tree, or `None` if there was no zero-length internal edge, or
/// the error the arena rejected the rebuild with.
fn collapse_zero_edges(tree: &Tree) -> Result<Option<Tree>, BonsaiErrors> {
    let n = tree.n_nodes();
    let n_leaves = tree.n_leaves();
    let root = tree.root();

    // A leaf is never collapsed, whatever its branch length: it carries an
    // observation and has nowhere to put it. The root has no upstream branch.
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
            // Left unreachable from the root, which is how `rebuild` drops it.
            continue;
        }
        // A chain of zero-length edges collapses onto the node above the whole
        // chain, so walk past every dropped ancestor rather than just one.
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

    rebuild(&parent, &branch, root, n_leaves).map(Some)
}

/// Resolve every polytomy in a tree (SPEC.md section 9.2).
///
/// Merging creates zero-length branches, which collapse into polytomies. A
/// zero-length edge does not change the likelihood, so the configuration was
/// optimal when it was made; by the time the root has moved it often is not, so
/// the star primitive is run again on every node carrying more members than it
/// stops at.
///
/// The collapse is [`collapse_zero_edges`] and it runs at the top of every
/// sweep, because it is what makes the polytomies structural: a merge that put
/// its new ancestor at zero distance from the centre leaves two nodes where the
/// model has one, and nothing downstream of here would ever notice.
///
/// ### Order
///
/// Ascending node index, which the arena invariant makes a post-order, so a
/// node is resolved only after everything below it has been. That is the same
/// direction [`NodeState::prune`] settles the tree in: a centre's downstream
/// members are then effective leaves of subtrees that have already been
/// improved, and only its upstream member is stale. The alternative, root
/// downwards, has it the other way round and stales the majority of the star.
/// The order is otherwise free, which is why the choice is documented rather
/// than defended: nothing in SPEC.md fixes it.
///
/// ### One pass or a fixed point
///
/// A fixed point, because a resolution changes the up-state of every node in
/// the tree and a centre that had nothing to gain earlier could have something
/// to gain later. Termination rests on the loglikelihood rather than on the
/// degrees: every accepted resolution raises it by more than the primitive's
/// `min_gain`, the collapse leaves it exactly alone, and it is bounded above by
/// the best of finitely many topologies. The degree argument that used to sit
/// here is no longer available, because the collapse can hand a node a degree
/// it did not have before. What it does bound is a single sweep: a resolution
/// only lowers its own centre's degree and every ancestor it creates is binary.
/// The sweep restarts after each one because [`Tree::from_parents`] renumbers
/// the internal nodes and a cursor into the old numbering means nothing.
///
/// **One pass is not enough, and the collapse is why.** Measured against the
/// same runs with the collapse done once on entry rather than every sweep,
/// collapsing every sweep makes three to four times as many resolutions, gains
/// more, and ends closer to the generating tree.
///
/// A resolution that puts its new ancestor at zero distance from its centre has
/// made another polytomy, and re-resolving it against the moved centre is worth
/// two to four times the loglikelihood of stopping there. It costs four to five
/// times as many resolutions, and a resolution is a sweep, so this is the
/// expensive half of step 3. Structural recovery follows the loglikelihood on
/// aggregate but not run by run: at the mildest collapse threshold two of six
/// seeds came out with a worse Robinson-Foulds despite a four-fold larger gain,
/// which is the data's noise rather than the search's doing.
///
/// ### What a sweep costs, and what is left in it
///
/// One resolution per sweep and one settling of the whole tree per sweep, so
/// the step is `O(sweeps * n * p)` and `sweeps` grows with the leaf count.
/// Almost all of a sweep is the down-and-up pass itself, and the share grows
/// with the tree.
///
/// Everything this module does per sweep is now the remaining 4 per cent, and
/// the exponent is the two settling sweeps against a resolution count that
/// grows. Getting it down needs one of two things, both outside this module.
/// Either [`NodeState`] and [`UpState`] gain a way to settle only the rows a
/// sweep actually reads, which is the down rows of a polytomy centre's children
/// and the up row of the centre itself, `O(n_polytomies * depth * p)` rather
/// than `O(n p)`; or they gain a way to be reused across sweeps, since
/// `NodeState::prune` asserts on the node count and every sweep therefore
/// reallocates and refills two `n * p` slabs. Resolving several polytomies per
/// sweep would do it too and is not available: a resolution changes every up
/// row in the tree, so the second centre of a sweep would be resolved against
/// stale rows and the answer would move.
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
    // The primitive itself terminates structurally, one merge a round down to
    // three members, so it takes any floor including a negative one. This loop
    // does not: it rests on every accepted resolution raising the loglikelihood
    // by more than `min_gain`. At zero the primitive accepts a merge whose gain
    // is a rounding artefact, the resolution lands the ancestor at zero distance
    // from its centre, the next sweep's collapse folds it back, and the same
    // merge is found again, on a six-leaf star, without ever terminating.
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
        Some(collapsed) => collapsed,
        None => tree.clone(),
    };
    let n_polytomies = count_polytomies(&tree);
    let mut gain = 0.0f64;
    let mut n_resolved = 0usize;
    let mut sweeps = 0usize;

    loop {
        sweeps += 1;
        // A no-op on the first sweep, since the entry tree was collapsed above.
        // Later sweeps need it because a resolution can itself place an
        // ancestor at zero distance from its centre.
        if let Some(collapsed) = collapse_zero_edges(&tree)? {
            tree = collapsed;
        }
        let (down, up, _) = crate::search::settle(&tree, leaves)?;

        let mut accepted: Option<Splice> = None;
        for node in tree.internal_postorder() {
            // The degree test first, off the tree, and the star only for a node
            // that passes it. `CentreStar::is_polytomy` reads nothing the tree
            // does not already hold, and building a star copies `O(deg * p)`
            // rows, so asking it the other way round copied the whole tree's
            // rows once a sweep to answer a question about node degrees.
            if !is_polytomy(&tree, node) {
                continue;
            }
            let star = centre_star(&tree, &down, &up, node)?;
            // Splicing builds a tree, which is `O(n)`; the primitive that
            // decides whether there is anything to splice is `O(deg^3 p)` over
            // a handful of members. So resolve first and splice only the
            // resolution that is kept.
            let result = resolve_star(star.view(), params)?;
            if !result.merges.is_empty() {
                accepted = Some(Splice {
                    gain: result.merges.iter().map(|x| x.gain).sum(),
                    n_merges: result.merges.len(),
                    tree: splice_result(&tree, &star, &result)?,
                });
                break;
            }
        }
        match accepted {
            None => break,
            Some(spliced) => {
                gain += spliced.gain;
                n_resolved += 1;
                tree = spliced.tree;
            }
        }
        if sweeps >= MAX_SWEEPS {
            break;
        }
    }

    Ok(PolytomyResult {
        tree,
        gain,
        n_polytomies,
        n_resolved,
        sweeps,
    })
}

/// Whether resolving a node's star could change anything, read off the tree.
///
/// The same test as [`CentreStar::is_polytomy`] and the reason that one exists
/// as well: a sweep needs the answer for every node and the star for almost
/// none of them.
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
            assert_relative_eq!(out.gain, after - before, max_relative = 1e-7);
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

        assert!(out.gain > 0.0);
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
        assert_eq!(out.gain, 0.0);
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
            .expect("the zero-length edge was not found");
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
            out.n_resolved > 0 && out.gain > 0.0,
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
        assert_eq!(twice.gain, 0.0);
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
