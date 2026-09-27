//! The tree an SPR cut leaves behind, walked without building it.
//!
//! Three quarters of SPR's proposals put the subtree back where it came from
//! and change no split. Building the remaining tree and the regrafted one to
//! find that out was an arena assembly and a node-by-node row map each, `O(n)`
//! per proposal and so `O(n^2)` per sweep. This module answers the same
//! question in `O(depth p)`: the remaining tree differs from the current one
//! only along the path above the cut, and the regrafted one only along the
//! path above the attachment, so both are views that override those paths and
//! read everything else off the current tree.
//!
//! ### Why the views have to reproduce the arena
//!
//! The beam search's answer depends on the arena, not just on the tree: its
//! spread start points are node *indices*, and it visits children in arena
//! order. And a row's bits depend on the order its children are summed in.
//! So the views reproduce what [`crate::search::spr`]'s arena assembly would
//! build: leaves first in index order, internal nodes by height and then by
//! original index, children in that order. Start points are found by rank in
//! that order without materialising it.
//!
//! What they cannot reproduce cheaply, they decline: a cut that suppresses a
//! degree-two root, or a current tree whose root is binary, falls back to the
//! built path. SPR checks every answer against that path in debug builds.
//!
//! ### Scoring and applying a move
//!
//! A move that changes a split is scored here too ([`score_move`]): the star
//! resolution is spliced into the regrafted view and the loglikelihood is the
//! current total less the terms of the nodes whose rows changed plus their
//! new terms. Only an accepted move is applied ([`apply_move`]), in one
//! relabel of the arena that lands on the numbering the built path's three
//! assemblies would, because the next proposal's beam starts and child orders
//! depend on it. Every assembly numbers a level by the order the nodes had
//! before, so a move reorders only the nodes whose height it changed.

use crate::model::merge::EffLeaf;
use crate::model::place::Walk;
use crate::search::live::{LiveTree, MoveEdit};
use crate::search::polytomy::{CentreStar, RESOLVED_STAR_MEMBERS, splice_edits};
use crate::search::split_hash;
use crate::search::spr::{Row, RowStore, UpSide, eff_step, fixed, root_up, up_step};
use crate::search::star::StarResult;
use crate::tree::NO_NODE;
use crate::utils::kernels::prune_general;
use crate::utils::simd::prune_binary;
use crate::utils::traits::BonsaiFloat;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::OnceLock;

////////////
// Shapes //
////////////

/// Where a node sat in the arena order of the tree the views start from.
///
/// A node a view made, whose id lies past the tree's, sorts after every node
/// the tree has, in id order, as the arena's assembly numbers it.
///
/// ### Params
///
/// * `tree` - The tree the views start from
/// * `v` - Node
///
/// ### Returns
///
/// The sort key.
fn old_order(tree: &LiveTree, v: u32) -> (u32, u32) {
    if v as usize >= tree.id_space() {
        (u32::MAX, v)
    } else {
        tree.order(v)
    }
}

/// A tree that differs from a base tree at a handful of nodes.
///
/// Node ids are the base tree's, plus one past its end for a node a regraft
/// creates. Anything not overridden is read off the base.
#[derive(Clone)]
struct Shape<'a> {
    /// The base tree.
    tree: &'a LiveTree,
    /// Overridden parents, [`NO_NODE`] for a root.
    parent: FxHashMap<u32, u32>,
    /// Overridden child lists, in arena order.
    children: FxHashMap<u32, Vec<u32>>,
    /// Overridden branch lengths.
    branch: FxHashMap<u32, f64>,
    /// Overridden heights.
    height: FxHashMap<u32, u32>,
}

impl<'a> Shape<'a> {
    /// A view identical to the base tree.
    ///
    /// ### Params
    ///
    /// * `tree` - The base tree
    ///
    /// ### Returns
    ///
    /// The view.
    fn new(tree: &'a LiveTree) -> Self {
        Self {
            tree,
            parent: FxHashMap::default(),
            children: FxHashMap::default(),
            branch: FxHashMap::default(),
            height: FxHashMap::default(),
        }
    }

    /// Parent of a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// Its parent, `None` at the root.
    fn parent(&self, v: u32) -> Option<u32> {
        match self.parent.get(&v) {
            Some(&up) => (up != NO_NODE).then_some(up),
            None => self.tree.parent(v),
        }
    }

    /// Children of a node, in arena order.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// Its children.
    fn children(&self, v: u32) -> &[u32] {
        match self.children.get(&v) {
            Some(kids) => kids,
            None => self.tree.children(v),
        }
    }

    /// Branch above a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The branch length.
    fn branch(&self, v: u32) -> f64 {
        match self.branch.get(&v) {
            Some(&t) => t,
            None => self.tree.branch(v),
        }
    }

    /// Height of a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The height.
    fn height(&self, v: u32) -> u32 {
        match self.height.get(&v) {
            Some(&h) => h,
            None => self.tree.height(v),
        }
    }

    /// Recompute heights and child order along a path, bottom-up.
    ///
    /// The arena orders children by height and then by their order before,
    /// leaves (height zero) first; only nodes on a path whose subtrees changed
    /// can move in that order, so only they are recomputed.
    ///
    /// ### Params
    ///
    /// * `path` - Nodes whose subtrees changed, each above the one before
    fn settle(&mut self, path: &[u32]) {
        for &v in path {
            let mut kids = self.children(v).to_vec();
            let h = kids.iter().map(|&c| self.height(c)).max().unwrap_or(0) + 1;
            kids.sort_by_key(|&c| (self.height(c), old_order(self.tree, c)));
            self.children.insert(v, kids);
            self.height.insert(v, h);
        }
    }

    /// A node and its ancestors, the node first.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The path to the root.
    fn path_up(&self, v: u32) -> Vec<u32> {
        let mut path = vec![v];
        while let Some(up) = self.parent(path[path.len() - 1]) {
            path.push(up);
        }
        path
    }
}

//////////
// Rows //
//////////

/// A recomputed down row and the node's loglikelihood term.
type Settled<T> = (Box<[T]>, Box<[T]>, f64);

/// Down rows of a view: recomputed along its changed path, the current tree's
/// everywhere else.
struct Down<'r, T> {
    /// The current tree's rows.
    store: &'r RowStore<T>,
    /// Rows recomputed by the views this one sits on, nearest first.
    lower: Vec<&'r FxHashMap<u32, Settled<T>>>,
    /// Rows this view recomputed, with their terms.
    dirty: FxHashMap<u32, Settled<T>>,
}

impl<'r, T: BonsaiFloat> Down<'r, T> {
    /// One node's down row.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// Its means and precisions.
    fn row(&self, v: u32) -> (&[T], &[T]) {
        if let Some(r) = self.dirty.get(&v) {
            return (&r.0, &r.1);
        }
        for layer in &self.lower {
            if let Some(r) = layer.get(&v) {
                return (&r.0, &r.1);
            }
        }
        (self.store.means(v), self.store.precisions(v))
    }

    /// Recompute the rows along a changed path, bottom-up.
    ///
    /// The same kernels, in the same child order, that
    /// [`crate::search::spr`]'s row map uses for a node whose subtree a move
    /// changed, so the rows are the bits a settle of the built tree gives.
    ///
    /// ### Params
    ///
    /// * `shape` - The view
    /// * `path` - Its changed nodes, each above the one before
    fn settle(&mut self, shape: &Shape<'_>, path: &[u32]) {
        let p = self.store.n_features();
        let mut scratch: Vec<f64> = Vec::new();
        for &v in path {
            let kids = shape.children(v);
            let children: Vec<(&[T], &[T], f64)> = kids
                .iter()
                .map(|&c| {
                    let (m, w) = self.row(c);
                    (m, w, shape.branch(c))
                })
                .collect();
            let mut m_out = vec![T::zero(); p];
            let mut w_out = vec![T::zero(); p];
            let contrib = if children.len() == 2 {
                prune_binary(
                    children[0].0,
                    children[0].1,
                    children[0].2,
                    children[1].0,
                    children[1].1,
                    children[1].2,
                    &mut m_out,
                    &mut w_out,
                )
            } else {
                if scratch.len() < p * children.len() {
                    scratch.resize(p * children.len(), 0.0);
                }
                prune_general(
                    &children,
                    &mut m_out,
                    &mut w_out,
                    &mut scratch[..p * children.len()],
                )
            };
            drop(children);
            self.dirty.insert(
                v,
                (m_out.into_boxed_slice(), w_out.into_boxed_slice(), contrib),
            );
        }
    }

    /// The up row of one node, from its parent's.
    ///
    /// ### Params
    ///
    /// * `shape` - The view
    /// * `v` - A non-root node
    /// * `up_a` - Its parent's up row
    ///
    /// ### Returns
    ///
    /// Its up row.
    fn up_from(&self, shape: &Shape<'_>, v: u32, up_a: (&[T], &[T])) -> Row<T> {
        let a = shape
            .parent(v)
            .expect("up_from is only asked about non-root nodes");
        let kids = shape.children(a);
        let side = if kids.len() == 2 {
            let other = if kids[0] == v { kids[1] } else { kids[0] };
            let (m_o, w_o) = self.row(other);
            UpSide::Sibling(shape.branch(other), m_o, w_o)
        } else {
            let (m_a, w_a) = self.row(a);
            let (m_c, w_c) = self.row(v);
            UpSide::Parent(m_a, w_a, shape.branch(v), m_c, w_c)
        };
        up_step(shape.parent(a).is_none(), shape.branch(a), up_a, side)
    }
}

/// Up rows and effective leaves of the pruned view, formed on demand.
///
/// Dense and reused across proposals, so that a proposal does not allocate
/// `n` cells to read a few dozen: [`ViewCache::reset`] clears exactly the
/// cells a proposal filled.
pub(crate) struct ViewCache<T> {
    /// Up rows by node.
    up: Vec<OnceLock<Row<T>>>,
    /// Effective leaves by node.
    eff: Vec<OnceLock<Row<T>>>,
    /// Nodes with a filled cell.
    touched: RefCell<Vec<u32>>,
}

impl<T> ViewCache<T> {
    /// Allocate for a tree.
    ///
    /// ### Params
    ///
    /// * `n` - Node id space to cover
    ///
    /// ### Returns
    ///
    /// An empty cache.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            up: (0..n).map(|_| OnceLock::new()).collect(),
            eff: (0..n).map(|_| OnceLock::new()).collect(),
            touched: RefCell::new(Vec::new()),
        }
    }

    /// Node id space covered.
    ///
    /// ### Returns
    ///
    /// The cell count.
    pub(crate) fn len(&self) -> usize {
        self.up.len()
    }

    /// Empty every cell a proposal filled.
    pub(crate) fn reset(&mut self) {
        for v in std::mem::take(self.touched.get_mut()) {
            self.up[v as usize].take();
            self.eff[v as usize].take();
        }
    }
}

//////////////////
// Pruned view //
//////////////////

/// The tree left behind by detaching a subtree, in the current tree's ids.
pub(crate) struct Pruned<'a> {
    /// The view.
    shape: Shape<'a>,
    /// The detached node.
    x: u32,
    /// The suppressed degree-two parent, if the cut left one.
    suppressed: Option<u32>,
    /// Changed nodes, bottom-up: the cut's parent, or its grandparent when the
    /// parent was suppressed, and every ancestor.
    path: Vec<u32>,
    /// Node count of the remaining tree.
    n_nodes: usize,
    /// Leaf count of the remaining tree.
    n_leaves: usize,
    /// Leaves detached with the subtree, ascending.
    gone_leaves: Vec<u32>,
    /// Internal nodes absent from their original level, by that level's
    /// height, in level order: the detached ones, the suppressed one, and path
    /// nodes whose height changed.
    gone: FxHashMap<u32, Vec<u32>>,
    /// Path nodes whose height changed, by their new height, in their order
    /// before; the assembly puts them behind the level's own nodes.
    moved_in: FxHashMap<u32, Vec<u32>>,
    /// Tallest height in the view.
    max_height: u32,
}

impl<'a> Pruned<'a> {
    /// The view for detaching `x`.
    ///
    /// ### Params
    ///
    /// * `tree` - The current tree
    /// * `x` - Node to detach
    ///
    /// ### Returns
    ///
    /// The view, or `None` where `x` may not be pruned or where the cut
    /// suppresses a degree-two root, which moves the root and is left to the
    /// built path.
    pub(crate) fn new(tree: &'a LiveTree, x: u32) -> Option<Self> {
        let par = tree.parent(x)?;
        if tree.parent(par).is_none() && tree.children(par).len() <= 3 {
            return None;
        }
        let kept: Vec<u32> = tree
            .children(par)
            .iter()
            .copied()
            .filter(|&c| c != x)
            .collect();
        let mut shape = Shape::new(tree);
        let (start, suppressed) = match (tree.parent(par), kept.as_slice()) {
            (Some(above), &[s]) => {
                shape.parent.insert(s, above);
                shape.branch.insert(s, tree.branch(s) + tree.branch(par));
                let kids: Vec<u32> = tree
                    .children(above)
                    .iter()
                    .map(|&c| if c == par { s } else { c })
                    .collect();
                shape.children.insert(above, kids);
                (above, Some(par))
            }
            _ => {
                shape.children.insert(par, kept);
                (par, None)
            }
        };
        let path = shape.path_up(start);
        shape.settle(&path);

        let mut gone_leaves = Vec::new();
        let mut gone: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        let mut subtree = 0usize;
        let mut stack = vec![x];
        while let Some(v) = stack.pop() {
            subtree += 1;
            if (v as usize) < tree.n_leaves() {
                gone_leaves.push(v);
            } else {
                gone.entry(tree.height(v)).or_default().push(v);
                stack.extend_from_slice(tree.children(v));
            }
        }
        if let Some(sp) = suppressed {
            gone.entry(tree.height(sp)).or_default().push(sp);
        }
        let mut moved_in: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for &v in &path {
            let (was, now) = (tree.height(v), shape.height(v));
            if was != now {
                gone.entry(was).or_default().push(v);
                moved_in.entry(now).or_default().push(v);
            }
        }
        gone_leaves.sort_unstable();
        for list in gone.values_mut().chain(moved_in.values_mut()) {
            list.sort_unstable_by_key(|&v| tree.order(v));
        }
        let max_height = moved_in
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .max(tree.n_levels() as u32);

        Some(Self {
            n_nodes: tree.n_nodes() - subtree - usize::from(suppressed.is_some()),
            n_leaves: tree.n_leaves() - gone_leaves.len(),
            shape,
            x,
            suppressed,
            path,
            gone_leaves,
            gone,
            moved_in,
            max_height,
        })
    }

    /// The node at one position of the remaining tree's arena.
    ///
    /// Leaves first by id, then internal nodes by height and then by their
    /// order before, as the built arena numbers them; found by rank, not by
    /// materialising the order.
    ///
    /// ### Params
    ///
    /// * `pos` - Arena index, below the remaining tree's node count
    ///
    /// ### Returns
    ///
    /// The node, in the current tree's ids.
    fn at(&self, pos: usize) -> u32 {
        let tree = self.shape.tree;
        if pos < self.n_leaves {
            // Smallest id with `pos + 1` surviving leaves at or below it.
            let (mut lo, mut hi) = (pos as u32, tree.n_leaves() as u32 - 1);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let kept = mid as usize + 1 - self.gone_leaves.partition_point(|&g| g <= mid);
                if kept > pos {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            return lo;
        }
        let empty: Vec<u32> = Vec::new();
        let mut rank = pos - self.n_leaves;
        for h in 1..=self.max_height {
            let gone = self.gone.get(&h).unwrap_or(&empty);
            let inn = self.moved_in.get(&h).unwrap_or(&empty);
            let stay = tree.level_len(h) - gone.len();
            if rank < stay {
                return select_skipping(tree, h, gone, rank);
            }
            if rank < stay + inn.len() {
                return inn[rank - stay];
            }
            rank -= stay + inn.len();
        }
        unreachable!("an arena position past the remaining tree's node count")
    }
}

/// The node at one rank of a level, not counting some of its nodes.
///
/// ### Params
///
/// * `tree` - The tree
/// * `h` - The level's height
/// * `gone` - Nodes of the level to skip, in level order
/// * `rank` - Position wanted among the rest, from zero
///
/// ### Returns
///
/// The node.
fn select_skipping(tree: &LiveTree, h: u32, gone: &[u32], rank: usize) -> u32 {
    // The least `t` with `t = rank + (skipped nodes at or before the t-th)`,
    // reached from below; the node there is never a skipped one.
    let mut t = rank;
    loop {
        let v = tree.level_select(h, t);
        let at = tree.order(v).1;
        let skipped = gone.partition_point(|&g| tree.order(g).1 <= at);
        if t == rank + skipped {
            return v;
        }
        t = rank + skipped;
    }
}

impl Walk for Pruned<'_> {
    fn id_space(&self) -> usize {
        self.shape.tree.id_space()
    }

    fn spread_starts(&self, n_starts: usize) -> Vec<u32> {
        // `model::place::start_points`, over the remaining tree's arena.
        let n = self.n_nodes;
        let k = n_starts.clamp(1, n);
        let mut out = Vec::with_capacity(k);
        out.push(self.shape.tree.root());
        for i in 1..k {
            out.push(self.at(i * n / k));
        }
        out
    }

    fn neighbours(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(self.shape.children(node));
        out.extend(self.shape.parent(node));
    }
}

/// Rows of the pruned view.
pub(crate) struct PrunedRows<'r, 'a, T> {
    /// The view.
    view: &'r Pruned<'a>,
    /// Its down rows.
    down: Down<'r, T>,
    /// Its up rows and effective leaves, formed on demand.
    cache: &'r ViewCache<T>,
}

impl<'r, 'a, T: BonsaiFloat> PrunedRows<'r, 'a, T> {
    /// Recompute the changed path's rows.
    ///
    /// ### Params
    ///
    /// * `view` - The pruned view
    /// * `store` - The current tree's rows
    /// * `cache` - Cleared cells for the up rows and effective leaves
    ///
    /// ### Returns
    ///
    /// The rows.
    pub(crate) fn new(
        view: &'r Pruned<'a>,
        store: &'r RowStore<T>,
        cache: &'r ViewCache<T>,
    ) -> Self {
        let mut down = Down {
            store,
            lower: Vec::new(),
            dirty: FxHashMap::default(),
        };
        down.settle(&view.shape, &view.path);
        Self { view, down, cache }
    }

    /// Up row of a node, filling the path above it on the way.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// Its up row.
    fn up_row(&self, v: u32) -> &Row<T> {
        let shape = &self.view.shape;
        let mut path = Vec::new();
        let mut here = v;
        while self.cache.up[here as usize].get().is_none() {
            path.push(here);
            match shape.parent(here) {
                Some(up) => here = up,
                None => break,
            }
        }
        for &u in path.iter().rev() {
            let row = match shape.parent(u) {
                None => root_up(self.down.store.n_features()),
                Some(a) => {
                    let above = self.cache.up[a as usize]
                        .get()
                        .expect("the path above is filled top-down");
                    self.down.up_from(shape, u, (&above.0, &above.1))
                }
            };
            if self.cache.up[u as usize].set(row).is_ok() {
                self.cache.touched.borrow_mut().push(u);
            }
        }
        self.cache.up[v as usize]
            .get()
            .expect("filled by the loop above")
    }

    /// The whole remaining tree collapsed onto one node.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// Its effective leaf, what the beam search scores an attachment against.
    pub(crate) fn eff_leaf(&self, v: u32) -> EffLeaf<'_, T> {
        let cell = &self.cache.eff[v as usize];
        if cell.get().is_none() {
            let shape = &self.view.shape;
            let above = self.up_row(v);
            let row = eff_step(
                shape.parent(v).is_none(),
                shape.branch(v),
                self.down.row(v),
                (&above.0, &above.1),
            );
            if cell.set(row).is_ok() {
                self.cache.touched.borrow_mut().push(v);
            }
        }
        let row = cell.get().expect("set above");
        EffLeaf {
            m: &row.0,
            w: &row.1,
        }
    }
}

////////////////////
// Attached view //
////////////////////

/// The regrafted tree's star at the attachment point, and what the split
/// fingerprint and [`score_move`] need to know about the move.
pub(crate) struct Attached<'a, T> {
    /// The regrafted tree, as a view of the current one.
    shape: Shape<'a>,
    /// Down rows the regraft recomputed, along the path above the centre.
    dirty: FxHashMap<u32, Settled<T>>,
    /// The star the polytomy resolution would run on, member for member what
    /// the built path reads.
    pub(crate) star: CentreStar<T>,
    /// Leaf word of each member, upstream last.
    pub(crate) member_word: Vec<u64>,
    /// Leaf count of each member, upstream last.
    pub(crate) member_count: Vec<usize>,
    /// Fingerprint of the regrafted tree before any resolution.
    pub(crate) print: u64,
}

/// What [`attach`] made of a regraft.
pub(crate) enum Regraft<'a, T> {
    /// The views decline it; the built path decides.
    Declined,
    /// It changes no split and its star is too small to need a resolution,
    /// which the fingerprint settles before any row is formed.
    Unchanged,
    /// The regraft, its rows and its star.
    Attached(Box<Attached<'a, T>>),
}

/// Build the regrafted tree's star and fingerprint without building the tree.
///
/// Most proposals put the subtree back where it came from, onto a star of
/// three members that needs no resolution, and the fingerprint alone says so.
/// With `stop_if_unchanged` those return before the path's rows and the
/// centre's up row are formed.
///
/// ### Params
///
/// * `rows` - The pruned view's rows
/// * `target` - Node to attach below, in the current tree's ids
/// * `branch` - Length of the new edge
/// * `word` - The current tree's leaf words
/// * `below` - The current tree's leaf counts
/// * `here` - The current tree's split fingerprint
/// * `stop_if_unchanged` - Return [`Regraft::Unchanged`] where it applies
///   rather than forming the rows
///
/// ### Returns
///
/// The star and the fingerprint terms; [`Regraft::Declined`] if the current
/// tree's root is binary, whose duplicate edge the fingerprint treats
/// specially and which is left to the built path.
pub(crate) fn attach<'a, T: BonsaiFloat>(
    rows: &PrunedRows<'_, 'a, T>,
    target: u32,
    branch: f64,
    word: &[u64],
    below: &[usize],
    here: u64,
    stop_if_unchanged: bool,
) -> Regraft<'a, T> {
    let view = rows.view;
    let tree = view.shape.tree;
    if tree.children(tree.root()).len() == 2 {
        return Regraft::Declined;
    }
    // A placement carried over from an earlier tree can name the parent this
    // cut suppresses, which is not in the remaining tree.
    if Some(target) == view.suppressed {
        return Regraft::Declined;
    }
    let x = view.x;
    let joint = tree.id_space() as u32;
    let mut shape = view.shape.clone();
    let centre = if (target as usize) < tree.n_leaves() {
        let above = shape
            .parent(target)
            .expect("a leaf of the remaining tree has a parent");
        shape.parent.insert(joint, above);
        shape.branch.insert(joint, shape.branch(target));
        shape.parent.insert(target, joint);
        shape.branch.insert(target, 0.0);
        shape.parent.insert(x, joint);
        shape.branch.insert(x, branch);
        let kids: Vec<u32> = shape
            .children(above)
            .iter()
            .map(|&c| if c == target { joint } else { c })
            .collect();
        shape.children.insert(above, kids);
        shape.children.insert(joint, vec![target, x]);
        joint
    } else {
        shape.parent.insert(x, target);
        shape.branch.insert(x, branch);
        let mut kids = shape.children(target).to_vec();
        kids.push(x);
        shape.children.insert(target, kids);
        target
    };
    let path = shape.path_up(centre);
    shape.settle(&path);

    // Leaf words and counts of the regrafted tree, where they differ.
    let old_path: FxHashSet<u32> =
        std::iter::successors(tree.parent(x), |&v| tree.parent(v)).collect();
    let new_path: FxHashSet<u32> = path.iter().copied().collect();
    let (wx, nx) = (word[x as usize], below[x as usize]);
    let word_now = |v: u32| -> u64 {
        if v == joint {
            return word[target as usize].wrapping_add(wx);
        }
        let mut w = word[v as usize];
        if old_path.contains(&v) {
            w = w.wrapping_sub(wx);
        }
        if new_path.contains(&v) {
            w = w.wrapping_add(wx);
        }
        w
    };
    let count_now = |v: u32| -> usize {
        if v == joint {
            return below[target as usize] + nx;
        }
        let mut c = below[v as usize];
        if old_path.contains(&v) {
            c -= nx;
        }
        if new_path.contains(&v) {
            c += nx;
        }
        c
    };

    let n_leaves = tree.n_leaves();
    let total = word[tree.root() as usize];
    let term = |w: u64, c: usize| -> u64 {
        if c >= 2 && n_leaves - c >= 2 {
            split_hash(w, total)
        } else {
            0
        }
    };
    // Every node whose leaf set moved is on exactly one of the two paths; the
    // root and whatever is above both paths' meeting point keep theirs.
    let mut print = here;
    for &v in &old_path {
        if new_path.contains(&v) || tree.parent(v).is_none() {
            continue;
        }
        print = print.wrapping_sub(term(word[v as usize], below[v as usize]));
        if Some(v) != view.suppressed {
            print = print.wrapping_add(term(word_now(v), count_now(v)));
        }
    }
    for &v in &path {
        if old_path.contains(&v) || shape.parent(v).is_none() {
            continue;
        }
        if v != joint {
            print = print.wrapping_sub(term(word[v as usize], below[v as usize]));
        }
        print = print.wrapping_add(term(word_now(v), count_now(v)));
    }

    let kids = shape.children(centre);
    let upstream = shape.parent(centre);
    let n = kids.len() + usize::from(upstream.is_some());
    if stop_if_unchanged && n <= RESOLVED_STAR_MEMBERS && print == here {
        return Regraft::Unchanged;
    }

    let mut down = Down {
        store: rows.down.store,
        lower: vec![&rows.down.dirty],
        dirty: FxHashMap::default(),
    };
    down.settle(&shape, &path);

    // The centre's up row, top-down along its path.
    let p = rows.down.store.n_features();
    let mut up = root_up(p);
    for &v in path.iter().rev().skip(1) {
        up = down.up_from(&shape, v, (&up.0, &up.1));
    }
    let mut star = CentreStar {
        centre,
        member_nodes: Vec::with_capacity(n),
        has_upstream: upstream.is_some(),
        deleted: Vec::new(),
        means: Vec::with_capacity(n * p),
        precisions: Vec::with_capacity(n * p),
        branch: Vec::with_capacity(n),
        n_features: p,
    };
    let mut member_word = Vec::with_capacity(n);
    let mut member_count = Vec::with_capacity(n);
    for &c in kids {
        let (m, w) = down.row(c);
        star.member_nodes.push(c);
        star.means.extend_from_slice(m);
        star.precisions.extend_from_slice(w);
        star.branch.push(shape.branch(c));
        member_word.push(word_now(c));
        member_count.push(count_now(c));
    }
    if let Some(par) = upstream {
        star.member_nodes.push(par);
        star.means.extend_from_slice(&up.0);
        star.precisions.extend_from_slice(&up.1);
        star.branch.push(shape.branch(centre));
        member_word.push(total.wrapping_sub(word_now(centre)));
        member_count.push(n_leaves - count_now(centre));
    }
    let dirty = down.dirty;
    Regraft::Attached(Box::new(Attached {
        shape,
        dirty,
        star,
        member_word,
        member_count,
        print,
    }))
}

/// A scored move's views, owned, so that accepting it recomputes nothing.
pub(crate) struct MoveData<T> {
    /// Parents the tree the move produces overrides, [`NO_NODE`] for a root.
    parent: FxHashMap<u32, u32>,
    /// Child lists it overrides, in its arena order.
    children: FxHashMap<u32, Vec<u32>>,
    /// Branches it overrides.
    branch: FxHashMap<u32, f64>,
    /// Heights it overrides.
    height: FxHashMap<u32, u32>,
    /// Heights the regrafted tree overrides, before any splice: the splice's
    /// assembly numbers a level in the regrafted arena's order.
    before_height: FxHashMap<u32, u32>,
    /// Rows the views recomputed, nearest first: the splice's, the regraft's,
    /// the cut's.
    layers: [FxHashMap<u32, Settled<T>>; 3],
    /// The parent the cut suppressed, if any.
    suppressed: Option<u32>,
    /// Split fingerprint of the tree the move produces.
    print: u64,
    /// Ancestors the splice made, `None` when there was no splice.
    n_made: Option<usize>,
}

/// The loglikelihood of the tree a move produces, as a [`fixed`] total,
/// without building it.
///
/// The built path splices the resolution into the regrafted tree, renumbers
/// the arena and settles the nodes whose rows changed. Here the splice is
/// applied to the view, and only the new ancestors, the centre and the path
/// above it are settled, with children in the order the built arena would put
/// them: by height, then by their order in the regrafted arena, the new
/// ancestors last in the order they were made. The total is the current one
/// less the terms of every node whose row changed or vanished plus the new
/// terms, which [`fixed`] makes the same integer the built path sums.
///
/// ### Params
///
/// * `rows` - The pruned view's rows
/// * `attached` - The regraft
/// * `result` - The resolution of the regraft's star, `None` when the star was
///   too small to need one
/// * `base` - The current tree's [`fixed`] total
/// * `print` - Split fingerprint of the tree the move produces
///
/// ### Returns
///
/// The total of the tree the move produces, and the views, for
/// [`apply_move`].
pub(crate) fn score_move<T: BonsaiFloat>(
    rows: PrunedRows<'_, '_, T>,
    attached: Attached<'_, T>,
    result: Option<&StarResult<T>>,
    base: i128,
    print: u64,
) -> (i128, MoveData<T>) {
    let view = rows.view;
    let tree = view.shape.tree;
    let store = rows.down.store;
    let (shape, spliced, before_height, n_made) = match result {
        Some(result) => {
            let (shape, spliced) = splice_rows(&rows, &attached, result);
            let before = attached.shape.height.clone();
            (
                shape,
                spliced,
                before,
                Some(result.parent.len() - result.n_members),
            )
        }
        None => (
            attached.shape.clone(),
            FxHashMap::default(),
            FxHashMap::default(),
            None,
        ),
    };

    let mut total = base;
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for layer in [&spliced, &attached.dirty, &rows.down.dirty] {
        for (&v, row) in layer {
            if seen.insert(v) {
                total += fixed(row.2);
                if (v as usize) < tree.id_space() {
                    total -= fixed(store.contribution(v));
                }
            }
        }
    }
    if let Some(sp) = view.suppressed {
        total -= fixed(store.contribution(sp));
    }
    let suppressed = view.suppressed;
    let Shape {
        parent,
        children,
        branch,
        height,
        ..
    } = shape;
    let data = MoveData {
        parent,
        children,
        branch,
        height,
        before_height,
        layers: [spliced, attached.dirty, rows.down.dirty],
        suppressed,
        print,
        n_made,
    };
    (total, data)
}

/// Apply a star resolution to the regrafted view and settle what it changed.
///
/// ### Params
///
/// * `rows` - The pruned view's rows
/// * `attached` - The regraft
/// * `result` - The resolution of its star
///
/// ### Returns
///
/// The spliced tree as a view, and the recomputed rows and terms of the new
/// ancestors, the centre and every node above it.
fn splice_rows<'a, T: BonsaiFloat>(
    rows: &PrunedRows<'_, 'a, T>,
    attached: &Attached<'a, T>,
    result: &StarResult<T>,
) -> (Shape<'a>, FxHashMap<u32, Settled<T>>) {
    let tree = rows.view.shape.tree;
    let before = &attached.shape;
    let mut shape = before.clone();
    // One past the regraft's own new node.
    let first_new = tree.id_space() as u32 + 1;
    let is_new = |v: u32| v >= first_new;

    let edits = splice_edits(&attached.star, result, first_new);
    for &(v, _, _) in &edits {
        if is_new(v) {
            shape.children.entry(v).or_default();
        }
    }
    for &(v, up, t) in &edits {
        let old = if is_new(v) { None } else { shape.parent(v) };
        if old != Some(up) {
            if let Some(op) = old {
                let mut kids = shape.children(op).to_vec();
                kids.retain(|&c| c != v);
                shape.children.insert(op, kids);
            }
            let mut kids = shape.children(up).to_vec();
            kids.push(v);
            shape.children.insert(up, kids);
            shape.parent.insert(v, up);
        }
        shape.branch.insert(v, t);
    }

    // Everything whose subtree changed is below the new root and reachable
    // from it through changed nodes only: the new ancestors, the centre, and
    // the centre's ancestors.
    let mut changed: FxHashSet<u32> = edits
        .iter()
        .map(|&(v, _, _)| v)
        .filter(|&v| is_new(v))
        .collect();
    changed.extend(shape.path_up(attached.star.centre));
    let mut top = attached.star.centre;
    while let Some(up) = shape.parent(top) {
        top = up;
    }
    let mut order = Vec::with_capacity(changed.len());
    let mut stack = vec![(top, false)];
    while let Some((v, expanded)) = stack.pop() {
        if expanded {
            order.push(v);
            continue;
        }
        stack.push((v, true));
        for &c in shape.children(v) {
            if changed.contains(&c) {
                stack.push((c, false));
            }
        }
    }
    debug_assert_eq!(order.len(), changed.len());

    // Post-order, so every changed child has its height before its parent.
    for &v in &order {
        let h = shape
            .children(v)
            .iter()
            .map(|&c| shape.height(c))
            .max()
            .unwrap_or(0)
            + 1;
        shape.height.insert(v, h);
    }
    for &v in &order {
        let mut kids = shape.children(v).to_vec();
        kids.sort_by_key(|&c| {
            if is_new(c) {
                (shape.height(c), 1u32, 0u32, (0u32, c))
            } else {
                (shape.height(c), 0, before.height(c), old_order(tree, c))
            }
        });
        shape.children.insert(v, kids);
    }

    let mut down = Down {
        store: rows.down.store,
        lower: vec![&attached.dirty, &rows.down.dirty],
        dirty: FxHashMap::default(),
    };
    down.settle(&shape, &order);
    let dirty = down.dirty;
    (shape, dirty)
}

/////////////////
// Applying it //
/////////////////

/// What applying a move changed, for the sweep's word index and revisit set.
pub(crate) struct Applied {
    /// Nodes whose rows the move changed or made, by height.
    pub(crate) changed: Vec<u32>,
    /// Word-index entries that no longer hold: the words the changed nodes
    /// had, and the suppressed node's, with the node each named.
    pub(crate) stale: Vec<(u64, u32)>,
    /// Split fingerprint of the tree the move produced.
    pub(crate) print: u64,
}

/// Apply a move the views scored, in place.
///
/// The tree takes the views' overrides ([`LiveTree::apply`]), the store takes
/// their rows, and the words and counts are recomputed for the nodes whose
/// rows changed, bottom-up. Nothing else is touched, so the cost is the
/// changed paths' and not the tree's.
///
/// ### Params
///
/// * `tree` - The current tree, updated
/// * `store` - Its rows, updated
/// * `data` - The views [`score_move`] scored the move on, against `tree`
/// * `word` - Leaf words per id, updated
/// * `below` - Leaf counts per id, updated
///
/// ### Returns
///
/// What changed.
pub(crate) fn apply_move<T: BonsaiFloat>(
    tree: &mut LiveTree,
    store: &mut RowStore<T>,
    data: MoveData<T>,
    word: &mut Vec<u64>,
    below: &mut Vec<usize>,
) -> Applied {
    let MoveData {
        parent,
        children,
        branch,
        height,
        before_height,
        layers,
        suppressed,
        print,
        n_made,
    } = data;
    let space = tree.id_space() as u32;
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut stale: Vec<(u64, u32)> = Vec::new();
    for layer in &layers {
        for &v in layer.keys() {
            if v < space && seen.insert(v) {
                stale.push((word[v as usize], v));
            }
        }
    }
    stale.extend(suppressed.map(|sp| (word[sp as usize], sp)));
    stale.sort_unstable_by_key(|&(_, v)| v);

    let real = tree.apply(MoveEdit {
        parent: &parent,
        children: &children,
        branch: &branch,
        height: &height,
        before_height: &before_height,
        suppressed,
        n_made,
    });
    let r = |v: u32| real.get(&v).copied().unwrap_or(v);
    word.resize(tree.id_space(), 0);
    below.resize(tree.id_space(), 0);

    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut changed: Vec<u32> = Vec::new();
    for layer in &layers {
        for (&v, row) in layer {
            if seen.insert(v) {
                let rv = r(v);
                store.write_row(rv, &row.0, &row.1, row.2);
                changed.push(rv);
            }
        }
    }
    changed.sort_unstable_by_key(|&v| (tree.height(v), v));
    for &v in &changed {
        let (mut w, mut c) = (0u64, 0usize);
        for &k in tree.children(v) {
            w = w.wrapping_add(word[k as usize]);
            c += below[k as usize];
        }
        word[v as usize] = w;
        below[v as usize] = c;
    }
    Applied {
        changed,
        stale,
        print,
    }
}
