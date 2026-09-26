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

use crate::model::merge::EffLeaf;
use crate::model::place::Walk;
use crate::search::polytomy::CentreStar;
use crate::search::split_hash;
use crate::search::spr::{Row, RowStore, UpSide, eff_step, root_up, up_step};
use crate::tree::{NO_NODE, Tree};
use crate::utils::kernels::prune_general;
use crate::utils::simd::prune_binary;
use crate::utils::traits::BonsaiFloat;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::OnceLock;

////////////
// Shapes //
////////////

/// Height of a node in a tree: zero for a leaf, one above its tallest child
/// otherwise.
///
/// The arena keeps internal nodes in contiguous levels by height, every height
/// from one up occupied, so this is a binary search over the level starts.
///
/// ### Params
///
/// * `tree` - The tree
/// * `v` - Node
///
/// ### Returns
///
/// The height.
fn tree_height(tree: &Tree, v: u32) -> u32 {
    let v = v as usize;
    if v < tree.n_leaves() {
        return 0;
    }
    let (mut lo, mut hi) = (0usize, tree.n_levels());
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if tree.level(mid).0 <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo as u32 + 1
}

/// A tree that differs from a base tree at a handful of nodes.
///
/// Node ids are the base tree's, plus one past its end for a node a regraft
/// creates. Anything not overridden is read off the base.
#[derive(Clone)]
struct Shape<'a> {
    /// The base tree.
    tree: &'a Tree,
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
    fn new(tree: &'a Tree) -> Self {
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
            None => tree_height(self.tree, v),
        }
    }

    /// Recompute heights and child order along a path, bottom-up.
    ///
    /// The arena orders children by height and then by id, leaves (height
    /// zero) first; only nodes on a path whose subtrees changed can move in
    /// that order, so only they are recomputed.
    ///
    /// ### Params
    ///
    /// * `path` - Nodes whose subtrees changed, each above the one before
    fn settle(&mut self, path: &[u32]) {
        for &v in path {
            let mut kids = self.children(v).to_vec();
            let h = kids.iter().map(|&c| self.height(c)).max().unwrap_or(0) + 1;
            kids.sort_by_key(|&c| (self.height(c), c));
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

/// Down rows of a view: recomputed along its changed path, the current tree's
/// everywhere else.
struct Down<'r, T> {
    /// The current tree's rows.
    store: &'r RowStore<T>,
    /// Rows recomputed by a view this one sits on, if any.
    lower: Option<&'r FxHashMap<u32, Row<T>>>,
    /// Rows this view recomputed.
    dirty: FxHashMap<u32, Row<T>>,
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
        if let Some(r) = self.lower.and_then(|lower| lower.get(&v)) {
            return (&r.0, &r.1);
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
            if children.len() == 2 {
                prune_binary(
                    children[0].0,
                    children[0].1,
                    children[0].2,
                    children[1].0,
                    children[1].1,
                    children[1].2,
                    &mut m_out,
                    &mut w_out,
                );
            } else {
                if scratch.len() < p * children.len() {
                    scratch.resize(p * children.len(), 0.0);
                }
                prune_general(
                    &children,
                    &mut m_out,
                    &mut w_out,
                    &mut scratch[..p * children.len()],
                );
            }
            drop(children);
            self.dirty
                .insert(v, (m_out.into_boxed_slice(), w_out.into_boxed_slice()));
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
    /// height, ascending: the detached ones, the suppressed one, and path
    /// nodes whose height changed.
    gone: FxHashMap<u32, Vec<u32>>,
    /// Path nodes whose height changed, by their new height, ascending.
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
    pub(crate) fn new(tree: &'a Tree, x: u32) -> Option<Self> {
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
                gone.entry(tree_height(tree, v)).or_default().push(v);
                stack.extend_from_slice(tree.children(v));
            }
        }
        if let Some(sp) = suppressed {
            gone.entry(tree_height(tree, sp)).or_default().push(sp);
        }
        let mut moved_in: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for &v in &path {
            let (was, now) = (tree_height(tree, v), shape.height(v));
            if was != now {
                gone.entry(was).or_default().push(v);
                moved_in.entry(now).or_default().push(v);
            }
        }
        gone_leaves.sort_unstable();
        for list in gone.values_mut().chain(moved_in.values_mut()) {
            list.sort_unstable();
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
    /// Leaves first by id, then internal nodes by height and then by id, as
    /// the built arena numbers them; found by rank, not by materialising the
    /// order.
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
            let (lo, hi) = if (h as usize) <= tree.n_levels() {
                tree.level(h as usize - 1)
            } else {
                (0, 0)
            };
            let gone = self.gone.get(&h).unwrap_or(&empty);
            let inn = self.moved_in.get(&h).unwrap_or(&empty);
            let count = hi - lo - gone.len() + inn.len();
            if rank < count {
                return select(lo as u32, hi as u32, gone, inn, rank);
            }
            rank -= count;
        }
        unreachable!("an arena position past the remaining tree's node count")
    }
}

/// The `rank`-th id, ascending, of `[lo, hi)` without `gone` and with `inn`.
///
/// ### Params
///
/// * `lo`, `hi` - The level's id range
/// * `gone` - Ids of the range that are absent, ascending
/// * `inn` - Ids from elsewhere that are present, ascending, outside the range
/// * `rank` - Position wanted, from zero
///
/// ### Returns
///
/// The id.
fn select(lo: u32, hi: u32, gone: &[u32], inn: &[u32], rank: usize) -> u32 {
    let count = |v: u32| -> usize {
        let in_range = if v < lo {
            0
        } else {
            ((v - lo + 1) as usize).min((hi - lo) as usize)
        };
        in_range - gone.partition_point(|&g| g <= v) + inn.partition_point(|&i| i <= v)
    };
    let mut a = inn.first().map_or(lo, |&i| i.min(lo));
    let mut b = inn
        .last()
        .map_or(hi.saturating_sub(1), |&i| i.max(hi.saturating_sub(1)));
    while a < b {
        let mid = a + (b - a) / 2;
        if count(mid) > rank {
            b = mid;
        } else {
            a = mid + 1;
        }
    }
    a
}

impl Walk for Pruned<'_> {
    fn id_space(&self) -> usize {
        self.shape.tree.n_nodes()
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
            lower: None,
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
/// fingerprint needs to know about the move.
pub(crate) struct Attached<T> {
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

/// Build the regrafted tree's star and fingerprint without building the tree.
///
/// ### Params
///
/// * `rows` - The pruned view's rows
/// * `target` - Node to attach below, in the current tree's ids
/// * `branch` - Length of the new edge
/// * `word` - The current tree's leaf words
/// * `below` - The current tree's leaf counts
/// * `here` - The current tree's split fingerprint
///
/// ### Returns
///
/// The star and the fingerprint terms, or `None` if the current tree's root is
/// binary, whose duplicate edge the fingerprint treats specially and which is
/// left to the built path.
pub(crate) fn attach<T: BonsaiFloat>(
    rows: &PrunedRows<'_, '_, T>,
    target: u32,
    branch: f64,
    word: &[u64],
    below: &[usize],
    here: u64,
) -> Option<Attached<T>> {
    let view = rows.view;
    let tree = view.shape.tree;
    if tree.children(tree.root()).len() == 2 {
        return None;
    }
    let x = view.x;
    let joint = tree.n_nodes() as u32;
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

    let mut down = Down {
        store: rows.down.store,
        lower: Some(&rows.down.dirty),
        dirty: FxHashMap::default(),
    };
    down.settle(&shape, &path);

    // The centre's up row, top-down along its path.
    let p = rows.down.store.n_features();
    let mut up = root_up(p);
    for &v in path.iter().rev().skip(1) {
        up = down.up_from(&shape, v, (&up.0, &up.1));
    }

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
    Some(Attached {
        star,
        member_word,
        member_count,
        print,
    })
}
