//! The tree an SPR sweep edits, with stable node ids.
//!
//! The arena [`Tree`] renumbers on every accepted move. Here a node keeps its id
//! while it exists, and the arena order the search depends on is kept per
//! level: child order, which sets the bits of every row, and rank, which sets
//! the beam search's start points.
//!
//! An assembly numbers a level by the previous order with new nodes last, so a
//! move reorders only nodes whose height changed, and each lands at the front
//! or back of its new level. A level is therefore a slot array with room at both
//! ends and a Fenwick tree over the occupied slots for rank and select.

use crate::errors::BonsaiErrors;
use crate::tree::{NO_NODE, Tree};
use rustc_hash::FxHashMap;

/// Least room kept free at each end of a level when it is laid out.
///
/// Leaving at least the level's own size free at each end makes relayout
/// amortised constant per insertion; the floor stops tiny levels relaying out
/// on every move.
const LEVEL_SLACK_MIN: usize = 16;

////////////
// Levels //
////////////

/// The internal nodes of one height, in arena order.
struct Level {
    /// Node per slot, [`NO_NODE`] where a slot is empty or its node left.
    slots: Vec<u32>,
    /// First slot in use.
    lo: usize,
    /// One past the last slot in use.
    hi: usize,
    /// Fenwick tree over the slots, one-based, counting occupied slots.
    fen: Vec<u32>,
    /// Occupied slots.
    alive: usize,
}

impl Level {
    /// Lay a level out afresh, in the middle of a slot array with room at
    /// both ends.
    ///
    /// ### Params
    ///
    /// * `nodes` - The level's nodes, in order
    /// * `pos` - Slot per node id, updated for every node laid out
    ///
    /// ### Returns
    ///
    /// The level.
    fn layout(nodes: &[u32], pos: &mut [u32]) -> Self {
        let n = nodes.len();
        let slack = n.max(LEVEL_SLACK_MIN);
        let cap = n + 2 * slack;
        let mut slots = vec![NO_NODE; cap];
        for (i, &v) in nodes.iter().enumerate() {
            slots[slack + i] = v;
            pos[v as usize] = (slack + i) as u32;
        }
        // Linear Fenwick build: each cell passes its count to its parent.
        let mut fen = vec![0u32; cap + 1];
        for i in 1..=cap {
            fen[i] += u32::from(slots[i - 1] != NO_NODE);
            let up = i + (i & i.wrapping_neg());
            if up <= cap {
                fen[up] += fen[i];
            }
        }
        Self {
            slots,
            lo: slack,
            hi: slack + n,
            fen,
            alive: n,
        }
    }

    /// The level's nodes, in order.
    ///
    /// ### Returns
    ///
    /// Occupied slots from front to back.
    fn nodes(&self) -> Vec<u32> {
        self.slots[self.lo..self.hi]
            .iter()
            .copied()
            .filter(|&v| v != NO_NODE)
            .collect()
    }

    /// Add to one slot's count.
    ///
    /// ### Params
    ///
    /// * `slot` - Zero-based slot
    /// * `add` - `1` for a node arriving, `-1` for one leaving
    fn bump(&mut self, slot: usize, add: i32) {
        let mut i = slot + 1;
        while i < self.fen.len() {
            self.fen[i] = self.fen[i].wrapping_add_signed(add);
            i += i & i.wrapping_neg();
        }
    }

    /// The slot of the node at one rank.
    ///
    /// ### Params
    ///
    /// * `rank` - Zero-based rank, below the occupied count
    ///
    /// ### Returns
    ///
    /// The slot.
    fn select(&self, rank: usize) -> usize {
        let cap = self.fen.len() - 1;
        let mut step = 1usize << (usize::BITS - 1 - cap.leading_zeros());
        let (mut at, mut left) = (0usize, rank as u32);
        while step > 0 {
            let next = at + step;
            if next <= cap && self.fen[next] <= left {
                at = next;
                left -= self.fen[next];
            }
            step >>= 1;
        }
        at
    }

    /// Empty one slot.
    ///
    /// ### Params
    ///
    /// * `slot` - The node's slot
    fn remove(&mut self, slot: usize) {
        debug_assert_ne!(self.slots[slot], NO_NODE);
        self.slots[slot] = NO_NODE;
        self.bump(slot, -1);
        self.alive -= 1;
    }

    /// Put a node in front of every other.
    ///
    /// ### Params
    ///
    /// * `v` - The node
    /// * `pos` - Slot per node id
    fn push_front(&mut self, v: u32, pos: &mut [u32]) {
        if self.lo == 0 {
            *self = Self::layout(&self.nodes(), pos);
        }
        self.lo -= 1;
        self.slots[self.lo] = v;
        pos[v as usize] = self.lo as u32;
        self.bump(self.lo, 1);
        self.alive += 1;
    }

    /// Put a node behind every other.
    ///
    /// ### Params
    ///
    /// * `v` - The node
    /// * `pos` - Slot per node id
    fn push_back(&mut self, v: u32, pos: &mut [u32]) {
        if self.hi == self.slots.len() {
            *self = Self::layout(&self.nodes(), pos);
        }
        self.slots[self.hi] = v;
        pos[v as usize] = self.hi as u32;
        self.bump(self.hi, 1);
        self.hi += 1;
        self.alive += 1;
    }
}

///////////////
// Live tree //
///////////////

/// A tree with stable node ids and the arena's order kept per level.
///
/// Leaves are ids `0..n_leaves`, as in the arena. Internal ids are the
/// arena's indices when the tree is built from one and stay put after that;
/// an id a move frees is handed to the next node a move makes.
pub(crate) struct LiveTree {
    /// Number of leaves.
    n_leaves: usize,
    /// Parent per id, [`NO_NODE`] at the root and at a free id.
    parent: Vec<u32>,
    /// Children per id, in arena order.
    children: Vec<Vec<u32>>,
    /// Branch above each id.
    branch: Vec<f64>,
    /// Height per id, zero at a leaf.
    height: Vec<u32>,
    /// Slot of each internal id in its level.
    pos: Vec<u32>,
    /// Internal nodes by height, `levels[h - 1]` for height `h`.
    levels: Vec<Level>,
    /// The root.
    root: u32,
    /// Ids a move freed, reused last in first out.
    free: Vec<u32>,
    /// Nodes in the tree, leaves included.
    n_nodes: usize,
}

/// A move as the views leave it, for [`LiveTree::apply`].
///
/// Ids are the tree's, plus `id_space` for the node a regraft below a leaf
/// makes and `id_space + 1` onwards for the ancestors a splice makes, in the
/// order it made them.
pub(crate) struct MoveEdit<'e> {
    /// Parents the tree the move produces overrides, [`NO_NODE`] for a root.
    pub(crate) parent: &'e FxHashMap<u32, u32>,
    /// Child lists it overrides.
    pub(crate) children: &'e FxHashMap<u32, Vec<u32>>,
    /// Branches it overrides.
    pub(crate) branch: &'e FxHashMap<u32, f64>,
    /// Heights it overrides.
    pub(crate) height: &'e FxHashMap<u32, u32>,
    /// Heights the regrafted tree overrides before the splice; only read when
    /// there was one.
    pub(crate) before_height: &'e FxHashMap<u32, u32>,
    /// The parent the cut suppressed, if any.
    pub(crate) suppressed: Option<u32>,
    /// Ancestors the splice made, `None` without a splice.
    pub(crate) n_made: Option<usize>,
}

impl LiveTree {
    /// Take over an arena tree, ids equal to its indices.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    ///
    /// ### Returns
    ///
    /// The live tree.
    pub(crate) fn from_tree(tree: &Tree) -> Self {
        let n = tree.n_nodes();
        let mut pos = vec![0u32; n];
        let mut height = vec![0u32; n];
        let mut levels = Vec::with_capacity(tree.n_levels());
        for level in 0..tree.n_levels() {
            let (lo, hi) = tree.level(level);
            height[lo..hi].fill(level as u32 + 1);
            let nodes: Vec<u32> = (lo as u32..hi as u32).collect();
            levels.push(Level::layout(&nodes, &mut pos));
        }
        Self {
            n_leaves: tree.n_leaves(),
            parent: (0..n as u32)
                .map(|v| tree.parent(v).unwrap_or(NO_NODE))
                .collect(),
            children: (0..n as u32).map(|v| tree.children(v).to_vec()).collect(),
            branch: tree.branches().to_vec(),
            height,
            pos,
            levels,
            root: tree.root(),
            free: Vec::new(),
            n_nodes: n,
        }
    }

    /// Number the tree as an arena again.
    ///
    /// ### Returns
    ///
    /// The arena tree, the id at each of its indices, and the index of each
    /// id, [`NO_NODE`] at a free id; or `MalformedTree` if the arena rejected
    /// the result, which would be a broken invariant here.
    pub(crate) fn to_tree(&self) -> Result<(Tree, Vec<u32>, Vec<u32>), BonsaiErrors> {
        let mut id_of: Vec<u32> = (0..self.n_leaves as u32).collect();
        for level in &self.levels {
            id_of.extend(level.nodes());
        }
        let mut arena_of = vec![NO_NODE; self.id_space()];
        for (i, &v) in id_of.iter().enumerate() {
            arena_of[v as usize] = i as u32;
        }
        let parent: Vec<u32> = id_of
            .iter()
            .map(|&v| match self.parent[v as usize] {
                NO_NODE => NO_NODE,
                a => arena_of[a as usize],
            })
            .collect();
        let branch: Vec<f64> = id_of.iter().map(|&v| self.branch[v as usize]).collect();
        let tree = Tree::from_level_ordered(parent, branch, self.n_leaves)?;
        Ok((tree, id_of, arena_of))
    }

    /// Number of leaves.
    ///
    /// ### Returns
    ///
    /// The count.
    #[inline]
    pub(crate) fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Number of nodes, leaves included.
    ///
    /// ### Returns
    ///
    /// The count.
    #[inline]
    pub(crate) fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// One past the largest id ever handed out.
    ///
    /// ### Returns
    ///
    /// The id space.
    #[inline]
    pub(crate) fn id_space(&self) -> usize {
        self.parent.len()
    }

    /// The root.
    ///
    /// ### Returns
    ///
    /// Its id.
    #[inline]
    pub(crate) fn root(&self) -> u32 {
        self.root
    }

    /// Whether an id names a node of the tree.
    ///
    /// ### Params
    ///
    /// * `v` - Id
    ///
    /// ### Returns
    ///
    /// False for an id past the id space or one a move freed.
    #[inline]
    pub(crate) fn contains(&self, v: u32) -> bool {
        (v as usize) < self.id_space() && (self.parent[v as usize] != NO_NODE || v == self.root)
    }

    /// Parent of a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node id
    ///
    /// ### Returns
    ///
    /// The parent, `None` at the root.
    #[inline]
    pub(crate) fn parent(&self, v: u32) -> Option<u32> {
        let a = self.parent[v as usize];
        (a != NO_NODE).then_some(a)
    }

    /// Children of a node, in arena order.
    ///
    /// ### Params
    ///
    /// * `v` - Node id
    ///
    /// ### Returns
    ///
    /// The children, empty at a leaf.
    #[inline]
    pub(crate) fn children(&self, v: u32) -> &[u32] {
        &self.children[v as usize]
    }

    /// Branch above a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node id
    ///
    /// ### Returns
    ///
    /// The length.
    #[inline]
    pub(crate) fn branch(&self, v: u32) -> f64 {
        self.branch[v as usize]
    }

    /// Height of a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node id
    ///
    /// ### Returns
    ///
    /// The height, zero at a leaf.
    #[inline]
    pub(crate) fn height(&self, v: u32) -> u32 {
        self.height[v as usize]
    }

    /// Where a node sits in the arena order: by height, then by slot.
    ///
    /// ### Params
    ///
    /// * `v` - Node id, a leaf or a node of the tree
    ///
    /// ### Returns
    ///
    /// A key that sorts nodes as the arena numbers them.
    #[inline]
    pub(crate) fn order(&self, v: u32) -> (u32, u32) {
        if (v as usize) < self.n_leaves {
            (0, v)
        } else {
            (self.height[v as usize], self.pos[v as usize])
        }
    }

    /// Number of levels above the leaves.
    ///
    /// ### Returns
    ///
    /// The count.
    #[inline]
    pub(crate) fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Nodes of one height.
    ///
    /// ### Params
    ///
    /// * `h` - Height, at least one
    ///
    /// ### Returns
    ///
    /// The count, zero above the tallest level.
    #[inline]
    pub(crate) fn level_len(&self, h: u32) -> usize {
        self.levels.get(h as usize - 1).map_or(0, |l| l.alive)
    }

    /// The nodes of one height, in arena order.
    ///
    /// ### Params
    ///
    /// * `h` - Height, at least one
    ///
    /// ### Returns
    ///
    /// The nodes, empty above the tallest level.
    pub(crate) fn level_nodes(&self, h: u32) -> Vec<u32> {
        self.levels
            .get(h as usize - 1)
            .map_or_else(Vec::new, Level::nodes)
    }

    /// The node at one rank of a level.
    ///
    /// ### Params
    ///
    /// * `h` - Height, at least one
    /// * `rank` - Zero-based, below [`LiveTree::level_len`]
    ///
    /// ### Returns
    ///
    /// The node.
    pub(crate) fn level_select(&self, h: u32, rank: usize) -> u32 {
        let level = &self.levels[h as usize - 1];
        level.slots[level.select(rank)]
    }

    /// Hand out an id, a freed one if there is one.
    ///
    /// ### Returns
    ///
    /// The id, with empty entries.
    fn alloc(&mut self) -> u32 {
        if let Some(v) = self.free.pop() {
            return v;
        }
        let v = self.parent.len() as u32;
        self.parent.push(NO_NODE);
        self.children.push(Vec::new());
        self.branch.push(0.0);
        self.height.push(0);
        self.pos.push(0);
        v
    }

    /// Apply a move the views describe.
    ///
    /// The regrafted arena orders a level by its height and then by the order
    /// before, the node a regraft makes last; the spliced arena orders a level
    /// by the regrafted order, the ancestors it makes last. So within its new
    /// level a node sorts by (regrafted height, height before, slot before),
    /// every node that kept its height at every stage sorts as it did, and a
    /// node that did not goes to the front of its level if its key is below
    /// the stayers' and to the back otherwise. Only those nodes move.
    ///
    /// ### Params
    ///
    /// * `edit` - The move, in the views' ids
    ///
    /// ### Returns
    ///
    /// Id per temporary id the views used for a node the move made.
    pub(crate) fn apply(&mut self, edit: MoveEdit<'_>) -> FxHashMap<u32, u32> {
        let space = self.id_space() as u32;
        let (joint, first_new) = (space, space + 1);
        let n_made = edit.n_made.unwrap_or(0);

        if let Some(sp) = edit.suppressed {
            let h = self.height[sp as usize];
            self.levels[h as usize - 1].remove(self.pos[sp as usize] as usize);
            self.parent[sp as usize] = NO_NODE;
            self.children[sp as usize].clear();
            self.free.push(sp);
            self.n_nodes -= 1;
        }
        let mut real: FxHashMap<u32, u32> = FxHashMap::default();
        let made = edit
            .parent
            .contains_key(&joint)
            .then_some(joint)
            .into_iter()
            .chain(first_new..first_new + n_made as u32);
        for temp in made {
            let v = self.alloc();
            real.insert(temp, v);
            self.n_nodes += 1;
        }
        let r = |v: u32| real.get(&v).copied().unwrap_or(v);

        // Every node whose height changed at any stage has an entry here, since
        // each view starts from a copy of the one before.
        let mut movers: Vec<(u32, (u32, u32, u32), u32)> = Vec::new();
        for (&v, &h2) in edit.height {
            if v < space {
                let (h0, p0) = (self.height[v as usize], self.pos[v as usize]);
                let h1 = match edit.n_made {
                    Some(_) => edit.before_height.get(&v).copied().unwrap_or(h0),
                    None => h2,
                };
                if h0 == h2 && h1 == h2 {
                    continue;
                }
                self.levels[h0 as usize - 1].remove(p0 as usize);
                movers.push((h2, (h1, h0, p0), v));
            } else if v == joint {
                let h1 = match edit.n_made {
                    Some(_) => edit.before_height.get(&v).copied().unwrap_or(h2),
                    None => h2,
                };
                movers.push((h2, (h1, u32::MAX, 0), r(v)));
            } else {
                movers.push((h2, (u32::MAX, u32::MAX, v - first_new), r(v)));
            }
        }
        movers.sort_unstable();
        for &(h2, _, v) in &movers {
            self.height[v as usize] = h2;
            while self.levels.len() < h2 as usize {
                self.levels.push(Level::layout(&[], &mut self.pos));
            }
        }
        // Front movers go in last first, so they end up in ascending order.
        for &(h2, key, v) in movers.iter().rev() {
            if (key.0, key.1) < (h2, h2) {
                self.levels[h2 as usize - 1].push_front(v, &mut self.pos);
            }
        }
        for &(h2, key, v) in &movers {
            if (key.0, key.1) >= (h2, h2) {
                self.levels[h2 as usize - 1].push_back(v, &mut self.pos);
            }
        }
        while self.levels.last().is_some_and(|l| l.alive == 0) {
            self.levels.pop();
        }

        for (&v, &a) in edit.parent {
            self.parent[r(v) as usize] = if a == NO_NODE { NO_NODE } else { r(a) };
        }
        for (&v, &t) in edit.branch {
            self.branch[r(v) as usize] = t;
        }
        for (&v, kids) in edit.children {
            let mut kids: Vec<u32> = kids.iter().map(|&c| r(c)).collect();
            kids.sort_unstable_by_key(|&c| self.order(c));
            self.children[r(v) as usize] = kids;
        }
        debug_assert_eq!(self.parent[self.root as usize], NO_NODE);
        real
    }
}

//////////////
// Topology //
//////////////

/// What an edge scan reads off a tree, so that one scan serves the arena and
/// the live tree alike.
pub(crate) trait Topology {
    /// Parent of a node, `None` at the root.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The parent.
    fn parent(&self, v: u32) -> Option<u32>;

    /// Children of a node, in arena order.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The children.
    fn children(&self, v: u32) -> &[u32];

    /// Branch above a node.
    ///
    /// ### Params
    ///
    /// * `v` - Node
    ///
    /// ### Returns
    ///
    /// The length.
    fn branch(&self, v: u32) -> f64;

    /// Number of leaves.
    ///
    /// ### Returns
    ///
    /// The count.
    fn n_leaves(&self) -> usize;
}

impl Topology for Tree {
    fn parent(&self, v: u32) -> Option<u32> {
        Tree::parent(self, v)
    }

    fn children(&self, v: u32) -> &[u32] {
        Tree::children(self, v)
    }

    fn branch(&self, v: u32) -> f64 {
        Tree::branch(self, v)
    }

    fn n_leaves(&self) -> usize {
        Tree::n_leaves(self)
    }
}

impl Topology for LiveTree {
    fn parent(&self, v: u32) -> Option<u32> {
        LiveTree::parent(self, v)
    }

    fn children(&self, v: u32) -> &[u32] {
        LiveTree::children(self, v)
    }

    fn branch(&self, v: u32) -> f64 {
        LiveTree::branch(self, v)
    }

    fn n_leaves(&self) -> usize {
        LiveTree::n_leaves(self)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A tree with levels of several sizes and a polytomy.
    fn fixture() -> Tree {
        let mut parent = vec![NO_NODE; 0];
        // Leaves 0..10, internal 10..: a ladder over a balanced part.
        let n_leaves = 10usize;
        parent.resize(n_leaves, NO_NODE);
        // Cherries (0,1)->10, (2,3)->11, (4,5)->12, (6,7,8)->13 polytomy
        for (leaf, up) in [
            (0, 10),
            (1, 10),
            (2, 11),
            (3, 11),
            (4, 12),
            (5, 12),
            (6, 13),
            (7, 13),
            (8, 13),
        ] {
            parent[leaf] = up;
        }
        parent.resize(18, NO_NODE);
        parent[10] = 14;
        parent[11] = 14;
        parent[12] = 15;
        parent[13] = 15;
        parent[14] = 16;
        parent[15] = 16;
        parent[16] = 17;
        parent[9] = 17;
        let branch = (0..18).map(|v| 0.1 + v as f64 * 0.01).collect();
        Tree::from_parents(parent, branch, n_leaves).expect("fixture")
    }

    #[test]
    fn test_a_tree_survives_the_round_trip() {
        let tree = fixture();
        let live = LiveTree::from_tree(&tree);
        let (back, id_of, arena_of) = live.to_tree().expect("to_tree");
        assert_eq!(back.n_nodes(), tree.n_nodes());
        for v in 0..tree.n_nodes() as u32 {
            assert_eq!(back.parent(v), tree.parent(v));
            assert_eq!(back.branch(v).to_bits(), tree.branch(v).to_bits());
            assert_eq!(id_of[v as usize], v);
            assert_eq!(arena_of[v as usize], v);
        }
    }

    #[test]
    fn test_rank_and_select_agree_with_the_arena() {
        let tree = fixture();
        let live = LiveTree::from_tree(&tree);
        for level in 0..tree.n_levels() {
            let (lo, hi) = tree.level(level);
            let h = level as u32 + 1;
            assert_eq!(live.level_len(h), hi - lo);
            for (r, v) in (lo..hi).enumerate() {
                assert_eq!(live.level_select(h, r), v as u32);
            }
        }
    }

    #[test]
    fn test_a_level_keeps_its_order_through_edits_at_both_ends() {
        let mut pos = vec![0u32; 200];
        let mut level = Level::layout(&[5, 6, 7], &mut pos);
        let mut want: std::collections::VecDeque<u32> = [5, 6, 7].into_iter().collect();
        for i in 0..60u32 {
            let v = 10 + i;
            if i % 3 == 0 {
                level.push_front(v, &mut pos);
                want.push_front(v);
            } else {
                level.push_back(v, &mut pos);
                want.push_back(v);
            }
            if i % 7 == 0 {
                let gone = want.remove(want.len() / 2).expect("non-empty");
                level.remove(pos[gone as usize] as usize);
            }
        }
        let want: Vec<u32> = want.into_iter().collect();
        assert_eq!(level.nodes(), want);
        for (r, &v) in want.iter().enumerate() {
            assert_eq!(level.slots[level.select(r)], v);
            assert_eq!(pos[v as usize] as usize, level.select(r));
        }
    }
}
