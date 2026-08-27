//! Two-dimensional layouts for a [`Tree`].
//!
//! Topology and branch lengths in, coordinates out. Nothing here touches the
//! likelihood machinery or any per-node data, so the tree a search returns can
//! be handed straight to a plotting layer.
//!
//! Three layouts and one post-processing step (SPEC.md section 14):
//!
//! * [`dendrogram`], the conventional ladderised rectangular tree,
//! * [`equal_angle`], Felsenstein's linear-time circular layout,
//! * [`equal_daylight`], his iterative refinement of it,
//! * [`Layout::hyperbolic`], the disk projection, which applies to any of them.
//!
//! [`has_edge_crossing`] checks a layout for crossings exactly. It is what
//! [`equal_daylight`] leans on, and it is worth reaching for whenever a
//! drawing looks wrong.
//!
//! Coordinates come back as two parallel `Vec<f64>` indexed by node, which is
//! what an R or Python wrapper wants: no per-node struct to unpack.
//!
//! Two things are worth stating before anyone reads a picture off these. The
//! root is a bookkeeping choice and not a feature of the tree (SPEC.md section
//! 2, S14), so rerooting changes every layout here without changing anything
//! about the tree. And the distance between two nodes is always the sum of the
//! branch lengths along the tree path, never the Euclidean distance between
//! their coordinates.
//!
//! Everything is written as flat scans or with an explicit stack. Bonsai trees
//! can be deep and laddery ([`Tree::ladder`] exists to exercise exactly that),
//! and a recursive layout would overflow the stack on a real dataset.

use std::f64::consts::{PI, TAU};

use crate::errors::BonsaiErrors;
use crate::tree::Tree;

////////////////
// Parameters //
////////////////

/// Node count above which [`equal_daylight`] declines to refine.
///
/// The rotation sweep, the acceptance test and the crossing check all measure
/// something about every node from every node, so one sweep is `O(n^2)` and no
/// reordering fixes that. Measured on this machine on 2026-08-27 (release
/// build, [`DAYLIGHT_MAX_SWEEPS`] sweeps, worst of balanced, ladder and random
/// trees): 4 ms at 63 nodes, 57 ms at 255, 0.63 s at 1023, 2.6 s at 2047 and
/// 4.3 s at 4095. A gate at 2048 nodes is the last doubling that keeps a
/// default `equal_daylight` call inside a few seconds; anyone who wants the
/// next one can raise `LayoutParams::daylight_max_nodes` and pay for it. Above
/// the gate the equal-angle layout comes back unchanged with
/// `DaylightReport::refined` set to `false`.
pub const DAYLIGHT_MAX_NODES: usize = 2048;

/// Hard cap on refinement sweeps in [`equal_daylight`].
///
/// The sweep is a coordinate descent with no convergence proof, so a cap is
/// what makes termination a fact rather than a hope, and since each sweep is
/// `O(n^2)` the cap is also the run time. Measured on this machine on
/// 2026-08-27 across balanced, ladder, star and random trees of 63 to 2047
/// nodes: twelve sweeps take the daylight discrepancy to within 2% of where
/// forty sweeps leave it on every shape tried, and on a 512-leaf ladder that
/// is 414 down to 7.9 against 414 down to 0.0 for three times the time. The
/// remaining 2% is not visible in a drawing.
pub const DAYLIGHT_MAX_SWEEPS: usize = 12;

/// Largest rotation, in radians, a sweep may apply and still count as
/// converged.
///
/// A ten-thousandth of a radian is six thousandths of a degree. On a drawing a
/// thousand pixels across, that moves the outermost node by roughly a
/// twentieth of a pixel, so a sweep that moves nothing by more than this has
/// stopped changing the picture and there is no point running another.
pub const DAYLIGHT_ANGLE_TOL: f64 = 1e-4;

/// Number of times a sweep may halve its rotation looking for a step that both
/// improves the daylight and keeps the drawing planar.
///
/// Ten halvings take [`DAYLIGHT_DAMPING`] from a half down to about five parts
/// in ten thousand. Measured on this machine on 2026-08-27, 512-leaf random
/// trees with branch lengths spread over a factor of forty need to come down
/// to about 0.005, which is seven halvings, before a sweep stops crossing; ten
/// leaves three halvings of margin. Past that the step is too small to be
/// worth another `O(n^2)` pass, and the run stops.
const DAYLIGHT_MAX_BACKTRACKS: usize = 10;

/// Largest fraction of the way towards equal daylight that one rotation may
/// move.
///
/// The starting point for the backtracking search inside [`equal_daylight`],
/// which halves it until the sweep both improves the daylight and leaves the
/// drawing planar. Half a step rather than a whole one because measured on
/// this machine on 2026-08-27, an undamped sweep on a 1024-leaf balanced tree
/// overshoots into a crossing on its first move and then has to backtrack from
/// there anyway; starting at a half saves that wasted pass and costs nothing,
/// since the search doubles the step back up whenever it is accepted.
pub const DAYLIGHT_DAMPING: f64 = 0.5;

/// Vertical spacing between adjacent leaves in [`dendrogram`].
///
/// Purely a display scale, since the vertical axis of a dendrogram carries no
/// information. One unit per leaf keeps leaf rows at integer coordinates,
/// which is the easiest thing for a plotting layer to label.
pub const DEFAULT_LEAF_SPACING: f64 = 1.0;

/// Fraction of the layout's largest coordinate below which a point counts as
/// coincident with the node it is being measured from.
///
/// A point sitting on the node it is measured from has no direction, so it
/// cannot contribute to an angular extent. The threshold is relative because
/// branch lengths carry the units of the input, which are arbitrary.
const COINCIDENT_REL_EPS: f64 = 1e-12;

/// Relative tolerance on the orientation determinant in [`has_edge_crossing`].
///
/// The determinant is a product of two coordinate differences, so it scales as
/// the square of the layout. Below this it is indistinguishable from zero and
/// the three points are treated as collinear.
const ORIENT_REL_EPS: f64 = 1e-12;

/// Tuning for the layouts and for the hyperbolic projection.
///
/// Every layout function takes this as `Option<LayoutParams>` and resolves it
/// with `unwrap_or_default()`, so the common case is `None`.
#[derive(Clone, Copy, Debug)]
pub struct LayoutParams {
    /// Vertical distance between adjacent leaves in [`dendrogram`].
    pub leaf_spacing: f64,
    /// Direction, in radians, of the first wedge boundary in the circular
    /// layouts. Rotates the whole picture and nothing else.
    pub start_angle: f64,
    /// Node count above which [`equal_daylight`] returns the equal-angle
    /// layout unrefined.
    pub daylight_max_nodes: usize,
    /// Cap on accepted refinement sweeps.
    pub daylight_max_sweeps: usize,
    /// Largest rotation a sweep may apply and still count as converged.
    pub daylight_angle_tol: f64,
    /// Fraction of the way towards equal daylight each rotation moves.
    pub daylight_damping: f64,
    /// Point mapped to the centre of the disk by [`Layout::hyperbolic`].
    pub hyperbolic_origin: (f64, f64),
    /// Scale applied after the translation by `hyperbolic_origin`. Larger
    /// zooms push more of the tree towards the rim.
    pub hyperbolic_zoom: f64,
}

impl LayoutParams {
    /// Build a parameter set explicitly.
    ///
    /// ### Params
    ///
    /// * `leaf_spacing` - Vertical distance between adjacent dendrogram leaves
    /// * `start_angle` - Direction of the first wedge boundary, in radians
    /// * `daylight_max_nodes` - Node count above which refinement is skipped
    /// * `daylight_max_sweeps` - Cap on accepted refinement sweeps
    /// * `daylight_angle_tol` - Rotation below which a sweep counts as
    ///   converged
    /// * `daylight_damping` - Fraction of each rotation actually applied
    /// * `hyperbolic_origin` - Point sent to the centre of the disk
    /// * `hyperbolic_zoom` - Scale applied after the translation
    ///
    /// ### Returns
    ///
    /// The parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        leaf_spacing: f64,
        start_angle: f64,
        daylight_max_nodes: usize,
        daylight_max_sweeps: usize,
        daylight_angle_tol: f64,
        daylight_damping: f64,
        hyperbolic_origin: (f64, f64),
        hyperbolic_zoom: f64,
    ) -> Self {
        Self {
            leaf_spacing,
            start_angle,
            daylight_max_nodes,
            daylight_max_sweeps,
            daylight_angle_tol,
            daylight_damping,
            hyperbolic_origin,
            hyperbolic_zoom,
        }
    }
}

impl Default for LayoutParams {
    /// The shipped defaults: one unit per leaf, wedges starting along the
    /// positive `x` axis, the measured daylight gate of
    /// [`DAYLIGHT_MAX_NODES`], and a hyperbolic projection centred on the
    /// layout origin at unit zoom, which is where [`equal_angle`] puts the
    /// root.
    ///
    /// ### Returns
    ///
    /// The default parameter set.
    fn default() -> Self {
        Self {
            leaf_spacing: DEFAULT_LEAF_SPACING,
            start_angle: 0.0,
            daylight_max_nodes: DAYLIGHT_MAX_NODES,
            daylight_max_sweeps: DAYLIGHT_MAX_SWEEPS,
            daylight_angle_tol: DAYLIGHT_ANGLE_TOL,
            daylight_damping: DAYLIGHT_DAMPING,
            hyperbolic_origin: (0.0, 0.0),
            hyperbolic_zoom: 1.0,
        }
    }
}

////////////
// Layout //
////////////

/// Node coordinates, as two parallel vectors indexed by node.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// Horizontal coordinate of each node.
    pub x: Vec<f64>,
    /// Vertical coordinate of each node.
    pub y: Vec<f64>,
}

impl Layout {
    /// Number of nodes the layout covers.
    ///
    /// ### Returns
    ///
    /// The node count.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.x.len()
    }

    /// Project onto the hyperbolic disk (SPEC.md section 14).
    ///
    /// Translate by the origin, scale by the zoom, convert to polar, leave the
    /// angle alone and map the radius
    ///
    /// ```text
    /// r -> r / (1 + sqrt(1 + r^2))
    /// ```
    ///
    /// which sends `0` to `0` and infinity to `1`, so everything finite lands
    /// strictly inside the unit disk. The map is strictly increasing in `r`
    /// (its derivative is `1 / (s * (1 + s))` with `s = sqrt(1 + r^2)`), so
    /// radial order survives and nothing turns inside out. This is a
    /// post-processing step rather than a fourth layout: apply it to whichever
    /// of the three is being drawn.
    ///
    /// ### Params
    ///
    /// * `params` - Supplies `hyperbolic_origin` and `hyperbolic_zoom`; `None`
    ///   for the defaults
    ///
    /// ### Returns
    ///
    /// A new layout with every node inside the open unit disk.
    pub fn hyperbolic(&self, params: Option<LayoutParams>) -> Layout {
        let p = params.unwrap_or_default();
        let (ox, oy) = p.hyperbolic_origin;
        let n = self.n_nodes();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let u = (self.x[i] - ox) * p.hyperbolic_zoom;
            let v = (self.y[i] - oy) * p.hyperbolic_zoom;
            let r = u.hypot(v);
            // Applied as a scale factor on (u, v) rather than as a polar round
            // trip. `atan2` followed by `sin_cos` would cost two transcendental
            // calls and perturb an angle the projection is supposed to leave
            // exactly alone.
            let scale = if r > 0.0 {
                1.0 / (1.0 + (1.0 + r * r).sqrt())
            } else {
                0.0
            };
            x.push(u * scale);
            y.push(v * scale);
        }
        Layout { x, y }
    }
}

//////////////////////
// Shared machinery //
//////////////////////

/// Reject branch lengths a layout cannot draw.
///
/// ### Params
///
/// * `tree` - Tree whose branch lengths are checked
///
/// ### Returns
///
/// `Ok(())`, or `MalformedTree` if a non-root branch is negative or not
/// finite.
fn check_branches(tree: &Tree) -> Result<(), BonsaiErrors> {
    let root = tree.root() as usize;
    for v in 0..tree.n_nodes() {
        if v == root {
            continue;
        }
        let t = tree.branch(v as u32);
        if !t.is_finite() || t < 0.0 {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("node {v} has branch length {t}, which cannot be drawn"),
            });
        }
    }
    Ok(())
}

/// Number of leaves below each node, itself included for a leaf.
///
/// One ascending scan: by the arena invariant, ascending index order is a
/// post-order, so every child is complete before its parent is reached.
///
/// ### Params
///
/// * `tree` - Tree to count over
///
/// ### Returns
///
/// The leaf count per node.
fn leaf_counts(tree: &Tree) -> Vec<u32> {
    let n = tree.n_nodes();
    let mut counts = vec![0u32; n];
    for c in counts.iter_mut().take(tree.n_leaves()) {
        *c = 1;
    }
    for v in 0..n {
        if let Some(p) = tree.parent(v as u32) {
            counts[p as usize] += counts[v];
        }
    }
    counts
}

/// A depth-first pre-order over the arena, held as index ranges.
///
/// Built with two flat scans and no stack. The point of it is that a subtree
/// becomes one contiguous slice of `order`, which is what turns rotating a
/// subtree in [`equal_daylight`] into a linear pass over memory rather than a
/// traversal.
struct Tour {
    /// Nodes in depth-first pre-order.
    order: Vec<u32>,
    /// Pre-order position of each node.
    tin: Vec<u32>,
    /// One past the last pre-order position in each node's subtree.
    tout: Vec<u32>,
}

/// Build the pre-order tour of a tree.
///
/// ### Params
///
/// * `tree` - Tree to walk
///
/// ### Returns
///
/// The tour.
fn tour(tree: &Tree) -> Tour {
    let n = tree.n_nodes();

    // Subtree sizes: ascending index order is a post-order.
    let mut size = vec![1u32; n];
    for v in 0..n {
        if let Some(p) = tree.parent(v as u32) {
            size[p as usize] += size[v];
        }
    }

    // Descending index order is a pre-order, since a parent's index always
    // exceeds its children's, so `tin[v]` is settled before any child reads it.
    let mut tin = vec![0u32; n];
    let mut tout = vec![0u32; n];
    let mut order = vec![0u32; n];
    for v in (0..n).rev() {
        order[tin[v] as usize] = v as u32;
        tout[v] = tin[v] + size[v];
        let mut cur = tin[v] + 1;
        for &c in tree.children(v as u32) {
            tin[c as usize] = cur;
            cur += size[c as usize];
        }
    }

    Tour { order, tin, tout }
}

/// Largest absolute coordinate in a layout, the scale the relative epsilons
/// are taken against.
///
/// ### Params
///
/// * `layout` - Layout to measure
///
/// ### Returns
///
/// The largest absolute coordinate, or zero for an empty layout.
fn layout_scale(layout: &Layout) -> f64 {
    let mut s = 0.0f64;
    for i in 0..layout.n_nodes() {
        s = s.max(layout.x[i].abs()).max(layout.y[i].abs());
    }
    s
}

/// Wrap an angle into `[-pi, pi)`.
///
/// ### Params
///
/// * `a` - Angle in radians
///
/// ### Returns
///
/// The equivalent angle in `[-pi, pi)`.
#[inline]
fn wrap_pi(a: f64) -> f64 {
    a - TAU * ((a + PI) / TAU).floor()
}

////////////////
// Dendrogram //
////////////////

/// Ladderised rectangular dendrogram.
///
/// **Only the horizontal axis carries meaning.** `x` is the summed branch
/// length from the root, so the horizontal separation between two nodes,
/// measured along the elbowed path through their common ancestor, is their
/// tree distance. The vertical axis is a display convenience: it exists to
/// stop the leaves overprinting and carries no information whatever. Reading
/// vertical proximity as similarity is the standard misreading of a
/// dendrogram, and rotating any node's children about it gives a different
/// picture of exactly the same tree.
///
/// "Ladderised" means the children of each node are ordered by the number of
/// leaves below them, smallest first, so small clades comb out along one side
/// instead of scattering. Ties break on node index, which keeps the output
/// deterministic. An internal node sits vertically centred on the span of its
/// children, taken as the midpoint of the extreme two rather than the mean of
/// all of them, so a polytomy stays centred on its bracket.
///
/// One descending scan for the horizontal coordinate, one explicit-stack
/// depth-first walk for the leaf order, one ascending scan for the internal
/// vertical coordinates. Nothing recurses.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies `leaf_spacing`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates, or `MalformedTree` if a branch length cannot be drawn.
pub fn dendrogram(tree: &Tree, params: Option<LayoutParams>) -> Result<Layout, BonsaiErrors> {
    check_branches(tree)?;
    let p = params.unwrap_or_default();
    let n = tree.n_nodes();
    let counts = leaf_counts(tree);

    // Ladderised children, as our own CSR alongside the arena's.
    let mut ptr = vec![0u32; n + 1];
    for v in 0..n {
        ptr[v + 1] = ptr[v] + tree.children(v as u32).len() as u32;
    }
    let mut kids: Vec<u32> = Vec::with_capacity(n.saturating_sub(1));
    for v in 0..n {
        let start = kids.len();
        kids.extend_from_slice(tree.children(v as u32));
        kids[start..].sort_unstable_by_key(|&c| (counts[c as usize], c));
    }

    // Horizontal: distance from the root. Descending index order is a
    // pre-order, so a parent is always placed before its children.
    let mut x = vec![0.0f64; n];
    for v in (0..n).rev() {
        for &c in &kids[ptr[v] as usize..ptr[v + 1] as usize] {
            x[c as usize] = x[v] + tree.branch(c);
        }
    }

    // Vertical: leaves in ladderised depth-first order. The stack is a `Vec`,
    // so a ladder of a million leaves costs a million heap slots rather than a
    // million call frames.
    let mut y = vec![0.0f64; n];
    let mut stack: Vec<u32> = vec![tree.root()];
    let mut next = 0usize;
    while let Some(v) = stack.pop() {
        let range = ptr[v as usize] as usize..ptr[v as usize + 1] as usize;
        if range.is_empty() {
            y[v as usize] = next as f64 * p.leaf_spacing;
            next += 1;
            continue;
        }
        // Reversed, because the stack pops the last push first.
        for &c in kids[range].iter().rev() {
            stack.push(c);
        }
    }

    // Internal nodes centred on their children's span. Ascending index order
    // is a post-order, so the children are already placed.
    for v in tree.n_leaves()..n {
        let first = kids[ptr[v] as usize] as usize;
        let last = kids[ptr[v + 1] as usize - 1] as usize;
        y[v] = 0.5 * (y[first] + y[last]);
    }

    Ok(Layout { x, y })
}

/////////////////
// Equal angle //
/////////////////

/// Felsenstein's equal-angle circular layout (*Inferring Phylogenies*,
/// pp. 578-584).
///
/// The root sits at the origin and owns the whole circle. Each node hands its
/// angular wedge down to its children in slices proportional to the number of
/// leaves each contains, and each child is placed at its parent's position
/// offset by its own branch length along the bisector of its slice. One
/// descending scan over the arena: no traversal, no recursion, `O(n)`.
///
/// The reason this is the default for large trees is that sibling subtrees are
/// confined to disjoint angular wedges seen from their parent, so no two edges
/// can cross. Be precise about that guarantee, because it is not quite
/// unconditional: the containment argument needs each wedge to be a convex
/// cone, which holds when the wedge is at most `pi` wide. A wedge is never
/// wider than its parent's and the root owns `2*pi`, so a wedge above `pi` can
/// only appear along a chain from the root whose leaf counts are extremely
/// lopsided. Nothing tried here produces one that crosses, ladders and random
/// trees with branch lengths spread over a factor of forty included, and
/// [`has_edge_crossing`] settles it exactly wherever it matters.
///
/// Unlike [`dendrogram`], both axes carry meaning here: the Euclidean distance
/// from a node to its parent is exactly that node's branch length. Distances
/// between non-adjacent nodes are still tree distances, not Euclidean ones.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies `start_angle`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates, or `MalformedTree` if a branch length cannot be drawn.
pub fn equal_angle(tree: &Tree, params: Option<LayoutParams>) -> Result<Layout, BonsaiErrors> {
    check_branches(tree)?;
    let p = params.unwrap_or_default();
    let n = tree.n_nodes();
    let counts = leaf_counts(tree);

    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let mut lo = vec![0.0f64; n];
    let mut width = vec![0.0f64; n];

    let root = tree.root() as usize;
    lo[root] = p.start_angle;
    width[root] = TAU;

    // Descending index order is a pre-order, so a node's wedge is settled
    // before its children need it.
    for v in (0..n).rev() {
        let kids = tree.children(v as u32);
        if kids.is_empty() {
            continue;
        }
        let denom = f64::from(counts[v]);
        let mut cur = lo[v];
        for &c in kids {
            let w = width[v] * f64::from(counts[c as usize]) / denom;
            lo[c as usize] = cur;
            width[c as usize] = w;
            let (sin, cos) = (cur + 0.5 * w).sin_cos();
            x[c as usize] = x[v] + tree.branch(c) * cos;
            y[c as usize] = y[v] + tree.branch(c) * sin;
            cur += w;
        }
    }

    Ok(Layout { x, y })
}

////////////////////
// Equal daylight //
////////////////////

/// What one [`equal_daylight`] run did.
#[derive(Clone, Copy, Debug)]
pub struct DaylightReport {
    /// Whether refinement was attempted at all. `false` means the tree was
    /// larger than `daylight_max_nodes` and the equal-angle layout came back
    /// untouched.
    pub refined: bool,
    /// Number of sweeps run, adopted or not. A sweep that no step size could
    /// make both cleaner and planar is counted and then ends the run.
    pub sweeps: usize,
    /// Daylight discrepancy of the starting equal-angle layout; `None` if it
    /// was never measured.
    pub initial_discrepancy: Option<f64>,
    /// Daylight discrepancy of the returned layout. Never larger than
    /// `initial_discrepancy`.
    pub final_discrepancy: Option<f64>,
    /// `Some(true)` if the returned layout was checked, edge pair by edge
    /// pair, and found free of crossings. `Some(false)` can only happen when
    /// no sweep was ever accepted and the equal-angle layout it fell back to
    /// crossed to begin with. `None` if the check was never run, which is the
    /// gated case.
    pub crossing_free: Option<bool>,
}

/// One subtree incident to a node, as the angular arc it occupies.
#[derive(Clone, Copy, Debug)]
struct Wedge {
    /// Start of the arc, in radians.
    lo: f64,
    /// Width of the arc, in radians, always below `2*pi`.
    width: f64,
    /// Pre-order range of the subtree, or `None` for the parent-side subtree,
    /// which is held fixed and used as the rotation anchor.
    span: Option<(u32, u32)>,
}

/// Scratch buffers reused across a whole refinement run.
///
/// The sweep touches every node from every node, so allocating any of these
/// per call would dominate the arithmetic.
#[derive(Default)]
struct Scratch {
    /// Incident wedges of the node currently being worked on.
    wedges: Vec<Wedge>,
    /// Gaps between those wedges, in cyclic order.
    gaps: Vec<f64>,
    /// Directions of one subtree's nodes, sorted, while its arc is measured.
    angles: Vec<f64>,
}

/// Smallest angular arc, seen from one point, containing every node in the
/// given slices.
///
/// Sorts the directions and takes the complement of the widest gap between
/// consecutive ones, which is the smallest containing arc by construction.
/// The `O(m log m)` sort is the price of getting it exactly, and it is worth
/// paying: a merely *containing* arc, taken relative to some reference
/// direction, can be far wider than the true extent, and [`sweep_node`] packs
/// subtrees using those widths. Measured on random 512-leaf trees on
/// 2026-08-27, referencing the subtree's own root inflated the daylight
/// discrepancy from 112 to 3800 on the first sweep and the refinement never
/// recovered; referencing the circular mean of the directions was worse again.
///
/// ### Params
///
/// * `layout` - Current coordinates
/// * `ox` - Horizontal coordinate of the point the arc is seen from
/// * `oy` - Vertical coordinate of the point the arc is seen from
/// * `parts` - Slices of node indices making up the subtree
/// * `eps` - Distance below which a point counts as coincident with the origin
/// * `angles` - Scratch buffer, clobbered
///
/// ### Returns
///
/// The `(start, width)` of the arc, or `None` if every point in the subtree is
/// coincident with the origin, which leaves the direction undefined.
fn arc(
    layout: &Layout,
    ox: f64,
    oy: f64,
    parts: &[&[u32]],
    eps: f64,
    angles: &mut Vec<f64>,
) -> Option<(f64, f64)> {
    angles.clear();
    for part in parts {
        for &v in *part {
            let dx = layout.x[v as usize] - ox;
            let dy = layout.y[v as usize] - oy;
            if dx.hypot(dy) > eps {
                angles.push(dy.atan2(dx));
            }
        }
    }
    if angles.is_empty() {
        return None;
    }
    angles.sort_unstable_by(f64::total_cmp);

    let k = angles.len();
    if k == 1 {
        return Some((angles[0], 0.0));
    }

    // Gap `j` runs from `angles[j]` to its cyclic successor. The widest one is
    // the daylight *inside* the subtree, so the arc is everything else.
    let mut widest = angles[0] + TAU - angles[k - 1];
    let mut after = 0usize;
    for j in 0..k - 1 {
        let gap = angles[j + 1] - angles[j];
        if gap > widest {
            widest = gap;
            after = j + 1;
        }
    }
    Some((angles[after], TAU - widest))
}

/// Angular arcs of every subtree incident to a node.
///
/// A node's neighbours are its children plus, unless it is the root, the whole
/// rest of the tree hanging off its parent. Both are contiguous in the
/// pre-order tour: a child's subtree is one slice, the parent side is the two
/// slices either side of the node's own.
///
/// ### Params
///
/// * `tree` - Tree being laid out
/// * `tour` - Pre-order tour of that tree
/// * `layout` - Current coordinates
/// * `node` - Node whose incident subtrees are wanted
/// * `eps` - Coincidence threshold, passed to [`arc`]
/// * `scratch` - Reused buffers; `wedges` comes back with one entry per
///   incident subtree
///
/// ### Returns
///
/// `true` on success, `false` if some incident subtree is entirely coincident
/// with the node, which leaves its direction undefined.
fn incident_wedges(
    tree: &Tree,
    tour: &Tour,
    layout: &Layout,
    node: u32,
    eps: f64,
    scratch: &mut Scratch,
) -> bool {
    scratch.wedges.clear();
    let (ox, oy) = (layout.x[node as usize], layout.y[node as usize]);

    for &c in tree.children(node) {
        let (a, b) = (
            tour.tin[c as usize] as usize,
            tour.tout[c as usize] as usize,
        );
        match arc(
            layout,
            ox,
            oy,
            &[&tour.order[a..b]],
            eps,
            &mut scratch.angles,
        ) {
            Some((lo, width)) => scratch.wedges.push(Wedge {
                lo,
                width,
                span: Some((a as u32, b as u32)),
            }),
            None => return false,
        }
    }

    if tree.parent(node).is_some() {
        let (a, b) = (
            tour.tin[node as usize] as usize,
            tour.tout[node as usize] as usize,
        );
        match arc(
            layout,
            ox,
            oy,
            &[&tour.order[..a], &tour.order[b..]],
            eps,
            &mut scratch.angles,
        ) {
            Some((lo, width)) => scratch.wedges.push(Wedge {
                lo,
                width,
                span: None,
            }),
            None => return false,
        }
    }

    true
}

/// Sort wedges into cyclic order and measure the daylight between them.
///
/// Gaps are signed. A negative one means the two wedges overlap, which is
/// normal rather than exceptional: the arcs computed by [`arc`] are containing
/// arcs and not minimal ones, and a subtree deep in a large tree genuinely can
/// subtend most of the circle seen from one of its own nodes. Overlap is
/// therefore not treated as a planarity failure; [`has_edge_crossing`] settles
/// that question exactly, and separately.
///
/// ### Params
///
/// * `wedges` - Incident wedges, sorted in place by their start angle
/// * `gaps` - Cleared and filled with the signed gap following each sorted
///   wedge
///
/// ### Returns
///
/// The total daylight, which sums to `2*pi` minus the summed wedge widths and
/// so can be negative.
fn daylight_gaps(wedges: &mut [Wedge], gaps: &mut Vec<f64>) -> f64 {
    for w in wedges.iter_mut() {
        w.lo = w.lo.rem_euclid(TAU);
    }
    wedges.sort_unstable_by(|a, b| a.lo.total_cmp(&b.lo));

    gaps.clear();
    let k = wedges.len();
    let mut total = 0.0f64;
    for i in 0..k {
        let next = if i + 1 == k {
            wedges[0].lo + TAU
        } else {
            wedges[i + 1].lo
        };
        let gap = next - (wedges[i].lo + wedges[i].width);
        gaps.push(gap);
        total += gap;
    }
    total
}

/// Rotate one subtree rigidly about a point.
///
/// ### Params
///
/// * `layout` - Coordinates, updated in place
/// * `tour` - Pre-order tour, whose `order` slice names the subtree
/// * `ox` - Horizontal coordinate of the centre of rotation
/// * `oy` - Vertical coordinate of the centre of rotation
/// * `range` - Pre-order range of the subtree
/// * `delta` - Rotation, in radians
///
/// ### Returns
///
/// Nothing; `layout` is updated.
fn rotate_subtree(
    layout: &mut Layout,
    tour: &Tour,
    ox: f64,
    oy: f64,
    range: (u32, u32),
    delta: f64,
) {
    let (sin, cos) = delta.sin_cos();
    for &v in &tour.order[range.0 as usize..range.1 as usize] {
        let i = v as usize;
        let dx = layout.x[i] - ox;
        let dy = layout.y[i] - oy;
        layout.x[i] = ox + cos * dx - sin * dy;
        layout.y[i] = oy + sin * dx + cos * dy;
    }
}

/// Move one node's subtrees a fraction of the way towards equal daylight.
///
/// The parent-side subtree is held fixed and the child subtrees are rotated
/// rigidly about the node towards the positions that would make the gaps
/// between consecutive wedges equal, travelling `damping` of the way there. A
/// rigid rotation cannot change anything inside the subtree it moves, but it
/// changes how that subtree looks from every node outside it, so the sweep
/// gives no guarantee about the tree as a whole and [`equal_daylight`] checks
/// the result rather than trusting it.
///
/// The node is left alone when the wedges already sum to more than the whole
/// circle. There is no daylight to share out, and packing them anyway means
/// choosing which subtrees to bury under which, which measurably sends the
/// descent off into oscillation.
///
/// ### Params
///
/// * `tree` - Tree being laid out
/// * `tour` - Pre-order tour of that tree
/// * `layout` - Coordinates, updated in place
/// * `node` - Internal node to work on
/// * `damping` - Fraction of the way towards equal daylight to travel
/// * `eps` - Coincidence threshold
/// * `scratch` - Reused buffers
///
/// ### Returns
///
/// The largest rotation applied, in radians; zero if the node was left alone.
fn sweep_node(
    tree: &Tree,
    tour: &Tour,
    layout: &mut Layout,
    node: u32,
    damping: f64,
    eps: f64,
    scratch: &mut Scratch,
) -> f64 {
    if !incident_wedges(tree, tour, layout, node, eps, scratch) {
        return 0.0;
    }
    let k = scratch.wedges.len();
    if k < 2 {
        return 0.0;
    }
    let daylight = daylight_gaps(&mut scratch.wedges, &mut scratch.gaps);
    if daylight <= 0.0 {
        return 0.0;
    }

    let target = daylight / k as f64;
    // The parent side anchors the node to the rest of the picture. At the root
    // there is no parent side, so the first child in cyclic order stands in.
    let fixed = scratch
        .wedges
        .iter()
        .position(|w| w.span.is_none())
        .unwrap_or(0);
    let (ox, oy) = (layout.x[node as usize], layout.y[node as usize]);

    let mut moved = 0.0f64;
    let mut cur = scratch.wedges[fixed].lo + scratch.wedges[fixed].width + target;
    for j in 1..k {
        let i = (fixed + j) % k;
        if let Some(range) = scratch.wedges[i].span {
            let delta = damping * wrap_pi(cur - scratch.wedges[i].lo);
            rotate_subtree(layout, tour, ox, oy, range, delta);
            moved = moved.max(delta.abs());
        }
        cur += scratch.wedges[i].width + target;
    }
    moved
}

/// Total daylight discrepancy of a layout: the objective the refinement
/// minimises.
///
/// At every internal node, the sum of squared deviations of the gaps between
/// consecutive subtree wedges from their own mean. It is zero exactly when the
/// daylight around every node is equal, which is the thing the algorithm is
/// named after. It says nothing about planarity, which is
/// [`has_edge_crossing`]'s job: a node whose subtrees overlap uniformly scores
/// zero here, and whether that overlap actually puts two edges across each
/// other is a separate question with an exact answer.
///
/// Nodes with a subtree entirely coincident with them are skipped, since they
/// have no angular extent to be uniform about.
///
/// ### Params
///
/// * `tree` - Tree being laid out
/// * `tour` - Pre-order tour of that tree
/// * `layout` - Coordinates to score
/// * `eps` - Coincidence threshold
/// * `scratch` - Reused buffers
///
/// ### Returns
///
/// The discrepancy, in squared radians.
fn daylight_discrepancy(
    tree: &Tree,
    tour: &Tour,
    layout: &Layout,
    eps: f64,
    scratch: &mut Scratch,
) -> f64 {
    let mut total = 0.0f64;
    for v in tree.n_leaves()..tree.n_nodes() {
        if !incident_wedges(tree, tour, layout, v as u32, eps, scratch) {
            continue;
        }
        let k = scratch.wedges.len();
        if k < 2 {
            continue;
        }
        let target = daylight_gaps(&mut scratch.wedges, &mut scratch.gaps) / k as f64;
        for &gap in scratch.gaps.iter() {
            total += (gap - target) * (gap - target);
        }
    }
    total
}

/// Felsenstein's equal-daylight refinement of the equal-angle layout.
///
/// Starts from [`equal_angle`] and repeatedly sweeps the internal nodes from
/// the root downwards, rotating each node's child subtrees rigidly about it so
/// the angular gaps between them come out equal. Equalising at one node
/// perturbs its neighbours, so this is a coordinate descent with no
/// convergence proof and it has to be bounded from outside. Two things bound
/// it, and they are what makes this honest rather than hopeful:
///
/// * a sweep is **adopted only if it lowers the discrepancy**, so the
///   objective falls monotonically and the run stops as soon as no step size
///   improves on where it already is, when a sweep moves nothing by more than
///   `daylight_angle_tol`, or when `daylight_max_sweeps` runs out;
/// * the naive form of this algorithm **introduces edge crossings**, and the
///   sweep here does too. It is prevented rather than merely reported: every
///   candidate is checked exactly by [`has_edge_crossing`], edge pair against
///   edge pair, and one that crosses is discarded rather than adopted, so a
///   crossing can never reach the caller. `DaylightReport::crossing_free` says
///   so for the layout actually returned.
///
/// How far each rotation actually travels is found by a backtracking search:
/// a sweep starts at `daylight_damping` and halves until it lands one that is
/// both cleaner and planar, up to ten times. That is not decoration. On trees
/// with uniform branch lengths a full-strength sweep is fine and the
/// discrepancy goes to zero; on 512-leaf trees whose branch lengths span a
/// factor of forty, a sweep at damping 0.1 crosses and the same sweep at 0.005
/// does not, so without the search the refinement is a no-op on exactly the
/// trees this crate produces. What it buys on those is real but modest: about
/// a tenth off the discrepancy, against effectively all of it on a ladder or a
/// balanced tree.
///
/// Every check costs a full `O(n^2)` pass, the same order as the sweep itself.
/// Above `daylight_max_nodes` that is not worth paying and the equal-angle
/// layout is returned unrefined, with `refined` set to `false` rather than
/// quietly.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies the daylight gate, sweep cap, tolerance and
///   `start_angle`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates and a [`DaylightReport`] saying what happened, or
/// `MalformedTree` if a branch length cannot be drawn.
pub fn equal_daylight(
    tree: &Tree,
    params: Option<LayoutParams>,
) -> Result<(Layout, DaylightReport), BonsaiErrors> {
    let p = params.unwrap_or_default();
    let base = equal_angle(tree, params)?;

    if tree.n_nodes() > p.daylight_max_nodes {
        return Ok((
            base,
            DaylightReport {
                refined: false,
                sweeps: 0,
                initial_discrepancy: None,
                final_discrepancy: None,
                crossing_free: None,
            },
        ));
    }

    let tour = tour(tree);
    let eps = COINCIDENT_REL_EPS * layout_scale(&base);
    let mut scratch = Scratch::default();

    let initial = daylight_discrepancy(tree, &tour, &base, eps, &mut scratch);
    // Only ever `false` if the equal-angle layout itself crossed and nothing
    // better was found: every candidate that is adopted has been checked.
    let mut clean = !has_edge_crossing(tree, &base)?;

    let mut best = base;
    let mut score = initial;
    let mut sweeps = 0usize;
    let mut step = p.daylight_damping;
    let mut converged = false;

    for _ in 0..p.daylight_max_sweeps {
        let mut accepted = false;
        // Backtracking line search on the rotation size. A full-strength sweep
        // is clean and improving on trees with uniform branch lengths and is
        // neither on trees without them: at 512 leaves with lengths spread
        // over a factor of forty, a sweep at damping 0.1 crosses while the
        // same sweep at 0.005 does not. Halving until a step lands is what
        // makes the refinement do anything at all on realistic trees.
        for _ in 0..DAYLIGHT_MAX_BACKTRACKS {
            let mut candidate = best.clone();
            let mut moved = 0.0f64;
            for v in (tree.n_leaves()..tree.n_nodes()).rev() {
                moved = moved.max(sweep_node(
                    tree,
                    &tour,
                    &mut candidate,
                    v as u32,
                    step,
                    eps,
                    &mut scratch,
                ));
            }
            let next = daylight_discrepancy(tree, &tour, &candidate, eps, &mut scratch);
            if next < score && !has_edge_crossing(tree, &candidate)? {
                best = candidate;
                score = next;
                accepted = true;
                clean = true;
                converged = moved < p.daylight_angle_tol;
                // Let the step grow back, so one awkward sweep does not pin
                // the rest of the run at a needlessly tiny rotation.
                step = (step * 2.0).min(p.daylight_damping);
                break;
            }
            step *= 0.5;
        }
        sweeps += 1;
        if !accepted || converged {
            break;
        }
    }

    Ok((
        best,
        DaylightReport {
            refined: true,
            sweeps,
            initial_discrepancy: Some(initial),
            final_discrepancy: Some(score),
            crossing_free: Some(clean),
        },
    ))
}

//////////////////////
// Crossing checker //
//////////////////////

/// Orientation determinant of three points.
///
/// ### Params
///
/// * `ax` - Horizontal coordinate of the first point
/// * `ay` - Vertical coordinate of the first point
/// * `bx` - Horizontal coordinate of the second point
/// * `by` - Vertical coordinate of the second point
/// * `cx` - Horizontal coordinate of the third point
/// * `cy` - Vertical coordinate of the third point
///
/// ### Returns
///
/// Twice the signed area of the triangle: positive if the points turn left,
/// negative if they turn right, zero if they are collinear.
#[inline]
fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Whether a point lies within the bounding box of a segment.
///
/// Only meaningful once the three points are known to be collinear.
///
/// ### Params
///
/// * `ax` - Horizontal coordinate of the segment start
/// * `ay` - Vertical coordinate of the segment start
/// * `bx` - Horizontal coordinate of the segment end
/// * `by` - Vertical coordinate of the segment end
/// * `cx` - Horizontal coordinate of the point tested
/// * `cy` - Vertical coordinate of the point tested
///
/// ### Returns
///
/// `true` if the point is inside the bounding box.
#[inline]
fn on_segment(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
    cx >= ax.min(bx) && cx <= ax.max(bx) && cy >= ay.min(by) && cy <= ay.max(by)
}

/// Whether a layout contains two edges that cross.
///
/// An edge joins a node to its parent, so a tree with `n` nodes has `n - 1` of
/// them and this compares every pair: `O(n^2)`, deliberately so. It is a
/// checker rather than a layout step, meant for tests and for spot-checking a
/// drawing that looks wrong, not for a hot path.
///
/// Pairs of edges sharing an endpoint are skipped, since meeting at a shared
/// node is what a tree does. Any other intersection counts, including a
/// collinear overlap and an edge passing through an unrelated node.
///
/// ### Params
///
/// * `tree` - Tree the layout belongs to
/// * `layout` - Coordinates to check
///
/// ### Returns
///
/// `true` if some pair of non-adjacent edges intersects, or `NodeOutOfRange`
/// if the layout does not cover the tree.
pub fn has_edge_crossing(tree: &Tree, layout: &Layout) -> Result<bool, BonsaiErrors> {
    let n = tree.n_nodes();
    if layout.x.len() != n || layout.y.len() != n {
        return Err(BonsaiErrors::NodeOutOfRange {
            index: layout.x.len().min(layout.y.len()),
            n_nodes: n,
        });
    }

    let edges: Vec<(u32, u32)> = (0..n as u32)
        .filter_map(|v| tree.parent(v).map(|p| (v, p)))
        .collect();

    let scale = layout_scale(layout);
    let eps = ORIENT_REL_EPS * scale * scale;

    for i in 0..edges.len() {
        let (a, b) = edges[i];
        let (ax, ay) = (layout.x[a as usize], layout.y[a as usize]);
        let (bx, by) = (layout.x[b as usize], layout.y[b as usize]);
        for j in i + 1..edges.len() {
            let (c, d) = edges[j];
            if a == c || a == d || b == c || b == d {
                continue;
            }
            let (cx, cy) = (layout.x[c as usize], layout.y[c as usize]);
            let (dx, dy) = (layout.x[d as usize], layout.y[d as usize]);

            let d1 = orient(cx, cy, dx, dy, ax, ay);
            let d2 = orient(cx, cy, dx, dy, bx, by);
            let d3 = orient(ax, ay, bx, by, cx, cy);
            let d4 = orient(ax, ay, bx, by, dx, dy);

            let straddles = |u: f64, v: f64| (u > eps && v < -eps) || (u < -eps && v > eps);
            if straddles(d1, d2) && straddles(d3, d4) {
                return Ok(true);
            }
            if (d1.abs() <= eps && on_segment(cx, cy, dx, dy, ax, ay))
                || (d2.abs() <= eps && on_segment(cx, cy, dx, dy, bx, by))
                || (d3.abs() <= eps && on_segment(ax, ay, bx, by, cx, cy))
                || (d4.abs() <= eps && on_segment(ax, ay, bx, by, dx, dy))
            {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::NO_NODE;
    use approx::assert_relative_eq;

    /// A linear congruential generator, so the fixtures are reproducible
    /// without pulling a distribution crate into a geometry test.
    struct Lcg(u64);

    impl Lcg {
        /// Next raw draw.
        ///
        /// ### Returns
        ///
        /// A 64-bit value.
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        /// Next draw in `[lo, hi)`.
        ///
        /// ### Params
        ///
        /// * `lo` - Lower bound
        /// * `hi` - Upper bound
        ///
        /// ### Returns
        ///
        /// A value in the half-open interval.
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + u * (hi - lo)
        }

        /// Next draw below `n`.
        ///
        /// ### Params
        ///
        /// * `n` - Exclusive upper bound, at least one
        ///
        /// ### Returns
        ///
        /// An index below `n`.
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// A star: every leaf hanging directly off the root.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves
    /// * `branch` - Branch length on every leaf
    ///
    /// ### Returns
    ///
    /// The tree.
    fn star(n_leaves: usize, branch: f64) -> Tree {
        let mut parent = vec![n_leaves as u32; n_leaves + 1];
        parent[n_leaves] = NO_NODE;
        match Tree::from_parents(parent, vec![branch; n_leaves + 1], n_leaves) {
            Ok(t) => t,
            Err(e) => panic!("star fixture is malformed: {e}"),
        }
    }

    /// A tree mixing a polytomy with resolved clades: the root has five
    /// children, three of them leaves, one a cherry and one a trifurcation.
    ///
    /// ### Returns
    ///
    /// The tree.
    fn polytomy() -> Tree {
        let parent = vec![8, 8, 9, 9, 9, 10, 10, 10, 10, 10, NO_NODE];
        let branch = vec![0.3, 0.7, 0.2, 0.9, 0.4, 1.1, 0.6, 0.8, 0.5, 1.3, 0.0];
        match Tree::from_parents(parent, branch, 8) {
            Ok(t) => t,
            Err(e) => panic!("polytomy fixture is malformed: {e}"),
        }
    }

    /// A random binary tree with random branch lengths.
    ///
    /// Repeatedly joins two randomly chosen active nodes under a fresh
    /// internal node, so the shapes range from near-balanced to near-ladder.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves, at least two
    /// * `seed` - Generator seed
    ///
    /// ### Returns
    ///
    /// The tree.
    fn random_tree(n_leaves: usize, seed: u64) -> Tree {
        let mut rng = Lcg(seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3));
        let n_nodes = 2 * n_leaves - 1;
        let mut parent = vec![NO_NODE; n_nodes];
        let mut branch = vec![0.0f64; n_nodes];
        let mut active: Vec<u32> = (0..n_leaves as u32).collect();
        let mut next_free = n_leaves as u32;
        while active.len() > 1 {
            let i = rng.below(active.len());
            let a = active.swap_remove(i);
            let j = rng.below(active.len());
            let b = active.swap_remove(j);
            parent[a as usize] = next_free;
            parent[b as usize] = next_free;
            branch[a as usize] = rng.uniform(0.05, 2.0);
            branch[b as usize] = rng.uniform(0.05, 2.0);
            active.push(next_free);
            next_free += 1;
        }
        match Tree::from_parents(parent, branch, n_leaves) {
            Ok(t) => t,
            Err(e) => panic!("random fixture is malformed: {e}"),
        }
    }

    /// The fixture set every "does it work at all" test runs over: a balanced
    /// tree, a ladder, a star and a polytomy.
    ///
    /// ### Returns
    ///
    /// Named trees.
    fn shapes() -> Vec<(&'static str, Tree)> {
        let balanced = match Tree::balanced_binary(16, 0.7) {
            Ok(t) => t,
            Err(e) => panic!("balanced fixture is malformed: {e}"),
        };
        let ladder = match Tree::ladder(17, 0.4) {
            Ok(t) => t,
            Err(e) => panic!("ladder fixture is malformed: {e}"),
        };
        vec![
            ("balanced", balanced),
            ("ladder", ladder),
            ("star", star(9, 1.25)),
            ("polytomy", polytomy()),
        ]
    }

    /// Lowest common ancestor of two nodes, by walking parents.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to walk
    /// * `a` - First node
    /// * `b` - Second node
    ///
    /// ### Returns
    ///
    /// The lowest common ancestor.
    fn lca(tree: &Tree, a: u32, b: u32) -> u32 {
        let mut chain = Vec::new();
        let mut v = a;
        loop {
            chain.push(v);
            match tree.parent(v) {
                Some(p) => v = p,
                None => break,
            }
        }
        let mut v = b;
        loop {
            if chain.contains(&v) {
                return v;
            }
            match tree.parent(v) {
                Some(p) => v = p,
                None => panic!("no common ancestor, which a tree cannot manage"),
            }
        }
    }

    /// Summed branch lengths along the tree path between two nodes.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to walk
    /// * `a` - First node
    /// * `b` - Second node
    ///
    /// ### Returns
    ///
    /// The path length.
    fn path_length(tree: &Tree, a: u32, b: u32) -> f64 {
        let top = lca(tree, a, b);
        let mut total = 0.0;
        for start in [a, b] {
            let mut v = start;
            while v != top {
                total += tree.branch(v);
                match tree.parent(v) {
                    Some(p) => v = p,
                    None => panic!("walked past the root"),
                }
            }
        }
        total
    }

    #[test]
    fn test_every_layout_gives_every_node_finite_coordinates() {
        for (name, tree) in shapes() {
            let dendro = dendrogram(&tree, None).expect(name);
            let angle = equal_angle(&tree, None).expect(name);
            let (daylight, report) = equal_daylight(&tree, None).expect(name);
            assert!(
                report.refined,
                "{name}: small tree should have been refined"
            );
            for layout in [&dendro, &angle, &daylight] {
                assert_eq!(layout.n_nodes(), tree.n_nodes(), "{name}");
                for i in 0..layout.n_nodes() {
                    assert!(layout.x[i].is_finite(), "{name}: x[{i}] not finite");
                    assert!(layout.y[i].is_finite(), "{name}: y[{i}] not finite");
                }
                let disk = layout.hyperbolic(None);
                for i in 0..disk.n_nodes() {
                    assert!(disk.x[i].is_finite(), "{name}: disk x[{i}] not finite");
                    assert!(disk.y[i].is_finite(), "{name}: disk y[{i}] not finite");
                }
            }
        }
    }

    #[test]
    fn test_equal_angle_never_crosses_edges() {
        // The load-bearing test: the whole reason equal-angle is the default
        // for large trees is that it is guaranteed planar.
        let mut trees: Vec<(String, Tree)> = shapes()
            .into_iter()
            .map(|(n, t)| (n.to_string(), t))
            .collect();
        for n in [2usize, 3, 5, 8, 13, 21] {
            if let Ok(t) = Tree::ladder(n, 0.9) {
                trees.push((format!("ladder{n}"), t));
            }
        }
        for n in [2usize, 4, 8, 32] {
            if let Ok(t) = Tree::balanced_binary(n, 1.0) {
                trees.push((format!("balanced{n}"), t));
            }
        }
        for seed in 0..32u64 {
            trees.push((format!("random{seed}"), random_tree(20, seed)));
        }
        // Bigger and more lopsided, where a wedge above `pi` is likely and the
        // containment argument stops being airtight.
        for seed in 0..8u64 {
            trees.push((format!("wide{seed}"), random_tree(120, 1000 + seed)));
        }
        for (name, tree) in trees {
            let layout = equal_angle(&tree, None).expect(&name);
            assert!(
                !has_edge_crossing(&tree, &layout).expect(&name),
                "{name}: equal-angle produced a crossing"
            );
        }
    }

    #[test]
    fn test_equal_daylight_never_crosses_edges() {
        for seed in 0..16u64 {
            let tree = random_tree(24, seed);
            let (layout, report) = equal_daylight(&tree, None).expect("layout");
            assert_eq!(report.crossing_free, Some(true), "seed {seed}");
            assert!(
                !has_edge_crossing(&tree, &layout).expect("check"),
                "seed {seed}: equal-daylight produced a crossing"
            );
        }
        for (name, tree) in shapes() {
            let (layout, _) = equal_daylight(&tree, None).expect(name);
            assert!(
                !has_edge_crossing(&tree, &layout).expect(name),
                "{name}: equal-daylight produced a crossing"
            );
        }
    }

    #[test]
    fn test_circular_layouts_place_each_node_its_branch_length_from_its_parent() {
        for (name, tree) in shapes() {
            let angle = equal_angle(&tree, None).expect(name);
            let (daylight, _) = equal_daylight(&tree, None).expect(name);
            for layout in [&angle, &daylight] {
                for v in 0..tree.n_nodes() as u32 {
                    let Some(p) = tree.parent(v) else { continue };
                    let dx = layout.x[v as usize] - layout.x[p as usize];
                    let dy = layout.y[v as usize] - layout.y[p as usize];
                    assert_relative_eq!(dx.hypot(dy), tree.branch(v), epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_dendrogram_horizontal_span_matches_summed_branch_lengths() {
        let tree = random_tree(24, 7);
        let layout = dendrogram(&tree, None).expect("layout");
        // The path between two nodes in a rectangular dendrogram goes out to
        // their common ancestor and back, so the horizontal distance covered
        // is the sum of the two legs.
        for a in [0u32, 3, 9, 17, 23, 30, 41] {
            for b in [1u32, 5, 11, 20, 22, 35, 44] {
                let top = lca(&tree, a, b) as usize;
                let horizontal =
                    (layout.x[a as usize] - layout.x[top]) + (layout.x[b as usize] - layout.x[top]);
                assert_relative_eq!(horizontal, path_length(&tree, a, b), epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_dendrogram_orders_children_by_subtree_leaf_count() {
        for (name, tree) in shapes() {
            let layout = dendrogram(&tree, None).expect(name);
            let counts = leaf_counts(&tree);
            for v in tree.n_leaves()..tree.n_nodes() {
                let mut kids: Vec<u32> = tree.children(v as u32).to_vec();
                kids.sort_by(|&a, &b| layout.y[a as usize].total_cmp(&layout.y[b as usize]));
                for w in kids.windows(2) {
                    let lo = (counts[w[0] as usize], w[0]);
                    let hi = (counts[w[1] as usize], w[1]);
                    assert!(lo < hi, "{name}: node {v} has children out of ladder order");
                }
            }
        }
    }

    #[test]
    fn test_dendrogram_leaves_occupy_consecutive_rows() {
        let tree = random_tree(32, 3);
        let layout = dendrogram(&tree, None).expect("layout");
        let mut rows: Vec<f64> = (0..tree.n_leaves()).map(|i| layout.y[i]).collect();
        rows.sort_by(f64::total_cmp);
        for (i, r) in rows.iter().enumerate() {
            assert_relative_eq!(*r, i as f64 * DEFAULT_LEAF_SPACING, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_equal_daylight_never_increases_the_discrepancy_and_terminates() {
        for seed in 0..12u64 {
            let tree = random_tree(28, seed);
            let (_, report) = equal_daylight(&tree, None).expect("layout");
            assert!(report.refined, "seed {seed}");
            let (Some(before), Some(after)) =
                (report.initial_discrepancy, report.final_discrepancy)
            else {
                panic!("seed {seed}: refined run reported no discrepancy");
            };
            assert!(
                after <= before,
                "seed {seed}: discrepancy rose from {before} to {after}"
            );
            assert!(
                report.sweeps <= DAYLIGHT_MAX_SWEEPS,
                "seed {seed}: {} sweeps exceeds the cap",
                report.sweeps
            );
        }
    }

    #[test]
    fn test_equal_daylight_actually_improves_on_equal_angle() {
        // A shape with obviously wasted daylight: a ladder, whose equal-angle
        // drawing crowds every rung into one narrowing wedge.
        let tree = Tree::ladder(24, 1.0).expect("ladder");
        let (_, report) = equal_daylight(&tree, None).expect("layout");
        let (Some(before), Some(after)) = (report.initial_discrepancy, report.final_discrepancy)
        else {
            panic!("refined run reported no discrepancy");
        };
        assert!(report.sweeps > 0, "no sweep was accepted");
        assert!(
            after < before,
            "discrepancy did not fall: {before} to {after}"
        );
    }

    #[test]
    fn test_equal_daylight_improves_trees_with_heterogeneous_branch_lengths() {
        // The realistic case, and the one the backtracking line search exists
        // for: an undamped sweep on these crosses immediately, so without the
        // search the refinement would be a no-op here.
        for leaves in [64usize, 128, 256] {
            let tree = random_tree(leaves, leaves as u64);
            let (layout, report) = equal_daylight(&tree, None).expect("layout");
            let (Some(before), Some(after)) =
                (report.initial_discrepancy, report.final_discrepancy)
            else {
                panic!("{leaves} leaves: refined run reported no discrepancy");
            };
            assert!(
                after < before,
                "{leaves} leaves: discrepancy did not fall, {before} to {after}"
            );
            assert_eq!(report.crossing_free, Some(true), "{leaves} leaves");
            assert!(!has_edge_crossing(&tree, &layout).expect("check"));
        }
    }

    #[test]
    fn test_equal_daylight_terminates_on_a_layout_it_cannot_improve() {
        // A star is already perfectly lit: every gap around the root is equal,
        // so no step of any size lowers the discrepancy and the run must stop
        // after the first sweep rather than grinding through the cap.
        let tree = star(64, 1.0);
        let (_, report) = equal_daylight(&tree, None).expect("layout");
        assert_eq!(report.sweeps, 1);
        assert_relative_eq!(
            report.final_discrepancy.unwrap_or(f64::NAN),
            0.0,
            epsilon = 1e-20
        );
    }

    #[test]
    fn test_equal_daylight_is_gated_by_node_count() {
        let tree = Tree::balanced_binary(64, 1.0).expect("balanced");
        let params = LayoutParams {
            daylight_max_nodes: 8,
            ..LayoutParams::default()
        };
        let (layout, report) = equal_daylight(&tree, Some(params)).expect("layout");
        assert!(!report.refined);
        assert_eq!(report.sweeps, 0);
        assert_eq!(report.crossing_free, None);
        assert_eq!(layout, equal_angle(&tree, Some(params)).expect("angle"));
    }

    #[test]
    fn test_hyperbolic_sends_the_origin_to_the_origin() {
        let layout = Layout {
            x: vec![0.0, 3.0],
            y: vec![0.0, -4.0],
        };
        let disk = layout.hyperbolic(None);
        assert_eq!(disk.x[0], 0.0);
        assert_eq!(disk.y[0], 0.0);
    }

    #[test]
    fn test_hyperbolic_is_monotone_in_radius() {
        let radii: Vec<f64> = (0..2000).map(|i| f64::from(i) * 0.37).collect();
        let layout = Layout {
            x: radii.clone(),
            y: vec![0.0; radii.len()],
        };
        let disk = layout.hyperbolic(None);
        for i in 1..radii.len() {
            assert!(
                disk.x[i] > disk.x[i - 1],
                "radius {} did not map above {}",
                radii[i],
                radii[i - 1]
            );
        }
        // The closed form of SPEC.md section 14, checked directly.
        for i in 0..radii.len() {
            let r = radii[i];
            assert_relative_eq!(disk.x[i], r / (1.0 + (1.0 + r * r).sqrt()), epsilon = 1e-14);
        }
    }

    #[test]
    fn test_hyperbolic_keeps_everything_strictly_inside_the_unit_disk() {
        let mut rng = Lcg(11);
        let n = 4096;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(rng.uniform(-1e12, 1e12));
            y.push(rng.uniform(-1e12, 1e12));
        }
        let disk = Layout { x, y }.hyperbolic(None);
        for i in 0..n {
            let r = disk.x[i].hypot(disk.y[i]);
            assert!(r < 1.0, "point {i} landed at radius {r}");
        }
    }

    #[test]
    fn test_hyperbolic_preserves_the_angle() {
        let mut rng = Lcg(29);
        let params = LayoutParams {
            hyperbolic_origin: (0.4, -1.7),
            hyperbolic_zoom: 2.5,
            ..LayoutParams::default()
        };
        let n = 512;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(rng.uniform(-50.0, 50.0));
            y.push(rng.uniform(-50.0, 50.0));
        }
        let source = Layout { x, y };
        let disk = source.hyperbolic(Some(params));
        for i in 0..n {
            let u = (source.x[i] - params.hyperbolic_origin.0) * params.hyperbolic_zoom;
            let v = (source.y[i] - params.hyperbolic_origin.1) * params.hyperbolic_zoom;
            assert_relative_eq!(disk.y[i].atan2(disk.x[i]), v.atan2(u), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hyperbolic_zoom_pushes_points_towards_the_rim() {
        let layout = Layout {
            x: vec![1.0],
            y: vec![0.0],
        };
        let near = layout.hyperbolic(Some(LayoutParams {
            hyperbolic_zoom: 1.0,
            ..LayoutParams::default()
        }));
        let far = layout.hyperbolic(Some(LayoutParams {
            hyperbolic_zoom: 100.0,
            ..LayoutParams::default()
        }));
        assert!(far.x[0] > near.x[0]);
        assert!(far.x[0] < 1.0);
    }

    #[test]
    fn test_layouts_are_deterministic() {
        for (name, tree) in shapes() {
            for _ in 0..4 {
                assert_eq!(
                    dendrogram(&tree, None).expect(name),
                    dendrogram(&tree, None).expect(name),
                    "{name}: dendrogram drifted"
                );
                assert_eq!(
                    equal_angle(&tree, None).expect(name),
                    equal_angle(&tree, None).expect(name),
                    "{name}: equal-angle drifted"
                );
                let (a, ra) = equal_daylight(&tree, None).expect(name);
                let (b, rb) = equal_daylight(&tree, None).expect(name);
                assert_eq!(a, b, "{name}: equal-daylight drifted");
                assert_eq!(ra.sweeps, rb.sweeps, "{name}: sweep count drifted");
            }
        }
    }

    #[test]
    fn test_deep_ladder_does_not_overflow_the_stack() {
        // Depth 100_000 is an order of magnitude past what a recursive layout
        // survives on a default stack, which is the point of the fixture.
        let tree = Tree::ladder(100_000, 0.001).expect("ladder");
        assert_eq!(tree.n_nodes(), 199_999);

        let dendro = dendrogram(&tree, None).expect("dendrogram");
        assert!(dendro.x.iter().all(|v| v.is_finite()));
        assert!(dendro.y.iter().all(|v| v.is_finite()));
        // The root is at zero and the deepest leaf is the furthest right.
        assert_relative_eq!(
            dendro.x.iter().fold(0.0f64, |a, &b| a.max(b)),
            path_length(&tree, tree.root(), 0),
            epsilon = 1e-9
        );

        let angle = equal_angle(&tree, None).expect("equal angle");
        assert!(angle.x.iter().all(|v| v.is_finite()));
        assert!(angle.y.iter().all(|v| v.is_finite()));

        // Far past the gate, so this must come back unrefined rather than
        // spending the afternoon in the quadratic.
        let (daylight, report) = equal_daylight(&tree, None).expect("equal daylight");
        assert!(!report.refined);
        assert_eq!(daylight, angle);

        let disk = angle.hyperbolic(None);
        assert!(
            (0..disk.n_nodes()).all(|i| disk.x[i].hypot(disk.y[i]) < 1.0),
            "a node escaped the unit disk"
        );
    }

    #[test]
    fn test_rejects_branch_lengths_that_cannot_be_drawn() {
        let mut tree = Tree::balanced_binary(4, 1.0).expect("balanced");
        tree.branches_mut()[1] = -0.5;
        assert!(matches!(
            dendrogram(&tree, None),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            equal_angle(&tree, None),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        tree.branches_mut()[1] = f64::NAN;
        assert!(matches!(
            equal_daylight(&tree, None),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_has_edge_crossing_detects_a_planted_crossing() {
        // Two cherries either side of the root: leaves 0,1 under node 4,
        // leaves 2,3 under node 5, both under root 6.
        let tree = Tree::balanced_binary(4, 1.0).expect("balanced");
        let mut layout = Layout {
            x: vec![2.0, 2.0, -2.0, -2.0, 1.0, -1.0, 0.0],
            y: vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        };
        assert!(!has_edge_crossing(&tree, &layout).expect("check"));
        // Fling leaf 2 across the root into the other cherry, so edge 2-5 and
        // edge 1-4 cross properly at (1.4, -0.4).
        layout.x[2] = 2.0;
        layout.y[2] = -0.5;
        assert!(has_edge_crossing(&tree, &layout).expect("check"));
    }

    #[test]
    fn test_has_edge_crossing_detects_a_collinear_overlap() {
        let tree = Tree::balanced_binary(4, 1.0).expect("balanced");
        // Leaf 0 dragged along the axis through both the root and node 5, so
        // edge 0-4 lies on top of edge 5-6 without ever properly crossing it.
        let layout = Layout {
            x: vec![-2.0, 2.0, -2.0, -2.0, 1.0, -1.0, 0.0],
            y: vec![0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        };
        assert!(has_edge_crossing(&tree, &layout).expect("check"));
    }

    #[test]
    fn test_has_edge_crossing_rejects_a_layout_of_the_wrong_size() {
        let tree = Tree::balanced_binary(4, 1.0).expect("balanced");
        let layout = Layout {
            x: vec![0.0; 3],
            y: vec![0.0; 3],
        };
        assert!(matches!(
            has_edge_crossing(&tree, &layout),
            Err(BonsaiErrors::NodeOutOfRange { .. })
        ));
    }

    #[test]
    fn test_start_angle_rotates_the_whole_circular_layout() {
        let tree = random_tree(12, 5);
        let base = equal_angle(&tree, None).expect("layout");
        let turned = equal_angle(
            &tree,
            Some(LayoutParams {
                start_angle: 0.75,
                ..LayoutParams::default()
            }),
        )
        .expect("layout");
        let (sin, cos) = 0.75f64.sin_cos();
        for i in 0..tree.n_nodes() {
            assert_relative_eq!(
                turned.x[i],
                cos * base.x[i] - sin * base.y[i],
                epsilon = 1e-12
            );
            assert_relative_eq!(
                turned.y[i],
                sin * base.x[i] + cos * base.y[i],
                epsilon = 1e-12
            );
        }
    }
}
