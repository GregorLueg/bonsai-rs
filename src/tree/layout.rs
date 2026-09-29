//! Two-dimensional layouts for a [`Tree`] (SPEC.md section 14).
//!
//! Topology and branch lengths in, coordinates out, as two parallel `Vec<f64>`
//! indexed by node.
//!
//! * [`dendrogram`], the ladderised rectangular tree,
//! * [`equal_angle`], Felsenstein's linear-time circular layout,
//! * [`equal_daylight`], his iterative refinement of it,
//! * [`Layout::hyperbolic`], the disk projection, which applies to any of them.
//!
//! The root is a bookkeeping choice (SPEC.md section 2, S14), so rerooting
//! changes every layout. The distance between two nodes is the summed branch
//! length along the tree path, never the Euclidean distance between their
//! coordinates.
//!
//! Everything is a flat scan or uses an explicit stack: deep ladders would
//! overflow the stack under recursion.

use std::f64::consts::{PI, TAU};

use crate::errors::BonsaiErrors;
use crate::tree::Tree;

////////////////
// Parameters //
////////////////

/// Node count above which [`equal_daylight`] declines to refine.
///
/// One sweep is `O(n^2)`. Ours, by measurement: 2048 nodes is the last doubling
/// that keeps a default call inside a few seconds on one core. Above the gate
/// the equal-angle layout comes back unchanged.
const DAYLIGHT_MAX_NODES: usize = 2048;

/// Hard cap on refinement sweeps in [`equal_daylight`].
///
/// The descent has no convergence proof, so the cap bounds the run time. Ours,
/// by measurement on balanced, ladder, star and random trees: twelve sweeps
/// land within two per cent of forty.
const DAYLIGHT_MAX_SWEEPS: usize = 12;

/// Largest rotation, in radians, a sweep may apply and still count as
/// converged.
///
/// A ten-thousandth of a radian moves the outermost node of a 1000 pixel
/// drawing by about a twentieth of a pixel.
const DAYLIGHT_ANGLE_TOL: f64 = 1e-4;

/// Number of times a sweep may halve its rotation looking for a step that both
/// improves the daylight and keeps the drawing planar.
///
/// Ours, by measurement: random trees with branch lengths spread over a factor
/// of forty need about seven halvings from [`DAYLIGHT_DAMPING`] before a sweep
/// stops crossing; ten leaves three of margin.
const DAYLIGHT_MAX_BACKTRACKS: usize = 10;

/// Largest fraction of the way towards equal daylight that one rotation may
/// move.
///
/// Starting step of the backtracking search in [`equal_daylight`]. An undamped
/// sweep on a large balanced tree overshoots into a crossing on its first move.
const DAYLIGHT_DAMPING: f64 = 0.5;

/// Vertical spacing between adjacent leaves in [`dendrogram`].
///
/// Display scale only; the vertical axis carries no information.
const DEFAULT_LEAF_SPACING: f64 = 1.0;

/// Fraction of the layout's largest coordinate below which a point counts as
/// coincident with the node it is being measured from.
///
/// Relative, because branch lengths carry the arbitrary units of the input.
const COINCIDENT_REL_EPS: f64 = 1e-12;

/// Relative tolerance on the orientation determinant in [`has_edge_crossing`].
///
/// The determinant scales as the product of the two segment lengths; below
/// this the three points are treated as collinear.
const ORIENT_REL_EPS: f64 = 1e-12;

/// Tuning for the layouts and for the hyperbolic projection.
///
/// Every layout function takes `Option<LayoutParams>`; `None` gives the defaults.
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

impl Default for LayoutParams {
    /// The shipped defaults: one unit per leaf, wedges starting along the
    /// positive `x` axis, and a hyperbolic projection centred on the origin at
    /// unit zoom.
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
    /// Translate by the origin, scale by the zoom, leave the angle alone and
    /// map the radius
    ///
    /// ```text
    /// r -> r / (1 + sqrt(1 + r^2))
    /// ```
    ///
    /// which is strictly increasing, sends `0` to `0` and infinity to `1`. In
    /// `f64`, past about `1e16` the result saturates at exactly `1` (radial
    /// order stays non-decreasing); `hypot` avoids overflow of `r^2`.
    ///
    /// Takes no `Result`: a non-finite `hyperbolic_origin` or `hyperbolic_zoom`
    /// gives non-finite coordinates back.
    ///
    /// ### Params
    ///
    /// * `params` - Supplies `hyperbolic_origin` and `hyperbolic_zoom`; `None`
    ///   for the defaults
    ///
    /// ### Returns
    ///
    /// A new layout with every node inside the closed unit disk, and strictly
    /// inside it for any radius a real tree produces.
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
            // Scale on (u, v), not a polar round trip, so the angle is untouched.
            // `hypot` also gives 0.5 at r = 0, so no branch.
            let scale = 1.0 / (1.0 + r.hypot(1.0));
            x.push(u * scale);
            y.push(v * scale);
        }
        Layout { x, y }
    }
}

//////////////////////
// Shared machinery //
//////////////////////

/// Reject layout parameters that cannot be drawn with.
///
/// The whole bundle is checked wherever any of it is read. Not used by
/// [`Layout::hyperbolic`], which returns no `Result`.
///
/// ### Params
///
/// * `params` - Parameters to check
///
/// ### Returns
///
/// Nothing, or `BadParameter` naming the offending field.
fn check_params(params: &LayoutParams) -> Result<(), BonsaiErrors> {
    let bad = |name: &'static str, value: f64| BonsaiErrors::BadParameter {
        name,
        value,
        expected: "a finite value in the field's documented range",
    };
    if !params.leaf_spacing.is_finite() || params.leaf_spacing <= 0.0 {
        return Err(bad("leaf_spacing", params.leaf_spacing));
    }
    if !params.start_angle.is_finite() {
        return Err(bad("start_angle", params.start_angle));
    }
    if !params.daylight_angle_tol.is_finite() || params.daylight_angle_tol < 0.0 {
        return Err(bad("daylight_angle_tol", params.daylight_angle_tol));
    }
    if !params.daylight_damping.is_finite()
        || params.daylight_damping <= 0.0
        || params.daylight_damping > 1.0
    {
        return Err(bad("daylight_damping", params.daylight_damping));
    }
    if !params.hyperbolic_origin.0.is_finite() {
        return Err(bad("hyperbolic_origin.0", params.hyperbolic_origin.0));
    }
    if !params.hyperbolic_origin.1.is_finite() {
        return Err(bad("hyperbolic_origin.1", params.hyperbolic_origin.1));
    }
    if !params.hyperbolic_zoom.is_finite() || params.hyperbolic_zoom <= 0.0 {
        return Err(bad("hyperbolic_zoom", params.hyperbolic_zoom));
    }
    Ok(())
}

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
/// One ascending scan (ascending index is a post-order).
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
/// Built with two flat scans and no stack. A subtree is one contiguous slice of
/// `order`.
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

/// Largest absolute coordinate in a layout, the scale for the relative epsilons.
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
/// Only the horizontal axis carries meaning: `x` is the summed branch length
/// from the root. The vertical axis is a display convenience.
///
/// Children of each node are ordered by leaf count, smallest first, ties on
/// node index. An internal node sits at the midpoint of its extreme two
/// children.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies `leaf_spacing`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates, or `MalformedTree` if a branch length or a layout
/// parameter cannot be drawn with.
pub fn dendrogram(tree: &Tree, params: Option<LayoutParams>) -> Result<Layout, BonsaiErrors> {
    check_branches(tree)?;
    let p = params.unwrap_or_default();
    check_params(&p)?;
    let n = tree.n_nodes();
    let counts = leaf_counts(tree);

    // Ladderised children as a CSR.
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

    // Descending index is a pre-order: parents are placed before children.
    let mut x = vec![0.0f64; n];
    for v in (0..n).rev() {
        for &c in &kids[ptr[v] as usize..ptr[v + 1] as usize] {
            x[c as usize] = x[v] + tree.branch(c);
        }
    }

    // Leaves in ladderised depth-first order.
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

    // Ascending index is a post-order: children are already placed.
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
/// wedge to its children in slices proportional to their leaf counts, and each
/// child is placed at its parent's position offset by its branch length along
/// the slice bisector. One descending scan, `O(n)`.
///
/// Sibling subtrees occupy disjoint wedges, so edges cannot cross provided each
/// wedge is at most `pi` wide (a convex cone). Wedges above `pi` occur at any
/// node holding more than half the leaves, e.g. in a caterpillar, so planarity
/// is not guaranteed there; no crossing has been produced on ladders, balanced
/// or random trees (`test_equal_angle_never_crosses_edges`).
///
/// Both axes carry meaning: the Euclidean distance from a node to its parent is
/// its branch length.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies `start_angle`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates, or `MalformedTree` if a branch length or a layout
/// parameter cannot be drawn with.
pub fn equal_angle(tree: &Tree, params: Option<LayoutParams>) -> Result<Layout, BonsaiErrors> {
    check_branches(tree)?;
    let p = params.unwrap_or_default();
    check_params(&p)?;
    let n = tree.n_nodes();
    let counts = leaf_counts(tree);

    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let mut lo = vec![0.0f64; n];
    let mut width = vec![0.0f64; n];

    let root = tree.root() as usize;
    lo[root] = p.start_angle;
    width[root] = TAU;

    // Descending index is a pre-order: a wedge is settled before its children.
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
/// Compares every pair of edges (node to parent), `O(n^2)`. Pairs sharing an
/// endpoint are skipped; any other intersection counts, including a collinear
/// overlap and an edge through an unrelated node.
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
fn has_edge_crossing(tree: &Tree, layout: &Layout) -> Result<bool, BonsaiErrors> {
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

    for i in 0..edges.len() {
        let (a, b) = edges[i];
        let (ax, ay) = (layout.x[a as usize], layout.y[a as usize]);
        let (bx, by) = (layout.x[b as usize], layout.y[b as usize]);
        let ab_len = (bx - ax).hypot(by - ay);
        for j in i + 1..edges.len() {
            let (c, d) = edges[j];
            if a == c || a == d || b == c || b == d {
                continue;
            }
            let (cx, cy) = (layout.x[c as usize], layout.y[c as usize]);
            let (dx, dy) = (layout.x[d as usize], layout.y[d as usize]);
            let cd_len = (dx - cx).hypot(dy - cy);

            // The determinant scales as |ab| * |cd|; a global epsilon would
            // call short far-out edges collinear and report false crossings.
            let eps = ORIENT_REL_EPS * ab_len * cd_len;

            // A degenerate segment is a point and cannot cross anything.
            if ab_len == 0.0 || cd_len == 0.0 {
                continue;
            }

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

////////////////////
// Equal daylight //
////////////////////

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
/// The sweep touches every node from every node; per-call allocation would
/// dominate.
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
/// consecutive ones. Exactness matters: [`sweep_node`] packs subtrees using
/// these widths, and a looser containing arc inflates the discrepancy by more
/// than an order of magnitude.
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

    // The widest cyclic gap is the daylight inside the subtree; the arc is the rest.
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
/// Neighbours are the children plus, unless root, the rest of the tree off the
/// parent: one tour slice per child, two slices for the parent side.
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
/// Gaps are signed; a negative one means overlapping wedges, which is normal
/// and not a planarity failure ([`has_edge_crossing`] decides that).
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
/// The parent-side subtree is held fixed and child subtrees are rotated
/// rigidly about the node, `damping` of the way towards equal gaps. The node is
/// left alone when the wedges already exceed the circle (packing them
/// oscillates).
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
    // The parent side anchors; at the root the first child stands in.
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
/// At every internal node, the sum of squared deviations of the wedge gaps from
/// their mean. Says nothing about planarity. Nodes with a subtree entirely
/// coincident with them are skipped.
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
/// Starts from [`equal_angle`] and sweeps the internal nodes from the root
/// down, rotating child subtrees rigidly so the gaps between them equalise. A
/// sweep is adopted only if it lowers the discrepancy and
/// [`has_edge_crossing`] finds no crossing, so a crossing never reaches the
/// caller. The step starts at `daylight_damping` and halves up to
/// [`DAYLIGHT_MAX_BACKTRACKS`] times until a sweep is accepted. The run stops
/// when no step improves, when a sweep moves nothing by more than
/// `daylight_angle_tol`, or after `daylight_max_sweeps`.
///
/// Each sweep and check is `O(n^2)`; above `daylight_max_nodes` the equal-angle
/// layout is returned unrefined.
///
/// ### Params
///
/// * `tree` - Tree to lay out
/// * `params` - Supplies the daylight gate, sweep cap, tolerance and
///   `start_angle`; `None` for the defaults
///
/// ### Returns
///
/// The coordinates, or `MalformedTree` if a branch length or a layout
/// parameter cannot be drawn with.
pub fn equal_daylight(tree: &Tree, params: Option<LayoutParams>) -> Result<Layout, BonsaiErrors> {
    let p = params.unwrap_or_default();
    let base = equal_angle(tree, params)?;

    if tree.n_nodes() > p.daylight_max_nodes {
        return Ok(base);
    }

    let tour = tour(tree);
    let eps = COINCIDENT_REL_EPS * layout_scale(&base);
    let mut scratch = Scratch::default();

    let mut score = daylight_discrepancy(tree, &tour, &base, eps, &mut scratch);
    let mut best = base;
    let mut step = p.daylight_damping;
    let mut converged = false;

    for _ in 0..p.daylight_max_sweeps {
        let mut accepted = false;
        // Backtracking line search on the rotation size.
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
                converged = moved < p.daylight_angle_tol;
                // Let the step grow back after an awkward sweep.
                step = (step * 2.0).min(p.daylight_damping);
                break;
            }
            step *= 0.5;
        }
        if !accepted || converged {
            break;
        }
    }

    Ok(best)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::NO_NODE;
    use crate::utils::rng::SplitMix64;
    use approx::assert_relative_eq;

    /// The daylight discrepancy of a layout, measured as `equal_daylight`
    /// measures it.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree the layout belongs to
    /// * `layout` - Coordinates to score
    ///
    /// ### Returns
    ///
    /// The discrepancy, in squared radians.
    fn discrepancy(tree: &Tree, layout: &Layout) -> f64 {
        let base = equal_angle(tree, None).expect("angle");
        let eps = COINCIDENT_REL_EPS * layout_scale(&base);
        daylight_discrepancy(tree, &tour(tree), layout, eps, &mut Scratch::default())
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
        Tree::from_parents(parent, vec![branch; n_leaves + 1], n_leaves)
            .expect("star fixture is malformed")
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
        Tree::from_parents(parent, branch, 8).expect("polytomy fixture is malformed")
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
        let mut rng = SplitMix64::new(seed);
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
            branch[a as usize] = rng.range(0.05, 2.0);
            branch[b as usize] = rng.range(0.05, 2.0);
            active.push(next_free);
            next_free += 1;
        }
        Tree::from_parents(parent, branch, n_leaves).expect("random fixture is malformed")
    }

    /// The fixture set every "does it work at all" test runs over: a balanced
    /// tree, a ladder, a star and a polytomy.
    ///
    /// ### Returns
    ///
    /// Named trees.
    fn shapes() -> Vec<(&'static str, Tree)> {
        let balanced = Tree::balanced_binary(16, 0.7).expect("balanced fixture is malformed");
        let ladder = Tree::ladder(17, 0.4).expect("ladder fixture is malformed");
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
    fn test_zero_length_branches_are_not_reported_as_crossings() {
        // Regression: a global crossing epsilon flagged zero-length branches.
        for n_leaves in [4usize, 8, 16] {
            let tree = Tree::balanced_binary(n_leaves, 0.0).expect("balanced fixture");
            let layout = equal_angle(&tree, None).expect("equal angle");
            assert!(
                !has_edge_crossing(&tree, &layout).expect("crossing check"),
                "{n_leaves} leaves on zero-length branches reported a crossing"
            );
        }
    }

    #[test]
    fn test_short_edges_far_from_the_origin_are_not_false_positives() {
        // Regression: a global epsilon swamped short edges far from the origin.
        let tree = Tree::from_parents(
            vec![4, 4, 5, 5, 6, 6, NO_NODE],
            vec![1e-3, 5.8e-3, 1e3, 0.24, 1e3, 1e-3, 0.0],
            4,
        )
        .expect("mixed-scale fixture");
        let layout = equal_angle(&tree, None).expect("equal angle");
        assert!(
            !has_edge_crossing(&tree, &layout).expect("crossing check"),
            "mixed branch-length scales produced a false crossing"
        );
    }

    #[test]
    fn test_hyperbolic_keeps_radial_order_at_extreme_radii() {
        // Regression: squaring r overflowed above 1.3e154 and inverted radial order.
        let radii = [0.0f64, 1.0, 1e10, 1e100, 1e160, 1e300];
        let layout = Layout {
            x: radii.to_vec(),
            y: vec![0.0; radii.len()],
        };
        let mapped = layout.hyperbolic(None);

        let mut previous = -1.0f64;
        for (i, &r) in radii.iter().enumerate() {
            let out = mapped.x[i].hypot(mapped.y[i]);
            assert!(out.is_finite(), "r = {r:e} mapped to {out}");
            // `<=`: the result saturates at exactly 1 past about 1e16.
            assert!(out <= 1.0, "r = {r:e} landed outside the rim at {out}");
            assert!(
                out >= previous,
                "r = {r:e} broke radial order: {out} after {previous}"
            );
            previous = out;
        }
        assert_eq!(mapped.x[0], 0.0, "the origin must map to itself");
    }

    #[test]
    fn test_every_layout_gives_every_node_finite_coordinates() {
        for (name, tree) in shapes() {
            let dendro = dendrogram(&tree, None).expect(name);
            let angle = equal_angle(&tree, None).expect(name);
            let daylight = equal_daylight(&tree, None).expect(name);
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
        // The test that matters: the whole reason equal-angle is the default
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
            let layout = equal_daylight(&tree, None).expect("layout");
            assert!(
                !has_edge_crossing(&tree, &layout).expect("check"),
                "seed {seed}: equal-daylight produced a crossing"
            );
        }
        for (name, tree) in shapes() {
            let layout = equal_daylight(&tree, None).expect(name);
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
            let daylight = equal_daylight(&tree, None).expect(name);
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
            let layout = equal_daylight(&tree, None).expect("layout");
            let before = discrepancy(&tree, &equal_angle(&tree, None).expect("angle"));
            let after = discrepancy(&tree, &layout);
            assert!(
                after <= before,
                "seed {seed}: discrepancy rose from {before} to {after}"
            );
        }
    }

    #[test]
    fn test_equal_daylight_actually_improves_on_equal_angle() {
        // A shape with obviously wasted daylight: a ladder, whose equal-angle
        // drawing crowds every rung into one narrowing wedge.
        let tree = Tree::ladder(24, 1.0).expect("ladder");
        let layout = equal_daylight(&tree, None).expect("layout");
        let before = discrepancy(&tree, &equal_angle(&tree, None).expect("angle"));
        let after = discrepancy(&tree, &layout);
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
            let layout = equal_daylight(&tree, None).expect("layout");
            let before = discrepancy(&tree, &equal_angle(&tree, None).expect("angle"));
            let after = discrepancy(&tree, &layout);
            assert!(
                after < before,
                "{leaves} leaves: discrepancy did not fall, {before} to {after}"
            );
            assert!(!has_edge_crossing(&tree, &layout).expect("check"));
        }
    }

    #[test]
    fn test_equal_daylight_terminates_on_a_layout_it_cannot_improve() {
        // A star is already perfectly lit: every gap around the root is equal,
        // so the run must stop there rather than grind through the cap.
        let tree = star(64, 1.0);
        let layout = equal_daylight(&tree, None).expect("layout");
        assert_relative_eq!(discrepancy(&tree, &layout), 0.0, epsilon = 1e-20);
    }

    #[test]
    fn test_equal_daylight_is_gated_by_node_count() {
        let tree = Tree::balanced_binary(64, 1.0).expect("balanced");
        let params = LayoutParams {
            daylight_max_nodes: 8,
            ..LayoutParams::default()
        };
        let layout = equal_daylight(&tree, Some(params)).expect("layout");
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
        let mut rng = SplitMix64::new(11);
        let n = 4096;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(rng.range(-1e12, 1e12));
            y.push(rng.range(-1e12, 1e12));
        }
        let disk = Layout { x, y }.hyperbolic(None);
        for i in 0..n {
            let r = disk.x[i].hypot(disk.y[i]);
            assert!(r < 1.0, "point {i} landed at radius {r}");
        }
    }

    #[test]
    fn test_hyperbolic_preserves_the_angle() {
        let mut rng = SplitMix64::new(29);
        let params = LayoutParams {
            hyperbolic_origin: (0.4, -1.7),
            hyperbolic_zoom: 2.5,
            ..LayoutParams::default()
        };
        let n = 512;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(rng.range(-50.0, 50.0));
            y.push(rng.range(-50.0, 50.0));
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
                assert_eq!(
                    equal_daylight(&tree, None).expect(name),
                    equal_daylight(&tree, None).expect(name),
                    "{name}: equal-daylight drifted"
                );
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
        let daylight = equal_daylight(&tree, None).expect("equal daylight");
        assert_eq!(daylight, angle);

        let disk = angle.hyperbolic(None);
        assert!(
            (0..disk.n_nodes()).all(|i| disk.x[i].hypot(disk.y[i]) < 1.0),
            "a node escaped the unit disk"
        );
    }

    #[test]
    fn test_rejects_layout_parameters_that_cannot_be_drawn_with() {
        // Regression: a non-finite `leaf_spacing` used to give an all-NaN layout.
        let tree = Tree::balanced_binary(4, 1.0).expect("balanced");
        let bad = [
            LayoutParams {
                leaf_spacing: f64::INFINITY,
                ..Default::default()
            },
            LayoutParams {
                leaf_spacing: 0.0,
                ..Default::default()
            },
            LayoutParams {
                start_angle: f64::NAN,
                ..Default::default()
            },
            LayoutParams {
                daylight_damping: 0.0,
                ..Default::default()
            },
            LayoutParams {
                daylight_angle_tol: f64::NAN,
                ..Default::default()
            },
            LayoutParams {
                hyperbolic_zoom: f64::INFINITY,
                ..Default::default()
            },
            LayoutParams {
                hyperbolic_origin: (0.0, f64::NAN),
                ..Default::default()
            },
        ];
        for params in bad {
            assert!(
                matches!(
                    dendrogram(&tree, Some(params)),
                    Err(BonsaiErrors::BadParameter { .. })
                ),
                "dendrogram accepted {params:?}"
            );
            assert!(
                matches!(
                    equal_angle(&tree, Some(params)),
                    Err(BonsaiErrors::BadParameter { .. })
                ),
                "equal_angle accepted {params:?}"
            );
            assert!(
                matches!(
                    equal_daylight(&tree, Some(params)),
                    Err(BonsaiErrors::BadParameter { .. })
                ),
                "equal_daylight accepted {params:?}"
            );
        }
        // The defaults, and a plausible non-default, still go through.
        assert!(dendrogram(&tree, None).is_ok());
        assert!(
            dendrogram(
                &tree,
                Some(LayoutParams {
                    leaf_spacing: 2.5,
                    start_angle: 0.3,
                    ..Default::default()
                })
            )
            .is_ok()
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
