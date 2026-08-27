//! Global branch-length optimisation.
//!
//! Implements the last paragraph of SPEC.md section 6. Collapsing everything
//! but one edge into an effective leaf on each side turns that edge into the
//! one-dimensional problem [`crate::model::branch::optimise_edge`] already
//! solves; doing it for every edge at once needs the effective leaf on *both*
//! sides of every edge, which is one sweep in each direction over the arena.
//!
//! The pruning recursion in [`crate::model::likelihood`] supplies one side: for
//! every node `k`, the subtree below `k`. This module supplies the other, the
//! "up" value: for a non-root node `k` with parent `a`, the effective leaf at
//! `a` of everything except `k`'s own subtree. Down values settle in post-order,
//! up values in pre-order, and the arena invariant makes both a bare linear
//! scan, the second one backwards.
//!
//! Optimising one edge moves the effective leaves every other edge sees, so the
//! two are iterated. The proposal is computed for all edges from one pair of
//! sweeps, which is not on its own guaranteed to improve the tree; see
//! [`optimise_branch_lengths`] for why it is safe anyway.

use crate::errors::BonsaiErrors;
use crate::model::branch::optimise_edge;
use crate::model::likelihood::NodeState;
use crate::tree::Tree;
use crate::utils::kernels::prep_edge;
use crate::utils::traits::{BonsaiFloat, narrow, wide};

/////////////////////
// Two-sided sweep //
/////////////////////

/// The effective leaf *above* every node: everything outside its own subtree.
///
/// Stored the same way as [`NodeState`], two flat row-major blocks of
/// `n_nodes * p`, so a node's row is one contiguous span and row indices agree
/// between the two sides. The root's row is zeroed: there is nothing above the
/// root, which is the same thing as an effective leaf of zero precision.
///
/// A row is meaningful only after [`UpState::sweep`], and only for the tree and
/// down rows it was swept against.
#[derive(Clone, Debug)]
pub struct UpState<T> {
    /// Up means, `[node][feature]`, row-major.
    m: Vec<T>,
    /// Up precisions, `[node][feature]`, row-major.
    w: Vec<T>,
    /// Number of features.
    p: usize,
    /// Number of nodes, leaves included.
    n_nodes: usize,
    /// Scratch for the polytomy path: the total precision and mean at a node,
    /// before any one child is removed. Grown on demand.
    scratch: Vec<f64>,
}

impl<T: BonsaiFloat> UpState<T> {
    /// Allocate up-state for a tree.
    ///
    /// ### Params
    ///
    /// * `n_nodes` - Total node count of the tree this state will serve
    /// * `p` - Number of features
    ///
    /// ### Returns
    ///
    /// Zeroed state; call [`UpState::sweep`] before reading any row.
    pub fn new(n_nodes: usize, p: usize) -> Self {
        Self {
            m: vec![T::zero(); n_nodes * p],
            w: vec![T::zero(); n_nodes * p],
            p,
            n_nodes,
            scratch: Vec::new(),
        }
    }

    /// Up means of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's row of up means, positioned at the node's parent.
    #[inline]
    pub fn means(&self, node: u32) -> &[T] {
        let lo = node as usize * self.p;
        &self.m[lo..lo + self.p]
    }

    /// Up precisions of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node whose row is wanted
    ///
    /// ### Returns
    ///
    /// The node's row of up precisions, positioned at the node's parent.
    #[inline]
    pub fn precisions(&self, node: u32) -> &[T] {
        let lo = node as usize * self.p;
        &self.w[lo..lo + self.p]
    }

    /// Fill every node's up row from a settled set of down rows.
    ///
    /// Writing `Wd[c] = W[c] / (1 + t[c] * W[c])` for the diffusion-corrected
    /// precision of a child and `Wup[a]` for the diffusion-corrected precision
    /// of `a`'s own up-part, the total effective precision seen at `a` is
    ///
    /// ```text
    /// Wtot[a] = sum_{c in C(a)} Wd[c] + Wup[a]
    /// Wup[a]  = 1 / (t[a] + 1 / Wu[a])          (zero at the root)
    /// ```
    ///
    /// with the matching weighted mean, and the up value of a child `k` is that
    /// total with `k`'s own contribution removed, `Wu[k] = Wtot[a] - Wd[k]`.
    /// Diffusing `a`'s up value along `t[a]` is the same operation the down
    /// recursion applies to a child, which is what makes this the mirror image
    /// of SPEC.md section 4 rather than a separate derivation.
    ///
    /// Parents are visited before children, so this is a descending scan over
    /// the internal nodes: the arena invariant puts every parent at a strictly
    /// larger index than its children, which makes descending index order a
    /// valid pre-order.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose topology and branch lengths to use
    /// * `down` - Down rows, already settled by [`NodeState::prune`] against
    ///   this same tree
    pub fn sweep(&mut self, tree: &Tree, down: &NodeState<T>) {
        debug_assert_eq!(tree.n_nodes(), self.n_nodes);
        debug_assert_eq!(down.n_features(), self.p);

        let root = tree.root() as usize * self.p;
        self.m[root..root + self.p].fill(T::zero());
        self.w[root..root + self.p].fill(T::zero());

        for a in (tree.n_leaves()..self.n_nodes).rev() {
            self.sweep_node(tree, down, a as u32);
        }
    }

    /// Write the up rows of one node's children.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree whose topology and branch lengths to use
    /// * `down` - Settled down rows
    /// * `a` - Internal node whose children are to be written; its own up row
    ///   must already be settled
    fn sweep_node(&mut self, tree: &Tree, down: &NodeState<T>, a: u32) {
        let p = self.p;
        let kids = tree.children(a);
        let is_root = tree.parent(a).is_none();
        let t_a = tree.branch(a);

        // Children sit strictly below their parent, so one split separates the
        // rows being written from the row being read.
        let split = a as usize * p;
        let (lo_m, hi_m) = self.m.split_at_mut(split);
        let (lo_w, hi_w) = self.w.split_at_mut(split);
        let (up_m, up_w) = (&hi_m[..p], &hi_w[..p]);

        if kids.len() == 2 {
            let (k, l) = (kids[0], kids[1]);
            let (t_k, t_l) = (tree.branch(k), tree.branch(l));
            let (m_k, w_k) = (down.means(k), down.precisions(k));
            let (m_l, w_l) = (down.means(l), down.precisions(l));

            // `children` is built by a scan in ascending node order, so the two
            // destination rows are ordered and disjoint.
            debug_assert!(k < l);
            let (first_m, second_m) = lo_m.split_at_mut(l as usize * p);
            let (first_w, second_w) = lo_w.split_at_mut(l as usize * p);
            let base = k as usize * p;
            let (out_m_k, out_w_k) =
                (&mut first_m[base..base + p], &mut first_w[base..base + p]);
            let (out_m_l, out_w_l) = (&mut second_m[..p], &mut second_w[..p]);

            for g in 0..p {
                let (w_up, m_up) = up_part(is_root, t_a, up_w[g], up_m[g]);
                let wk = wide(w_k[g]);
                let wl = wide(w_l[g]);
                let wdk = wk / (1.0 + t_k * wk);
                let wdl = wl / (1.0 + t_l * wl);
                let (mk, ml) = (wide(m_k[g]), wide(m_l[g]));

                // Two children is the common case, so each leave-one-out total
                // is formed directly rather than as `Wtot - Wd[c]`: the two are
                // similar in magnitude whenever one child dominates, and the
                // subtraction would then lose the leading digits of the answer.
                let tot_k = wdl + w_up;
                let tot_l = wdk + w_up;
                out_w_k[g] = narrow(tot_k);
                out_w_l[g] = narrow(tot_l);
                // Convex combinations, for the reason given in
                // `prune_binary_scalar`: the weight lies in `[0, 1]`, so the
                // mean is pinned between the two it interpolates and cannot
                // cancel. At the root the weight is exactly zero and the answer
                // is the sibling's mean untouched.
                out_m_k[g] = narrow(ml + (m_up - ml) * (w_up / tot_k));
                out_m_l[g] = narrow(mk + (m_up - mk) * (w_up / tot_l));
            }
            return;
        }

        // Polytomy. Here there is no leave-one-out total that avoids a
        // subtraction without carrying a prefix and a suffix scan per feature,
        // which costs more memory traffic than the case is worth: polytomies
        // are what the search is actively removing, and the subtraction is
        // benign unless one child holds nearly all the precision at `a`.
        if self.scratch.len() < 2 * p {
            self.scratch.resize(2 * p, 0.0);
        }
        let (w_tot, m_tot) = self.scratch.split_at_mut(p);
        let (m_a, w_a) = (down.means(a), down.precisions(a));
        for g in 0..p {
            let (w_up, m_up) = up_part(is_root, t_a, up_w[g], up_m[g]);
            // The down sweep has already summed the children's diffused
            // precisions into `W[a]`, so the total costs one addition.
            let ma = wide(m_a[g]);
            let tot = wide(w_a[g]) + w_up;
            w_tot[g] = tot;
            m_tot[g] = ma + (m_up - ma) * (w_up / tot);
        }

        for &c in kids {
            let (m_c, w_c) = (down.means(c), down.precisions(c));
            let t_c = tree.branch(c);
            let base = c as usize * p;
            for g in 0..p {
                let wc = wide(w_c[g]);
                let wdc = wc / (1.0 + t_c * wc);
                let rest = w_tot[g] - wdc;
                // `m_tot` is the convex combination of `M[c]` and the answer,
                // so extrapolating away from `M[c]` recovers the answer exactly
                // and never forms a ratio of sums.
                let mc = wide(m_c[g]);
                lo_w[base + g] = narrow(rest);
                lo_m[base + g] = narrow(m_tot[g] + (m_tot[g] - mc) * (wdc / rest));
            }
        }
    }
}

/// The up-part of a node as its children see it: its own up value diffused
/// along the branch above it.
///
/// ### Params
///
/// * `is_root` - Whether the node is the root, which has no up-part at all
/// * `t_a` - Length of the branch above the node
/// * `w_up` - The node's own up precision
/// * `m_up` - The node's own up mean
///
/// ### Returns
///
/// The diffused precision and the mean, which diffusion leaves alone. Zero
/// precision at the root, where the mean is arbitrary and is returned as zero
/// so that the convex combinations downstream stay finite.
#[inline]
fn up_part<T: BonsaiFloat>(is_root: bool, t_a: f64, w_up: T, m_up: T) -> (f64, f64) {
    if is_root {
        (0.0, 0.0)
    } else {
        (1.0 / (t_a + 1.0 / wide(w_up)), wide(m_up))
    }
}

/////////////////////////
// Global optimisation //
/////////////////////////

/// Backtracking budget for one iteration.
///
/// The proposal is an ascent direction (see [`optimise_branch_lengths`]), so
/// some step size along it improves the tree unless the tree is already
/// stationary. Twenty halvings take the step to `1e-6` of its length, far below
/// the point at which a failure to improve means stationarity rather than an
/// overlong step. Measured on this module's fixtures, 2026-08-27: about one
/// iteration in five takes a single halving and the rest take the full step,
/// and the budget is only ever exhausted on the last iteration of all, which is
/// how the loop discovers it has converged.
const MAX_BACKTRACK: usize = 20;

/// Factor by which a rejected step is shrunk.
///
/// Plain halving. Nothing here is sensitive to the schedule: the accepted step
/// is usually the full one, and the shrinking exists to survive the cases where
/// it is not.
const BACKTRACK_SHRINK: f64 = 0.5;

/// Floor on the scale used by the relative convergence test.
///
/// Loglikelihoods here are defined up to an additive constant (SPEC.md section
/// 3.2), so one can sit arbitrarily close to zero on a small enough fixture.
/// Without a floor the relative test would then demand an improvement of zero
/// and never terminate.
const LOGLIK_SCALE_FLOOR: f64 = 1.0;

/// Stopping rule for [`optimise_branch_lengths`].
#[derive(Clone, Copy, Debug)]
pub struct GlobalBranchParams {
    /// Maximum number of sweep-and-optimise iterations.
    pub max_iter: usize,
    /// Relative improvement in the tree loglikelihood below which the iteration
    /// stops, compared against `max(|L|, 1)`.
    pub tol: f64,
}

impl GlobalBranchParams {
    /// Build a stopping rule.
    ///
    /// ### Params
    ///
    /// * `max_iter` - Maximum number of iterations
    /// * `tol` - Relative improvement below which to stop
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }
}

impl Default for GlobalBranchParams {
    /// Defaults chosen by measurement, 2026-08-27.
    ///
    /// Each edge's proposal is that edge's exact conditional optimum rather
    /// than a small step, so the outer iteration converges linearly at a rate
    /// set only by how strongly the edges couple. Measured on the fixtures in
    /// this module, 2026-08-27 (7 to 16 leaves, 64 to 1024 features, branch
    /// lengths started a factor of four either side of the optimum): 7
    /// iterations for a star, 16 for a ladder and 19 to 22 for balanced binary
    /// trees, from which `max_iter` is a runaway guard at roughly three times
    /// the worst case rather than a working limit. The starting point barely
    /// matters, which is what one would expect of an iteration whose every
    /// component step is an exact solve.
    ///
    /// `tol` is far tighter than the search needs. Branch lengths feed the
    /// merge scores, and where this loop stops should never be what separates
    /// two candidate topologies.
    fn default() -> Self {
        Self {
            max_iter: 64,
            tol: 1e-10,
        }
    }
}

/// Optimise every branch length in the tree.
///
/// One iteration sweeps down (the pruning recursion), sweeps up, and then
/// solves every edge with [`crate::model::branch::optimise_edge`] on
///
/// ```text
/// s[g] = 1 / W_down[k][g] + 1 / W_up[k][g]
/// d[g] = (M_down[k][g] - M_up[k][g])^2
/// ```
///
/// Each of those solves is the exact optimum of that edge with every other
/// branch length held fixed, because the whole tree loglikelihood as a function
/// of `t[k]` alone is the edge form of SPEC.md section 6 plus a constant. Taken
/// one at a time they would each improve the tree; taken together, as they are
/// here, they need not, because moving one edge moves the effective leaves the
/// others were solved against.
///
/// What survives is that the proposal is an *ascent direction*. The
/// loglikelihood's partial derivative in `t[k]` is `-f(t[k])/2` for that edge's
/// `f`, and `optimise_edge` returns a length above the current one exactly when
/// `f(t[k])` is negative, so every component of the proposed move agrees in
/// sign with its own partial derivative. A step that fails to improve the tree
/// is therefore too long rather than wrong, and is retried shorter. That is
/// what makes the loop monotone in the loglikelihood.
///
/// ### Params
///
/// * `tree` - Tree whose branch lengths are updated in place
/// * `state` - Node state for this tree, left settled against the returned
///   branch lengths
/// * `params` - Stopping rule, or `None` for [`GlobalBranchParams::default`]
///
/// ### Returns
///
/// The tree loglikelihood at the returned branch lengths, which is never below
/// its value at the supplied ones. `RootFindDiverged` if an edge solve fails.
pub fn optimise_branch_lengths<T: BonsaiFloat>(
    tree: &mut Tree,
    state: &mut NodeState<T>,
    params: Option<GlobalBranchParams>,
) -> Result<f64, BonsaiErrors> {
    let params = params.unwrap_or_default();
    let p = state.n_features();
    let n_nodes = tree.n_nodes();
    let root = tree.root() as usize;

    let mut up = UpState::new(n_nodes, p);
    let mut s = vec![0.0f64; p];
    let mut d = vec![0.0f64; p];
    let mut proposal = vec![0.0f64; n_nodes];
    let mut current: Vec<f64> = tree.branches().to_vec();
    let mut best = state.prune(tree);

    for _ in 0..params.max_iter {
        up.sweep(tree, state);
        for k in 0..n_nodes {
            if k == root {
                proposal[k] = current[k];
                continue;
            }
            let node = k as u32;
            let upper = prep_edge(
                state.means(node),
                state.precisions(node),
                up.means(node),
                up.precisions(node),
                &mut s,
                &mut d,
            );
            proposal[k] = optimise_edge(&s, &d, upper)?;
        }
        if proposal == current {
            return Ok(best);
        }

        let mut alpha = 1.0f64;
        let mut improved = None;
        for _ in 0..MAX_BACKTRACK {
            let branches = tree.branches_mut();
            for k in 0..n_nodes {
                branches[k] = current[k] + alpha * (proposal[k] - current[k]);
            }
            let l = state.prune(tree);
            if l > best {
                improved = Some(l);
                break;
            }
            alpha *= BACKTRACK_SHRINK;
        }

        // A step worth less than the tolerance is undone rather than kept, so
        // that a second call on a converged tree returns the identical tree
        // instead of one nudged by a further sub-tolerance step.
        match improved {
            Some(l) if l - best > params.tol * best.abs().max(LOGLIK_SCALE_FLOOR) => {
                current.copy_from_slice(tree.branches());
                best = l;
            }
            _ => {
                tree.branches_mut().copy_from_slice(&current);
                // Resettle the rows the rejected step left behind, so the state
                // the caller keeps still describes the tree it holds.
                state.prune(tree);
                return Ok(best);
            }
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
    use crate::tree::{NO_NODE, Tree};
    use crate::utils::kernels::edge_newton;
    use approx::assert_relative_eq;

    /// Deterministic uniform stream, so the fixtures need no rng dependency.
    struct Rng(u64);

    impl Rng {
        /// Start the stream.
        ///
        /// ### Params
        ///
        /// * `seed` - Non-zero seed
        ///
        /// ### Returns
        ///
        /// The generator.
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// Next uniform on `(0, 1)`.
        ///
        /// ### Returns
        ///
        /// The variate.
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        /// Next standard normal, by Box-Muller.
        ///
        /// ### Returns
        ///
        /// The variate.
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Leaf means and precisions with no relation to any tree.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - Number of leaves
    /// * `p` - Number of features
    /// * `seed` - Stream seed
    ///
    /// ### Returns
    ///
    /// Row-major means and precisions, `[leaf][feature]`.
    fn leaf_data(n_leaves: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let mut m = Vec::with_capacity(n_leaves * p);
        let mut w = Vec::with_capacity(n_leaves * p);
        for _ in 0..n_leaves * p {
            m.push(rng.uniform() * 4.0 - 2.0);
            w.push(0.25 + rng.uniform() * 3.0);
        }
        (m, w)
    }

    /// Leaf data generated by Brownian motion on a tree, then observed with
    /// noise.
    ///
    /// SPEC.md section 13.1, in the transformed units of section 3.1 where
    /// every feature has unit diffusion variance: a step of variance `t[k]`
    /// along each branch, then a per-leaf observation of variance `1 / w`.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to simulate on
    /// * `p` - Number of features
    /// * `precision` - Observation precision given to every leaf and feature
    /// * `seed` - Stream seed
    ///
    /// ### Returns
    ///
    /// Row-major leaf means and precisions.
    fn simulate(tree: &Tree, p: usize, precision: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let n = tree.n_nodes();
        let mut x = vec![0.0f64; n * p];
        // Descending index order is a pre-order, so a parent is always drawn
        // before its children.
        for k in (0..n - 1).rev() {
            let a = tree.parent(k as u32).expect("only the root has no parent") as usize;
            let sd = tree.branch(k as u32).sqrt();
            for g in 0..p {
                x[k * p + g] = x[a * p + g] + sd * rng.normal();
            }
        }
        let sd = precision.sqrt().recip();
        let n_leaves = tree.n_leaves();
        let mut m = vec![0.0f64; n_leaves * p];
        for i in 0..n_leaves * p {
            m[i] = x[i] + sd * rng.normal();
        }
        (m, vec![precision; n_leaves * p])
    }

    /// A six-leaf tree whose root has three children, so that every edge is
    /// separately identifiable.
    ///
    /// ### Params
    ///
    /// * `branch` - Length of each of the nine non-root branches
    ///
    /// ### Returns
    ///
    /// The tree.
    fn three_cherries(branch: &[f64; 9]) -> Tree {
        let mut b = branch.to_vec();
        b.push(0.0);
        Tree::from_parents(vec![6, 6, 7, 7, 8, 8, 9, 9, 9, NO_NODE], b, 6).unwrap()
    }

    /// Rebuild a tree as if rooted at `new_root` and with the subtree below
    /// `exclude` deleted, suppressing every degree-two node that leaves behind.
    ///
    /// The likelihood is root-independent (SPEC.md section 2, S14), and a
    /// degree-two internal node contributes nothing to it and merely adds its
    /// two branch lengths, so pruning this tree gives the up value of `exclude`
    /// at its parent by a route sharing no code with [`UpState::sweep`].
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to rebuild
    /// * `new_root` - Node to root the result at, which must be the parent of
    ///   `exclude`
    /// * `exclude` - Node whose subtree is deleted
    ///
    /// ### Returns
    ///
    /// The rebuilt tree and the original index of each of its leaves, or `None`
    /// if the deletion leaves the new root with fewer than two neighbours and
    /// so no tree this arena can hold.
    fn reroot_without(tree: &Tree, new_root: u32, exclude: u32) -> Option<(Tree, Vec<usize>)> {
        let n = tree.n_nodes();
        let n_leaves = tree.n_leaves();
        let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for k in 0..n as u32 {
            if k == exclude {
                continue;
            }
            if let Some(a) = tree.parent(k) {
                adj[k as usize].push((a, tree.branch(k)));
                adj[a as usize].push((k, tree.branch(k)));
            }
        }

        // Dropping that one edge disconnects the whole excluded subtree.
        let mut alive = vec![false; n];
        alive[new_root as usize] = true;
        let mut stack = vec![new_root];
        while let Some(v) = stack.pop() {
            for i in 0..adj[v as usize].len() {
                let u = adj[v as usize][i].0 as usize;
                if !alive[u] {
                    alive[u] = true;
                    stack.push(u as u32);
                }
            }
        }
        for v in 0..n {
            if alive[v] {
                adj[v].retain(|&(u, _)| alive[u as usize]);
            } else {
                adj[v].clear();
            }
        }
        if adj[new_root as usize].len() < 2 {
            return None;
        }

        loop {
            let mut changed = false;
            for v in n_leaves..n {
                if !alive[v] || v == new_root as usize || adj[v].len() != 2 {
                    continue;
                }
                let (a, ta) = adj[v][0];
                let (b, tb) = adj[v][1];
                adj[a as usize].retain(|&(u, _)| u != v as u32);
                adj[b as usize].retain(|&(u, _)| u != v as u32);
                adj[a as usize].push((b, ta + tb));
                adj[b as usize].push((a, ta + tb));
                adj[v].clear();
                alive[v] = false;
                changed = true;
            }
            if !changed {
                break;
            }
        }

        // Root it: breadth-first from `new_root`, recording depth so that the
        // arena's ordering invariant can be satisfied by construction.
        let mut old_parent = vec![NO_NODE; n];
        let mut old_branch = vec![0.0f64; n];
        let mut depth = vec![0usize; n];
        let mut seen = vec![false; n];
        seen[new_root as usize] = true;
        let mut order = vec![new_root];
        let mut head = 0usize;
        while head < order.len() {
            let v = order[head];
            head += 1;
            for i in 0..adj[v as usize].len() {
                let (u, t) = adj[v as usize][i];
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    old_parent[u as usize] = v;
                    old_branch[u as usize] = t;
                    depth[u as usize] = depth[v as usize] + 1;
                    order.push(u);
                }
            }
        }

        // Only original leaves can end up with a single neighbour: the one node
        // that lost one is `new_root`, which is never suppressed.
        let leaf_map: Vec<usize> = (0..n_leaves).filter(|&i| alive[i]).collect();
        let mut internals: Vec<usize> = (n_leaves..n).filter(|&v| alive[v]).collect();
        internals.sort_by_key(|&v| (std::cmp::Reverse(depth[v]), v));

        let mut index = vec![usize::MAX; n];
        for (slot, &old) in leaf_map.iter().enumerate() {
            index[old] = slot;
        }
        for (slot, &old) in internals.iter().enumerate() {
            index[old] = leaf_map.len() + slot;
        }

        let n_new = leaf_map.len() + internals.len();
        let mut parent = vec![NO_NODE; n_new];
        let mut branch = vec![0.0f64; n_new];
        for old in 0..n {
            if !alive[old] {
                continue;
            }
            let new = index[old];
            parent[new] = match old_parent[old] {
                NO_NODE => NO_NODE,
                a => index[a as usize] as u32,
            };
            branch[new] = old_branch[old];
        }
        let rebuilt = Tree::from_parents(parent, branch, leaf_map.len()).unwrap();
        Some((rebuilt, leaf_map))
    }

    /// Worst scaled stationarity residual a converged tree is allowed.
    ///
    /// The loglikelihood is quadratic at its maximum, so its gradient can only
    /// be driven down to the square root of the smallest change in the
    /// loglikelihood the iteration can see; past that the line search cannot
    /// separate an improving step from a rounding error. Measured on the
    /// fixtures below, 2026-08-27: the iteration stalls at a worst scaled
    /// residual of `4.2e-7` and stays there whatever tolerance it is given, so
    /// this is the floor and not a stopping rule that could be tightened.
    const STATIONARY_RESIDUAL: f64 = 1e-6;

    /// Relative displacement used to show a branch length is at a maximum.
    ///
    /// Large enough that the quadratic term dominates the loglikelihood's own
    /// rounding noise, which is around `1e-10` absolute on these fixtures
    /// against a curvature of order ten; small enough to stay well inside the
    /// quadratic region.
    const PERTURBATION: f64 = 1e-3;

    /// Every edge's stationarity residual, scaled by its value at `t = 0`.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree to check
    /// * `state` - Node state, settled against that tree
    ///
    /// ### Returns
    ///
    /// Per non-root node, its branch length, `|f(t)| / |f(0)|`, and `f(0)`.
    fn residuals(tree: &Tree, state: &NodeState<f64>) -> Vec<(f64, f64, f64)> {
        let p = state.n_features();
        let mut up = UpState::new(tree.n_nodes(), p);
        up.sweep(tree, state);
        let (mut s, mut d) = (vec![0.0; p], vec![0.0; p]);
        (0..tree.n_nodes() as u32 - 1)
            .map(|k| {
                prep_edge(
                    state.means(k),
                    state.precisions(k),
                    up.means(k),
                    up.precisions(k),
                    &mut s,
                    &mut d,
                );
                let t = tree.branch(k);
                let at_zero = edge_newton(&s, &d, 0.0).0;
                let f = edge_newton(&s, &d, t).0;
                (t, f.abs() / at_zero.abs().max(f64::MIN_POSITIVE), at_zero)
            })
            .collect()
    }

    /// Assert that no single edge of a converged tree can be improved.
    ///
    /// Two statements. The stationarity residual of every edge is at the floor
    /// described on [`STATIONARY_RESIDUAL`], with a branch pinned at zero
    /// counting as stationary when the loglikelihood is already decreasing in
    /// its length. And, the statement that actually says "optimum" without
    /// reusing any of the optimiser's own machinery, displacing any single
    /// branch either way does not increase the tree loglikelihood.
    ///
    /// ### Params
    ///
    /// * `tree` - Converged tree
    /// * `state` - Node state, settled against that tree
    /// * `m` - Leaf means the tree was optimised against
    /// * `w` - Leaf precisions the tree was optimised against
    fn assert_no_edge_can_be_improved(
        tree: &Tree,
        state: &NodeState<f64>,
        m: &[f64],
        w: &[f64],
    ) {
        for (k, (t, relative, at_zero)) in residuals(tree, state).into_iter().enumerate() {
            if t == 0.0 {
                assert!(at_zero >= 0.0, "edge {k} sits at zero with f(0) = {at_zero:e}");
            } else {
                assert!(
                    relative < STATIONARY_RESIDUAL,
                    "edge {k}: residual {relative:e} at t = {t}"
                );
            }
        }

        let p = state.n_features();
        let mut probe = tree.clone();
        let mut probe_state = NodeState::new(tree.n_nodes(), p, m, w).unwrap();
        let best = probe_state.prune(&probe);
        for k in 0..tree.n_nodes() - 1 {
            let t = tree.branch(k as u32);
            let step = if t == 0.0 { PERTURBATION } else { t * PERTURBATION };
            for delta in [step, -step] {
                let moved = t + delta;
                if moved < 0.0 {
                    continue;
                }
                probe.branches_mut()[k] = moved;
                let l = probe_state.prune(&probe);
                assert!(
                    l <= best,
                    "edge {k}: moving {t} to {moved} gained {}",
                    l - best
                );
            }
            probe.branches_mut()[k] = t;
        }
    }

    #[test]
    fn test_up_values_match_an_explicit_reroot() {
        // The load-bearing test. Rerooting at the parent and deleting the
        // subtree is an independent route to the same effective leaf, sharing
        // nothing with the sweep but the pruning recursion itself.
        let p = 24usize;
        for (tree, seed) in [
            (Tree::balanced_binary(8, 0.7).unwrap(), 11u64),
            (
                three_cherries(&[0.4, 1.3, 0.9, 0.2, 1.8, 0.5, 0.7, 1.1, 0.3]),
                12,
            ),
            (Tree::ladder(7, 0.6).unwrap(), 13),
            (
                Tree::from_parents(vec![5, 5, 5, 5, 5, NO_NODE], vec![0.9; 6], 5).unwrap(),
                14,
            ),
        ] {
            let (m, w) = leaf_data(tree.n_leaves(), p, seed);
            let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
            state.prune(&tree);
            let mut up = UpState::new(tree.n_nodes(), p);
            up.sweep(&tree, &state);

            let mut checked = 0usize;
            for k in 0..tree.n_nodes() as u32 - 1 {
                let a = tree.parent(k).expect("k is not the root");
                let Some((sub, leaf_map)) = reroot_without(&tree, a, k) else {
                    continue;
                };
                let mut sub_m = Vec::with_capacity(leaf_map.len() * p);
                let mut sub_w = Vec::with_capacity(leaf_map.len() * p);
                for &i in &leaf_map {
                    sub_m.extend_from_slice(&m[i * p..i * p + p]);
                    sub_w.extend_from_slice(&w[i * p..i * p + p]);
                }
                let mut sub_state = NodeState::new(sub.n_nodes(), p, &sub_m, &sub_w).unwrap();
                sub_state.prune(&sub);

                for g in 0..p {
                    assert_relative_eq!(
                        up.precisions(k)[g],
                        sub_state.precisions(sub.root())[g],
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        up.means(k)[g],
                        sub_state.means(sub.root())[g],
                        epsilon = 1e-12
                    );
                }
                checked += 1;
            }
            assert!(checked >= tree.n_nodes() - 3, "only checked {checked} nodes");
        }
    }

    #[test]
    fn test_up_value_of_a_binary_root_child_is_its_sibling_diffused() {
        // The one case the reroot cannot reach: deleting a child of a two-child
        // root leaves a root with one child, which the arena cannot hold. The
        // answer is the sibling's subtree diffused along the sibling's branch.
        let p = 16usize;
        let (m, w) = leaf_data(4, p, 21);
        let tree = Tree::from_parents(
            vec![4, 4, 5, 5, 6, 6, NO_NODE],
            vec![0.3, 1.2, 0.8, 0.4, 1.5, 0.6, 0.0],
            4,
        )
        .unwrap();
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        state.prune(&tree);
        let mut up = UpState::new(tree.n_nodes(), p);
        up.sweep(&tree, &state);

        for (k, sib) in [(4u32, 5u32), (5, 4)] {
            let t = tree.branch(sib);
            for g in 0..p {
                let wl = state.precisions(sib)[g];
                assert_relative_eq!(
                    up.precisions(k)[g],
                    wl / (1.0 + t * wl),
                    max_relative = 1e-13
                );
                assert_relative_eq!(up.means(k)[g], state.means(sib)[g], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_loglikelihood_never_decreases_across_iterations() {
        // Sign errors show up here first. Run the optimiser with a growing
        // iteration cap: the returned value must be monotone in the cap and
        // never below the starting tree's.
        let (n_leaves, p) = (16usize, 96usize);
        let truth = Tree::balanced_binary(n_leaves, 1.0).unwrap();
        let (m, w) = simulate(&truth, p, 4.0, 31);

        for start in [0.05f64, 1.0, 6.0] {
            let tree = Tree::balanced_binary(n_leaves, start).unwrap();
            let mut base = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
            let l0 = base.prune(&tree);

            let mut previous = l0;
            for cap in 1..=8 {
                let mut t = tree.clone();
                let mut state = NodeState::new(t.n_nodes(), p, &m, &w).unwrap();
                let l = optimise_branch_lengths(
                    &mut t,
                    &mut state,
                    Some(GlobalBranchParams::new(cap, 0.0)),
                )
                .unwrap();
                assert!(l >= l0, "start {start}, cap {cap}: {l} below the start {l0}");
                assert!(l >= previous, "start {start}, cap {cap}: {l} below {previous}");
                previous = l;
            }
        }
    }

    #[test]
    fn test_returned_loglikelihood_matches_the_returned_tree() {
        let (n_leaves, p) = (8usize, 64usize);
        let truth = Tree::balanced_binary(n_leaves, 0.8).unwrap();
        let (m, w) = simulate(&truth, p, 4.0, 41);
        let mut tree = Tree::balanced_binary(n_leaves, 3.0).unwrap();
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let l = optimise_branch_lengths(&mut tree, &mut state, None).unwrap();

        let mut fresh = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        assert_relative_eq!(l, fresh.prune(&tree), max_relative = 1e-14);
    }

    #[test]
    fn test_every_edge_is_stationary_at_convergence() {
        let p = 128usize;
        let truth = three_cherries(&[0.4, 1.3, 0.9, 0.2, 1.8, 0.5, 0.7, 1.1, 0.3]);
        let (m, w) = simulate(&truth, p, 4.0, 51);
        let mut tree = three_cherries(&[2.0; 9]);
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        // A zero tolerance runs the iteration until the line search can no
        // longer find an improving step at all, which is where the residual
        // floor of `STATIONARY_RESIDUAL` is measured.
        optimise_branch_lengths(
            &mut tree,
            &mut state,
            Some(GlobalBranchParams::new(200, 0.0)),
        )
        .unwrap();

        assert_no_edge_can_be_improved(&tree, &state, &m, &w);
    }

    #[test]
    fn test_perturbed_branch_lengths_move_back_towards_the_truth() {
        let p = 1024usize;
        let true_lengths = [0.4f64, 1.3, 0.9, 0.2, 1.8, 0.5, 0.7, 1.1, 0.3];
        let truth = three_cherries(&true_lengths);
        let (m, w) = simulate(&truth, p, 16.0, 61);

        for factor in [0.25f64, 4.0] {
            let start: [f64; 9] = std::array::from_fn(|i| true_lengths[i] * factor);
            let mut tree = three_cherries(&start);
            let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
            optimise_branch_lengths(&mut tree, &mut state, None).unwrap();

            let before: f64 = (0..9).map(|i| (start[i] - true_lengths[i]).abs()).sum();
            let after: f64 = (0..9)
                .map(|i| (tree.branch(i as u32) - true_lengths[i]).abs())
                .sum();
            assert!(
                after < 0.25 * before,
                "factor {factor}: error went {before} -> {after}"
            );
            // Sampling noise on one realisation is what stops this being
            // tighter: a branch length is identified only to about
            // `t * sqrt(2/p)` even with the topology known.
            for i in 0..9 {
                let got = tree.branch(i as u32);
                assert!(
                    (got - true_lengths[i]).abs() < 0.5 * true_lengths[i],
                    "factor {factor}, edge {i}: {got} against {}",
                    true_lengths[i]
                );
            }
        }
    }

    #[test]
    fn test_a_second_call_changes_nothing() {
        let (n_leaves, p) = (16usize, 64usize);
        let truth = Tree::balanced_binary(n_leaves, 1.2).unwrap();
        let (m, w) = simulate(&truth, p, 4.0, 71);
        let mut tree = Tree::balanced_binary(n_leaves, 0.3).unwrap();
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();

        let first = optimise_branch_lengths(&mut tree, &mut state, None).unwrap();
        let settled = tree.branches().to_vec();
        let second = optimise_branch_lengths(&mut tree, &mut state, None).unwrap();

        assert_eq!(first, second);
        assert_eq!(settled, tree.branches());
    }

    #[test]
    fn test_two_leaf_tree_recovers_the_single_edge_solve() {
        // Two leaves is one edge split across the root, so only the total is
        // identified. That total must match the plain one-dimensional solve on
        // the raw leaf data.
        let p = 48usize;
        let (m, w) = leaf_data(2, p, 81);
        let mut tree = Tree::from_parents(vec![2, 2, NO_NODE], vec![1.5, 0.4, 0.0], 2).unwrap();
        let mut state = NodeState::new(3, p, &m, &w).unwrap();
        optimise_branch_lengths(&mut tree, &mut state, None).unwrap();

        let (mut s, mut d) = (vec![0.0; p], vec![0.0; p]);
        let upper = prep_edge(&m[..p], &w[..p], &m[p..], &w[p..], &mut s, &mut d);
        let expect = optimise_edge(&s, &d, upper).unwrap();
        assert_relative_eq!(tree.branch(0) + tree.branch(1), expect, max_relative = 1e-8);
    }

    #[test]
    fn test_star_tree_optimises_every_spoke() {
        // One internal node, so the polytomy path in the sweep is the only one
        // exercised and every spoke is independent given the others.
        let (n_leaves, p) = (7usize, 96usize);
        let mut parent: Vec<u32> = vec![n_leaves as u32; n_leaves];
        parent.push(NO_NODE);
        let truth = Tree::from_parents(parent.clone(), vec![0.9; n_leaves + 1], n_leaves).unwrap();
        let (m, w) = simulate(&truth, p, 4.0, 91);

        let mut tree = Tree::from_parents(parent, vec![4.0; n_leaves + 1], n_leaves).unwrap();
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        let l0 = state.prune(&tree);
        let l = optimise_branch_lengths(
            &mut tree,
            &mut state,
            Some(GlobalBranchParams::new(200, 0.0)),
        )
        .unwrap();

        assert!(l > l0);
        assert_no_edge_can_be_improved(&tree, &state, &m, &w);
    }

    #[test]
    fn test_indistinguishable_leaves_collapse_their_branch_to_zero() {
        // Two leaves closer together than their own error bars want no branch
        // at all, which is where the polytomies of SPEC.md section 9.2 come
        // from. The rest of the tree must still get a positive branch.
        let p = 64usize;
        let (mut m, mut w) = leaf_data(4, p, 101);
        for g in 0..p {
            m[p + g] = m[g] + 1e-4;
            w[g] = 0.5;
            w[p + g] = 0.5;
            // Push the other cherry well away, so its own branch is not zero.
            m[2 * p + g] += 6.0;
            m[3 * p + g] += 6.0;
        }
        let mut tree = Tree::from_parents(
            vec![4, 4, 5, 5, 6, 6, NO_NODE],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            4,
        )
        .unwrap();
        let mut state = NodeState::new(tree.n_nodes(), p, &m, &w).unwrap();
        optimise_branch_lengths(&mut tree, &mut state, None).unwrap();

        assert_eq!(tree.branch(0), 0.0);
        assert_eq!(tree.branch(1), 0.0);
        assert!(tree.branch(4) + tree.branch(5) > 0.0);
    }

    #[test]
    fn test_f32_storage_tracks_f64_storage() {
        let (n_leaves, p) = (16usize, 128usize);
        let truth = Tree::balanced_binary(n_leaves, 1.0).unwrap();
        let (m, w) = simulate(&truth, p, 4.0, 111);

        let mut t64 = Tree::balanced_binary(n_leaves, 2.5).unwrap();
        let mut s64 = NodeState::new(t64.n_nodes(), p, &m, &w).unwrap();
        let l64 = optimise_branch_lengths(&mut t64, &mut s64, None).unwrap();

        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let mut t32 = Tree::balanced_binary(n_leaves, 2.5).unwrap();
        let mut s32 = NodeState::new(t32.n_nodes(), p, &m32, &w32).unwrap();
        let l32 = optimise_branch_lengths(&mut t32, &mut s32, None).unwrap();

        assert_relative_eq!(l32, l64, max_relative = 1e-3);
        for k in 0..t64.n_nodes() as u32 - 1 {
            assert!(
                (t32.branch(k) - t64.branch(k)).abs() < 0.02 * (1.0 + t64.branch(k)),
                "edge {k}: {} against {}",
                t32.branch(k),
                t64.branch(k)
            );
        }
    }
}
