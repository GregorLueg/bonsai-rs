//! Per-step attribution for the seven-step search.
//!
//! `benches/pipeline.rs` gives the total; this says which step owns it. Written
//! to find a cubic term in the merge, which turned out to be the candidate
//! restrictions not being wired into the pipeline at all.
//!
//! Measured 2026-09-06 on a quiet machine, 200 features, seconds at 2048 leaves
//! and the exponent over the last doubling:
//!
//! | step | seconds | `n^` |
//! |---|---|---|
//! | 1 star | 0.07 | 0.99 |
//! | 2 merge | 7.53 | 1.65 |
//! | 3 polytomy | 0.11 | 2.83 |
//! | 4 branch | 0.36 | 1.00 |
//! | 5 spr | 6.71 | 1.47 |
//! | 6 nni | 0.75 | 2.60 |
//!
//! The merge and SPR dominate at roughly equal cost and both are sub-quadratic.
//! Every step is exact: the trees are byte identical to what the unoptimised
//! search produced.
//!
//! ### Two exponents that need reading carefully
//!
//! **NNI's 2.60 is its round count, not its cost per round.** Over 256 to 1024
//! the times are 0.03, 0.06 and 0.12, exactly linear; the jump at 2048 is the
//! phase running three rounds where the smaller sizes ran one. Timing only the
//! sizes that run a single round gives `n^1.02`. The round count is
//! data-dependent, so the fitted exponent of the whole step swings with it.
//!
//! **Polytomy resolution's 2.83 is real and is the last structural item.**
//! Between 86 and 96 per cent of one sweep is a full `NodeState::prune` plus a
//! full `UpState::sweep`, both `O(n p)`, over a sweep count growing 8, 30, 107
//! across those sizes. Fixing it needs those two able to settle only the rows a
//! sweep actually reads: the down rows of a polytomy centre's children and the
//! up row of the centre. That is a change in `model::likelihood` and
//! `model::global`, not in `search::polytomy`. Resolving several polytomies per
//! sweep is *not* the shortcut it looks like, because a resolution changes every
//! up row and the second centre would be scored against stale ones.
//!
//! ### What the two rewrites actually removed
//!
//! Both were diagnosed wrongly first, and in both cases the profile was worth
//! more than the fix.
//!
//! Step 5's cost was **proposing**, not accepting. Each candidate settled the
//! tree the cut left behind and the tree the regraft built, then collapsed the
//! first onto every node: five `O(n p)` sweeps for rows of which the beam search
//! reads a few dozen. The `O(n p)` re-prune that accepts or rejects, which was
//! the suspected culprit, is 0.03 per cent of the step, because the split filter
//! discards almost every candidate before it runs. 93.19 s to 6.71 s.
//!
//! Step 6's cost was **the filter itself**. `interchange_at` built a whole tree
//! and `split_fingerprint` walked it, `O(n)` each, for every one of `O(n)`
//! candidate edges, which was 1.47 s of 1.98 s at 4096 leaves. Instrumenting the
//! loop showed the re-prune ran at most once per round, since between zero and
//! one candidate per round survived the filter at all. Replacing the filter with
//! a structural test over the star result removed it. 1.98 s to 0.75 s.
//!
//! **Run on a quiet machine.** Check `uptime` first.
//!
//! ```sh
//! cargo bench --bench steps
//! ```

use bonsai_rs::model::global::optimise_branch_lengths;
use bonsai_rs::model::likelihood::NodeState;
use bonsai_rs::search::Leaves;
use bonsai_rs::search::bounds::EllipsoidBounds;
use bonsai_rs::search::candidates::KnnCandidates;
use bonsai_rs::search::nni::nni;
use bonsai_rs::search::polytomy::resolve_polytomies;
use bonsai_rs::search::spr::spr;
use bonsai_rs::search::star::{Star, star_tree_with};
use bonsai_rs::tree::Tree;
use bonsai_rs::tree::simulate::{SimulationParams, simulate_binary};
use std::time::Instant;

fn main() {
    let p = 200usize;
    println!(
        "{:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "leaves", "1 star", "2 merge", "3 poly", "4 branch", "5 spr", "6 nni"
    );

    let mut previous: Option<(usize, [f64; 6])> = None;
    for n in [256usize, 512, 1024, 2048] {
        let d = simulate_binary::<f64>(Some(SimulationParams {
            n_leaves: n,
            n_features: p,
            noise_sd: 0.3,
            seed: 31,
            ..Default::default()
        }))
        .expect("simulation");
        let precisions = d.precisions();
        let leaves = Leaves {
            means: &d.means,
            precisions: &precisions,
            n_features: p,
        };

        let mut t = [0.0f64; 6];

        // Step 1.
        let mut parent = vec![n as u32; n];
        parent.push(bonsai_rs::tree::NO_NODE);
        let mut branch = vec![1.0f64; n];
        branch.push(0.0);
        let mut star = Tree::from_parents(parent, branch, n).expect("star");
        let mut state =
            NodeState::new(star.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
        let t0 = Instant::now();
        optimise_branch_lengths(&mut star, &mut state, None).expect("step 1");
        t[0] = t0.elapsed().as_secs_f64();

        // Step 2.
        let t0 = Instant::now();
        let mut candidates = EllipsoidBounds::new(KnnCandidates::new(None), None);
        let (mut tree, _) = star_tree_with(
            Star {
                means: leaves.means,
                precisions: leaves.precisions,
                branch: &star.branches()[..n],
                n_features: p,
            },
            None,
            &mut candidates,
        )
        .expect("step 2");
        t[1] = t0.elapsed().as_secs_f64();

        // Step 3.
        let t0 = Instant::now();
        tree = resolve_polytomies(&tree, leaves, None)
            .expect("step 3")
            .tree;
        t[2] = t0.elapsed().as_secs_f64();

        // Step 4.
        let mut state =
            NodeState::new(tree.n_nodes(), p, leaves.means, leaves.precisions).expect("state");
        let t0 = Instant::now();
        optimise_branch_lengths(&mut tree, &mut state, None).expect("step 4");
        t[3] = t0.elapsed().as_secs_f64();

        // Step 5.
        let t0 = Instant::now();
        tree = spr(&tree, leaves, None).expect("step 5").tree;
        t[4] = t0.elapsed().as_secs_f64();

        // Step 6.
        let t0 = Instant::now();
        let _ = nni(&tree, leaves, None).expect("step 6");
        t[5] = t0.elapsed().as_secs_f64();

        println!(
            "{n:>7} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            t[0], t[1], t[2], t[3], t[4], t[5]
        );
        if let Some((prev_n, prev)) = previous {
            let factor = (n / prev_n) as f64;
            print!("{:>7}", "exps");
            for k in 0..6 {
                let e = if prev[k] > 1e-6 {
                    (t[k] / prev[k]).ln() / factor.ln()
                } else {
                    f64::NAN
                };
                print!(" {e:>9.2}");
            }
            println!("   (n^exponent)");
        }
        previous = Some((n, t));
    }
}
