//! Per-step attribution for the seven-step search.
//!
//! `benches/pipeline.rs` gives the total; this says which step owns it. Written
//! to find a cubic term in the merge, which turned out to be the candidate
//! restrictions not being wired into the pipeline at all.
//!
//! Measured 2026-09-05, 200 features, seconds at 2048 leaves and the exponent
//! over the last doubling, before and against [`search::spr::LazyRows`]:
//!
//! | step | before | `n^` | after | `n^` |
//! |---|---|---|---|---|
//! | 1 star | 0.07 | 1.02 | 0.07 | 0.99 |
//! | 2 merge | 7.52 | 1.65 | 7.56 | 1.67 |
//! | 3 polytomy | 0.12 | 2.86 | 0.12 | 2.84 |
//! | 4 branch | 0.36 | 1.00 | 0.35 | 0.99 |
//! | **5 spr** | **93.19** | **1.97** | **6.75** | **1.49** |
//! | 6 nni | 1.97 | 3.02 | 1.98 | 3.03 |
//!
//! Over the whole range 256 to 2048 step 5 fits `n^1.93` before and `n^1.36`
//! after. The trees are unchanged, byte for byte.
//!
//! What was expensive was **proposing**, not accepting. Step 5 sweeps every
//! candidate subtree each round, and each candidate used to settle the tree the
//! cut left behind and the tree the regraft built and then collapse the first
//! onto every node: five `O(n p)` sweeps for rows of which a few dozen are ever
//! read. The `O(n p)` re-prune that accepts or rejects is 0.03 per cent of the
//! step, because the split filter discards almost every candidate before it
//! ever runs.
//!
//! **Steps 3 and 6 are the ones now left with a bad exponent, and it is real.**
//! Both are reproducible over five runs and both have the same cause: one
//! accepted move per full `O(n p)` resettle of the whole tree, over a move
//! count that grows with `n`. Step 6 is 1.98 seconds here and, at `n^3`, would
//! be the leading term of the whole search well before thirty thousand cells.
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
