//! Which of the seven steps is the cubic one?
//!
//! The end-to-end sweep in `benches/pipeline.rs` shows `O(n^3)`. This times each
//! step separately so the term can be named rather than guessed at.
//!
//! Temporary; delete once the answer is recorded.

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
