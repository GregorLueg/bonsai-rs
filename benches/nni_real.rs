//! What step 6 does on a tree step 5 leaves behind, and what the knobs around
//! it buy in loglikelihood and Robinson-Foulds rather than in seconds.
//!
//! Written for the first comparison on realistic data, where step 6 turned out
//! to gain far less than the equivalent stage of the published implementation;
//! `docs/COMPARISON.md` has the numbers. Steps 1 to
//! 5 are run once per fixture and the trees after steps 4 and 5 are cached as
//! Newick in the scratch directory, so every variant below starts from the
//! identical step 5 tree and only the part under test is rerun.
//!
//! ```sh
//! cargo bench --bench nni_real -- real <dir> <variant> [args]
//! cargo bench --bench nni_real -- sim <binary|unbalanced> <n> <p> <noise> <seed> <variant> [args]
//! ```
//!
//! `<dir>` holds `ours/means.csv`, `ours/sds.csv` (transformed units) and
//! `truth.nwk` with leaves labelled `cell<i>`. Variants:
//!
//! * `base` - step 6 at the defaults, then step 7; prints the per-round trace
//! * `random <n> [seed]` - `NniParams::n_random` set
//! * `mingain <g>` - `StarParams::min_gain` on the interchange star
//! * `reopt` - step 4 again between steps 5 and 6
//! * `alternate [cycles]` - steps 5 and 6 repeated until neither moves
//!
//! Every variant reports the loglikelihood after step 7 and the RF to the
//! generating tree, which are the gate.

use bonsai_rs::bonsai::BonsaiParams;
use bonsai_rs::bonsai::StartTree;
use bonsai_rs::model::global::optimise_branch_lengths;
use bonsai_rs::model::likelihood::NodeState;
use bonsai_rs::search::Leaves;
use bonsai_rs::search::bounds::EllipsoidBounds;
use bonsai_rs::search::candidates::KnnCandidates;
use bonsai_rs::search::nni::{NniParams, NniResult, nni, nni_random};
use bonsai_rs::search::polytomy::resolve_polytomies;
use bonsai_rs::search::spr::spr;
use bonsai_rs::search::star::{Star, star_tree_with};
use bonsai_rs::tree::linkage::linkage_tree;
use bonsai_rs::tree::newick::{parse_newick, write_newick};
use bonsai_rs::tree::simulate::{
    SimulationParams, robinson_foulds, simulate_binary, simulate_unbalanced,
};
use bonsai_rs::tree::{NO_NODE, Tree};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Fixture {
    means: Vec<f64>,
    precisions: Vec<f64>,
    n: usize,
    p: usize,
    truth: Tree,
    key: String,
}

impl Fixture {
    fn leaves(&self) -> Leaves<'_, f64> {
        Leaves {
            means: &self.means,
            precisions: &self.precisions,
            n_features: self.p,
        }
    }
}

fn read_csv(path: &Path) -> (Vec<f64>, usize, usize) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut values = Vec::new();
    let mut rows = 0usize;
    let mut cols = 0usize;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let before = values.len();
        values.extend(
            line.split(',')
                .map(|f| f.trim().parse::<f64>().expect("number")),
        );
        cols = values.len() - before;
        rows += 1;
    }
    (values, rows, cols)
}

/// Labels of the form `<prefix><i>`, so the leaves come back in index order.
fn labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("L{i:08}")).collect()
}

/// Rebuild a parsed tree on the canonical leaf indexing: the parser numbers
/// leaves in string order, the metrics match them by index.
fn relabel(tree: &Tree, found: &[String], index_of: &HashMap<&str, usize>) -> Tree {
    let n_nodes = tree.n_nodes();
    let n_leaves = tree.n_leaves();
    let mut parent = vec![NO_NODE; n_nodes];
    let mut branch = vec![0.0f64; n_nodes];
    for node in 0..n_nodes {
        let dst = if node < n_leaves {
            index_of[found[node].as_str()]
        } else {
            node
        };
        parent[dst] = tree.parent(node as u32).unwrap_or(NO_NODE);
        branch[dst] = tree.branch(node as u32);
    }
    Tree::from_parents(parent, branch, n_leaves).expect("relabel")
}

fn load_tree(path: &Path, cell_labels: &[String]) -> Tree {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (tree, found) = parse_newick(&text).expect("newick");
    let index_of: HashMap<&str, usize> = cell_labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    relabel(&tree, &found, &index_of)
}

fn save_tree(path: &Path, tree: &Tree) {
    fs::write(
        path,
        write_newick(tree, &labels(tree.n_leaves())).expect("newick") + "\n",
    )
    .expect("write");
}

fn scratch() -> PathBuf {
    let dir = env::var("NNI_REAL_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/private/tmp/claude-501/-Users-gregorlueg-repos-shared-bonsai-rs/e844eef2-43f9-4bd6-a9c6-3d0db6c72a46/scratchpad/nni_real")
    });
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn real_fixture(dir: &Path) -> Fixture {
    let (means, n, p) = read_csv(&dir.join("ours").join("means.csv"));
    let (sds, _, _) = read_csv(&dir.join("ours").join("sds.csv"));
    let precisions: Vec<f64> = sds.iter().map(|&s| 1.0 / (s * s)).collect();
    let cell_labels: Vec<String> = (0..n).map(|i| format!("cell{i}")).collect();
    let truth = load_tree(&dir.join("truth.nwk"), &cell_labels);
    let key = dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "real".to_string());
    Fixture {
        means,
        precisions,
        n,
        p,
        truth,
        key: format!("real_{key}"),
    }
}

fn sim_fixture(kind: &str, n: usize, p: usize, noise: f64, seed: u64) -> Fixture {
    let params = SimulationParams {
        n_leaves: n,
        n_features: p,
        noise_sd: noise,
        seed,
        ..Default::default()
    };
    let data = match kind {
        "binary" => simulate_binary::<f64>(Some(params)),
        "unbalanced" => simulate_unbalanced::<f64>(Some(params)),
        other => panic!("unknown generator {other}"),
    }
    .expect("simulate");
    let precisions = data.precisions();
    Fixture {
        means: data.means,
        precisions,
        n,
        p,
        truth: data.tree,
        key: format!("sim_{kind}_n{n}_p{p}_noise{noise}_s{seed}"),
    }
}

fn loglik(tree: &Tree, leaves: Leaves<'_, f64>) -> f64 {
    let mut state = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )
    .expect("state");
    state.prune(tree)
}

fn optimise(tree: &mut Tree, leaves: Leaves<'_, f64>, params: &BonsaiParams) -> f64 {
    let mut state = NodeState::new(
        tree.n_nodes(),
        leaves.n_features,
        leaves.means,
        leaves.precisions,
    )
    .expect("state");
    optimise_branch_lengths(tree, &mut state, Some(params.branch)).expect("branch")
}

/// Relabel internal nodes into a post-order so that every parent index
/// exceeds its children's, which is what the arena requires and what a hand
/// swap breaks.
fn renumber(parent: &[u32], branch: &[f64], n_leaves: usize) -> (Vec<u32>, Vec<f64>) {
    let n = parent.len();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut root = 0u32;
    for (v, &p) in parent.iter().enumerate() {
        if p == NO_NODE {
            root = v as u32;
        } else {
            children[p as usize].push(v as u32);
        }
    }
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut stack: Vec<(u32, bool)> = vec![(root, false)];
    while let Some((v, done)) = stack.pop() {
        if done {
            order.push(v);
        } else {
            stack.push((v, true));
            for &c in &children[v as usize] {
                if c as usize >= n_leaves {
                    stack.push((c, false));
                }
            }
        }
    }
    let mut new_id = vec![0u32; n];
    for (v, slot) in new_id.iter_mut().enumerate().take(n_leaves) {
        *slot = v as u32;
    }
    for (i, &v) in order.iter().enumerate() {
        new_id[v as usize] = (n_leaves + i) as u32;
    }
    let mut out_parent = vec![NO_NODE; n];
    let mut out_branch = vec![0.0f64; n];
    for v in 0..n {
        let nv = new_id[v] as usize;
        out_parent[nv] = if parent[v] == NO_NODE {
            NO_NODE
        } else {
            new_id[parent[v] as usize]
        };
        out_branch[nv] = branch[v];
    }
    (out_parent, out_branch)
}

fn load() -> String {
    let out = std::process::Command::new("uptime")
        .output()
        .expect("uptime");
    let s = String::from_utf8_lossy(&out.stdout);
    s.split("load averages:")
        .nth(1)
        .map(|x| x.trim().to_string())
        .unwrap_or_default()
}

/// Steps 1 to 5 at the defaults, cached.
fn through_step_five(fx: &Fixture, params: &BonsaiParams) -> (Tree, Tree) {
    let dir = scratch();
    let start = match params.start {
        StartTree::GreedyMerge => "greedy",
        StartTree::Linkage => "linkage",
    };
    let path4 = dir.join(format!("{}_{start}_step4.nwk", fx.key));
    let path5 = dir.join(format!("{}_{start}_step5.nwk", fx.key));
    let lab = labels(fx.n);
    if path4.exists() && path5.exists() {
        let t4 = load_tree(&path4, &lab);
        let t5 = load_tree(&path5, &lab);
        println!(
            "cached: step 4 loglik {:.3} rf {}   step 5 loglik {:.3} rf {}",
            loglik(&t4, fx.leaves()),
            robinson_foulds(&t4, &fx.truth).expect("rf"),
            loglik(&t5, fx.leaves()),
            robinson_foulds(&t5, &fx.truth).expect("rf")
        );
        return (t4, t5);
    }
    let leaves = fx.leaves();
    let n = fx.n;
    let t0 = Instant::now();
    let mut tree = match params.start {
        StartTree::GreedyMerge => {
            let mut parent = vec![n as u32; n];
            parent.push(NO_NODE);
            let mut branch = vec![1.0; n];
            branch.push(0.0);
            let mut star = Tree::from_parents(parent, branch, n).expect("star");
            let mut state = NodeState::new(star.n_nodes(), fx.p, leaves.means, leaves.precisions)
                .expect("state");
            optimise_branch_lengths(&mut star, &mut state, Some(params.branch)).expect("branch");
            let mut candidates =
                EllipsoidBounds::new(KnnCandidates::new(Some(params.knn)), Some(params.bounds));
            star_tree_with(
                Star {
                    means: leaves.means,
                    precisions: leaves.precisions,
                    branch: &star.branches()[..n],
                    n_features: fx.p,
                },
                Some(params.star),
                &mut candidates,
            )
            .expect("merge")
            .0
        }
        StartTree::Linkage => {
            linkage_tree(leaves.means, n, fx.p, Some(params.linkage)).expect("linkage")
        }
    };
    println!(
        "steps 1-2 ({start}): {:.1} s  loglik {:.3}  rf {}",
        t0.elapsed().as_secs_f64(),
        loglik(&tree, leaves),
        robinson_foulds(&tree, &fx.truth).expect("rf")
    );
    let t0 = Instant::now();
    tree = resolve_polytomies(&tree, leaves, Some(params.star))
        .expect("polytomy")
        .tree;
    let l4 = optimise(&mut tree, leaves, params);
    println!(
        "steps 3-4: {:.1} s  loglik {:.3}  rf {}",
        t0.elapsed().as_secs_f64(),
        l4,
        robinson_foulds(&tree, &fx.truth).expect("rf")
    );
    save_tree(&path4, &tree);
    let t4 = tree.clone();
    let t0 = Instant::now();
    let out = spr(&tree, leaves, Some(params.spr)).expect("spr");
    println!(
        "step 5: {:.1} s  loglik {:.3}  rf {}  moves {} rounds {}  (load {})",
        t0.elapsed().as_secs_f64(),
        out.loglik,
        robinson_foulds(&out.tree, &fx.truth).expect("rf"),
        out.gains.len(),
        out.rounds,
        load()
    );
    save_tree(&path5, &out.tree);
    (t4, out.tree)
}

fn print_trace(out: &NniResult) {
    let t = &out.trace;
    let sum =
        |f: fn(&bonsai_rs::search::nni::NniRound) -> usize| -> usize { t.iter().map(f).sum() };
    println!(
        "nni: {} moves over {} rounds; totals eligible {} proposed {} changed {} improving {}",
        out.n_moves,
        out.rounds,
        sum(|r| r.eligible),
        sum(|r| r.proposed),
        sum(|r| r.changed),
        sum(|r| r.improving)
    );
    let show: Vec<usize> = if t.len() <= 12 {
        (0..t.len()).collect()
    } else {
        (0..6).chain(t.len() - 6..t.len()).collect()
    };
    println!("round  eligible  proposed  changed  improving  best_gain");
    for i in show {
        let r = &t[i];
        println!(
            "{:>5}  {:>8}  {:>8}  {:>7}  {:>9}  {:>10.3}",
            i + 1,
            r.eligible,
            r.proposed,
            r.changed,
            r.improving,
            r.best_gain
        );
    }
}

fn finish(label: &str, mut tree: Tree, fx: &Fixture, params: &BonsaiParams, secs: f64) {
    let leaves = fx.leaves();
    let before = loglik(&tree, leaves);
    let l7 = optimise(&mut tree, leaves, params);
    let rf = robinson_foulds(&tree, &fx.truth).expect("rf");
    println!(
        "RESULT {label}: loglik before step 7 {before:.3}, after {l7:.3}, rf {rf}, {secs:.1} s (load {})",
        load()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // The linkage start is the baseline; the greedy merge is
    // kept reachable because it is still the crate's default.
    let mut params = BonsaiParams {
        start: match env::var("START").as_deref() {
            Ok("greedy") => StartTree::GreedyMerge,
            _ => StartTree::Linkage,
        },
        ..BonsaiParams::default()
    };
    let (fx, rest) = match args.get(1).map(String::as_str) {
        Some("real") => (real_fixture(Path::new(&args[2])), &args[3..]),
        Some("sim") => (
            sim_fixture(
                &args[2],
                args[3].parse().expect("n"),
                args[4].parse().expect("p"),
                args[5].parse().expect("noise"),
                args[6].parse().expect("seed"),
            ),
            &args[7..],
        ),
        _ => {
            eprintln!(
                "usage: nni_real real <dir> <variant> | sim <kind> <n> <p> <noise> <seed> <variant>"
            );
            std::process::exit(2);
        }
    };
    println!(
        "fixture {}: {} cells by {} features (load {})",
        fx.key,
        fx.n,
        fx.p,
        load()
    );
    let leaves = fx.leaves();
    let (t4, t5) = through_step_five(&fx, &params);
    let variant = rest.first().map(String::as_str).unwrap_or("base");

    match variant {
        "base" | "random" | "ils" | "mingain" | "reopt" => {
            let mut start = t5.clone();
            let mut label = variant.to_string();
            match variant {
                "random" => {
                    params.nni.n_random = rest[1].parse().expect("n_random");
                    params.nni.seed = rest.get(2).map(|s| s.parse().expect("seed")).unwrap_or(0);
                    params.nni.temperature =
                        rest.get(3).map(|s| s.parse().expect("tau")).unwrap_or(1.0);
                    label = format!(
                        "random{} seed{} tau{}",
                        params.nni.n_random, params.nni.seed, params.nni.temperature
                    );
                }
                "ils" => {
                    params.nni.n_restarts = rest[1].parse().expect("n_restarts");
                    params.nni.n_random = rest[2].parse().expect("n_random");
                    params.nni.temperature =
                        rest.get(3).map(|s| s.parse().expect("tau")).unwrap_or(1.0);
                    params.nni.seed = rest.get(4).map(|s| s.parse().expect("seed")).unwrap_or(0);
                    label = format!(
                        "ils restarts{} moves{} tau{} seed{}",
                        params.nni.n_restarts,
                        params.nni.n_random,
                        params.nni.temperature,
                        params.nni.seed
                    );
                }
                "mingain" => {
                    params.nni.star.min_gain = rest[1].parse().expect("min_gain");
                    label = format!("mingain{:e}", params.nni.star.min_gain);
                }
                "reopt" => {
                    let l = optimise(&mut start, leaves, &params);
                    println!("step 4 again after step 5: loglik {l:.3}");
                }
                _ => {}
            }
            let t0 = Instant::now();
            let out = nni(&start, leaves, Some(params.nni)).expect("nni");
            let secs = t0.elapsed().as_secs_f64();
            print_trace(&out);
            println!(
                "nni: loglik {:.3} rf {}  ({:.1} s)",
                out.loglik,
                robinson_foulds(&out.tree, &fx.truth).expect("rf"),
                secs
            );
            finish(&label, out.tree, &fx, &params, secs);
        }
        "alternate" => {
            // Steps 5 and 6 to a joint fixed point, with step 4 between them
            // when asked, starting from the cached step 5 tree so the first
            // step 6 is the shipped one.
            let cycles: usize = rest
                .get(1)
                .map(|s| s.parse().expect("cycles"))
                .unwrap_or(20);
            let reopt = rest.get(2).map(|s| s == "reopt").unwrap_or(false);
            let mut tree = t5.clone();
            let t0 = Instant::now();
            for cycle in 1..=cycles {
                if reopt {
                    optimise(&mut tree, leaves, &params);
                }
                let out = nni(&tree, leaves, Some(params.nni)).expect("nni");
                let nni_moves = out.n_moves;
                tree = out.tree;
                println!(
                    "cycle {cycle}: nni {} moves {} rounds -> loglik {:.3} rf {}",
                    nni_moves,
                    out.rounds,
                    out.loglik,
                    robinson_foulds(&tree, &fx.truth).expect("rf")
                );
                if reopt {
                    optimise(&mut tree, leaves, &params);
                }
                let out = spr(&tree, leaves, Some(params.spr)).expect("spr");
                let spr_moves = out.gains.len();
                tree = out.tree;
                println!(
                    "cycle {cycle}: spr {} moves {} rounds -> loglik {:.3} rf {}  ({:.0} s, load {})",
                    spr_moves,
                    out.rounds,
                    out.loglik,
                    robinson_foulds(&tree, &fx.truth).expect("rf"),
                    t0.elapsed().as_secs_f64(),
                    load()
                );
                if nni_moves == 0 && spr_moves == 0 {
                    break;
                }
            }
            let secs = t0.elapsed().as_secs_f64();
            finish(
                &format!("alternate{}", if reopt { " reopt" } else { "" }),
                tree,
                &fx,
                &params,
                secs,
            );
        }
        "perturb" => {
            // How much the random phase actually moves the tree: RF between
            // the step 5 tree and its perturbation, and what the climb gets
            // back.
            params.nni.n_random = rest[1].parse().expect("n_random");
            params.nni.temperature = rest.get(2).map(|s| s.parse().expect("tau")).unwrap_or(1.0);
            params.nni.seed = rest.get(3).map(|s| s.parse().expect("seed")).unwrap_or(0);
            let before = loglik(&t5, leaves);
            let moved = nni_random(&t5, leaves, Some(params.nni)).expect("random");
            println!(
                "perturb: {} moves at tau {} -> rf from start {}, rf to truth {}, loglik {:.3} ({:+.3})",
                moved.n_moves,
                params.nni.temperature,
                robinson_foulds(&moved.tree, &t5).expect("rf"),
                robinson_foulds(&moved.tree, &fx.truth).expect("rf"),
                moved.loglik,
                moved.loglik - before
            );
            let climbed = nni(&moved.tree, leaves, Some(NniParams::default())).expect("nni");
            println!(
                "perturb: climbed with {} moves -> loglik {:.3} rf {}",
                climbed.n_moves,
                climbed.loglik,
                robinson_foulds(&climbed.tree, &fx.truth).expect("rf")
            );
            finish(
                &format!(
                    "perturb{} tau{}",
                    params.nni.n_random, params.nni.temperature
                ),
                climbed.tree,
                &fx,
                &params,
                0.0,
            );
        }
        "foreign" => {
            // A tree from elsewhere, with `cell<i>`
            // labels: its loglikelihood as it stands, after our step 7, and
            // whether our steps 5 and 6 find anything left in it.
            let cell_labels: Vec<String> = (0..fx.n).map(|i| format!("cell{i}")).collect();
            let mut tree = load_tree(Path::new(&rest[1]), &cell_labels);
            println!(
                "foreign: {} nodes, loglik {:.3} rf {}",
                tree.n_nodes(),
                loglik(&tree, leaves),
                robinson_foulds(&tree, &fx.truth).expect("rf")
            );
            let l = optimise(&mut tree, leaves, &params);
            println!("foreign after step 7: loglik {l:.3}");
            let out = nni(&tree, leaves, Some(params.nni)).expect("nni");
            print_trace(&out);
            println!(
                "foreign after our nni: loglik {:.3} rf {}",
                out.loglik,
                robinson_foulds(&out.tree, &fx.truth).expect("rf")
            );
            let out = spr(&out.tree, leaves, Some(params.spr)).expect("spr");
            println!(
                "foreign after our spr: loglik {:.3} rf {} moves {}",
                out.loglik,
                robinson_foulds(&out.tree, &fx.truth).expect("rf"),
                out.gains.len()
            );
            finish("foreign", out.tree, &fx, &params, 0.0);
        }
        "neighbours" => {
            // Is the step 7 tree a local optimum of the interchange
            // neighbourhood when every branch length is re-optimised, rather
            // than only the three the star primitive creates? Every binary
            // interchange, spliced by hand and handed to step 4.
            let mut tree = t5.clone();
            let out = nni(&tree, leaves, Some(params.nni)).expect("nni");
            tree = out.tree;
            let base = optimise(&mut tree, leaves, &params);
            println!("neighbours: from the step 7 tree at loglik {base:.3}");
            let n_nodes = tree.n_nodes();
            let parents: Vec<u32> = (0..n_nodes as u32)
                .map(|v| tree.parent(v).unwrap_or(NO_NODE))
                .collect();
            let mut gains: Vec<(f64, u32, u32, u32)> = Vec::new();
            let t0 = Instant::now();
            let mut tried = 0usize;
            for k in tree.internal_postorder() {
                let Some(l) = tree.parent(k) else { continue };
                for &x in tree.children(k) {
                    for &y in tree.children(l) {
                        if y == k {
                            continue;
                        }
                        let mut p = parents.clone();
                        p[x as usize] = l;
                        p[y as usize] = k;
                        let (p, b) = renumber(&p, tree.branches(), fx.n);
                        let mut t = Tree::from_parents(p, b, fx.n).expect("swap");
                        let ll = optimise(&mut t, leaves, &params);
                        gains.push((ll - base, k, x, y));
                        tried += 1;
                        if tried.is_multiple_of(100) {
                            println!(
                                "  {tried} tried, {:.0} s, {} improving so far",
                                t0.elapsed().as_secs_f64(),
                                gains.iter().filter(|g| g.0 > 1e-6).count()
                            );
                        }
                    }
                }
            }
            gains.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite"));
            let improving = gains.iter().filter(|g| g.0 > 1e-6).count();
            println!(
                "neighbours: {} interchanges, {improving} improve the fully re-optimised tree ({:.0} s)",
                gains.len(),
                t0.elapsed().as_secs_f64()
            );
            for g in gains.iter().take(25) {
                println!(
                    "  gain {:>10.3}  edge {}-{}  swap {} <-> {}",
                    g.0, g.1, parents[g.1 as usize], g.2, g.3
                );
            }
        }
        "spr-from-4" => {
            // Sanity check of the cache: step 5 again from the step 4 tree.
            let out = spr(&t4, leaves, Some(params.spr)).expect("spr");
            println!(
                "spr from step 4: loglik {:.3} rf {} moves {}",
                out.loglik,
                robinson_foulds(&out.tree, &fx.truth).expect("rf"),
                out.gains.len()
            );
        }
        other => panic!("unknown variant {other}"),
    }
}
