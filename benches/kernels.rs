//! Single-threaded throughput of each feature-axis kernel, so the wall each one
//! hits can be identified before anything is done to it.
//!
//! Reports achieved FLOP/s and achieved bandwidth per kernel. Comparing those
//! against what one core can do says which lever applies:
//!
//! - near peak FLOP/s: compute bound, widen the vectors
//! - near peak bandwidth: memory bound, move fewer bytes
//! - far below both: latency bound on a dependency chain, break the chain
//!
//! Working sets are sized to stay in L2 on purpose. The question here is what
//! the arithmetic can do, not what the memory system can feed it; the sweep in
//! `prune_sweep` covers the out-of-cache case.
//!
//! ```sh
//! cargo bench --bench kernels
//! ```
//!
//! **Run this on a quiet machine.** These kernels are short and the numbers are
//! in nanoseconds per element, so anything else competing for the cores moves
//! them by more than the effects being looked for. Check `uptime` first.
//!
//! ### Open question
//!
//! `edge_newton` is the hottest kernel in the crate and it is not obvious which
//! wall it hits. At roughly three cycles per feature it sits close to both the
//! two-chain FMA latency bound and the `f64` division throughput bound, and
//! those want opposite fixes: more accumulator chains for the first, fewer
//! divisions for the second. Splitting it into four independent chains was
//! tried and appeared to change nothing, but that measurement was taken on a
//! loaded machine and is worthless. Redo it, and settle the question by timing
//! a variant with the division replaced by a multiply: if that is much faster,
//! the division is the wall and extra chains will never help.

use bonsai_rs::utils::kernels::{edge_loglik, edge_newton, prep_edge, prune_binary_scalar};
use bonsai_rs::utils::rng::splitmix64_at;
use std::hint::black_box;
use std::time::Instant;

/// Features per call, chosen to keep the working set inside L2.
const P: usize = 2000;

/// Calls per timed repeat.
const CALLS: usize = 20_000;

/// Repeats; the reported time is the best.
const REPEATS: usize = 5;

/// Time a kernel and report its achieved rates.
///
/// ### Params
///
/// * `name` - Kernel name
/// * `flops` - Floating-point operations per feature, counting a division as one
/// * `bytes` - Bytes read and written per feature
/// * `f` - The kernel, returning something to keep alive
fn report<F: FnMut() -> f64>(name: &str, flops: f64, bytes: f64, mut f: F) {
    let mut best = f64::INFINITY;
    let mut keep = 0.0f64;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        for _ in 0..CALLS {
            keep += black_box(f());
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    assert!(keep.is_finite(), "kernel produced a non-finite result");

    let elems = (CALLS * P) as f64;
    println!(
        "{:<22} {:>9.2} {:>12.2} {:>12.2} {:>12.1}",
        name,
        best * 1e9 / elems,
        elems / best / 1e6,
        flops * elems / best / 1e9,
        bytes * elems / best / 1e9,
    );
}

fn main() {
    let m_k: Vec<f64> = (0..P).map(|g| splitmix64_at(g as u64) * 4.0 - 2.0).collect();
    let w_k: Vec<f64> = (0..P)
        .map(|g| 0.3 + splitmix64_at(P as u64 + g as u64) * 3.0)
        .collect();
    let m_l: Vec<f64> = (0..P)
        .map(|g| splitmix64_at(2 * P as u64 + g as u64) * 4.0 - 2.0)
        .collect();
    let w_l: Vec<f64> = (0..P)
        .map(|g| 0.3 + splitmix64_at(3 * P as u64 + g as u64) * 3.0)
        .collect();

    let mut m_out = vec![0.0f64; P];
    let mut w_out = vec![0.0f64; P];
    let mut s = vec![0.0f64; P];
    let mut d = vec![0.0f64; P];
    prep_edge(&m_k, &w_k, &m_l, &w_l, &mut s, &mut d);

    println!(
        "{:<22} {:>9} {:>12} {:>12} {:>12}",
        "kernel", "ns/elem", "Melem/s", "GFLOP/s", "GB/s"
    );

    // Roughly: 2 divisions, 1 log, ~10 multiply-adds; 4 reads and 2 writes.
    report("prune_binary (1 log)", 13.0, 48.0, || {
        prune_binary_scalar(
            black_box(&m_k),
            black_box(&w_k),
            0.37,
            black_box(&m_l),
            black_box(&w_l),
            1.9,
            &mut m_out,
            &mut w_out,
        )
    });

    // 1 division, 6 multiply-adds; 2 reads, no writes. The hot one.
    report("edge_newton (no log)", 7.0, 16.0, || {
        let (f, fp) = edge_newton(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    // 1 division, 1 log, 2 multiply-adds; 2 reads.
    report("edge_loglik (1 log)", 4.0, 16.0, || {
        edge_loglik(black_box(&s), black_box(&d), 0.83)
    });

    // 2 divisions, 5 multiply-adds; 4 reads, 2 writes.
    report("prep_edge", 7.0, 48.0, || {
        prep_edge(
            black_box(&m_k),
            black_box(&w_k),
            black_box(&m_l),
            black_box(&w_l),
            &mut s,
            &mut d,
        )
    });
}
