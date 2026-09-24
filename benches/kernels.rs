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
//! ### `edge_newton` is the kernel to beat, and lanes are what beat it
//!
//! It is the hottest kernel in the crate and not only here: instrumenting
//! `benches/merge_scan.rs` gives 35 to 45 calls per candidate pair, because
//! `model::branch::optimise_edge` is a bracketed Newton and every iteration is
//! one pass over the feature axis. Nothing else comes close;
//! `model::merge::MergeScratch::split_derivative` reads like a hot loop and runs
//! 0.07 times per pair.
//!
//! At roughly three cycles per feature it sat close to both the two-chain FMA
//! latency bound and the `f64` division throughput bound, which want opposite
//! fixes. The variants below settle it. Reference figures,
//! `P = 2000`, load average 3.5 rather than a quiet machine, so read the small
//! differences as noise:
//!
//! | variant | ns/elem |
//! |---|---|
//! | `edge_newton`, scalar | 0.95 |
//! | division replaced by a multiply | 0.92 |
//! | four accumulator chains | 1.00 |
//! | one division per four features | 1.23 |
//! | **`utils::simd::edge_newton_simd`, `f64x4`** | **0.71** |
//!
//! Neither algebraic fix does anything: the division is not the wall and the
//! dependency chain is not the wall. Explicit lanes take a third off, and that
//! carries through to 14 per cent of the merge scan and 8 per cent of a whole
//! run at 2048 cells by 2000 features, with identical trees.
//!
//! The compiler will not find this. `edge_newton` accumulates into two
//! floating-point reductions that LLVM may not reorder, so the scalar tier is
//! genuinely scalar: one feature per iteration with a scalar divide on aarch64,
//! and the same on x86-64 at baseline, at `x86-64-v3` and at `x86-64-v4`.
//! Vectorising a reduction is work only a human is allowed to do here.
//!
//! `prune_binary` is the other kernel with a vector tier, and it is the other
//! side of the same lesson in the opposite direction: `benches/prune_sweep.rs`
//! puts the whole prune at 2.4 per cent of the pipeline, so its 1.7x is worth
//! about one per cent of a run.

use bonsai_rs::utils::kernels::{edge_loglik, edge_newton, prep_edge, prune_binary_scalar};
use bonsai_rs::utils::rng::splitmix64_at;
use bonsai_rs::utils::simd::edge_newton_simd;
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
    let m_k: Vec<f64> = (0..P)
        .map(|g| splitmix64_at(g as u64) * 4.0 - 2.0)
        .collect();
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

    println!();
    println!("diagnostics for edge_newton, not shipped kernels:");

    // Same shape with the division replaced by a multiply. If this is much
    // faster than `edge_newton` the division is the wall and more accumulator
    // chains cannot help; if it is not, the two-chain latency is the wall.
    report("  no division", 7.0, 16.0, || {
        let (f, fp) = edge_newton_no_division(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    // The division kept, the two dependency chains split into eight. The other
    // half of the same question.
    report("  eight chains", 7.0, 16.0, || {
        let (f, fp) = edge_newton_eight_chains(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    // Both, so the two effects can be told apart from their combination.
    report("  batched division", 7.0, 16.0, || {
        let (f, fp) = edge_newton_batched(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    // Explicit lanes, which none of the three above touch.
    report("  f64x4 lanes", 7.0, 16.0, || {
        let (f, fp) = edge_newton_simd(black_box(&s), black_box(&d), 0.83);
        f + fp
    });
}

/// [`edge_newton`] with the reciprocal replaced by a multiply.
///
/// Numerically meaningless; it exists only to price the division. Everything
/// else about the loop, including the two dependency chains, is unchanged.
///
/// ### Params
///
/// * `s` - Summed inverse precisions
/// * `d` - Squared separations
/// * `t` - Branch length
///
/// ### Returns
///
/// Two numbers of no significance, shaped like `(f, f')`.
fn edge_newton_no_division(s: &[f64], d: &[f64], t: f64) -> (f64, f64) {
    let mut f = 0.0f64;
    let mut fp = 0.0f64;
    for g in 0..s.len() {
        let r = 0.5 * (s[g] + t);
        let dr = d[g] * r;
        f += r * (1.0 - dr);
        fp += r * r * (2.0 * dr - 1.0);
    }
    (f, fp)
}

/// [`edge_newton`] with the two accumulators split into eight.
///
/// Exact same arithmetic per feature and the same division; only the
/// dependency structure of the reduction changes. Chains are summed in a fixed
/// order, so this is deterministic, but it does not agree with `edge_newton` to
/// the bit.
///
/// ### Params
///
/// * `s` - Summed inverse precisions
/// * `d` - Squared separations
/// * `t` - Branch length
///
/// ### Returns
///
/// The pair `(f(t), f'(t))`.
fn edge_newton_eight_chains(s: &[f64], d: &[f64], t: f64) -> (f64, f64) {
    let mut f = [0.0f64; 4];
    let mut fp = [0.0f64; 4];
    let n = s.len() - s.len() % 4;
    for g in (0..n).step_by(4) {
        for k in 0..4 {
            let r = 1.0 / (s[g + k] + t);
            let dr = d[g + k] * r;
            f[k] += r * (1.0 - dr);
            fp[k] += r * r * (2.0 * dr - 1.0);
        }
    }
    let (mut f_tot, mut fp_tot) = (f[0] + f[1] + f[2] + f[3], fp[0] + fp[1] + fp[2] + fp[3]);
    for g in n..s.len() {
        let r = 1.0 / (s[g] + t);
        let dr = d[g] * r;
        f_tot += r * (1.0 - dr);
        fp_tot += r * r * (2.0 * dr - 1.0);
    }
    (f_tot, fp_tot)
}

/// [`edge_newton`] with one division per four features and eight chains.
///
/// Four reciprocals from one division by the same identity
/// `model::merge::batched_reciprocals` uses, extended to four terms. The
/// products of three `r` values overflow far sooner than the merge kernel's
/// triple product does, so this is a measurement variant and not a candidate
/// for shipping as written.
///
/// ### Params
///
/// * `s` - Summed inverse precisions
/// * `d` - Squared separations
/// * `t` - Branch length
///
/// ### Returns
///
/// The pair `(f(t), f'(t))`.
fn edge_newton_batched(s: &[f64], d: &[f64], t: f64) -> (f64, f64) {
    let mut f = [0.0f64; 4];
    let mut fp = [0.0f64; 4];
    let n = s.len() - s.len() % 4;
    for g in (0..n).step_by(4) {
        let r0 = s[g] + t;
        let r1 = s[g + 1] + t;
        let r2 = s[g + 2] + t;
        let r3 = s[g + 3] + t;
        let p01 = r0 * r1;
        let p23 = r2 * r3;
        let x = 1.0 / (p01 * p23);
        let a = [r1 * p23 * x, r0 * p23 * x, r3 * p01 * x, r2 * p01 * x];
        for k in 0..4 {
            let dr = d[g + k] * a[k];
            f[k] += a[k] * (1.0 - dr);
            fp[k] += a[k] * a[k] * (2.0 * dr - 1.0);
        }
    }
    let (mut f_tot, mut fp_tot) = (f[0] + f[1] + f[2] + f[3], fp[0] + fp[1] + fp[2] + fp[3]);
    for g in n..s.len() {
        let r = 1.0 / (s[g] + t);
        let dr = d[g] * r;
        f_tot += r * (1.0 - dr);
        fp_tot += r * r * (2.0 * dr - 1.0);
    }
    (f_tot, fp_tot)
}
