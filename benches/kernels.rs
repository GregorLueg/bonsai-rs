//! Single-threaded throughput of each feature-axis kernel (achieved FLOP/s and
//! bandwidth), with working sets sized to stay in L2. Also prices the scalar
//! `edge_newton` against diagnostic variants and the `f64x4` tier.
//!
//! ```sh
//! cargo bench --bench kernels
//! ```

use bonsai_rs::utils::kernels::{edge_loglik, edge_newton, prep_edge, prune_binary_scalar};
use bonsai_rs::utils::rng::splitmix64_at;
use bonsai_rs::utils::simd::edge_newton_simd;
use std::hint::black_box;
use std::time::Instant;

/// Features per call, keeping the working set inside L2.
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

/// [`edge_newton`] with the reciprocal replaced by a multiply.
///
/// Numerically meaningless; it only prices the division.
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
/// Same arithmetic and division; only the reduction's dependency structure
/// changes, so it does not agree with `edge_newton` to the bit.
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
/// Four reciprocals from one division, as in `model::merge::batched_reciprocals`.
/// Overflows far sooner than that kernel, so it is a measurement variant only.
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

    report("  no division", 7.0, 16.0, || {
        let (f, fp) = edge_newton_no_division(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    report("  eight chains", 7.0, 16.0, || {
        let (f, fp) = edge_newton_eight_chains(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    report("  batched division", 7.0, 16.0, || {
        let (f, fp) = edge_newton_batched(black_box(&s), black_box(&d), 0.83);
        f + fp
    });

    report("  f64x4 lanes", 7.0, 16.0, || {
        let (f, fp) = edge_newton_simd(black_box(&s), black_box(&d), 0.83);
        f + fp
    });
}
