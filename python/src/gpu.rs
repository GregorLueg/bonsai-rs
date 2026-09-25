//! Sanity on the GPU, when this build and this machine can reach one.
//!
//! Two separate questions. Whether the extension was compiled with the `gpu`
//! feature is fixed when the wheel is built; it is on by default, because the
//! CubeCL/wgpu code adds little to the binary. Whether there is an adapter is
//! only answerable at runtime, and is legitimately "no" on a headless box or in
//! a container without the Vulkan loader. [`gpu_available`] answers both at
//! once, and asking for the GPU where it is `False` raises rather than falling
//! back silently.
//!
//! The device path is `f32` throughout, whatever dtype the caller asked for:
//! wgpu has no `f64`. `sanity-sc-rs` documents what that costs.

use pyo3::prelude::*;
use sanity_sc_rs::SanityOutput;
use sanity_sc_rs::config::SanityParams;
use sanity_sc_rs::float::SanityFloat;
use sanity_sc_rs::input::CountMatrix;

use crate::error::{BErr, BonsaiError};

////////////
// Probe  //
////////////

/// Whether Sanity can run on the GPU here.
///
/// ### Params
///
/// * `py` - Attached interpreter token
///
/// ### Returns
///
/// `True` only when this build has the `gpu` feature **and** wgpu resolves an
/// adapter. Safe to call on any machine.
#[pyfunction]
pub fn gpu_available(py: Python<'_>) -> bool {
    py.detach(probe)
}

/// Try to stand up a wgpu client on the default device.
///
/// Acquiring a client panics rather than erroring when there is no adapter, so
/// the attempt is caught. That is sound because the release profile pins
/// `panic = "unwind"`, which pyo3 needs anyway. The panic hook is silenced for
/// the duration, or merely asking prints a backtrace.
///
/// ### Returns
///
/// `true` if a client came back, `false` if the attempt panicked.
#[cfg(feature = "gpu")]
fn probe() -> bool {
    use cubecl::Runtime;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(|| {
        let _ = WgpuRuntime::client(&WgpuDevice::default());
    })
    .is_ok();
    std::panic::set_hook(hook);
    ok
}

/// Stub for builds without the `gpu` feature.
///
/// ### Returns
///
/// Always `false`.
#[cfg(not(feature = "gpu"))]
fn probe() -> bool {
    false
}

/// Refuse a GPU request this build or machine cannot serve.
///
/// Called with the interpreter attached, before the work is detached, so the
/// refusal is a Python exception rather than a panic on a worker.
///
/// ### Params
///
/// * `gpu` - Whether the caller asked for the GPU
///
/// ### Returns
///
/// `Ok` when the request can be served, or a `BonsaiError` saying why not.
pub(crate) fn check(gpu: bool) -> PyResult<()> {
    if !gpu || probe() {
        return Ok(());
    }
    let why = if cfg!(feature = "gpu") {
        "gpu=True, but wgpu found no adapter on this machine"
    } else {
        "gpu=True, but this build of bonsai-rs has no GPU support compiled in"
    };
    Err(BonsaiError::new_err(format!(
        "{why}; check bonsai_rs.gpu_available() first"
    )))
}

////////////
// Sanity //
////////////

/// Run Sanity on the device the caller chose.
///
/// ### Params
///
/// * `counts` - The counts
/// * `totals` - Total UMIs per cell over all genes
/// * `sp` - Sanity parameters
/// * `gpu` - Run on the GPU; [`check`] must have passed for it
///
/// ### Returns
///
/// The Sanity output, or the error either path raised.
pub(crate) fn run_sanity<T: SanityFloat>(
    counts: &CountMatrix,
    totals: &[f64],
    sp: SanityParams,
    gpu: bool,
) -> Result<SanityOutput<T>, BErr> {
    #[cfg(feature = "gpu")]
    if gpu {
        use cubecl::Runtime;
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        let client = WgpuRuntime::client(&WgpuDevice::default());
        return Ok(sanity_sc_rs::gpu::sanity_gpu::<T, WgpuRuntime>(
            counts,
            totals,
            Some(sp),
            &client,
        )?);
    }
    let _ = gpu;
    Ok(sanity_sc_rs::sanity::<T>(counts, totals, Some(sp))?)
}
