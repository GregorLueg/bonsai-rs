//! Verbosity levels and the progress printer the pipeline reports through.

use std::time::Duration;

/// Percentage step between two progress lines of a long sweep.
///
/// Bounds a sweep at ten progress lines.
const PROGRESS_STEP_PCT: usize = 10;

/// How much the pipeline prints while it runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verbosity {
    /// Nothing at all.
    #[default]
    Quiet,
    /// One line per pipeline step.
    Normal,
    /// Step lines plus progress inside the merge, SPR and NNI steps.
    Detailed,
}

impl Verbosity {
    /// Whether normal or detailed verbosity is set.
    ///
    /// ### Returns
    ///
    /// `true` for [`Verbosity::Normal`] and [`Verbosity::Detailed`].
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Whether detailed verbosity is set.
    ///
    /// ### Returns
    ///
    /// `true` for [`Verbosity::Detailed`].
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parse a verbosity level from an integer, for the FFI.
///
/// ### Params
///
/// * `level` - `0` quiet, `1` normal, `2` detailed; anything else is quiet
///
/// ### Returns
///
/// The matching [`Verbosity`].
pub fn parse_verbosity_level(level: usize) -> Verbosity {
    match level {
        1 => Verbosity::Normal,
        2 => Verbosity::Detailed,
        _ => Verbosity::Quiet,
    }
}

/// Print a progress line whenever a sweep crosses a decile of its work.
///
/// Cheap to call per unit of work; prints only across a [`PROGRESS_STEP_PCT`]
/// boundary or at completion. The verbosity check stays with the caller.
///
/// ### Params
///
/// * `done` - Units of work finished, including the one just completed
/// * `prev_done` - Units of work finished before it
/// * `total` - Units of work in the whole sweep; nothing prints if this is `0`
/// * `unit` - What is being counted, e.g. `"merges"`
/// * `elapsed` - Time since the sweep started
pub fn report_decile_progress(
    done: usize,
    prev_done: usize,
    total: usize,
    unit: &str,
    elapsed: Duration,
) {
    if total == 0 {
        return;
    }
    let pct = done * 100 / total;
    let prev_pct = prev_done * 100 / total;
    if pct / PROGRESS_STEP_PCT > prev_pct / PROGRESS_STEP_PCT || done == total {
        println!("    {pct}% ({done} / {total} {unit}, {elapsed:.2?})");
    }
}
