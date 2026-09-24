//! The crate's deterministic pseudo-random source.
//!
//! Fixtures, simulated datasets and benchmarks all need reproducible
//! pseudo-randomness and nothing else: no distributional quality beyond
//! plausible-looking spread, no cryptographic property, and above all no
//! dependency whose next release would silently rewrite every expected value in
//! the test suite. Splitmix64 is three lines of arithmetic, produces identical
//! output on every platform, and is independent of thread count.
//!
//! Two forms of the same stream are offered. [`SplitMix64`] is stateful and is
//! what a generator loop wants. [`splitmix64_at`] is stateless, so element `i`
//! is a pure function of `i`, which is what a benchmark cross-checked against
//! an external reference wants: the other side can produce the same stream
//! without replicating a loop. `splitmix64_at(i)` is by construction the first
//! draw of `SplitMix64::new(i)`.

///////////////
// Constants //
///////////////

/// Golden-ratio increment of the splitmix64 stream, `floor(2^64 / phi)`.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First multiplicative mixing constant of splitmix64.
const SPLITMIX_MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second multiplicative mixing constant of splitmix64.
const SPLITMIX_MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// Mantissa bits used when turning a `u64` into a double in `[0, 1)`.
const MANTISSA_BITS: u32 = 53;

/// Mantissa bits used for the open interval `(0, 1)`.
///
/// One fewer, so that the half-bit offset that opens the upper end is still
/// representable; see [`SplitMix64::uniform_nonzero`].
const OPEN_MANTISSA_BITS: u32 = 52;

//////////////
// The PRNG //
//////////////

/// A splitmix64 stream.
///
/// Three lines of state advance and mixing, no dependency, identical output
/// everywhere. Seeded directly with the caller's seed, so distinct seeds give
/// distinct streams from the first draw.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitMix64 {
    /// Stream position; advanced by [`SPLITMIX_GAMMA`] per draw.
    state: u64,
}

impl SplitMix64 {
    /// Start a stream at a seed.
    ///
    /// ### Params
    ///
    /// * `seed` - Seed value; any `u64` is valid, including zero
    ///
    /// ### Returns
    ///
    /// The stream.
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next raw 64-bit word.
    ///
    /// ### Returns
    ///
    /// A uniformly distributed `u64`.
    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MIX_B);
        z ^ (z >> 31)
    }

    /// Draw a uniform on the half-open interval `[0, 1)`.
    ///
    /// ### Returns
    ///
    /// The variate.
    #[inline]
    pub(crate) fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> (64 - MANTISSA_BITS);
        bits as f64 / (1u64 << MANTISSA_BITS) as f64
    }

    /// Draw a uniform on the open interval `(0, 1)`.
    ///
    /// Needed wherever a logarithm is taken of the variate, which is both
    /// Box-Muller and the exponential inverse transform.
    ///
    /// One bit shorter than [`SplitMix64::uniform`], and offset by half a bit,
    /// which is what keeps both ends open. `(bits + 1) / 2^53` is exactly `1.0`
    /// on the largest of the `2^53` mantissas, whose logarithm is zero, and an
    /// exponential of exactly zero is a variance of zero that the simulator
    /// divides by. Half a bit does not fix that at 53:
    /// the spacing just below `2^53` is `1`, so `2^53 - 0.5` rounds straight
    /// back up to `2^53`. At 52 it is `0.5`, the offset survives, and the
    /// extreme draws are `2^-53` and `1 - 2^-53`. One draw in `2^53` either way,
    /// so this was never reachable in practice; it is fixed because the
    /// alternative is a documented guarantee that is not one.
    ///
    /// ### Returns
    ///
    /// The variate, never zero and never one.
    #[inline]
    pub(crate) fn uniform_nonzero(&mut self) -> f64 {
        let bits = self.next_u64() >> (64 - OPEN_MANTISSA_BITS);
        (bits as f64 + 0.5) / (1u64 << OPEN_MANTISSA_BITS) as f64
    }

    /// Draw a standard normal variate by Box-Muller.
    ///
    /// The second variate of the pair is discarded rather than cached. Caching
    /// would halve the cost but would make the stream position depend on the
    /// parity of previous calls, and this module interleaves normal and uniform
    /// draws; a fixed two-uniforms-per-normal cost keeps the stream trivially
    /// auditable.
    ///
    /// ### Returns
    ///
    /// A draw from `N(0, 1)`.
    #[inline]
    pub(crate) fn normal(&mut self) -> f64 {
        let radial = (-2.0 * self.uniform_nonzero().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        radial * angle.cos()
    }

    /// Draw an exponential variate with a given mean, by inverse transform.
    ///
    /// ### Params
    ///
    /// * `mean` - Mean of the distribution, strictly positive
    ///
    /// ### Returns
    ///
    /// The variate, strictly positive.
    #[inline]
    pub(crate) fn exponential(&mut self, mean: f64) -> f64 {
        -mean * self.uniform_nonzero().ln()
    }

    /// Draw a uniform variate on `[lo, hi)`.
    ///
    /// ### Params
    ///
    /// * `lo` - Lower bound
    /// * `hi` - Upper bound, at least `lo`
    ///
    /// ### Returns
    ///
    /// The variate.
    #[inline]
    pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }

    /// Draw a log-uniform variate on `[lo, hi]`.
    ///
    /// ### Params
    ///
    /// * `lo` - Lower bound, strictly positive
    /// * `hi` - Upper bound, at least `lo`
    ///
    /// ### Returns
    ///
    /// The variate.
    #[inline]
    pub(crate) fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.range(lo.ln(), hi.ln()).exp()
    }

    /// Draw an index uniformly from `0..n`.
    ///
    /// Uses the low bits of a 64-bit draw. The modulo bias is on the order of
    /// `n / 2^64` and is irrelevant for the leaf counts this module handles.
    ///
    /// ### Params
    ///
    /// * `n` - Exclusive upper bound, strictly positive
    ///
    /// ### Returns
    ///
    /// An index in `0..n`.
    #[inline]
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// One draw from the counter-based form of the stream.
///
/// Stateless: element `index` is a pure function of `index`, so an external
/// reference implementation can reproduce the stream without a loop. Equal to
/// the first [`SplitMix64::uniform`] draw of a stream seeded at `index`.
///
/// ### Params
///
/// * `index` - Position in the stream
///
/// ### Returns
///
/// A uniform in `[0, 1)`.
#[inline]
pub fn splitmix64_at(index: u64) -> f64 {
    SplitMix64::new(index).uniform()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Invert `y = x ^ (x >> s)`.
    ///
    /// ### Params
    ///
    /// * `y` - The xor-shifted value
    /// * `s` - Shift it was made with
    ///
    /// ### Returns
    ///
    /// The `x` that produces `y`.
    fn unxorshift(y: u64, s: u32) -> u64 {
        let mut x = y;
        let mut done = s;
        while done < 64 {
            x = y ^ (x >> s);
            done += s;
        }
        x
    }

    /// Multiplicative inverse modulo `2^64`, by Newton iteration.
    ///
    /// ### Params
    ///
    /// * `a` - An odd multiplier
    ///
    /// ### Returns
    ///
    /// The `b` with `a * b == 1` in wrapping arithmetic.
    fn inverse(a: u64) -> u64 {
        let mut x = 1u64;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        }
        x
    }

    /// The seed whose first raw draw is a given word.
    ///
    /// Splitmix64's mixing is a bijection, so the corners of the mantissa are
    /// reachable by inverting it rather than by searching for a seed.
    ///
    /// ### Params
    ///
    /// * `out` - Wanted first draw
    ///
    /// ### Returns
    ///
    /// The seed.
    fn seed_for(out: u64) -> u64 {
        let mut z = unxorshift(out, 31);
        z = z.wrapping_mul(inverse(SPLITMIX_MIX_B));
        z = unxorshift(z, 27);
        z = z.wrapping_mul(inverse(SPLITMIX_MIX_A));
        z = unxorshift(z, 30);
        z.wrapping_sub(SPLITMIX_GAMMA)
    }

    #[test]
    fn test_the_open_uniform_never_reaches_either_end() {
        // `(bits + 1) / 2^53` is exactly `1.0` on the largest of the `2^53`
        // mantissas, and `ln(1) == 0` makes
        // `exponential` return zero from a routine documented as strictly
        // positive. The simulator then divides by the square root of it.
        for out in [u64::MAX, 0, 2048, u64::MAX - 2047] {
            let seed = seed_for(out);
            assert_eq!(
                SplitMix64::new(seed).next_u64(),
                out,
                "the seed inversion is wrong, so this test proves nothing"
            );
            let u = SplitMix64::new(seed).uniform_nonzero();
            assert!(u > 0.0 && u < 1.0, "uniform_nonzero returned {u}");
            assert!(
                SplitMix64::new(seed).exponential(2.0) > 0.0,
                "exponential returned a non-positive variate"
            );
            assert!(SplitMix64::new(seed).normal().is_finite());
        }
    }
}
