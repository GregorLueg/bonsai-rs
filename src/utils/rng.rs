//! The crate's deterministic pseudo-random source.
//!
//! Splitmix64: reproducible on every platform and independent of thread count,
//! with no dependency to rewrite expected test values. [`SplitMix64`] is
//! stateful; [`splitmix64_at`] is stateless (element `i` is the first draw of
//! `SplitMix64::new(i)`), so an external reference can reproduce a stream.

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
/// One fewer, so the half-bit offset stays representable; see
/// [`SplitMix64::uniform_nonzero`].
const OPEN_MANTISSA_BITS: u32 = 52;

//////////////
// The PRNG //
//////////////

/// A splitmix64 stream.
///
/// Seeded directly with the caller's seed.
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
    /// For logarithms of the variate. Uses 52 mantissa bits plus a half-bit
    /// offset: at 53 bits `2^53 - 0.5` rounds back to `2^53` and would return
    /// `1.0`. Extremes are `2^-53` and `1 - 2^-53`.
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
    /// The second variate is discarded so the stream position does not depend
    /// on call parity.
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
    /// Modulo bias is of order `n / 2^64`.
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
/// Equal to the first [`SplitMix64::uniform`] draw of a stream seeded at
/// `index`.
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
    /// The mixing is a bijection, so it can be inverted.
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
        // Corner mantissas must not give `ln(u) == 0`, else `exponential`
        // returns zero.
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
