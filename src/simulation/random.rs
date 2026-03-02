/// Fast xoshiro256++ pseudo-random number generator.
///
/// - 256-bit state → period 2^256 − 1, far exceeding any simulation need.
/// - No mutex, no atomic — each batch gets its own instance seeded from
///   `(batch_id, global_seed)`, so there is zero lock contention.
/// - Box-Muller transform produces standard normal pairs from uniform pairs.
///
/// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
/// Generators", 2021 (xoshiro256++ variant).
pub struct BoxMullerRng {
    state: [u64; 4],
    // Box-Muller produces two normals per call; cache the second.
    cached: Option<f64>,
}

impl BoxMullerRng {
    /// Create a new RNG from a 64-bit seed.
    /// Uses SplitMix64 to initialise the 256-bit xoshiro state.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let state = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self { state, cached: None }
    }

    /// Derive a seed for batch `batch_id` given a `global_seed`.
    /// Simple bijection: avoids all batches sharing the same sequence.
    pub fn batch_seed(global_seed: u64, batch_id: u64) -> u64 {
        global_seed
            .wrapping_add(batch_id.wrapping_mul(0x9e3779b97f4a7c15))
    }

    /// Generate a uniform [0, 1) sample via xoshiro256++.
    #[inline]
    pub fn next_uniform(&mut self) -> f64 {
        let result = rotl(self.state[0].wrapping_add(self.state[3]), 23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = rotl(self.state[3], 45);
        // Map to [0, 1) — use upper 53 bits for float precision
        (result >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Generate a standard normal deviate via Box-Muller.
    ///
    /// Produces two normals per uniform pair; caches the second to avoid waste.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.cached.take() {
            return z;
        }
        // Box-Muller transform
        let u1 = loop {
            let u = self.next_uniform();
            if u > 1e-300 { break u; } // avoid log(0)
        };
        let u2 = self.next_uniform();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z1 = mag * theta.cos();
        let z2 = mag * theta.sin();
        self.cached = Some(z2);
        z1
    }

    /// Fill a slice with independent standard normal deviates.
    #[inline]
    pub fn fill_normal(&mut self, buf: &mut [f64]) {
        for x in buf.iter_mut() {
            *x = self.next_normal();
        }
    }
}

#[inline(always)]
fn rotl(x: u64, k: u64) -> u64 {
    (x << k) | (x >> (64 - k))
}

/// SplitMix64 — used to seed xoshiro from a single u64.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
