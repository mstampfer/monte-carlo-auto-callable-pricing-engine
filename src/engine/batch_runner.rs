use crate::simulation::BoxMullerRng;

/// Configuration for a single simulation batch.
///
/// Passed by value into `spawn_blocking` / `rayon::spawn` — must be `Send + 'static`.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of paths to simulate in this batch.
    pub n_paths: usize,
    /// RNG seed for this batch (unique per batch, derived from global seed + batch id).
    pub seed: u64,
}

impl BatchConfig {
    pub fn new(n_paths: usize, seed: u64) -> Self {
        Self { n_paths, seed }
    }
}

/// Aggregated result from a single simulation batch.
///
/// Designed to be merged across batches without storing individual path payoffs.
#[derive(Debug, Clone, Default)]
pub struct PartialResult {
    /// Number of paths included in this result.
    pub n_paths: usize,
    /// Sum of discounted weighted payoffs: Σ w_i · payoff_i · disc
    pub sum_payoff: f64,
    /// Sum of squares for standard error calculation: Σ (w_i · payoff_i · disc)²
    pub sum_payoff_sq: f64,
}

impl PartialResult {
    pub fn new(n_paths: usize, sum_payoff: f64, sum_payoff_sq: f64) -> Self {
        Self { n_paths, sum_payoff, sum_payoff_sq }
    }

    /// Merge two partial results (combine batches).
    pub fn merge(self, other: PartialResult) -> PartialResult {
        PartialResult {
            n_paths:      self.n_paths + other.n_paths,
            sum_payoff:   self.sum_payoff + other.sum_payoff,
            sum_payoff_sq: self.sum_payoff_sq + other.sum_payoff_sq,
        }
    }

    /// Monte Carlo price estimate.
    pub fn price(&self) -> f64 {
        if self.n_paths == 0 { return 0.0; }
        self.sum_payoff / self.n_paths as f64
    }

    /// Standard error of the price estimate.
    pub fn std_err(&self) -> f64 {
        if self.n_paths < 2 { return f64::NAN; }
        let n = self.n_paths as f64;
        let mean = self.sum_payoff / n;
        let variance = (self.sum_payoff_sq / n) - mean * mean;
        // Clamp to avoid negative variance from floating-point cancellation
        (variance.max(0.0) / n).sqrt()
    }

    /// 95% confidence interval [lower, upper].
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        let p  = self.price();
        let se = self.std_err();
        (p - 1.96 * se, p + 1.96 * se)
    }

    /// Create an RNG from the batch config seed.
    pub fn rng_from_seed(seed: u64) -> BoxMullerRng {
        BoxMullerRng::from_seed(seed)
    }
}
