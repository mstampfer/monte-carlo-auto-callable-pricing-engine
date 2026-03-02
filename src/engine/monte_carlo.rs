use std::sync::Arc;

use crate::domain::{DualTimeGrid, MarketData, Product, Propagator};
use crate::domain::time_grid::StepKind;
use crate::simulation::{BoxMullerRng, BrownianBridge, OneStepSurvival, PathState};
use crate::simulation::path_state::BatchBuffers;
use crate::engine::batch_runner::{BatchConfig, PartialResult};

/// Core Monte Carlo simulation engine.
///
/// Generic over:
/// - `P: Product`    — the financial instrument being priced
/// - `Pr: Propagator`— the spot evolution model (e.g. Black-Scholes)
///
/// `run_batch` is synchronous (CPU-bound work). Async coordination lives
/// entirely in the concurrency layer above this engine.
pub struct MonteCarloEngine<P: Product, Pr: Propagator> {
    /// Cloned once per batch; reset (not reallocated) per path.
    product_template: P,

    /// Shared read-only propagator.
    propagator: Arc<Pr>,

    /// Market data (rates, vol, spot).
    market_data: MarketData,

    /// Dual-frequency time grid.
    time_grid: DualTimeGrid,

    /// Autocall barrier (absolute level, e.g. 100.0).
    barrier_call: f64,

    /// Knock-in barrier (absolute level, e.g. 70.0).
    barrier_ki: f64,

    /// Number of monthly coarse steps.
    n_monthly: usize,

    /// Number of daily fine steps per monthly interval.
    n_daily_per_month: usize,
}

impl<P: Product, Pr: Propagator> MonteCarloEngine<P, Pr> {
    pub fn new(
        product_template:   P,
        propagator:         Arc<Pr>,
        market_data:        MarketData,
        time_grid:          DualTimeGrid,
        barrier_call:       f64,
        barrier_ki:         f64,
        n_monthly:          usize,
        n_daily_per_month:  usize,
    ) -> Self {
        Self {
            product_template,
            propagator,
            market_data,
            time_grid,
            barrier_call,
            barrier_ki,
            n_monthly,
            n_daily_per_month,
        }
    }

    /// Synchronous hot path — runs `n_paths` simulations and returns aggregated result.
    ///
    /// ### OSS Estimator (full autocallable)
    ///
    /// At each monthly step k, instead of possibly terminating the path,
    /// we condition on NOT autocalling and accumulate:
    ///
    ///   contribution_k = W_{k-1} · (1 − p_k) · autocall_payoff_k · disc(t_k)
    ///
    /// At maturity:
    ///
    ///   contribution_T = W_T · maturity_payoff · disc(T)
    ///
    /// Per-path total = sum of all contributions.
    /// Final price = mean(per-path total) over all paths.
    pub fn run_batch(&self, config: &BatchConfig) -> PartialResult {
        let mut rng = BoxMullerRng::from_seed(config.seed);
        let oss     = OneStepSurvival::new();

        // Pre-allocate buffers — reused every path, zero alloc in hot loop
        let mut bufs = BatchBuffers::new(self.n_monthly, self.n_daily_per_month);

        // Clone product once per batch; reset per path (cheap zeroing)
        let mut product = self.product_template.clone();

        let discount_t = (-self.market_data.r * self.time_grid.maturity).exp();
        let drift       = self.market_data.log_drift();
        let vol         = self.market_data.vol;
        let s0          = self.market_data.spot;
        let r           = self.market_data.r;

        let mut sum_payoff    = 0.0_f64;
        let mut sum_payoff_sq = 0.0_f64;

        // Pre-collect monthly steps to avoid repeated filtering
        let monthly_steps: Vec<_> = self.time_grid.steps.iter()
            .filter(|s| s.kind == StepKind::Monthly)
            .cloned()
            .collect();

        for _ in 0..config.n_paths {
            product.reset();

            let mut state = PathState::new(s0);

            // Per-path accumulated payoff (OSS: sum of autocall + maturity contributions)
            let mut path_total = 0.0_f64;

            let mut prev_spot = s0;
            let mut prev_t    = 0.0_f64;

            for (m_idx, monthly_step) in monthly_steps.iter().enumerate() {
                let dt_coarse = monthly_step.t - prev_t;

                // Weight BEFORE this step (W_{k-1})
                let w_prev = state.weight;

                // --- One-Step Survival: draw conditioned on NOT autocalling ---
                let u = rng.next_uniform();
                let (spot_monthly, p_k) = oss.apply(
                    prev_spot,
                    self.barrier_call,
                    drift,
                    vol,
                    dt_coarse,
                    u,
                );

                // OSS autocall contribution:
                //   W_{k-1} · (1 − p_k) · autocall_payoff_k · disc(t_k)
                if let Some(autocall_pay) = product.oss_autocall_payoff(m_idx) {
                    let disc_k = (-r * monthly_step.t).exp();
                    path_total += w_prev * (1.0 - p_k) * autocall_pay * disc_k;
                }

                // Update path weight: W_k = W_{k-1} · p_k
                state.weight *= p_k;

                // --- Brownian Bridge: fill daily sub-steps ---
                let n_sub = self.n_daily_per_month;
                rng.fill_normal(&mut bufs.z_fine[..n_sub]);
                let dt_fine = dt_coarse / n_sub as f64;

                BrownianBridge::fill(
                    prev_spot,
                    spot_monthly,
                    vol,
                    dt_fine,
                    &bufs.z_fine[..n_sub],
                    &mut bufs.daily_spots[..n_sub],
                );

                // Check knock-in on each daily sub-spot (excluding the monthly boundary)
                for &s in &bufs.daily_spots[..n_sub - 1] {
                    if s < self.barrier_ki {
                        product.set_knock_in();
                    }
                }

                // Check knock-in at monthly boundary
                if spot_monthly < self.barrier_ki {
                    product.set_knock_in();
                }

                state.spot = spot_monthly;

                // Notify product at the monthly date; with OSS, spot < B_c always,
                // so autocall never fires — just updates knock-in / obs counter.
                product.notify(monthly_step.t, spot_monthly);

                prev_spot = spot_monthly;
                prev_t    = monthly_step.t;
            }

            // --- Maturity contribution: W_T · maturity_payoff · disc(T) ---
            let maturity_pay = product.terminal_payoff(state.spot);
            path_total += state.weight * maturity_pay * discount_t;

            sum_payoff    += path_total;
            sum_payoff_sq += path_total * path_total;
        }

        PartialResult::new(config.n_paths, sum_payoff, sum_payoff_sq)
    }

    /// Convenience: run_batch from a config directly (clones engine internally).
    pub fn run_batch_owned(engine: Arc<Self>, config: BatchConfig) -> PartialResult {
        engine.run_batch(&config)
    }
}

impl<P: Product + Clone, Pr: Propagator> Clone for MonteCarloEngine<P, Pr> {
    fn clone(&self) -> Self {
        Self {
            product_template:  self.product_template.clone(),
            propagator:        Arc::clone(&self.propagator),
            market_data:       self.market_data,
            time_grid:         self.time_grid.clone(),
            barrier_call:      self.barrier_call,
            barrier_ki:        self.barrier_ki,
            n_monthly:         self.n_monthly,
            n_daily_per_month: self.n_daily_per_month,
        }
    }
}
