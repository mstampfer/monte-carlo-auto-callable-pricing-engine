/// Per-path mutable state during Monte Carlo simulation.
///
/// Kept small and cache-friendly — only three scalars.
/// The engine resets this between paths (no allocation).
#[derive(Debug, Clone, Copy)]
pub struct PathState {
    /// Current spot price.
    pub spot: f64,

    /// Accumulated path weight from one-step survival conditioning.
    /// Starts at 1.0; multiplied by Φ(d_k) at each monthly boundary.
    pub weight: f64,

    /// True if spot has breached the knock-in barrier on any day.
    pub knock_in: bool,
}

impl PathState {
    pub fn new(spot: f64) -> Self {
        Self { spot, weight: 1.0, knock_in: false }
    }

    pub fn reset(&mut self, spot: f64) {
        self.spot    = spot;
        self.weight  = 1.0;
        self.knock_in = false;
    }
}

/// Pre-allocated scratch buffers reused across all paths in a batch.
///
/// Allocated **once** per batch; zero allocation in the hot path loop.
pub struct BatchBuffers {
    /// Standard normal deviates for the coarse monthly steps.
    pub z_coarse: Vec<f64>,
    /// Standard normal deviates for daily sub-steps in current monthly interval.
    pub z_fine:   Vec<f64>,
    /// Intermediate spot values for Brownian bridge reconstruction.
    pub daily_spots: Vec<f64>,
}

impl BatchBuffers {
    pub fn new(n_monthly: usize, n_daily_per_month: usize) -> Self {
        Self {
            z_coarse:    vec![0.0; n_monthly],
            z_fine:      vec![0.0; n_daily_per_month],
            daily_spots: vec![0.0; n_daily_per_month],
        }
    }
}
