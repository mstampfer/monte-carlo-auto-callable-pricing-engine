use crate::domain::MarketData;

/// Trait for evolving the underlying spot price over a single time step.
///
/// Implementations are stateless — all market parameters are baked in at
/// construction time. The hot loop passes only `(spot, dt, z)`.
pub trait Propagator: Send + Sync + 'static {
    /// Evolve spot from current value by one time step of length `dt`.
    ///
    /// `z` is a pre-drawn standard normal deviate.
    /// Returns the new spot value.
    fn propagate(&self, spot: f64, dt: f64, z: f64) -> f64;
}

/// Black-Scholes GBM propagator.
///
/// Exact log-normal discretisation:
/// ```text
/// S_{n+1} = S_n · exp( (r − q − σ²/2)·Δt + σ·√Δt · Z )
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BlackScholes {
    drift: f64,  // (r - q - σ²/2)
    vol:   f64,  // σ
}

impl BlackScholes {
    pub fn new(market: &MarketData) -> Self {
        Self {
            drift: market.log_drift(),
            vol:   market.vol,
        }
    }
}

impl Propagator for BlackScholes {
    #[inline(always)]
    fn propagate(&self, spot: f64, dt: f64, z: f64) -> f64 {
        let sqrt_dt = dt.sqrt();
        spot * (self.drift * dt + self.vol * sqrt_dt * z).exp()
    }
}
