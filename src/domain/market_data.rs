/// Market data value object — immutable snapshot of market inputs.
///
/// All fields expressed in consistent units:
/// - `spot`: current underlying price (e.g. 100.0)
/// - `vol`:  annualised Black-Scholes implied volatility (e.g. 0.25 = 25%)
/// - `r`:    continuously compounded risk-free rate (e.g. 0.05 = 5%)
/// - `q`:    continuous dividend yield (e.g. 0.02 = 2%)
#[derive(Debug, Clone, Copy)]
pub struct MarketData {
    pub spot: f64,
    pub vol:  f64,
    pub r:    f64,
    pub q:    f64,
}

impl MarketData {
    pub fn new(spot: f64, vol: f64, r: f64, q: f64) -> Self {
        Self { spot, vol, r, q }
    }

    /// Risk-neutral drift: (r - q - σ²/2)
    pub fn log_drift(&self) -> f64 {
        self.r - self.q - 0.5 * self.vol * self.vol
    }
}
