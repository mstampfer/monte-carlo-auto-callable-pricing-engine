use statrs::distribution::{Normal, ContinuousCDF};

/// One-Step Survival (Glasserman-Staum) for autocall barrier conditioning.
///
/// At each monthly boundary t_k, instead of simulating paths that may cross
/// the autocall barrier (which would terminate the path), we:
///
/// 1. Compute the probability of NOT crossing: p_k = Φ(d_k)
/// 2. Multiply the path weight by p_k
/// 3. Draw the spot from a truncated normal (conditioned on NOT crossing)
///
/// This keeps ALL paths alive until maturity, weighted by their survival
/// probability. The estimator remains unbiased:
///   V̂ = (1/M) Σ_m  w_m · payoff(path_m) · exp(−r·T)
///
/// The key benefit: the payoff function becomes smooth in S_0, enabling
/// stable finite-difference Greeks.
pub struct OneStepSurvival {
    normal: Normal,
}

impl OneStepSurvival {
    pub fn new() -> Self {
        Self {
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    /// Apply one-step survival conditioning at a monthly boundary.
    ///
    /// # Arguments
    /// * `spot_prev`    — spot at the previous monthly date (or t=0)
    /// * `barrier_call` — autocall barrier level (absolute, not fractional)
    /// * `drift`        — log drift = (r - q - σ²/2)
    /// * `vol`          — annualised volatility σ
    /// * `dt`           — time step length in years
    /// * `u`            — uniform [0,1) draw for the truncated normal
    ///
    /// # Returns
    /// `(new_spot, survival_prob)` where:
    /// - `new_spot` is drawn conditioned on NOT autocalling (S < B_c)
    /// - `survival_prob` is Φ(d_k) — multiply into path weight
    pub fn apply(
        &self,
        spot_prev:    f64,
        barrier_call: f64,
        drift:        f64,
        vol:          f64,
        dt:           f64,
        u:            f64,   // uniform [0,1) used for truncated draw
    ) -> (f64, f64) {
        let sqrt_dt = dt.sqrt();

        // d_k = (ln(B_c / S_prev) - drift·Δt) / (σ·√Δt)
        // This is the standard-normal quantile of the autocall boundary.
        let d_k = (barrier_call / spot_prev).ln() - drift * dt;
        let d_k = d_k / (vol * sqrt_dt);

        // Survival probability: Φ(d_k)
        let p_k = self.normal.cdf(d_k);

        // Truncated draw: Z ~ N(0,1) | Z < d_k
        // Φ⁻¹(u · Φ(d_k))
        let z_k = self.normal.inverse_cdf(u * p_k);

        // New spot conditioned on survival (always < B_c)
        let new_spot = spot_prev * (drift * dt + vol * sqrt_dt * z_k).exp();

        (new_spot, p_k)
    }
}

impl Default for OneStepSurvival {
    fn default() -> Self {
        Self::new()
    }
}
