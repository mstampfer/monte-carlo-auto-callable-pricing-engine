/// Brownian bridge for daily knock-in monitoring.
///
/// Between consecutive monthly dates (t_{k-1}, S_prev) and (t_k, S_end),
/// we reconstruct the daily path using a Brownian bridge conditioned on
/// both endpoints. This preserves the correct continuous-time distribution
/// for the minimum/maximum between the endpoints.
///
/// Bridge formula for sub-step j out of n total sub-steps:
/// ```text
/// S(t_j) = S_prev^{(n-j)/n} · S_end^{j/n} · exp( σ · √Δt_fine · ε_j )
/// ```
/// where ε_j are i.i.d. N(0,1) bridge residuals.
///
/// In practice we use the equivalent log-space formulation:
/// ```text
/// log S(t_j) = ((n-j)/n)·log(S_prev) + (j/n)·log(S_end) + σ·√Δt_fine·B_j
/// ```
/// where B_j is the j-th Brownian bridge increment (mean 0, variance j(n-j)/n · Δt_fine).
pub struct BrownianBridge;

impl BrownianBridge {
    /// Fill `spots` with n_sub daily spot values bridged between `s_prev` and `s_end`.
    ///
    /// # Arguments
    /// * `s_prev`   — spot at the start of the monthly interval
    /// * `s_end`    — spot at the end of the monthly interval (already drawn)
    /// * `vol`      — annualised volatility σ
    /// * `dt_fine`  — fine (daily) time step size in years
    /// * `z_fine`   — slice of n_sub standard normal deviates (pre-drawn)
    /// * `spots`    — output buffer; length must equal n_sub
    ///
    /// # Note
    /// `spots[n_sub-1]` corresponds to `s_end` (the monthly boundary).
    /// The engine uses `spots[0..n_sub-1]` for daily knock-in checking.
    pub fn fill(
        s_prev:  f64,
        s_end:   f64,
        vol:     f64,
        dt_fine: f64,
        z_fine:  &[f64],
        spots:   &mut [f64],
    ) {
        let n = z_fine.len();
        debug_assert_eq!(spots.len(), n);
        debug_assert!(n >= 1);

        let log_prev = s_prev.ln();
        let log_end  = s_end.ln();

        // σ · √Δt_fine — standard deviation of each bridge residual term
        let vol_sqrt_dt = vol * dt_fine.sqrt();

        // Build the bridge using the sequential conditioning formula.
        // At step j (1-indexed), the conditional mean of log S(t_j) given
        // log S(t_{j-1}) and log S(t_end) is:
        //   μ_j = log S(t_{j-1}) + (log_end - log S(t_{j-1})) / (n - j + 1)
        // and the conditional std dev is:
        //   σ_j = σ · √( Δt_fine · (n-j) / (n-j+1) )
        //
        // This is the standard backward-induction bridge (Glasserman §3.3).

        let mut log_curr = log_prev;

        for j in 0..n {
            let remaining = (n - j) as f64; // steps remaining including this one
            // Conditional mean increment toward log_end
            let mean_log = log_curr + (log_end - log_curr) / remaining;
            // Conditional std dev
            let bridge_vol = if remaining > 1.0 {
                vol_sqrt_dt * ((remaining - 1.0) / remaining).sqrt()
            } else {
                0.0 // last step lands exactly on log_end
            };
            log_curr = mean_log + bridge_vol * z_fine[j];
            spots[j] = log_curr.exp();
        }

        // Ensure the last point exactly equals s_end (eliminate floating-point drift)
        if n > 0 {
            spots[n - 1] = s_end;
        }
    }
}
