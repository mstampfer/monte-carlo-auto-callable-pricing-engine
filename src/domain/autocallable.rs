use crate::domain::product::Product;

/// An autocallable structured note with:
/// - Monthly autocall (knock-out) monitoring: if S(t_k) >= barrier_call → pay coupon, terminate
/// - Daily knock-in monitoring: if S(t) < barrier_ki on any day → set knock-in flag
/// - Maturity payoff depends on knock-in status and final spot
///
/// Payoff at maturity (if never autocalled):
/// - No KI:               pay notional (capital protected)
/// - KI AND S(T) >= B_c:  pay notional * (1 + coupon_final)
/// - KI AND S(T) <  B_c:  pay notional * S(T)/S(0)
#[derive(Debug, Clone)]
pub struct AutoCallable {
    /// Spot at inception (S_0), for participation calculation.
    pub spot_initial: f64,

    /// Notional amount.
    pub notional: f64,

    /// Autocall (knock-out) barrier as fraction of S_0 (e.g. 1.0 = at-the-money).
    pub barrier_call_frac: f64,

    /// Knock-in barrier as fraction of S_0 (e.g. 0.7 = 70% of S_0).
    pub barrier_ki_frac: f64,

    /// Coupon paid at each autocall date (as fraction of notional, e.g. 0.05 = 5%).
    /// Length must equal the number of monthly observation dates.
    pub coupons: Vec<f64>,

    /// Observation dates in years (coarse/monthly grid).
    pub obs_dates: Vec<f64>,

    // --- mutable path state ---
    knock_in: bool,
    autocall_triggered: bool,
    autocall_payoff: f64,
    current_obs_idx: usize,
}

impl AutoCallable {
    pub fn new(
        spot_initial:      f64,
        notional:          f64,
        barrier_call_frac: f64,
        barrier_ki_frac:   f64,
        coupons:           Vec<f64>,
        obs_dates:         Vec<f64>,
    ) -> Self {
        Self {
            spot_initial,
            notional,
            barrier_call_frac,
            barrier_ki_frac,
            coupons,
            obs_dates,
            knock_in: false,
            autocall_triggered: false,
            autocall_payoff: 0.0,
            current_obs_idx: 0,
        }
    }

    fn barrier_call(&self) -> f64 {
        self.spot_initial * self.barrier_call_frac
    }

    fn barrier_ki(&self) -> f64 {
        self.spot_initial * self.barrier_ki_frac
    }
}

impl Product for AutoCallable {
    fn observation_dates(&self) -> &[f64] {
        &self.obs_dates
    }

    /// Called at each monthly observation date.
    /// Returns `true` if autocall triggers (path should terminate).
    fn notify(&mut self, _t: f64, spot: f64) -> bool {
        // Check knock-in first (spot below barrier)
        if spot < self.barrier_ki() {
            self.knock_in = true;
        }

        // Check autocall condition
        if spot >= self.barrier_call() {
            let coupon = self.coupons.get(self.current_obs_idx).copied().unwrap_or(0.0);
            self.autocall_payoff = self.notional * (1.0 + coupon);
            self.autocall_triggered = true;
            self.current_obs_idx += 1;
            return true; // terminate path
        }

        self.current_obs_idx += 1;
        false
    }

    fn terminal_payoff(&self, spot_at_maturity: f64) -> f64 {
        if self.autocall_triggered {
            return self.autocall_payoff;
        }

        // Maturity payoff
        let barrier_call = self.barrier_call();
        let knock_in = self.knock_in;

        if !knock_in {
            // Capital protected: return notional
            self.notional
        } else if spot_at_maturity >= barrier_call {
            // KI happened but recovered: pay notional + final coupon
            let coupon_final = self.coupons.last().copied().unwrap_or(0.0);
            self.notional * (1.0 + coupon_final)
        } else {
            // KI and below call: participate in downside
            self.notional * (spot_at_maturity / self.spot_initial)
        }
    }

    fn reset(&mut self) {
        self.knock_in           = false;
        self.autocall_triggered = false;
        self.autocall_payoff    = 0.0;
        self.current_obs_idx    = 0;
    }

    fn knock_in_triggered(&self) -> bool {
        self.knock_in
    }

    fn set_knock_in(&mut self) {
        self.knock_in = true;
    }

    /// OSS autocall contribution: the payoff at monthly step `step_idx`
    /// if the autocall barrier were crossed.
    fn oss_autocall_payoff(&self, step_idx: usize) -> Option<f64> {
        let coupon = self.coupons.get(step_idx).copied().unwrap_or(0.0);
        Some(self.notional * (1.0 + coupon))
    }
}
