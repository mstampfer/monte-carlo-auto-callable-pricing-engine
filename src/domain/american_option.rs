use crate::domain::product::Product;

/// Stub American option — demonstrates Product trait plug-in genericity.
///
/// Uses a simple exercise decision: exercise if intrinsic value > hold value
/// (approximated here as exercise whenever in-the-money at observation dates,
/// i.e., a Bermudan approximation).
///
/// This is NOT a production-quality American option pricer — it is included
/// solely to show that `MonteCarloEngine` is instrument-agnostic.
#[derive(Debug, Clone)]
pub struct AmericanOption {
    pub strike:    f64,
    pub is_call:   bool,
    pub notional:  f64,
    pub obs_dates: Vec<f64>,

    // mutable path state
    exercised:       bool,
    exercise_payoff: f64,
}

impl AmericanOption {
    pub fn new(strike: f64, is_call: bool, notional: f64, obs_dates: Vec<f64>) -> Self {
        Self {
            strike,
            is_call,
            notional,
            obs_dates,
            exercised:       false,
            exercise_payoff: 0.0,
        }
    }

    fn intrinsic(&self, spot: f64) -> f64 {
        if self.is_call {
            (spot - self.strike).max(0.0)
        } else {
            (self.strike - spot).max(0.0)
        }
    }
}

impl Product for AmericanOption {
    fn observation_dates(&self) -> &[f64] {
        &self.obs_dates
    }

    /// Simple early exercise: exercise whenever intrinsic > 0.
    fn notify(&mut self, _t: f64, spot: f64) -> bool {
        let intrinsic = self.intrinsic(spot);
        if intrinsic > 0.0 && !self.exercised {
            self.exercised       = true;
            self.exercise_payoff = self.notional * intrinsic;
            return true;
        }
        false
    }

    fn terminal_payoff(&self, spot_at_maturity: f64) -> f64 {
        if self.exercised {
            self.exercise_payoff
        } else {
            self.notional * self.intrinsic(spot_at_maturity)
        }
    }

    fn reset(&mut self) {
        self.exercised       = false;
        self.exercise_payoff = 0.0;
    }

    fn knock_in_triggered(&self) -> bool {
        false
    }

    fn set_knock_in(&mut self) {
        // American option has no knock-in concept; ignore
    }
}
