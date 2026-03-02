/// A dual-frequency time grid combining:
///
/// - **Coarse grid**: monthly knock-out / autocall observation dates
/// - **Fine grid**: daily knock-in monitoring dates (~21 business days per month)
///
/// The engine iterates over `steps`, calling into the product at coarse dates
/// and checking the knock-in barrier at every fine step.
#[derive(Debug, Clone)]
pub struct DualTimeGrid {
    /// All simulation time points (union of coarse and fine), sorted ascending.
    /// Each entry: (time_in_years, step_type)
    pub steps: Vec<TimeStep>,

    /// Maturity in years.
    pub maturity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepKind {
    /// A daily fine step — check knock-in barrier.
    Daily,
    /// A monthly coarse step — check autocall barrier AND knock-in barrier.
    Monthly,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeStep {
    pub t:    f64,       // time in years from today
    pub dt:   f64,       // length of this step
    pub kind: StepKind,
}

impl DualTimeGrid {
    /// Build a dual grid for a given maturity with monthly coarse dates and
    /// daily fine dates (approximately `business_days_per_month` per interval).
    ///
    /// # Arguments
    /// * `maturity_years`         — product maturity (e.g. 1.0 = 1 year)
    /// * `n_monthly`              — number of monthly observation dates
    /// * `business_days_per_month`— number of daily sub-steps per monthly interval
    pub fn new(maturity_years: f64, n_monthly: usize, business_days_per_month: usize) -> Self {
        let mut times: Vec<(f64, StepKind)> = Vec::new();

        let monthly_dt = maturity_years / n_monthly as f64;
        let daily_dt   = monthly_dt / business_days_per_month as f64;

        for m in 0..n_monthly {
            let t_month_start = m as f64 * monthly_dt;
            // Insert daily sub-steps inside this monthly interval
            // (all except the last sub-step, which is the monthly boundary itself)
            for d in 1..business_days_per_month {
                let t = t_month_start + d as f64 * daily_dt;
                times.push((t, StepKind::Daily));
            }
            // Monthly boundary
            let t_monthly = (m + 1) as f64 * monthly_dt;
            times.push((t_monthly, StepKind::Monthly));
        }

        // Sort by time (should already be sorted, but be safe)
        times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Build steps with dt
        let mut steps = Vec::with_capacity(times.len());
        let mut t_prev = 0.0_f64;
        for (t, kind) in &times {
            steps.push(TimeStep {
                t:    *t,
                dt:   t - t_prev,
                kind: *kind,
            });
            t_prev = *t;
        }

        Self {
            steps,
            maturity: maturity_years,
        }
    }

    /// Total number of time steps in the simulation.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }
}
