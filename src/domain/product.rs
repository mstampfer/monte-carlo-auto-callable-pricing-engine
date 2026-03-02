/// Core trait representing a financial product that can be simulated.
///
/// The Product owns its state and is mutated at each observation date.
/// The engine merges product observation dates into the full propagation grid
/// and calls `notify` at each such date.
pub trait Product: Send + Sync + Clone + 'static {
    /// The set of dates on which this product requires a spot reading.
    /// Expressed as times in years from today (t=0).
    fn observation_dates(&self) -> &[f64];

    /// Called by the engine each time simulation reaches an observation date.
    ///
    /// `t`    — current simulation time (years from today)
    /// `spot` — current spot value
    ///
    /// Updates internal state (e.g., records knock-in event, stores coupon trigger).
    /// Returns `true` if the path should terminate early (e.g., autocall triggered).
    fn notify(&mut self, t: f64, spot: f64) -> bool;

    /// Called once at the end of a surviving path (after all dates).
    /// Returns the path's unweighted cash payoff (in currency units).
    fn terminal_payoff(&self, spot_at_maturity: f64) -> f64;

    /// Reset product state — called before each simulated path.
    fn reset(&mut self);

    /// Whether a knock-in event has been registered on this path.
    fn knock_in_triggered(&self) -> bool;

    /// Register a knock-in event (called by engine during daily monitoring).
    fn set_knock_in(&mut self);

    /// **OSS autocall contribution** (optional).
    ///
    /// For products with an autocall (knock-out) barrier, this returns the
    /// cash payoff the product would deliver IF the autocall barrier were
    /// crossed at the given monthly step index.
    ///
    /// The engine uses this to add the OSS autocall component:
    ///   contribution_k = W_{k-1} · (1 − p_k) · autocall_payoff_k · disc(t_k)
    ///
    /// Return `None` (default) if the product has no autocall mechanism.
    fn oss_autocall_payoff(&self, step_idx: usize) -> Option<f64> {
        let _ = step_idx;
        None
    }
}
