pub mod rayon_bridge_baseline;

use std::sync::Arc;
use std::time::Instant;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};
use crate::analytics::{BatchCollector, PriceResult, ProfiledResult};
use crate::simulation::BoxMullerRng;

/// Concurrency strategy variants for the hybrid (tokio controller + rayon
/// workers) architecture. Add new variants here as you experiment with
/// optimisations; `RayonBridgeBaseline` is the seed and the reference for
/// price-equivalence checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyStrategy {
    /// Baseline: rayon::spawn dispatches batches to rayon's work-stealing
    /// pool, oneshot channels feed tokio_stream aggregation.
    RayonBridgeBaseline,
}

impl ConcurrencyStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::RayonBridgeBaseline => "rayon_bridge_baseline",
        }
    }
}

/// Split `n_paths` into `n_batches` roughly equal batches.
pub fn make_batch_configs(n_paths: usize, n_batches: usize, global_seed: u64) -> Vec<BatchConfig> {
    let base = n_paths / n_batches;
    let rem  = n_paths % n_batches;
    (0..n_batches)
        .map(|i| {
            let paths = if i < rem { base + 1 } else { base };
            let seed  = BoxMullerRng::batch_seed(global_seed, i as u64);
            BatchConfig::new(i, paths, seed)
        })
        .collect()
}

/// Run simulation with the specified strategy.
///
/// Per-batch timing and thread identity are captured via `tracing::info_span!` inside
/// each strategy closure. If a [`crate::analytics::BatchCollectorLayer`] has been
/// registered as the global tracing subscriber (as the profiler binary does), the
/// events are collected and returned in `ProfiledResult::events`. Otherwise (e.g. in
/// the benchmark binary) the drain returns an empty `Vec` at zero cost.
pub async fn run_simulation<P, Pr>(
    strategy:    ConcurrencyStrategy,
    engine:      Arc<MonteCarloEngine<P, Pr>>,
    n_paths:     usize,
    _n_threads:  usize,
    n_batches:   usize,
    global_seed: u64,
) -> ProfiledResult
where
    P:  Product,
    Pr: Propagator,
{
    let batch_configs = make_batch_configs(n_paths, n_batches, global_seed);
    let t0 = Instant::now();

    let partial: PartialResult = match strategy {
        ConcurrencyStrategy::RayonBridgeBaseline =>
            rayon_bridge_baseline::run(Arc::clone(&engine), batch_configs).await,
    };

    let elapsed = t0.elapsed();
    let events  = BatchCollector::global().drain(t0);

    ProfiledResult {
        price_result: PriceResult::from_partial(strategy.name(), &partial, elapsed),
        events,
    }
}
