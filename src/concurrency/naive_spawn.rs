/// S1 — naive_spawn: anti-pattern demonstration.
///
/// Uses `tokio::spawn` for CPU-bound work — this blocks async executor threads,
/// starves I/O tasks, and underperforms compared to proper `spawn_blocking`.
///
/// Included explicitly as an anti-pattern so the benchmark table makes the
/// performance cost visible and quantifiable.
use std::sync::Arc;
use futures::StreamExt;
use futures::stream::FuturesUnordered;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};

pub async fn run<P, Pr>(
    engine:  Arc<MonteCarloEngine<P, Pr>>,
    configs: Vec<BatchConfig>,
) -> PartialResult
where
    P:  Product,
    Pr: Propagator,
{
    let mut set = FuturesUnordered::new();

    for cfg in configs {
        let eng = Arc::clone(&engine);
        // ANTI-PATTERN: CPU work inside tokio::spawn blocks the async executor.
        set.push(tokio::spawn(async move {
            eng.run_batch(&cfg)
        }));
    }

    let mut total = PartialResult::default();
    while let Some(result) = set.next().await {
        total = total.merge(result.expect("task panicked"));
    }
    total
}
