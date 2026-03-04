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
            let span = tracing::info_span!("batch",
                batch_id = cfg.batch_id as u64,
                n_paths  = cfg.n_paths  as u64,
                price    = tracing::field::Empty,
                std_err  = tracing::field::Empty,
            );
            let _guard = span.enter();
            let result  = eng.run_batch(&cfg);
            span.record("price",   result.price());
            span.record("std_err", result.std_err());
            result
        }));
    }

    let mut total = PartialResult::default();
    while let Some(result) = set.next().await {
        total = total.merge(result.expect("task panicked"));
    }
    total
}
