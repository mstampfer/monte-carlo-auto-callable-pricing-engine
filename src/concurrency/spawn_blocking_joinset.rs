/// S2 — spawn_blocking_joinset: correct CPU-work concurrency.
///
/// Uses `tokio::task::spawn_blocking` to move CPU work off the async executor
/// into tokio's dedicated blocking thread pool.
///
/// `JoinSet` provides structured task lifecycle: all tasks are tracked and
/// can be cancelled via `abort_all()` if needed (unlike S1's raw futures).
///
/// `tokio_stream` is NOT used here because `JoinSet` already provides clean
/// task lifecycle management that a stream combinator cannot match.
use std::sync::Arc;
use tokio::task::JoinSet;

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
    let mut set = JoinSet::new();

    for cfg in configs {
        let eng = Arc::clone(&engine);
        set.spawn_blocking(move || {
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
        });
    }

    let mut total = PartialResult::default();
    while let Some(result) = set.join_next().await {
        total = total.merge(result.expect("task panicked"));
    }
    total
}
