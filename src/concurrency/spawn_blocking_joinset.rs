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
        set.spawn_blocking(move || eng.run_batch(&cfg));
    }

    let mut total = PartialResult::default();
    while let Some(result) = set.join_next().await {
        total = total.merge(result.expect("task panicked"));
    }
    total
}
