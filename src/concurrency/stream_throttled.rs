/// S7 — stream_throttled: rate-limited submission for cloud billing scenarios.
///
/// Adds `StreamExt::throttle` before `buffer_unordered` to limit how fast
/// new batch configs are submitted to the compute pool.
///
/// `throttle` is unique to `tokio_stream::StreamExt` — there is no equivalent
/// in `futures::StreamExt`. It is called via UFCS here to avoid trait method
/// conflicts, while `buffer_unordered` (from `futures::StreamExt`) handles
/// bounded concurrency over the throttled stream.
///
/// Note: S7 is intentionally slower than S3–S6. The artificial delay
/// demonstrates the mechanism, not optimal throughput.
use std::sync::Arc;
use std::time::Duration;
use futures::StreamExt as _;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};

/// Rate limit: at most 1 batch submitted per RATE_LIMIT_MS milliseconds.
const RATE_LIMIT_MS: u64 = 10;

pub async fn run<P, Pr>(
    engine:    Arc<MonteCarloEngine<P, Pr>>,
    configs:   Vec<BatchConfig>,
    n_threads: usize,
) -> PartialResult
where
    P:  Product,
    Pr: Propagator,
{
    // throttle is unique to tokio_stream::StreamExt — call via UFCS
    let throttled = tokio_stream::StreamExt::throttle(
        tokio_stream::iter(configs),
        Duration::from_millis(RATE_LIMIT_MS),
    );

    // Use futures::StreamExt for buffer_unordered + fold
    throttled
        .map(|cfg| {
            let eng = Arc::clone(&engine);
            tokio::task::spawn_blocking(move || {
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
            })
        })
        .buffer_unordered(n_threads)
        .map(|r: Result<PartialResult, tokio::task::JoinError>| {
            r.expect("batch panicked")
        })
        .fold(PartialResult::default(), |acc, r| async move { acc.merge(r) })
        .await
}
