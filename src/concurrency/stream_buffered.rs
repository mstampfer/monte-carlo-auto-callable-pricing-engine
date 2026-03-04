/// S6 — stream_buffered: pure stream-first bounded concurrency.
///
/// Uses `tokio_stream::iter` + `futures::StreamExt::buffer_unordered(n)` as
/// the sole concurrency controller — no Semaphore, no JoinSet.
///
/// `buffer_unordered(n)` internally drives a `FuturesUnordered` pool capped
/// at `n` concurrent futures, providing back-pressure without additional state.
///
/// Expected performance: closely matches S2 (`spawn_blocking` + JoinSet),
/// demonstrating that the stream combinator is a zero-overhead abstraction
/// over the manual JoinSet pattern.
use std::sync::Arc;
use futures::StreamExt as _;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};

pub async fn run<P, Pr>(
    engine:    Arc<MonteCarloEngine<P, Pr>>,
    configs:   Vec<BatchConfig>,
    n_threads: usize,
) -> PartialResult
where
    P:  Product,
    Pr: Propagator,
{
    tokio_stream::iter(configs)
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
