/// S3 — rayon_bridge: fastest strategy — rayon compute, tokio_stream aggregation.
///
/// Architecture:
/// - `rayon::spawn` dispatches all batches to rayon's work-stealing thread pool
///   (optimal for CPU-bound work: avoids context-switch overhead of tokio tasks)
/// - Each batch sends its result via a `oneshot` channel back to the async world
/// - `tokio_stream::iter` + `.then()` + `.fold()` aggregate results as they arrive
///
/// Why `then` (not `buffer_unordered`):
/// Rayon is already running all batches in parallel. The tokio side only needs
/// to *await* the oneshot completions — there is no point adding another
/// concurrency layer on the tokio executor.
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_stream::StreamExt as _;

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
    // Spawn all rayon tasks up front; collect oneshot receivers.
    let receivers: Vec<oneshot::Receiver<PartialResult>> = configs
        .into_iter()
        .map(|cfg| {
            let (tx, rx) = oneshot::channel();
            let eng = Arc::clone(&engine);
            rayon::spawn(move || {
                let result = eng.run_batch(&cfg);
                let _ = tx.send(result);
            });
            rx
        })
        .collect();

    // Aggregate via tokio_stream: iter → then (await each oneshot) → fold
    tokio_stream::iter(receivers)
        .then(|rx| async move {
            rx.await.expect("rayon sender dropped — task panicked?")
        })
        .fold(PartialResult::default(), |acc, r| acc.merge(r))
        .await
}
