/// S5 — channel_pipeline: mpsc worker pool with ReceiverStream aggregation.
///
/// Architecture:
/// - A producer task sends batch configs into a bounded mpsc channel.
/// - A pool of `spawn_blocking` workers drain the batch channel, execute
///   `run_batch`, and send results into a result channel.
/// - `ReceiverStream::new(result_rx)` bridges the mpsc receiver to a Stream,
///   enabling idiomatic `.fold()` aggregation without knowing worker count
///   or completion order.
///
/// SSE-ready: the ReceiverStream can be forwarded to an SSE endpoint
/// (`axum::response::Sse`) so partial prices stream to the client.
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};

const QUEUE_DEPTH: usize = 32;

pub async fn run<P, Pr>(
    engine:    Arc<MonteCarloEngine<P, Pr>>,
    configs:   Vec<BatchConfig>,
    n_workers: usize,
) -> PartialResult
where
    P:  Product,
    Pr: Propagator,
{
    let (batch_tx,  batch_rx)  = mpsc::channel::<BatchConfig>(QUEUE_DEPTH);
    let (result_tx, result_rx) = mpsc::channel::<PartialResult>(QUEUE_DEPTH);

    // Producer: send all batch configs into the batch channel.
    tokio::spawn(async move {
        for cfg in configs {
            if batch_tx.send(cfg).await.is_err() {
                break; // receivers dropped
            }
        }
        // batch_tx dropped here → channel closes → workers see None → stop
    });

    // Shared receiver for the batch channel (multiple workers drain it).
    let batch_rx = Arc::new(Mutex::new(batch_rx));

    // Worker pool: each worker runs in a blocking thread.
    for _ in 0..n_workers {
        let rx  = Arc::clone(&batch_rx);
        let tx  = result_tx.clone();
        let eng = Arc::clone(&engine);

        tokio::task::spawn_blocking(move || {
            loop {
                // blocking_lock + blocking_recv to bridge sync ↔ async boundary
                let cfg = {
                    let mut guard = rx.blocking_lock();
                    guard.blocking_recv()
                };
                match cfg {
                    Some(cfg) => {
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
                        if tx.blocking_send(result).is_err() {
                            break;
                        }
                    }
                    None => break, // batch channel exhausted
                }
            }
        });
    }

    // Drop the original result_tx — all worker clones will drop when they finish,
    // causing the ReceiverStream to terminate naturally.
    drop(result_tx);

    // Aggregate via ReceiverStream — idiomatic channel → Stream bridge.
    ReceiverStream::new(result_rx)
        .fold(PartialResult::default(), |acc, r| acc.merge(r))
        .await
}
