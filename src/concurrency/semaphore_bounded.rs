/// S4 — semaphore_bounded: cloud vCPU budget control.
///
/// Uses `Arc<Semaphore>` to enforce a hard limit on the number of concurrently
/// running blocking tasks. Relevant in cloud environments where the pricing
/// service shares CPU with other workloads and must respect a vCPU quota.
///
/// Contrast with S6 (`buffer_unordered`): the Semaphore is the single,
/// explicit source of back-pressure. Adding `buffer_unordered` on top would
/// create two independent concurrency controllers — confusing semantics
/// with no benefit.
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};

pub async fn run<P, Pr>(
    engine:         Arc<MonteCarloEngine<P, Pr>>,
    configs:        Vec<BatchConfig>,
    max_concurrent: usize,
) -> PartialResult
where
    P:  Product,
    Pr: Propagator,
{
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let mut handles = Vec::with_capacity(configs.len());

    for cfg in configs {
        let eng    = Arc::clone(&engine);
        let permit = Arc::clone(&sem).acquire_owned().await
            .expect("semaphore closed");

        handles.push(tokio::task::spawn_blocking(move || {
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
            drop(permit); // release slot on completion
            result
        }));
    }

    let mut total = PartialResult::default();
    for handle in handles {
        total = total.merge(handle.await.expect("task panicked"));
    }
    total
}
