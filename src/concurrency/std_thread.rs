/// S8 — std_thread: pure std::thread::scope parallelism.
///
/// Spawns OS threads directly via `std::thread::scope`, bypassing both the
/// tokio async runtime and rayon's work-stealing pool. Scoped threads
/// guarantee all spawned threads join before the scope exits.
///
/// Batches are chunked into groups of `n_threads` to bound concurrency,
/// matching the semaphore/buffer_unordered pattern used in S4/S6.

use std::sync::{Arc, Mutex};

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
    let results = Mutex::new(Vec::with_capacity(configs.len()));

    for chunk in configs.chunks(n_threads) {
        std::thread::scope(|s| {
            for cfg in chunk {
                let eng = Arc::clone(&engine);
                let results = &results;
                let cfg = cfg.clone();
                s.spawn(move || {
                    let span = tracing::info_span!("batch",
                        batch_id = cfg.batch_id as u64,
                        n_paths  = cfg.n_paths  as u64,
                        price    = tracing::field::Empty,
                        std_err  = tracing::field::Empty,
                    );
                    let _guard = span.enter();
                    let result = eng.run_batch(&cfg);
                    span.record("price",   result.price());
                    span.record("std_err", result.std_err());
                    results.lock().unwrap().push(result);
                });
            }
        });
    }

    results.into_inner().unwrap()
        .into_iter()
        .fold(PartialResult::default(), |acc, r| acc.merge(r))
}
