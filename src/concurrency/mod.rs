pub mod naive_spawn;
pub mod spawn_blocking_joinset;
pub mod rayon_bridge;
pub mod semaphore_bounded;
pub mod channel_pipeline;
pub mod stream_buffered;
pub mod stream_throttled;
pub mod std_thread;

use std::sync::Arc;
use std::time::Instant;

use crate::domain::{Product, Propagator};
use crate::engine::{BatchConfig, MonteCarloEngine, PartialResult};
use crate::analytics::{BatchCollector, PriceResult, ProfiledResult};
use crate::simulation::BoxMullerRng;

/// Available concurrency strategies for parallel Monte Carlo simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyStrategy {
    /// S1: tokio::spawn (anti-pattern — CPU work on async executor)
    NaiveSpawn,
    /// S2: spawn_blocking + JoinSet (correct, structured task lifecycle)
    SpawnBlockingJoinSet,
    /// S3: rayon work-stealing + oneshot + tokio_stream aggregation
    RayonBridge,
    /// S4: Arc<Semaphore> + spawn_blocking (bounded concurrency, back-pressure)
    SemaphoreBounded,
    /// S5: mpsc worker pool + ReceiverStream (streaming / SSE-ready)
    ChannelPipeline,
    /// S6: tokio_stream::iter + buffer_unordered (pure stream concurrency)
    StreamBuffered,
    /// S7: throttle + buffer_unordered (cloud rate-limiting scenario)
    StreamThrottled,
    /// S8: std::thread::scope (pure OS threads, no async runtime)
    StdThread,
}

impl ConcurrencyStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NaiveSpawn            => "S1  naive_spawn",
            Self::SpawnBlockingJoinSet  => "S2  spawn_blocking_joinset",
            Self::RayonBridge           => "S3  rayon_bridge",
            Self::SemaphoreBounded      => "S4  semaphore_bounded(8)",
            Self::ChannelPipeline       => "S5  channel_pipeline(8)",
            Self::StreamBuffered        => "S6  stream_buffered(8)",
            Self::StreamThrottled       => "S7  stream_throttled(8,10ms)",
            Self::StdThread            => "S8  std_thread(8)",
        }
    }
}

/// Split `n_paths` into `n_batches` roughly equal batches.
pub fn make_batch_configs(n_paths: usize, n_batches: usize, global_seed: u64) -> Vec<BatchConfig> {
    let base = n_paths / n_batches;
    let rem  = n_paths % n_batches;
    (0..n_batches)
        .map(|i| {
            let paths = if i < rem { base + 1 } else { base };
            let seed  = BoxMullerRng::batch_seed(global_seed, i as u64);
            BatchConfig::new(i, paths, seed)
        })
        .collect()
}

/// Run simulation with the specified strategy.
///
/// Per-batch timing and thread identity are captured via `tracing::info_span!` inside
/// each strategy closure. If a [`crate::analytics::BatchCollectorLayer`] has been
/// registered as the global tracing subscriber (as the profiler binary does), the
/// events are collected and returned in `ProfiledResult::events`. Otherwise (e.g. in
/// the benchmark binary) the drain returns an empty `Vec` at zero cost.
pub async fn run_simulation<P, Pr>(
    strategy:    ConcurrencyStrategy,
    engine:      Arc<MonteCarloEngine<P, Pr>>,
    n_paths:     usize,
    n_threads:   usize,
    n_batches:   usize,
    global_seed: u64,
) -> ProfiledResult
where
    P:  Product,
    Pr: Propagator,
{
    let batch_configs = make_batch_configs(n_paths, n_batches, global_seed);
    let t0 = Instant::now();

    let partial: PartialResult = match strategy {
        ConcurrencyStrategy::NaiveSpawn =>
            naive_spawn::run(Arc::clone(&engine), batch_configs).await,
        ConcurrencyStrategy::SpawnBlockingJoinSet =>
            spawn_blocking_joinset::run(Arc::clone(&engine), batch_configs).await,
        ConcurrencyStrategy::RayonBridge =>
            rayon_bridge::run(Arc::clone(&engine), batch_configs).await,
        ConcurrencyStrategy::SemaphoreBounded =>
            semaphore_bounded::run(Arc::clone(&engine), batch_configs, n_threads).await,
        ConcurrencyStrategy::ChannelPipeline =>
            channel_pipeline::run(Arc::clone(&engine), batch_configs, n_threads).await,
        ConcurrencyStrategy::StreamBuffered =>
            stream_buffered::run(Arc::clone(&engine), batch_configs, n_threads).await,
        ConcurrencyStrategy::StreamThrottled =>
            stream_throttled::run(Arc::clone(&engine), batch_configs, n_threads).await,
        ConcurrencyStrategy::StdThread =>
            std_thread::run(Arc::clone(&engine), batch_configs, n_threads).await,
    };

    let elapsed = t0.elapsed();
    let events  = BatchCollector::global().drain(t0);

    ProfiledResult {
        price_result: PriceResult::from_partial(strategy.name(), &partial, elapsed),
        events,
    }
}
