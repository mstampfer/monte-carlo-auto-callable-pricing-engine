# Monte Carlo Auto-Callable Pricing Engine

A high-performance structured-product pricing engine written in Rust, demonstrating how to build a correct-from-the-start Monte Carlo framework around three core abstractions — **Product**, **Propagator**, and **Engine** — with emphasis on memory layout, CPU efficiency, and cloud-friendly parallelism. Includes a comparative study of **8 concurrency strategies** with a profiler TUI, an AWS hybrid architecture design, and use-case-specific strategy recommendations.

The instrument priced is an **autocallable note with daily knock-in monitoring and monthly knock-out (autocall) observations**, valued with the Glasserman-Staum one-step survival technique.

---

## Quick Start

```bash
# Build optimised binary (fat LTO, codegen-units=1)
cargo build --release

# Run the benchmark harness (all 8 concurrency strategies)
cargo run --release

# Run with a specific strategy or path count
cargo run --release -- --npaths 2_000_000 s3

# Launch the post-run profiler TUI
cargo run --release --bin profiler

# Profiler with 64 batches across selected strategies
cargo run --release --bin profiler -- --nbatches 64 s1 s3 s6 s7
```

Sample benchmark output:

```
╔══════════════════════════════════════════════════════════════════════════╗
║    HSBC Monte Carlo Auto-Callable Pricing Engine — Benchmark Harness     ║
╚══════════════════════════════════════════════════════════════════════════╝

  Instrument : Autocallable note, maturity = 1Y
  S_0        : 100, σ = 25%, r = 5%, q = 2%
  Barriers   : Call = 100% of S_0, KI = 70% of S_0
  Grid       : 12 monthly x 21 daily sub-steps
  Paths      : 200000, Threads = 8
  Method     : One-Step Survival (Glasserman-Staum) + Brownian Bridge

╔════════════════════════════════╦══════════╦═══════════╦══════════╦══════════════════╦═════════╗
║ Strategy                       ║    Paths ║ Time (ms) ║    Price ║      95% CI      ║ Speedup ║
╠════════════════════════════════╬══════════╬═══════════╬══════════╬══════════════════╬═════════╣
║ S1  naive_spawn                ║     200K ║       127 ║   96.734 ║ [ 96.71, 96.76] ║    1.0× ║
║ S2  spawn_blocking_joinset     ║     200K ║       124 ║   96.734 ║ [ 96.71, 96.76] ║    1.2× ║
║ S3  rayon_bridge               ║     200K ║       123 ║   96.734 ║ [ 96.71, 96.76] ║    1.0× ║
║ S4  semaphore_bounded(8)       ║     200K ║       124 ║   96.734 ║ [ 96.71, 96.76] ║    1.0× ║
║ S5  channel_pipeline(8)        ║     200K ║       126 ║   96.734 ║ [ 96.71, 96.76] ║    1.0× ║
║ S6  stream_buffered(8)         ║     200K ║       126 ║   96.734 ║ [ 96.71, 96.76] ║    1.0× ║
║ S7  stream_throttled(8,10ms)   ║     200K ║       183 ║   96.734 ║ [ 96.71, 96.76] ║    0.7× ║
╚════════════════════════════════╩══════════╩═══════════╩══════════╩══════════════════╩═════════╝

── AmericanOption stub (Bermudan approximation, same engine) ────────────
  AmericanOption (Bermudan approx): price = 3.226,  95% CI = [3.173, 3.280]
```

All eight strategies converge to the same price within each other's 95% confidence intervals, validating the OSS estimator. S7 is intentionally slower — the 10 ms/batch throttle adds overhead proportional to the batch count to demonstrate the rate-limiting mechanism. S8 uses pure `std::thread::scope` with no async runtime dependency.

---

## Profiler TUI

The `profiler` binary runs the same simulation and renders a post-run ratatui TUI that reveals *how* each strategy used its threads. Instrumentation uses `tracing::info_span!` inside each batch closure; a custom `BatchCollectorLayer` subscriber captures timing and thread identity with negligible overhead (two `Instant::now()` calls per batch ≈ 0.0002% perturbation).

```bash
# Default: all 8 strategies, 200K paths, 32 batches (4× threads — exposes work-stealing)
cargo run --release --bin profiler

# More paths for sharper timelines
cargo run --release --bin profiler -- --npaths 2_000_000

# More batches to make strategy differences visible
cargo run --release --bin profiler -- --nbatches 64 s1 s3 s6 s7
```

### Tab 1 — Thread Timelines

Gantt chart for each strategy. Each row is one OS thread; each coloured block is one batch (colour = `batch_id`). Grey `░` = idle.

```
▶ S3  rayon_bridge   111 ms
  T0 [████████░░░████████████░░░░░████████████████░░░]
  T1 [░░████████████████░░░░████████████████████████░]
  ...
  CPU eff: 95%  Imbalance: 1.95  Batches: 64  Price: 96.740
```

Keys: `↑`/`↓` scroll, `Tab`/`1`/`2`/`3` switch tabs, `q`/`Esc` quit.

### Tab 2 — Batch Analysis

For a selected strategy (chosen with `↑`/`↓`):

- **Duration histogram** — sparkline of batch compute-time distribution
- **Batch-to-thread matrix** — which thread ran which batch (`●`)
- **Completion order** — batch IDs sorted by finish time; out-of-order = work-stealing or async scheduling visible

### Tab 3 — Convergence & Comparison

- **Price convergence sparkline** — running price after each batch completion
- **Strategy comparison table** — wall time, price, CPU efficiency %, load imbalance index, speedup vs S1

### Key metrics

| Metric | Formula |
|---|---|
| CPU efficiency | Σ(batch durations) / (wall time × unique threads) |
| Load imbalance | max(batch duration) / mean(batch duration) |
| Throughput | total paths / wall time |

### What `--nbatches` reveals

With the default `n_batches = n_threads = 8`, all strategies look identical: one batch per thread, all running simultaneously. Setting `--nbatches 32` or `--nbatches 64` makes structural differences visible:

| Strategy | What appears |
|---|---|
| S3 rayon | Work-stealing: multiple colour segments per thread row, out-of-order completion |
| S1 naive_spawn | Uses all CPU cores (tokio defaults to `num_cpus`), higher scheduling overhead than S3 |
| S6 buffered | Hard concurrency cap at 8 — leaves spare cores idle on a 10-core machine |
| S7 throttled | Wave pattern: thin coloured squares separated by large idle gaps; CPU eff ≈ 10% |

---

## Mathematical Model

### Underlying process

Black-Scholes single underlying, risk-neutral measure:

```
dS = (r − q) S dt + σ S dW
```

Discretised on the log-price grid (exact):

```
S_{n+1} = S_n · exp( (r − q − σ²/2)·Δt + σ·√Δt · Z )
Z ~ N(0,1)  drawn via Box-Muller from xoshiro256++ uniforms
```

### Autocallable payoff

1. **At each monthly date** t_k: if S(t_k) ≥ B_c → pay `notional · (1 + coupon_k)`, terminate.
2. **At maturity T** (if never called):
   - No knock-in ever occurred → pay `notional` (capital protected)
   - Knock-in AND S(T) ≥ B_c → pay `notional · (1 + coupon_N)`
   - Knock-in AND S(T) < B_c → pay `notional · S(T)/S(0)` (full downside participation)

### Dual-frequency time grid

| Grid level | Purpose | Typical spacing |
|---|---|---|
| Coarse (monthly) | Autocall observation, OSS step | ~1/12 year |
| Fine (daily) | Knock-in barrier monitoring | ~1/252 year |

The engine builds the coarse grid first, then inserts `business_days_per_month − 1` daily sub-steps per monthly interval using a Brownian bridge to reconstruct the intra-period path.

### One-Step Survival (Glasserman-Staum)

Standard Monte Carlo for autocallable notes suffers two problems: (1) paths terminate on autocall events, creating indicator-function discontinuities that make finite-difference Greeks noisy; (2) the surviving paths represent only the no-call scenario.

OSS resolves both. At each monthly boundary t_k, instead of possibly terminating:

```
d_k   = ( ln(B_c / S_prev) − drift·Δt ) / ( σ·√Δt )
p_k   = Φ(d_k)                    // probability of NOT autocalling
w    *= p_k                        // accumulate path weight
Z_k   = Φ⁻¹( U_k · Φ(d_k) )      // truncated-normal draw: Z | Z < d_k
S_ko  = S_prev · exp( drift·Δt + σ·√Δt · Z_k )   // always < B_c
```

The full OSS estimator includes both components in a single unbiased expression:

```
path_total = Σ_k [ W_{k-1} · (1 − p_k) · coupon_payoff_k · disc(t_k) ]   (autocall)
           +      W_T · maturity_payoff · disc(T)                          (no-call)

V̂ = (1/M) Σ_m  path_total_m
```

Because the payoff is now a smooth function of S_0 (no barrier-crossing indicator), delta via central finite difference is stable even for bump sizes as small as 0.1% of S_0. Common random numbers (same seed for up/down bumps) reduce variance further.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  DOMAIN  (pure value types, no allocation in hot path)           │
│                                                                  │
│   Product (trait)        Propagator (trait)    DualTimeGrid      │
│   └─ AutoCallable        └─ BlackScholes        MarketData       │
│   └─ AmericanOption stub                                         │
└──────────────────────────────────────────────────────────────────┘
                  │                   │
                  ▼                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  ENGINE                                                          │
│   MonteCarloEngine<P: Product, Pr: Propagator>                   │
│   └─ run_batch(&BatchConfig) -> PartialResult  (synchronous)     │
│       ├─ OSS step at each monthly boundary                       │
│       ├─ Brownian bridge for daily sub-steps                     │
│       └─ Knock-in monitoring on every daily spot                 │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  CONCURRENCY LAYER  (8 strategies, same engine unit of work)     │
│   S1 naive_spawn · S2 JoinSet · S3 rayon_bridge                  │
│   S4 semaphore · S5 channel_pipeline · S6 stream_buffered        │
│   S7 stream_throttled · S8 std_thread                            │
│                                                                  │
│   run_simulation() → ProfiledResult                              │
│     tracing::info_span!("batch", …) in each closure             │
│     BatchCollector::global().drain(t0) → Vec<BatchEvent>        │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  PROFILER BINARY  (src/bin/profiler.rs)                          │
│   BatchCollectorLayer (tracing subscriber) → Vec<ProfiledResult> │
│   ratatui TUI: Thread Timelines · Batch Analysis · Convergence  │
└──────────────────────────────────────────────────────────────────┘
```

### `Product` trait

The key extensibility point. Owns mutable path state; the engine resets it between paths.

```rust
pub trait Product: Send + Sync + Clone + 'static {
    fn observation_dates(&self) -> &[f64];
    fn notify(&mut self, t: f64, spot: f64) -> bool;   // true = early termination
    fn terminal_payoff(&self, spot_at_maturity: f64) -> f64;
    fn reset(&mut self);
    fn set_knock_in(&mut self);
    fn knock_in_triggered(&self) -> bool;

    // OSS autocall contribution at step k (None = no autocall mechanism)
    fn oss_autocall_payoff(&self, step_idx: usize) -> Option<f64> { None }
}
```

### `Propagator` trait

Stateless spot evolution. The entire Black-Scholes model is two fields and one multiplication.

```rust
pub trait Propagator: Send + Sync + 'static {
    fn propagate(&self, spot: f64, dt: f64, z: f64) -> f64;
}
```

### `MonteCarloEngine<P, Pr>`

`run_batch` is **synchronous**. It takes `n_paths` and an RNG seed, returns an aggregated `PartialResult`. All async coordination happens in the concurrency layer above — the engine itself is runtime-agnostic.

---

## Performance Design

### Zero allocation in the hot loop

`BatchBuffers` is allocated once per batch and reused for every path within the batch:

```rust
struct BatchBuffers {
    z_fine:      Vec<f64>,   // normal draws for daily sub-steps
    daily_spots: Vec<f64>,   // Brownian bridge scratch space
}
```

### Per-batch seeded RNG — no lock contention

Each batch receives a unique seed derived from `(batch_id, global_seed)` via SplitMix64. `BoxMullerRng` holds only 256 bits of xoshiro256++ state. No mutex, no atomic. Batches are fully independent.

### Clone-once Product template

`product_template` is cloned once per **batch** (not per path). Inside the batch loop, `product.reset()` performs a cheap field-zeroing instead of a heap allocation.

### Fat LTO + single codegen unit

```toml
[profile.release]
opt-level = 3
lto       = "fat"
codegen-units = 1
```

Enables cross-crate inlining of the `Propagator::propagate` hot path, which is a single `fma`-friendly expression.

---

## Concurrency Strategies

All eight strategies use `MonteCarloEngine::run_batch` as the unit of work. Paths are split into `n_batches` independent batches (default: equal to `n_threads`; increase via `--nbatches` to expose work-stealing behaviour in the profiler).

| # | Module | tokio_stream role | CPU model | Key characteristic |
|---|---|---|---|---|
| S1 | `naive_spawn` | none — raw `FuturesUnordered` | `tokio::spawn` (wrong) | Anti-pattern baseline |
| S2 | `spawn_blocking_joinset` | none — `JoinSet` owns lifecycle | blocking thread pool | Structured task cancellation |
| S3 | `rayon_bridge` | `iter` + `then` + `fold` | rayon work-stealing | Lowest CPU overhead |
| S4 | `semaphore_bounded` | none — Semaphore is the sole controller | blocking thread pool | Hard vCPU budget |
| S5 | `channel_pipeline` | `ReceiverStream` + `fold` | mpsc worker pool | SSE / streaming ready |
| S6 | `stream_buffered` | `buffer_unordered(n)` sole control | blocking thread pool | Pure stream concurrency |
| S7 | `stream_throttled` | `throttle` + `buffer_unordered` | blocking thread pool | Cloud rate/billing quota |
| S8 | `std_thread` | none — pure stdlib | `std::thread::scope` | Zero-dependency baseline |

### S1 — Anti-pattern

`tokio::spawn` puts CPU work on the async executor's thread pool, stealing threads from I/O tasks. Included to make the cost measurable against the correct alternatives.

### S2 — spawn_blocking + JoinSet

The correct foundation. `JoinSet` provides per-task `abort_all()` — useful for risk limit checks that need to cancel in-flight batches.

### S3 — rayon_bridge

Rayon's work-stealing scheduler minimises context-switch overhead versus tokio's general-purpose pool. The `oneshot` channels relay results back to the async world; `tokio_stream::iter` + `.then()` + `.fold()` aggregate them cleanly. No `buffer_unordered` is needed here — rayon is already running all batches in parallel.

### S4 — semaphore_bounded

A single `Arc<Semaphore>` enforces a hard cap on concurrent blocking tasks. Relevant when the pricing service shares CPU with other workloads on a cloud node. A stream-based `buffer_unordered` would create a second, redundant concurrency controller.

### S5 — channel_pipeline

Producer–consumer via bounded `mpsc` channels. Workers call `blocking_recv` and `blocking_send` to bridge the sync/async boundary. `ReceiverStream::new(result_rx)` is the idiomatic `tokio_stream` bridge from a channel to a stream. Can be wired directly into an `axum::response::Sse` endpoint to stream partial prices to a client.

### S6 — stream_buffered

The most concise form. `buffer_unordered(n)` internally maintains a `FuturesUnordered` pool of size `n`, replacing both the manual `JoinSet` loop and the Semaphore with a single composable operator.

```rust
tokio_stream::iter(configs)
    .map(|cfg| tokio::task::spawn_blocking(move || engine.run_batch(&cfg)))
    .buffer_unordered(n_threads)
    .map(|r| r.expect("batch panicked"))
    .fold(PartialResult::default(), |acc, r| async move { acc.merge(r) })
    .await
```

### S7 — stream_throttled

`StreamExt::throttle` is unique to `tokio_stream` — there is no equivalent in `futures`. It limits the rate at which new batch configs are submitted to the compute pool, directly modelling a cloud billing quota without external middleware. Called via UFCS (`tokio_stream::StreamExt::throttle(stream, dur)`) to avoid method-name conflicts when `futures::StreamExt` is also in scope for `buffer_unordered`.

### S8 — std_thread (scoped)

Pure standard library — no Tokio, no Rayon. `std::thread::scope` spawns 8 OS threads per chunk, processing 8 chunks sequentially. Results are collected via `Mutex<Vec>`. The scope barrier guarantees all threads in a chunk join before the next chunk starts, giving 100% window purity in the completion order. Measures the abstraction cost of the runtime-based strategies and serves as a zero-dependency baseline.

---

## Greeks

Delta is computed by central finite difference with **common random numbers** (same RNG seed for up/down bumps) and **fixed absolute barriers** (barriers are set at note inception and do not move with the spot):

```
Δ = ( V(S_0 + ε) − V(S_0 − ε) ) / ( 2ε )
```

OSS smoothing eliminates the indicator-function discontinuity at the barrier, so the estimator is differentiable in S_0. The result is stable across bump sizes spanning an order of magnitude:

```
bump = 1.0%  →  Δ = 0.4809
bump = 0.1%  →  Δ = 0.4819
```

A standard (non-OSS) estimator would show substantial noise at 0.1% bump due to the barrier discontinuity.

---

## File Structure

```
src/
├── lib.rs
├── main.rs                       # Benchmark harness
│
├── bin/
│   └── profiler.rs               # Post-run TUI profiler (ratatui)
│
├── domain/
│   ├── product.rs                # Product trait — extensibility point
│   ├── propagator.rs             # Propagator trait + BlackScholes
│   ├── market_data.rs            # MarketData (spot, vol, r, q)
│   ├── time_grid.rs              # DualTimeGrid: coarse + fine
│   ├── autocallable.rs           # AutoCallable implements Product
│   └── american_option.rs        # AmericanOption stub
│
├── simulation/
│   ├── random.rs                 # xoshiro256++ + Box-Muller
│   ├── path_state.rs             # PathState + BatchBuffers
│   ├── one_step_survival.rs      # OSS weight + truncated draw
│   └── brownian_bridge.rs        # Intra-period daily path reconstruction
│
├── engine/
│   ├── monte_carlo.rs            # MonteCarloEngine<P, Pr>
│   └── batch_runner.rs           # BatchConfig, PartialResult
│
├── concurrency/
│   ├── mod.rs                    # ConcurrencyStrategy enum + run_simulation()
│   ├── naive_spawn.rs            # S1: anti-pattern
│   ├── spawn_blocking_joinset.rs # S2: structured tasks
│   ├── rayon_bridge.rs           # S3: rayon + oneshot + stream
│   ├── semaphore_bounded.rs      # S4: bounded concurrency
│   ├── channel_pipeline.rs       # S5: mpsc + ReceiverStream
│   ├── stream_buffered.rs        # S6: buffer_unordered
│   ├── stream_throttled.rs       # S7: throttle + buffer_unordered
│   └── std_thread.rs             # S8: std::thread::scope, zero deps
│
└── analytics/
    ├── results.rs                # PriceResult, BenchmarkReport
    └── profiling.rs              # BatchEvent, ProfiledResult, BatchCollector,
                                  # BatchCollectorLayer (tracing subscriber)
```

---

## Dependencies

| Crate | Role |
|---|---|
| `tokio` | Async runtime, `spawn_blocking`, `JoinSet`, `Semaphore`, `mpsc`, `oneshot` |
| `tokio-stream` | `StreamExt::throttle`, `ReceiverStream`, `iter` |
| `futures` | `StreamExt::buffer_unordered`, `FuturesUnordered` |
| `rayon` | Work-stealing thread pool for CPU-bound batches |
| `statrs` | Normal CDF (Φ) and quantile (Φ⁻¹) for OSS |
| `rand` | Seeding utilities |
| `tracing` | `info_span!` in each batch closure — structured per-batch instrumentation |
| `tracing-subscriber` | `BatchCollectorLayer` registry — collects spans into `Vec<BatchEvent>` |
| `ratatui` + `crossterm` | Terminal UI for the profiler binary |
| `thiserror` / `anyhow` | Error handling |
| `criterion` (dev) | Micro-benchmark harness |

`async-trait` is **not** needed. The engine uses compile-time generics (`MonteCarloEngine<P, Pr>`) rather than dynamic dispatch, so there is no object-safety concern and native `async fn` in traits (stable since Rust 1.75) covers any remaining need.

---

## Adding a New Instrument

Implement the `Product` trait for your type, then pass it to `MonteCarloEngine`:

```rust
#[derive(Clone)]
struct MyBarrierOption { /* ... */ }

impl Product for MyBarrierOption {
    fn observation_dates(&self) -> &[f64] { &self.dates }
    fn notify(&mut self, _t: f64, spot: f64) -> bool {
        // update state; return true to terminate early
        false
    }
    fn terminal_payoff(&self, spot: f64) -> f64 { /* ... */ }
    fn reset(&mut self) { /* zero mutable fields */ }
    fn set_knock_in(&mut self) { self.ki = true; }
    fn knock_in_triggered(&self) -> bool { self.ki }
    // optionally override oss_autocall_payoff for autocall products
}

let engine = MonteCarloEngine::new(
    MyBarrierOption { /* ... */ },
    Arc::new(BlackScholes::new(&market_data)),
    market_data,
    time_grid,
    barrier_call,
    barrier_ki,
    n_monthly,
    business_days_per_month,
);
```

The same seven concurrency strategies, the OSS variance-reduction machinery, and the profiler TUI are available to the new instrument without modification.

---

## Strategy Recommendations

The profiler data leads to clear use-case-specific guidance:

| Use Case | Best | Runner-up | Avoid |
|---|---|---|---|
| Interactive pricing UI | S5 Channel Pipeline | S3 Rayon Bridge | S1, S2 |
| End-of-day batch run | S3 Rayon Bridge | S5 Channel Pipeline | S7 |
| Lowest cloud CPU cost | S7 Stream Throttled | S4 Semaphore | S1, S2 |
| Lowest memory | S7 Stream Throttled | S5, S8 | S2 |
| Best generic default | S6 Stream Buffered | S5 Channel Pipeline | S1 |
| Highest throughput | S3 Rayon Bridge | S2* | S7 |
| Deterministic ordering | S8 std::thread | S7, S5 | S1, S2 |
| Mixed workload server | S5 Channel Pipeline | S4 Semaphore | S1, S2, S3 |

*S2 is fastest (112 ms) but has pathological tail latency and thread explosion.

**Decision tree for most teams:**

1. Need streaming partial results? → **S5**
2. Dedicated batch server? → **S3**
3. Minimal code with good defaults? → **S6**
4. Resource-constrained or billing-sensitive? → **S7**
5. Zero external dependencies? → **S8**

Full analysis with measured data in [`presentation/notes/strategy_recommendations.md`](presentation/notes/strategy_recommendations.md).

---

## AWS Hybrid Architecture

The local hybrid architecture (Tokio controller + Rayon workers on one machine) maps to AWS as:

| Local | AWS | Role |
|---|---|---|
| Tokio Controller | EC2 / Fargate Task | Async I/O, gRPC fan-out, aggregation |
| FuturesUnordered | gRPC streaming | Out-of-order completion, backpressure |
| Rayon thread pool | EKS / ECS worker pods | CPU-bound MC simulation, work-stealing |
| In-memory aggregation | Controller-side reducer | Streaming partials, convergence check |

The gRPC boundary decouples runtimes — the controller and workers scale independently. Workers run on Spot/Fargate Spot (60–70% savings), and idempotent batches make Spot interruption a non-event. HPA scales from 0 to 100+ pods based on queue depth.

Estimated cost: ~$0.003 per 1M-path pricing run. Full portfolio risk (10K instruments) completes in minutes for ~$30.

Architecture diagram: [`presentation/svg/images/aws_hybrid_architecture.svg`](presentation/svg/images/aws_hybrid_architecture.svg)
Design rationale: [`presentation/notes/slide24_aws_hybrid_architecture.md`](presentation/notes/slide24_aws_hybrid_architecture.md)

---

## Presentation

A 25-slide deck with speaker notes covers the full study:

```
presentation/
├── concurrency_optimization.pptx    # Main deck (25 slides)
├── svg/images/                      # Architecture diagrams (SVG)
│   ├── architecture_overview.svg
│   ├── hybrid_architecture.svg
│   └── aws_hybrid_architecture.svg
└── notes/                           # Speaker notes (md + pdf)
    ├── strategies.md                # All 8 strategies — code + analysis
    ├── strategy_recommendations.md  # Use-case matrix + decision tree
    ├── slide24_aws_hybrid_architecture.md
    ├── slide05–slide22_*.md         # Per-slide deep-dive notes
    └── *.pdf                        # PDF exports of all notes
```

---

## References

- P. Glasserman & J. Staum, *Conditioning on one-step survival for barrier option simulations*, Operations Research 49(6), 2001.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004. Chapter 6 (variance reduction), Chapter 8 (Greeks).
- D. Blackman & S. Vigna, *Scrambled Linear Pseudorandom Number Generators*, ACM TOMACS 32(2), 2022. (xoshiro256++)
