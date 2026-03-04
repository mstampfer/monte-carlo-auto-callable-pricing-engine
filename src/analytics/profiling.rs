use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use tracing::span;
use tracing_subscriber::{layer::Context, Layer};

use crate::analytics::PriceResult;

// ── Public types (unchanged interface) ───────────────────────────────────────

/// Per-batch timing and result captured during a simulation run.
#[derive(Debug, Clone)]
pub struct BatchEvent {
    /// Batch index (matches `BatchConfig::batch_id`).
    pub batch_id:  usize,
    /// OS thread that executed this batch.
    pub thread_id: ThreadId,
    /// Start time relative to `run_simulation` t0.
    pub start:     Duration,
    /// Wall-clock time for this batch only.
    pub duration:  Duration,
    /// Number of paths in this batch.
    pub n_paths:   usize,
    /// Batch-local price estimate (sum_payoff / n_paths).
    pub price:     f64,
    /// Batch-local standard error.
    pub std_err:   f64,
}

/// Result from `run_simulation` — wraps `PriceResult` plus per-batch profiling data.
#[derive(Debug)]
pub struct ProfiledResult {
    pub price_result: PriceResult,
    /// One event per batch, in collection order (not necessarily batch_id order).
    pub events:       Vec<BatchEvent>,
}

// ── Internal in-flight state ──────────────────────────────────────────────────

struct InFlightSpan {
    batch_id:   usize,
    n_paths:    usize,
    thread_id:  ThreadId,
    entered_at: Instant,
    price:      f64,
    std_err:    f64,
}

struct CompletedBatch {
    batch_id:   usize,
    n_paths:    usize,
    thread_id:  ThreadId,
    entered_at: Instant,  // absolute; relativised in drain()
    duration:   Duration,
    price:      f64,
    std_err:    f64,
}

// ── Global collector singleton ────────────────────────────────────────────────

/// Accumulates completed batch spans.  Access via [`BatchCollector::global()`].
///
/// The profiler binary registers [`BatchCollectorLayer`] as the global tracing subscriber.
/// The main benchmark binary registers nothing — tracing spans become no-ops and
/// `drain()` returns an empty `Vec`.
#[derive(Default)]
pub struct BatchCollector {
    in_flight: Mutex<HashMap<span::Id, InFlightSpan>>,
    completed: Mutex<Vec<CompletedBatch>>,
}

static BATCH_COLLECTOR: OnceLock<BatchCollector> = OnceLock::new();

impl BatchCollector {
    pub fn global() -> &'static Self {
        BATCH_COLLECTOR.get_or_init(BatchCollector::default)
    }

    /// Drain all completed batches, computing `start` relative to `run_t0`.
    pub fn drain(&self, run_t0: Instant) -> Vec<BatchEvent> {
        self.completed.lock().unwrap().drain(..)
            .map(|b| BatchEvent {
                batch_id:  b.batch_id,
                thread_id: b.thread_id,
                start:     b.entered_at.checked_duration_since(run_t0).unwrap_or_default(),
                duration:  b.duration,
                n_paths:   b.n_paths,
                price:     b.price,
                std_err:   b.std_err,
            })
            .collect()
    }
}

// ── tracing Layer ─────────────────────────────────────────────────────────────

/// Register with `tracing_subscriber::registry()` to populate [`BatchCollector::global()`].
///
/// Only spans named `"batch"` are collected; all others are ignored.
pub struct BatchCollectorLayer;

impl<S: tracing::Subscriber> Layer<S> for BatchCollectorLayer {
    /// Span created: extract `batch_id` and `n_paths` fields.
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, _ctx: Context<'_, S>) {
        if attrs.metadata().name() != "batch" {
            return;
        }
        let mut v = NewSpanVisitor::default();
        attrs.record(&mut v);
        BatchCollector::global().in_flight.lock().unwrap().insert(
            id.clone(),
            InFlightSpan {
                batch_id:   v.batch_id,
                n_paths:    v.n_paths,
                thread_id:  std::thread::current().id(), // overwritten on_enter
                entered_at: Instant::now(),              // overwritten on_enter
                price:      0.0,
                std_err:    0.0,
            },
        );
    }

    /// Span entered: record the executing thread and wall-clock start time.
    fn on_enter(&self, id: &span::Id, _ctx: Context<'_, S>) {
        if let Some(s) = BatchCollector::global().in_flight.lock().unwrap().get_mut(id) {
            s.thread_id  = std::thread::current().id();
            s.entered_at = Instant::now();
        }
    }

    /// `span.record(…)` called: capture `price` and `std_err` fields.
    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, _ctx: Context<'_, S>) {
        if let Some(s) = BatchCollector::global().in_flight.lock().unwrap().get_mut(id) {
            let mut v = RecordVisitor::default();
            values.record(&mut v);
            if let Some(p) = v.price   { s.price   = p; }
            if let Some(e) = v.std_err { s.std_err = e; }
        }
    }

    /// Span exited (guard dropped): finalise duration and move to completed list.
    fn on_exit(&self, id: &span::Id, _ctx: Context<'_, S>) {
        let col = BatchCollector::global();
        if let Some(s) = col.in_flight.lock().unwrap().remove(id) {
            col.completed.lock().unwrap().push(CompletedBatch {
                batch_id:   s.batch_id,
                n_paths:    s.n_paths,
                thread_id:  s.thread_id,
                entered_at: s.entered_at,
                duration:   s.entered_at.elapsed(),
                price:      s.price,
                std_err:    s.std_err,
            });
        }
    }
}

// ── Field visitors ────────────────────────────────────────────────────────────

#[derive(Default)]
struct NewSpanVisitor {
    batch_id: usize,
    n_paths:  usize,
}

impl tracing::field::Visit for NewSpanVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "batch_id" => self.batch_id = value as usize,
            "n_paths"  => self.n_paths  = value as usize,
            _ => {}
        }
    }
    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
}

#[derive(Default)]
struct RecordVisitor {
    price:   Option<f64>,
    std_err: Option<f64>,
}

impl tracing::field::Visit for RecordVisitor {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        match field.name() {
            "price"   => self.price   = Some(value),
            "std_err" => self.std_err = Some(value),
            _ => {}
        }
    }
    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
}
