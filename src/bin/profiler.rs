//! Monte Carlo Profiler TUI
//!
//! Post-run visualiser for all 7 concurrency strategies.
//!
//! Usage:
//!   cargo run --release --bin profiler
//!   cargo run --release --bin profiler -- --npaths 2_000_000
//!   cargo run --release --bin profiler -- --nbatches 64
//!   cargo run --release --bin profiler -- s3 s6 s7
//!
//! Tabs:
//!   1 / Tab  — Thread Timelines (Gantt)
//!   2        — Batch Analysis   (histogram, thread matrix, completion order)
//!   3        — Memory Analysis  (per-batch alloc sparklines, peak heap)
//!   4        — Convergence & Comparison
//!   q / Esc  — Quit
//!   ↑ / ↓   — Scroll (Tab 1) or select strategy (Tabs 2/3/4)

use std::collections::HashSet;
use std::io;
use std::process;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Sparkline,
        Table, Tabs,
    },
    Frame, Terminal,
};

use hsbc_monte_carlo_auto_callable::{
    analytics::{BatchEvent, ProfiledResult, TrackingAllocator},
    concurrency::{run_simulation, ConcurrencyStrategy},
    domain::{BlackScholes, DualTimeGrid, MarketData, AutoCallable},
    engine::MonteCarloEngine,
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

// ── Simulation parameters (identical to main.rs) ─────────────────────────────

const DEFAULT_N_PATHS:         usize = 200_000;
const DEFAULT_N_BATCHES:       usize = 32;   // 4× N_THREADS — exposes work-stealing
const N_THREADS:               usize = 8;
const GLOBAL_SEED:             u64   = 42;
const SPOT_INITIAL:            f64   = 100.0;
const NOTIONAL:                f64   = 100.0;
const VOLATILITY:              f64   = 0.25;
const RISK_FREE_RATE:          f64   = 0.05;
const DIVIDEND_YIELD:          f64   = 0.02;
const BARRIER_CALL_FRAC:       f64   = 1.00;
const BARRIER_KI_FRAC:         f64   = 0.70;
const MATURITY_YEARS:          f64   = 1.0;
const N_MONTHLY:               usize = 12;
const BUSINESS_DAYS_PER_MONTH: usize = 21;

const ALL_STRATEGIES: &[ConcurrencyStrategy] = &[
    ConcurrencyStrategy::NaiveSpawn,
    ConcurrencyStrategy::SpawnBlockingJoinSet,
    ConcurrencyStrategy::RayonBridge,
    ConcurrencyStrategy::SemaphoreBounded,
    ConcurrencyStrategy::ChannelPipeline,
    ConcurrencyStrategy::StreamBuffered,
    ConcurrencyStrategy::StreamThrottled,
];

/// Colors cycled per batch_id in the Gantt chart.
const BATCH_COLORS: [Color; 8] = [
    Color::Blue,
    Color::Green,
    Color::Red,
    Color::Yellow,
    Color::Magenta,
    Color::Cyan,
    Color::LightBlue,
    Color::LightGreen,
];

// ── App state ────────────────────────────────────────────────────────────────

struct App {
    /// Active tab index: 0 = Timeline, 1 = Batch Analysis, 2 = Memory, 3 = Convergence
    tab: usize,
    /// Selected strategy index (used by tabs 2–4, and for convergence overlay in tab 3)
    selected: usize,
    /// Scroll offset for Tab 1 Gantt (lines)
    scroll: usize,
    /// All profiling results, one per strategy
    results: Vec<ProfiledResult>,
    /// Peak heap bytes (live) captured per strategy
    peak_heap: Vec<usize>,
}

impl App {
    fn new(results: Vec<ProfiledResult>, peak_heap: Vec<usize>) -> Self {
        Self { tab: 0, selected: 0, scroll: 0, results, peak_heap }
    }

    fn next_tab(&mut self) {
        self.tab = (self.tab + 1) % 4;
    }

    fn prev_tab(&mut self) {
        self.tab = (self.tab + 3) % 4;
    }

    fn scroll_down(&mut self) {
        match self.tab {
            0 => self.scroll = self.scroll.saturating_add(1),
            _ => self.selected = (self.selected + 1).min(self.results.len().saturating_sub(1)),
        }
    }

    fn scroll_up(&mut self) {
        match self.tab {
            0 => self.scroll = self.scroll.saturating_sub(1),
            _ => self.selected = self.selected.saturating_sub(1),
        }
    }
}

// ── Metric helpers ────────────────────────────────────────────────────────────

fn unique_threads(events: &[BatchEvent]) -> Vec<ThreadId> {
    let mut seen: HashSet<ThreadId> = HashSet::new();
    let mut ordered: Vec<ThreadId>  = Vec::new();
    // Sort by first appearance (start time) to get stable T0, T1, ... labels
    let mut sorted: Vec<&BatchEvent> = events.iter().collect();
    sorted.sort_by_key(|e| e.start);
    for e in sorted {
        if seen.insert(e.thread_id) {
            ordered.push(e.thread_id);
        }
    }
    ordered
}

/// Returns (cpu_efficiency 0–1, load_imbalance_index).
fn compute_metrics(result: &ProfiledResult) -> (f64, f64) {
    let wall_ns = result.price_result.wall_time.as_nanos() as f64;
    if wall_ns == 0.0 || result.events.is_empty() {
        return (0.0, 1.0);
    }
    let n_threads = unique_threads(&result.events).len() as f64;
    let total_compute: f64 = result.events.iter().map(|e| e.duration.as_nanos() as f64).sum();
    let cpu_eff = total_compute / (wall_ns * n_threads);

    let durations_ms: Vec<f64> = result.events.iter().map(|e| e.duration.as_secs_f64() * 1000.0).collect();
    let max_dur  = durations_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_dur = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
    let imbalance = if mean_dur > 0.0 { max_dur / mean_dur } else { 1.0 };

    (cpu_eff, imbalance)
}

/// Running price convergence: cumulative price after each batch (sorted by completion).
fn compute_convergence(result: &ProfiledResult) -> Vec<f64> {
    let mut sorted: Vec<&BatchEvent> = result.events.iter().collect();
    sorted.sort_by_key(|e| e.start + e.duration);

    let mut cumsum = 0.0f64;
    let mut total_paths = 0usize;
    sorted.iter().map(|e| {
        cumsum      += e.price * e.n_paths as f64;
        total_paths += e.n_paths;
        cumsum / total_paths as f64
    }).collect()
}

/// Compute the shared x-axis range across all strategy results.
///
/// Returns `(global_min_ms, global_max_ms)`. The y-axis max and bucket count
/// are computed inside `render_tab2` once the widget width is known, so that
/// the sparkline always fills the full width of the histogram pane.
fn global_histogram_range(results: &[ProfiledResult]) -> (f64, f64) {
    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;
    for r in results {
        for e in &r.events {
            let ms = e.duration.as_secs_f64() * 1000.0;
            global_min = global_min.min(ms);
            global_max = global_max.max(ms);
        }
    }
    if global_min > global_max { (0.0, 1.0) } else { (global_min, global_max) }
}

/// Events sorted by completion time (start + duration).
fn completion_order(events: &[BatchEvent]) -> Vec<usize> {
    let mut indexed: Vec<(usize, Duration)> = events.iter()
        .map(|e| (e.batch_id, e.start + e.duration))
        .collect();
    indexed.sort_by_key(|(_, t)| *t);
    indexed.into_iter().map(|(id, _)| id).collect()
}

// ── Tab 1: Thread Timelines ───────────────────────────────────────────────────

fn render_tab1(f: &mut Frame, area: Rect, app: &App) {
    // Pre-build all lines, then render with scroll offset via Paragraph::scroll
    let mut lines: Vec<Line<'static>> = Vec::new();

    for result in &app.results {
        let wall_ms = result.price_result.wall_time.as_millis();
        let (cpu_eff, imbalance) = compute_metrics(result);
        let threads = unique_threads(&result.events);

        // Strategy header
        lines.push(Line::from(vec![
            Span::styled(
                format!(" ▶ {}   {} ms", result.price_result.strategy_name, wall_ms),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]));

        let wall_ns = result.price_result.wall_time.as_nanos() as u64;
        // Leave room for "  T0 [" (6 chars) and "]" (1 char) + right label (8 chars)
        let bar_width = (area.width as usize).saturating_sub(16).max(4);

        for (tid_idx, tid) in threads.iter().enumerate() {
            let thread_events: Vec<&BatchEvent> = result.events.iter()
                .filter(|e| &e.thread_id == tid)
                .collect();

            let mut spans: Vec<Span<'static>> = Vec::new();
            spans.push(Span::raw(format!("  T{tid_idx:<1} [")));

            for col in 0..bar_width {
                let t_ns = if wall_ns > 0 {
                    (col as u128 * wall_ns as u128 / bar_width as u128) as u64
                } else {
                    0
                };
                let t = Duration::from_nanos(t_ns);

                let active = thread_events.iter().find(|e| t >= e.start && t < e.start + e.duration);
                let (ch, color) = match active {
                    Some(e) => ('█', BATCH_COLORS[e.batch_id % BATCH_COLORS.len()]),
                    None    => ('░', Color::DarkGray),
                };
                spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            }
            spans.push(Span::raw("]"));
            lines.push(Line::from(spans));
        }

        // Metrics line
        lines.push(Line::from(vec![
            Span::styled(
                format!(
                    "  CPU eff: {:.0}%  Imbalance: {:.2}  Batches: {}  Price: {:.3}",
                    cpu_eff * 100.0, imbalance, result.events.len(),
                    result.price_result.price
                ),
                Style::default().fg(Color::Cyan),
            ),
        ]));
        lines.push(Line::from(""));  // spacer
    }

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(
            " Tab 1: Thread Timelines  [↑↓ scroll] [Tab/2/3 switch] [q quit] "
        ))
        .scroll((app.scroll as u16, 0));
    f.render_widget(para, area);
}

// ── Tab 2: Batch Analysis ─────────────────────────────────────────────────────

fn render_tab2(f: &mut Frame, area: Rect, app: &App, hist_min: f64, hist_max: f64) {
    // Split: left = strategy list, right = details
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(0)])
        .split(area);

    // ── Left: strategy list ──────────────────────────────────────────────────
    let items: Vec<ListItem> = app.results.iter().enumerate().map(|(i, r)| {
        let prefix = if i == app.selected { "● " } else { "  " };
        let wall_ms = r.price_result.wall_time.as_millis();
        ListItem::new(format!("{}{} ({}ms)", prefix, r.price_result.strategy_name.trim(), wall_ms))
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Strategy [↑↓] "))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));
    f.render_stateful_widget(list, chunks[0], &mut list_state);

    // ── Right: batch details for selected strategy ───────────────────────────
    if app.results.is_empty() {
        return;
    }
    let result = &app.results[app.selected.min(app.results.len() - 1)];
    let threads = unique_threads(&result.events);

    let n_comp_rows = (result.events.len().div_ceil(20) + 3).min(10) as u16;
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // histogram
            Constraint::Length(threads.len() as u16 + 3),  // thread matrix
            Constraint::Min(n_comp_rows),                   // completion order
        ])
        .split(chunks[1]);

    // ── Duration histogram (shared x/y scale, width-filling) ─────────────────
    // Use the inner widget width as bucket count: Sparkline renders one bar per
    // data point, so n_buckets == inner_width guarantees the bars fill the pane.
    let n_buckets = (right_chunks[0].width as usize).saturating_sub(2).max(8);
    let range = (hist_max - hist_min).max(0.001);

    // Compute per-strategy peak bucket counts.
    let peaks: Vec<u64> = app.results.iter().map(|r| {
        let mut b = vec![0u64; n_buckets];
        for e in &r.events {
            let ms = e.duration.as_secs_f64() * 1000.0;
            let i = ((ms - hist_min) / range * (n_buckets - 1) as f64).round() as usize;
            b[i.min(n_buckets - 1)] += 1;
        }
        b.into_iter().max().unwrap_or(0)
    }).collect();

    // Y-axis cap: max peak of all strategies *except* the selected one.
    // If the selected strategy's peak exceeds this cap it is truncated,
    // preventing an outlier (e.g. S7 throttled) from crushing the scale.
    let selected_idx = app.selected.min(app.results.len() - 1);
    let others_y_max: u64 = peaks.iter().enumerate()
        .filter(|&(i, _)| i != selected_idx)
        .map(|(_, &p)| p)
        .max()
        .unwrap_or(1);
    let selected_peak = peaks[selected_idx];
    let truncated = selected_peak > others_y_max && app.results.len() > 1;
    let hist_y_max = if truncated { others_y_max } else { selected_peak.max(1) };

    let mut buckets = vec![0u64; n_buckets];
    for e in &result.events {
        let ms = e.duration.as_secs_f64() * 1000.0;
        let i = ((ms - hist_min) / range * (n_buckets - 1) as f64).round() as usize;
        buckets[i.min(n_buckets - 1)] += 1;
    }

    let trunc_note = if truncated { " (y truncated)" } else { "" };
    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(
            format!(" Duration histogram  [{:.1}ms – {:.1}ms]{} ", hist_min, hist_max, trunc_note)
        ))
        .data(&buckets)
        .max(hist_y_max)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, right_chunks[0]);

    // ── Batch-to-thread mapping (compact list) ────────────────────────────────
    // A sparse matrix with one column per batch requires ~200 chars for 64 batches;
    // ratatui scales columns proportionally when they overflow, shrinking cells to
    // 1 char and making the ● dots invisible. Instead, list each thread's batches.
    let mut matrix_lines: Vec<Line<'static>> = Vec::new();
    matrix_lines.push(Line::from(Span::styled(
        "Thread  Batches run (sorted by start time, colored by batch id)",
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
    )));
    for (tid_idx, tid) in threads.iter().enumerate() {
        let mut batch_events: Vec<(usize, std::time::Duration)> = result.events.iter()
            .filter(|e| &e.thread_id == tid)
            .map(|e| (e.batch_id, e.start))
            .collect();
        batch_events.sort_by_key(|&(_, s)| s);
        let mut spans: Vec<Span<'static>> = vec![
            Span::styled(format!("T{:<5} ", tid_idx), Style::default().fg(Color::Gray)),
        ];
        for (bid, _) in &batch_events {
            spans.push(Span::styled(
                format!(" {:>3}", bid),
                Style::default().fg(BATCH_COLORS[bid % BATCH_COLORS.len()]),
            ));
        }
        matrix_lines.push(Line::from(spans));
    }
    let matrix_para = Paragraph::new(matrix_lines)
        .block(Block::default().borders(Borders::ALL).title(" Batch-to-thread mapping "));
    f.render_widget(matrix_para, right_chunks[1]);

    // ── Completion order (wrapped) ────────────────────────────────────────────
    let order = completion_order(&result.events);
    let n_per_line = 20usize;
    let mut comp_lines: Vec<Line<'static>> = vec![
        Line::from(Span::styled(
            "Completion order (by finish time):",
            Style::default().fg(Color::White),
        )),
    ];
    for chunk in order.chunks(n_per_line) {
        let spans: Vec<Span<'static>> = std::iter::once(Span::raw("  "))
            .chain(chunk.iter().map(|&id| Span::styled(
                format!("{:>3}", id),
                Style::default().fg(BATCH_COLORS[id % BATCH_COLORS.len()]),
            )))
            .collect();
        comp_lines.push(Line::from(spans));
    }
    comp_lines.push(Line::from(Span::styled(
        "  (out-of-order = work-stealing / async scheduling visible)",
        Style::default().fg(Color::DarkGray),
    )));
    let completion_para = Paragraph::new(comp_lines)
        .block(Block::default().borders(Borders::ALL).title(" Completion Order "));
    f.render_widget(completion_para, right_chunks[2]);
}

// ── Tab 3: Convergence & Comparison ──────────────────────────────────────────

fn render_tab3(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(0)])
        .split(area);

    // ── Convergence sparkline for selected strategy ───────────────────────────
    if !app.results.is_empty() {
        let result = &app.results[app.selected.min(app.results.len() - 1)];
        let conv = compute_convergence(result);

        if !conv.is_empty() {
            let min_p = conv.iter().cloned().fold(f64::MAX, f64::min);
            let max_p = conv.iter().cloned().fold(f64::MIN, f64::max);
            let range = (max_p - min_p).max(0.001);

            // Interpolate convergence curve to fill the widget width.
            let inner_w = (chunks[0].width as usize).saturating_sub(2).max(1);
            let spark_data: Vec<u64> = (0..inner_w).map(|col| {
                // Map column to a fractional index into conv[]
                let t = col as f64 / (inner_w - 1).max(1) as f64;
                let idx_f = t * (conv.len() - 1) as f64;
                let lo = (idx_f.floor() as usize).min(conv.len() - 1);
                let hi = (lo + 1).min(conv.len() - 1);
                let frac = idx_f - lo as f64;
                let p = conv[lo] * (1.0 - frac) + conv[hi] * frac;
                ((p - min_p) / range * 1000.0) as u64
            }).collect();

            let final_price = conv.last().copied().unwrap_or(0.0);
            let sparkline = Sparkline::default()
                .block(Block::default().borders(Borders::ALL).title(format!(
                    " Price convergence: {} — {:.3} final price  [{:.3}–{:.3}]  [↑↓ select] ",
                    result.price_result.strategy_name.trim(), final_price, min_p, max_p
                )))
                .data(&spark_data)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(sparkline, chunks[0]);
        }
    }

    // ── Comparison table ──────────────────────────────────────────────────────
    let baseline_ms = app.results.first()
        .map(|r| r.price_result.wall_time.as_millis() as f64)
        .unwrap_or(1.0);

    let header = Row::new(vec![
        Cell::from("Strategy").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("ms").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Price").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("CPU eff").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Imbalance").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Speedup").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Alloc").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Peak heap").style(Style::default().add_modifier(Modifier::BOLD)),
    ]).style(Style::default().fg(Color::Yellow));

    let rows: Vec<Row> = app.results.iter().enumerate().map(|(i, r)| {
        let ms = r.price_result.wall_time.as_millis();
        let speedup = baseline_ms / ms as f64;
        let (cpu_eff, imbalance) = compute_metrics(r);
        let total_alloc: usize = r.events.iter().map(|e| e.alloc_bytes).sum();
        let peak = app.peak_heap.get(i).copied().unwrap_or(0);

        let style = if i == app.selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else if cpu_eff >= 0.95 {
            Style::default().fg(Color::Green)
        } else if cpu_eff < 0.70 {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };

        Row::new(vec![
            Cell::from(r.price_result.strategy_name.trim().to_string()),
            Cell::from(ms.to_string()),
            Cell::from(format!("{:.3}", r.price_result.price)),
            Cell::from(format!("{:.0}%", cpu_eff * 100.0)),
            Cell::from(format!("{:.2}", imbalance)),
            Cell::from(format!("{:.2}×", speedup)),
            Cell::from(format_bytes(total_alloc)),
            Cell::from(format_bytes(peak)),
        ]).style(style)
    }).collect();

    let widths = [
        Constraint::Length(30),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(
            " Strategy Comparison  [↑↓ select convergence] "
        ))
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_widget(table, chunks[1]);
}

// ── Tab 4: Memory Analysis ────────────────────────────────────────────────

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Linearly interpolate a sparse data series to exactly `width` points,
/// so the sparkline fills the full widget width (one bar per column).
fn interpolate_to_width(data: &[u64], width: usize) -> Vec<u64> {
    if data.is_empty() {
        return vec![0; width];
    }
    if data.len() == 1 {
        return vec![data[0]; width];
    }
    (0..width).map(|col| {
        let t = col as f64 / (width - 1).max(1) as f64;
        let idx_f = t * (data.len() - 1) as f64;
        let lo = (idx_f.floor() as usize).min(data.len() - 1);
        let hi = (lo + 1).min(data.len() - 1);
        let frac = idx_f - lo as f64;
        let v = data[lo] as f64 * (1.0 - frac) + data[hi] as f64 * frac;
        v as u64
    }).collect()
}

fn render_tab4(f: &mut Frame, area: Rect, app: &App) {
    if app.results.is_empty() {
        return;
    }
    let selected_idx = app.selected.min(app.results.len() - 1);
    let result = &app.results[selected_idx];

    // Vertical layout: full-width sparklines on top, strategy list + summary at bottom
    let n_strat_rows = (app.results.len() as u16 + 2).min(12);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),           // alloc bytes sparkline
            Constraint::Length(12),           // alloc count sparkline
            Constraint::Min(n_strat_rows),    // bottom pane
        ])
        .split(area);

    // ── Alloc bytes sparkline (interpolated to full width, global y-scale) ─
    let global_max_bytes: u64 = app.results.iter()
        .flat_map(|r| r.events.iter().map(|e| e.alloc_bytes as u64))
        .max()
        .unwrap_or(1);

    let raw_bytes: Vec<u64> = result.events.iter()
        .map(|e| e.alloc_bytes as u64)
        .collect();

    let inner_w = (rows[0].width as usize).saturating_sub(2).max(1);
    let bytes_data = interpolate_to_width(&raw_bytes, inner_w);

    let max_batch_bytes = raw_bytes.iter().copied().max().unwrap_or(0);
    let sparkline_bytes = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Allocation volume per batch — {}  [max: {}] ",
            result.price_result.strategy_name.trim(),
            format_bytes(max_batch_bytes as usize)
        )))
        .data(&bytes_data)
        .max(global_max_bytes)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(sparkline_bytes, rows[0]);

    // ── Alloc count sparkline (interpolated to full width, global y-scale) ─
    let global_max_count: u64 = app.results.iter()
        .flat_map(|r| r.events.iter().map(|e| e.alloc_count as u64))
        .max()
        .unwrap_or(1);

    let raw_count: Vec<u64> = result.events.iter()
        .map(|e| e.alloc_count as u64)
        .collect();

    let inner_w = (rows[1].width as usize).saturating_sub(2).max(1);
    let count_data = interpolate_to_width(&raw_count, inner_w);

    let max_batch_count = raw_count.iter().copied().max().unwrap_or(0);
    let sparkline_count = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Allocation count per batch — {}  [max: {}] ",
            result.price_result.strategy_name.trim(),
            max_batch_count
        )))
        .data(&count_data)
        .max(global_max_count)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline_count, rows[1]);

    // ── Bottom: strategy list (left) + memory summary (right) ────────────────
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(38), Constraint::Min(0)])
        .split(rows[2]);

    // Strategy list
    let items: Vec<ListItem> = app.results.iter().enumerate().map(|(i, r)| {
        let prefix = if i == app.selected { "● " } else { "  " };
        let total_alloc: usize = r.events.iter().map(|e| e.alloc_bytes).sum();
        ListItem::new(format!("{}{} ({})", prefix, r.price_result.strategy_name.trim(), format_bytes(total_alloc)))
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Strategy [↑↓] "))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));
    f.render_stateful_widget(list, bottom[0], &mut list_state);

    // Memory summary
    let total_alloc_bytes: usize = result.events.iter().map(|e| e.alloc_bytes).sum();
    let total_alloc_count: usize = result.events.iter().map(|e| e.alloc_count).sum();
    let n_batches = result.events.len().max(1);
    let avg_bytes = total_alloc_bytes / n_batches;
    let avg_count = total_alloc_count / n_batches;
    let peak = app.peak_heap.get(selected_idx).copied().unwrap_or(0);

    let summary_lines = vec![
        Line::from(vec![
            Span::styled("  Total alloc bytes:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>14}  ({} avg/batch)", format_bytes(total_alloc_bytes), format_bytes(avg_bytes)),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Total alloc count:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>14}  ({} avg/batch)", total_alloc_count, avg_count),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Peak heap (live):   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>14}", format_bytes(peak)),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Batches:            ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>14}", result.events.len()),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    let summary = Paragraph::new(summary_lines)
        .block(Block::default().borders(Borders::ALL).title(" Memory Summary "));
    f.render_widget(summary, bottom[1]);
}

// ── Main render ───────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Outer layout: tabs bar + content + help bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(size);

    // Tab bar
    let tab_titles: Vec<Line> = vec![
        Line::from(" 1: Thread Timelines "),
        Line::from(" 2: Batch Analysis "),
        Line::from(" 3: Memory Analysis "),
        Line::from(" 4: Convergence & Compare "),
    ];
    let tabs = Tabs::new(tab_titles)
        .block(Block::default().borders(Borders::ALL).title(" Monte Carlo Profiler "))
        .select(app.tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, chunks[0]);

    // Content
    match app.tab {
        0 => render_tab1(f, chunks[1], app),
        1 => {
            let (h_min, h_max) = global_histogram_range(&app.results);
            render_tab2(f, chunks[1], app, h_min, h_max);
        }
        2 => render_tab4(f, chunks[1], app),
        3 => render_tab3(f, chunks[1], app),
        _ => {}
    }

    // Help bar
    let help = Paragraph::new(Line::from(vec![
        Span::styled(" Tab/1/2/3/4 ", Style::default().fg(Color::Cyan)),
        Span::raw("switch  "),
        Span::styled(" ↑/↓ ", Style::default().fg(Color::Cyan)),
        Span::raw("scroll/select  "),
        Span::styled(" q/Esc ", Style::default().fg(Color::Cyan)),
        Span::raw("quit"),
    ])).style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[2]);
}

// ── CLI parsing ───────────────────────────────────────────────────────────────

fn parse_npaths(s: &str) -> Result<usize, String> {
    let cleaned: String = s.chars().filter(|&c| c != '_').collect();
    cleaned.parse::<usize>()
        .map_err(|_| format!("'{}' is not a valid integer", s))
        .and_then(|n| if n == 0 { Err("--npaths must be > 0".into()) } else { Ok(n) })
}

fn parse_strategy(arg: &str) -> Option<ConcurrencyStrategy> {
    match arg.to_lowercase().trim_start_matches('s') {
        "1" | "naive_spawn"            => Some(ConcurrencyStrategy::NaiveSpawn),
        "2" | "spawn_blocking_joinset" => Some(ConcurrencyStrategy::SpawnBlockingJoinSet),
        "3" | "rayon_bridge"           => Some(ConcurrencyStrategy::RayonBridge),
        "4" | "semaphore_bounded"      => Some(ConcurrencyStrategy::SemaphoreBounded),
        "5" | "channel_pipeline"       => Some(ConcurrencyStrategy::ChannelPipeline),
        "6" | "stream_buffered"        => Some(ConcurrencyStrategy::StreamBuffered),
        "7" | "stream_throttled"       => Some(ConcurrencyStrategy::StreamThrottled),
        _ => None,
    }
}

// ── Tracing setup ─────────────────────────────────────────────────────────────

fn setup_tracing() {
    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry()
        .with(hsbc_monte_carlo_auto_callable::analytics::BatchCollectorLayer)
        .init();
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_tracing();

    // ── Parse args ────────────────────────────────────────────────────────────
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = raw.as_slice();

    let n_paths: usize = if args.first().map(|s| s == "--npaths").unwrap_or(false) {
        match args.get(1) {
            Some(val) => match parse_npaths(val) {
                Ok(n)  => { args = &args[2..]; n }
                Err(e) => { eprintln!("error: {e}"); process::exit(1); }
            },
            None => { eprintln!("error: --npaths requires a value"); process::exit(1); }
        }
    } else {
        DEFAULT_N_PATHS
    };

    let n_batches: usize = if args.first().map(|s| s == "--nbatches").unwrap_or(false) {
        match args.get(1) {
            Some(val) => match parse_npaths(val) {  // reuse same integer parser
                Ok(n)  => { args = &args[2..]; n }
                Err(e) => { eprintln!("error: {e}"); process::exit(1); }
            },
            None => { eprintln!("error: --nbatches requires a value"); process::exit(1); }
        }
    } else {
        DEFAULT_N_BATCHES
    };

    let selected_strategies: Vec<ConcurrencyStrategy> = if args.is_empty() {
        ALL_STRATEGIES.to_vec()
    } else {
        let mut strats = Vec::new();
        for arg in args {
            match parse_strategy(arg) {
                Some(s) => strats.push(s),
                None => {
                    eprintln!("error: unknown strategy '{arg}'");
                    process::exit(1);
                }
            }
        }
        strats
    };

    // ── Build engine ──────────────────────────────────────────────────────────
    let market_data = MarketData::new(SPOT_INITIAL, VOLATILITY, RISK_FREE_RATE, DIVIDEND_YIELD);

    let coupons: Vec<f64>   = (1..=N_MONTHLY).map(|k| 0.05 * k as f64 / N_MONTHLY as f64).collect();
    let obs_dates: Vec<f64> = (1..=N_MONTHLY).map(|k| MATURITY_YEARS * k as f64 / N_MONTHLY as f64).collect();

    let product = AutoCallable::new(
        SPOT_INITIAL, NOTIONAL,
        BARRIER_CALL_FRAC, BARRIER_KI_FRAC,
        coupons, obs_dates,
    );

    let propagator = Arc::new(BlackScholes::new(&market_data));
    let time_grid  = DualTimeGrid::new(MATURITY_YEARS, N_MONTHLY, BUSINESS_DAYS_PER_MONTH);

    let engine: Arc<MonteCarloEngine<AutoCallable, BlackScholes>> = Arc::new(
        MonteCarloEngine::new(
            product, propagator, market_data, time_grid,
            SPOT_INITIAL * BARRIER_CALL_FRAC,
            SPOT_INITIAL * BARRIER_KI_FRAC,
            N_MONTHLY, BUSINESS_DAYS_PER_MONTH,
        )
    );

    // ── Run simulations ───────────────────────────────────────────────────────
    eprintln!("Running {} strategies — {} paths, {} batches ({} threads)...",
        selected_strategies.len(), n_paths, n_batches, N_THREADS);
    let t_total = Instant::now();

    let mut results: Vec<ProfiledResult> = Vec::new();
    let mut peak_heap: Vec<usize> = Vec::new();
    for strategy in &selected_strategies {
        eprint!("  {:35} ... ", strategy.name());
        TrackingAllocator::reset_peak();
        let profiled = run_simulation(*strategy, Arc::clone(&engine), n_paths, N_THREADS, n_batches, GLOBAL_SEED).await;
        let peak = TrackingAllocator::peak_bytes();
        eprintln!("{} ms  price = {:.3}",
            profiled.price_result.wall_time.as_millis(),
            profiled.price_result.price);
        results.push(profiled);
        peak_heap.push(peak);
    }

    eprintln!("Total run time: {} ms\n", t_total.elapsed().as_millis());
    eprintln!("Starting TUI — press q or Esc to quit...");
    std::thread::sleep(Duration::from_millis(300)); // let user read the output

    // ── Setup terminal ────────────────────────────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend  = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(results, peak_heap);

    // ── Event loop ────────────────────────────────────────────────────────────
    loop {
        terminal.draw(|f| ui(f, &app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Tab                      => app.next_tab(),
                    KeyCode::BackTab                  => app.prev_tab(),
                    KeyCode::Char('1')                => app.tab = 0,
                    KeyCode::Char('2')                => app.tab = 1,
                    KeyCode::Char('3')                => app.tab = 2,
                    KeyCode::Char('4')                => app.tab = 3,
                    KeyCode::Down                     => app.scroll_down(),
                    KeyCode::Up                       => app.scroll_up(),
                    _ => {}
                }
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}
