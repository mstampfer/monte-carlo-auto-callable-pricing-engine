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
//!   3        — Convergence & Comparison
//!   q / Esc  — Quit
//!   ↑ / ↓   — Scroll (Tab 1) or select strategy (Tabs 2/3)

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
    analytics::{BatchEvent, ProfiledResult},
    concurrency::{run_simulation, ConcurrencyStrategy},
    domain::{BlackScholes, DualTimeGrid, MarketData, AutoCallable},
    engine::MonteCarloEngine,
};

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
    /// Active tab index: 0 = Timeline, 1 = Batch Analysis, 2 = Convergence
    tab: usize,
    /// Selected strategy index (used by tabs 2 & 3, and for convergence overlay in tab 3)
    selected: usize,
    /// Scroll offset for Tab 1 Gantt (lines)
    scroll: usize,
    /// All profiling results, one per strategy
    results: Vec<ProfiledResult>,
}

impl App {
    fn new(results: Vec<ProfiledResult>) -> Self {
        Self { tab: 0, selected: 0, scroll: 0, results }
    }

    fn next_tab(&mut self) {
        self.tab = (self.tab + 1) % 3;
    }

    fn prev_tab(&mut self) {
        self.tab = (self.tab + 2) % 3;
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

fn render_tab2(f: &mut Frame, area: Rect, app: &App) {
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

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // histogram
            Constraint::Length(threads.len() as u16 + 4),  // thread matrix
            Constraint::Min(3),      // completion order
        ])
        .split(chunks[1]);

    // ── Duration histogram ───────────────────────────────────────────────────
    let durations_ms: Vec<f64> = result.events.iter()
        .map(|e| e.duration.as_secs_f64() * 1000.0)
        .collect();
    let min_ms = durations_ms.iter().cloned().fold(f64::MAX, f64::min);
    let max_ms = durations_ms.iter().cloned().fold(f64::MIN, f64::max);

    // Build histogram with 8 buckets
    let n_buckets = 8usize;
    let range = (max_ms - min_ms).max(0.001);
    let mut buckets = vec![0u64; n_buckets];
    for &d in &durations_ms {
        let bucket = ((d - min_ms) / range * (n_buckets - 1) as f64).round() as usize;
        buckets[bucket.min(n_buckets - 1)] += 1;
    }

    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(
            format!(" Duration histogram  [{:.1}ms – {:.1}ms] ", min_ms, max_ms)
        ))
        .data(&buckets)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, right_chunks[0]);

    // ── Batch-to-thread matrix ────────────────────────────────────────────────
    let n_batches = result.events.len();
    let col_w = 3usize;
    let mut header_cells: Vec<Cell> = vec![Cell::from("Thread")];
    for b in 0..n_batches {
        header_cells.push(Cell::from(format!("{:>2}", b)).style(
            Style::default().fg(BATCH_COLORS[b % BATCH_COLORS.len()])
        ));
    }

    let mut rows: Vec<Row> = Vec::new();
    rows.push(Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)));

    for (tid_idx, tid) in threads.iter().enumerate() {
        let mut cells: Vec<Cell> = vec![Cell::from(format!("T{}", tid_idx))];
        for b in 0..n_batches {
            // Find event for batch b on this thread
            let ran = result.events.iter().any(|e| e.batch_id == b && &e.thread_id == tid);
            cells.push(Cell::from(if ran { " ●" } else { "  " }).style(
                if ran {
                    Style::default().fg(BATCH_COLORS[b % BATCH_COLORS.len()])
                } else {
                    Style::default().fg(Color::DarkGray)
                }
            ));
        }
        rows.push(Row::new(cells));
    }

    let mut widths: Vec<Constraint> = vec![Constraint::Length(6)];
    for _ in 0..n_batches {
        widths.push(Constraint::Length(col_w as u16));
    }

    let matrix = Table::new(rows, &widths)
        .block(Block::default().borders(Borders::ALL).title(" Batch-to-thread matrix "));
    f.render_widget(matrix, right_chunks[1]);

    // ── Completion order ──────────────────────────────────────────────────────
    let order = completion_order(&result.events);
    let order_str: Vec<String> = order.iter().map(|id| id.to_string()).collect();
    let completion_para = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Completion order: ", Style::default().fg(Color::White)),
            Span::styled(order_str.join("  "), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled(
                "(out-of-order = work-stealing / async scheduling visible)",
                Style::default().fg(Color::DarkGray),
            )
        ]),
    ]).block(Block::default().borders(Borders::ALL).title(" Completion Order "));
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
            let spark_data: Vec<u64> = conv.iter()
                .map(|&p| ((p - min_p) / range * 1000.0) as u64)
                .collect();

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
    ]).style(Style::default().fg(Color::Yellow));

    let rows: Vec<Row> = app.results.iter().enumerate().map(|(i, r)| {
        let ms = r.price_result.wall_time.as_millis();
        let speedup = baseline_ms / ms as f64;
        let (cpu_eff, imbalance) = compute_metrics(r);

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
        ]).style(style)
    }).collect();

    let widths = [
        Constraint::Length(30),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(8),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(
            " Strategy Comparison  [↑↓ select convergence] "
        ))
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_widget(table, chunks[1]);
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
        Line::from(" 3: Convergence & Compare "),
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
        1 => render_tab2(f, chunks[1], app),
        2 => render_tab3(f, chunks[1], app),
        _ => {}
    }

    // Help bar
    let help = Paragraph::new(Line::from(vec![
        Span::styled(" Tab/1/2/3 ", Style::default().fg(Color::Cyan)),
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
    for strategy in &selected_strategies {
        eprint!("  {:35} ... ", strategy.name());
        let profiled = run_simulation(*strategy, Arc::clone(&engine), n_paths, N_THREADS, n_batches, GLOBAL_SEED).await;
        eprintln!("{} ms  price = {:.3}",
            profiled.price_result.wall_time.as_millis(),
            profiled.price_result.price);
        results.push(profiled);
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

    let mut app = App::new(results);

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
