//! SVG export for profiler visualisations.
//!
//! Generates per-strategy SVGs for Thread Timelines, Duration Histograms,
//! Batch-to-thread mappings, and Completion Order panels.
//! No external crate dependencies — builds SVG XML directly.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::profiling::{BatchEvent, ProfiledResult, unique_threads, compute_metrics};

// ── Layout constants ────────────────────────────────────────────────────────

const SVG_WIDTH: u32 = 600;
const LEFT_PAD: u32 = 60;
const RIGHT_PAD: u32 = 12;
const TOP_PAD: u32 = 12;
const HEADER_HEIGHT: u32 = 28;
const ROW_HEIGHT: u32 = 24;
const METRICS_HEIGHT: u32 = 22;
const BOTTOM_PAD: u32 = 8;
const BAR_HEIGHT: u32 = 18;
const BAR_WIDTH: u32 = SVG_WIDTH - LEFT_PAD - RIGHT_PAD;

const LINE_HEIGHT: u32 = 16;

// ── Colors (dark theme) ─────────────────────────────────────────────────────

const BG_COLOR: &str = "#1E1E1E";
const HEADER_COLOR: &str = "#FBBC04";
const METRICS_COLOR: &str = "#24C1E0";
const LABEL_COLOR: &str = "#9AA0A6";
const IDLE_COLOR: &str = "#3C4043";
const HIST_COLOR: &str = "#34A853";
const DIM_COLOR: &str = "#5F6368";

const BATCH_COLORS: [&str; 8] = [
    "#4285F4", "#34A853", "#EA4335", "#FBBC04",
    "#A142F4", "#24C1E0", "#8AB4F8", "#81C995",
];

// ── Helpers ─────────────────────────────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn svg_open(width: u32, height: u32) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="'Courier New', Courier, monospace" font-size="12"><rect width="100%" height="100%" fill="{BG_COLOR}" rx="6" />"#
    )
}

fn completion_order(events: &[BatchEvent]) -> Vec<usize> {
    let mut indexed: Vec<(usize, Duration)> = events.iter()
        .map(|e| (e.batch_id, e.start + e.duration))
        .collect();
    indexed.sort_by_key(|(_, t)| *t);
    indexed.into_iter().map(|(id, _)| id).collect()
}

fn global_histogram_range(results: &[ProfiledResult]) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for r in results {
        for e in &r.events {
            let ms = e.duration.as_secs_f64() * 1000.0;
            lo = lo.min(ms);
            hi = hi.max(ms);
        }
    }
    if lo > hi { (0.0, 1.0) } else { (lo, hi) }
}

/// Convert strategy name like "S1  naive_spawn" to "s1_naive_spawn".
pub fn strategy_filename(strategy_name: &str) -> String {
    let name = strategy_name.trim().to_lowercase();
    let cleaned: String = name.chars()
        .map(|c| if c == ' ' || c == '(' || c == ')' || c == ',' { '_' } else { c })
        .collect();
    let mut result = String::new();
    let mut prev_underscore = false;
    for c in cleaned.chars() {
        if c == '_' {
            if !prev_underscore { result.push('_'); }
            prev_underscore = true;
        } else {
            result.push(c);
            prev_underscore = false;
        }
    }
    result.trim_end_matches('_').to_string()
}

// ── 1. Thread Timeline ─────────────────────────────────────────────────────

/// Render a single strategy's Thread Timeline as an SVG string.
pub fn render_timeline(result: &ProfiledResult) -> String {
    let threads = unique_threads(&result.events);
    let (cpu_eff, imbalance) = compute_metrics(result);
    let wall_ns = result.price_result.wall_time.as_nanos() as f64;
    let wall_ms = result.price_result.wall_time.as_millis();
    let n_threads = threads.len() as u32;

    let svg_height = TOP_PAD + HEADER_HEIGHT + n_threads * ROW_HEIGHT + METRICS_HEIGHT + BOTTOM_PAD;
    let mut svg = String::with_capacity(4096);

    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} {} ms</text>"#,
        xml_escape(result.price_result.strategy_name.trim()), wall_ms);

    for (tid_idx, tid) in threads.iter().enumerate() {
        let y_row = TOP_PAD + HEADER_HEIGHT + tid_idx as u32 * ROW_HEIGHT;
        let bar_y = y_row + (ROW_HEIGHT - BAR_HEIGHT) / 2;

        let _ = write!(svg,
            r#"<rect x="{LEFT_PAD}" y="{bar_y}" width="{BAR_WIDTH}" height="{BAR_HEIGHT}" fill="{IDLE_COLOR}" rx="2" />"#);

        let label_y = y_row + 16;
        let _ = write!(svg,
            r#"<text x="12" y="{label_y}" fill="{LABEL_COLOR}" font-size="12">T{tid_idx}</text>"#);

        let thread_events: Vec<&BatchEvent> = result.events.iter()
            .filter(|e| e.thread_id == *tid)
            .collect();

        if wall_ns > 0.0 {
            for event in &thread_events {
                let start_ns = event.start.as_nanos() as f64;
                let dur_ns = event.duration.as_nanos() as f64;
                let x = LEFT_PAD as f64 + start_ns / wall_ns * BAR_WIDTH as f64;
                let w = (dur_ns / wall_ns * BAR_WIDTH as f64).max(1.0);
                let color = BATCH_COLORS[event.batch_id % BATCH_COLORS.len()];
                let _ = write!(svg,
                    r#"<rect x="{x:.1}" y="{bar_y}" width="{w:.1}" height="{BAR_HEIGHT}" fill="{color}" rx="2" />"#);
            }
        }
    }

    let metrics_y = TOP_PAD + HEADER_HEIGHT + n_threads * ROW_HEIGHT + 16;
    let _ = write!(svg,
        r#"<text x="12" y="{metrics_y}" fill="{METRICS_COLOR}" font-size="11">CPU eff: {:.0}%  Imbalance: {:.2}  Batches: {}  Price: {:.3}</text>"#,
        cpu_eff * 100.0, imbalance, result.events.len(), result.price_result.price);

    svg.push_str("</svg>");
    svg
}

// ── 2. Duration Histogram ──────────────────────────────────────────────────

/// Render a duration histogram SVG.  `hist_min`/`hist_max` set the shared
/// x-axis range across all strategies (from [`global_histogram_range`]).
pub fn render_histogram(result: &ProfiledResult, hist_min: f64, hist_max: f64) -> String {
    let name = result.price_result.strategy_name.trim();
    let n_buckets: usize = 80;
    let range = (hist_max - hist_min).max(0.001);

    let mut buckets = vec![0u64; n_buckets];
    for e in &result.events {
        let ms = e.duration.as_secs_f64() * 1000.0;
        let i = ((ms - hist_min) / range * (n_buckets - 1) as f64).round() as usize;
        buckets[i.min(n_buckets - 1)] += 1;
    }
    let y_max = buckets.iter().copied().max().unwrap_or(1).max(1);

    let hist_area_height: u32 = 120;
    let axis_height: u32 = 20;
    let svg_height = TOP_PAD + HEADER_HEIGHT + hist_area_height + axis_height + BOTTOM_PAD;
    let hist_left = LEFT_PAD;
    let hist_width = BAR_WIDTH;
    let col_w = hist_width as f64 / n_buckets as f64;

    let mut svg = String::with_capacity(4096);
    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} — Duration histogram</text>"#,
        xml_escape(name));

    let base_y = TOP_PAD + HEADER_HEIGHT + hist_area_height;

    // Bars
    for (i, &count) in buckets.iter().enumerate() {
        if count == 0 { continue; }
        let h = (count as f64 / y_max as f64 * hist_area_height as f64).max(1.0);
        let x = hist_left as f64 + i as f64 * col_w;
        let y = base_y as f64 - h;
        let _ = write!(svg,
            r#"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{h:.1}" fill="{HIST_COLOR}" />"#,
            (col_w - 0.5).max(1.0));
    }

    // X-axis labels
    let axis_y = base_y + 14;
    let _ = write!(svg,
        r#"<text x="{hist_left}" y="{axis_y}" fill="{LABEL_COLOR}" font-size="10">{hist_min:.1}ms</text>"#);
    let mid_ms = (hist_min + hist_max) / 2.0;
    let mid_x = hist_left as f64 + hist_width as f64 / 2.0;
    let _ = write!(svg,
        r#"<text x="{mid_x:.0}" y="{axis_y}" fill="{LABEL_COLOR}" font-size="10" text-anchor="middle">{mid_ms:.1}ms</text>"#);
    let right_x = hist_left + hist_width;
    let _ = write!(svg,
        r#"<text x="{right_x}" y="{axis_y}" fill="{LABEL_COLOR}" font-size="10" text-anchor="end">{hist_max:.1}ms</text>"#);

    // Y-axis label
    let _ = write!(svg,
        r#"<text x="8" y="{}" fill="{LABEL_COLOR}" font-size="10">{y_max}</text>"#,
        TOP_PAD + HEADER_HEIGHT + 10);

    svg.push_str("</svg>");
    svg
}

// ── 3. Batch-to-thread mapping ─────────────────────────────────────────────

/// Render a batch-to-thread mapping SVG.
pub fn render_thread_mapping(result: &ProfiledResult) -> String {
    let name = result.price_result.strategy_name.trim();
    let threads = unique_threads(&result.events);

    // header + title line + one line per thread + bottom pad
    let n_lines = threads.len() as u32 + 1;
    let svg_height = TOP_PAD + HEADER_HEIGHT + n_lines * LINE_HEIGHT + BOTTOM_PAD;

    let mut svg = String::with_capacity(4096);
    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} — Batch-to-thread mapping</text>"#,
        xml_escape(name));

    // Column title
    let title_y = TOP_PAD + HEADER_HEIGHT + 12;
    let _ = write!(svg,
        r#"<text x="12" y="{title_y}" fill="{LABEL_COLOR}" font-size="10" font-weight="bold">Thread   Batches (sorted by start time)</text>"#);

    for (tid_idx, tid) in threads.iter().enumerate() {
        let y = TOP_PAD + HEADER_HEIGHT + (tid_idx as u32 + 1) * LINE_HEIGHT + 12;

        // Thread label
        let _ = write!(svg,
            r#"<text x="12" y="{y}" fill="{LABEL_COLOR}" font-size="11">T{tid_idx:<3}</text>"#);

        // Batch IDs for this thread, sorted by start time
        let mut batch_events: Vec<(usize, Duration)> = result.events.iter()
            .filter(|e| e.thread_id == *tid)
            .map(|e| (e.batch_id, e.start))
            .collect();
        batch_events.sort_by_key(|&(_, s)| s);

        let mut x_offset: f64 = 52.0;
        for (bid, _) in &batch_events {
            let color = BATCH_COLORS[bid % BATCH_COLORS.len()];
            let _ = write!(svg,
                r#"<text x="{x_offset:.0}" y="{y}" fill="{color}" font-size="11">{bid:>3}</text>"#);
            x_offset += 28.0;
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── 4. Completion Order ────────────────────────────────────────────────────

/// Render a completion order SVG.
pub fn render_completion_order(result: &ProfiledResult) -> String {
    let name = result.price_result.strategy_name.trim();
    let order = completion_order(&result.events);
    let ids_per_row: usize = 20;

    let n_rows = order.len().div_ceil(ids_per_row);
    // header + title + data rows + footer note
    let n_lines = n_rows as u32 + 2;
    let svg_height = TOP_PAD + HEADER_HEIGHT + n_lines * LINE_HEIGHT + BOTTOM_PAD;

    let mut svg = String::with_capacity(2048);
    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} — Completion order</text>"#,
        xml_escape(name));

    let title_y = TOP_PAD + HEADER_HEIGHT + 12;
    let _ = write!(svg,
        r#"<text x="12" y="{title_y}" fill="{LABEL_COLOR}" font-size="10" font-weight="bold">Batch IDs by finish time:</text>"#);

    for (row_idx, chunk) in order.chunks(ids_per_row).enumerate() {
        let y = TOP_PAD + HEADER_HEIGHT + (row_idx as u32 + 1) * LINE_HEIGHT + 12;
        let mut x_offset: f64 = 16.0;
        for &id in chunk {
            let color = BATCH_COLORS[id % BATCH_COLORS.len()];
            let _ = write!(svg,
                r#"<text x="{x_offset:.0}" y="{y}" fill="{color}" font-size="11">{id:>3}</text>"#);
            x_offset += 28.0;
        }
    }

    // Footer note
    let footer_y = TOP_PAD + HEADER_HEIGHT + (n_rows as u32 + 1) * LINE_HEIGHT + 12;
    let _ = write!(svg,
        r#"<text x="12" y="{footer_y}" fill="{DIM_COLOR}" font-size="9">(out-of-order = work-stealing / async scheduling visible)</text>"#);

    svg.push_str("</svg>");
    svg
}

// ── 5. Convergence ─────────────────────────────────────────────────────────

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

/// Render a price convergence SVG — running mean price after each batch completes,
/// plus a summary line with key strategy metrics.
pub fn render_convergence(result: &ProfiledResult) -> String {
    let name = result.price_result.strategy_name.trim();
    let conv = compute_convergence(result);
    let (cpu_eff, imbalance) = compute_metrics(result);
    let wall_ms = result.price_result.wall_time.as_millis();
    let total_alloc: u64 = result.events.iter().map(|e| e.alloc_bytes as u64).sum();

    let chart_height: u32 = 100;
    let summary_lines: u32 = 3;
    let svg_height = TOP_PAD + HEADER_HEIGHT + chart_height + 8 + summary_lines * LINE_HEIGHT + BOTTOM_PAD;

    let mut svg = String::with_capacity(4096);
    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let final_price = conv.last().copied().unwrap_or(0.0);
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} — Convergence</text>"#,
        xml_escape(name));

    if !conv.is_empty() {
        let min_p = conv.iter().cloned().fold(f64::MAX, f64::min);
        let max_p = conv.iter().cloned().fold(f64::MIN, f64::max);
        let range = (max_p - min_p).max(0.001);

        let chart_top = TOP_PAD + HEADER_HEIGHT;
        let chart_base = chart_top + chart_height;

        // Background
        let _ = write!(svg,
            r#"<rect x="{LEFT_PAD}" y="{chart_top}" width="{BAR_WIDTH}" height="{chart_height}" fill="{IDLE_COLOR}" rx="2" />"#);

        // Build polyline points
        let n = conv.len();
        let mut points = String::new();
        for (i, &p) in conv.iter().enumerate() {
            let x = LEFT_PAD as f64 + (i as f64 / (n - 1).max(1) as f64) * BAR_WIDTH as f64;
            let y = chart_base as f64 - ((p - min_p) / range) * chart_height as f64;
            if !points.is_empty() { points.push(' '); }
            let _ = write!(points, "{x:.1},{y:.1}");
        }
        let _ = write!(svg,
            "<polyline points=\"{points}\" fill=\"none\" stroke=\"{METRICS_COLOR}\" stroke-width=\"2\" />");

        // Y-axis range labels
        let _ = write!(svg,
            r#"<text x="8" y="{}" fill="{LABEL_COLOR}" font-size="9">{max_p:.2}</text>"#,
            chart_top + 10);
        let _ = write!(svg,
            r#"<text x="8" y="{}" fill="{LABEL_COLOR}" font-size="9">{min_p:.2}</text>"#,
            chart_base - 2);

        // Final price marker
        let final_y = chart_base as f64 - ((final_price - min_p) / range) * chart_height as f64;
        let final_x = LEFT_PAD as f64 + BAR_WIDTH as f64;
        let _ = write!(svg,
            "<circle cx=\"{final_x:.1}\" cy=\"{final_y:.1}\" r=\"3\" fill=\"{HEADER_COLOR}\" />");
    }

    // Summary metrics
    let sy = TOP_PAD + HEADER_HEIGHT + chart_height + 20;
    let lines = [
        format!("Price: {final_price:.3}  CI: [{:.3}, {:.3}]",
            result.price_result.ci_lower, result.price_result.ci_upper),
        format!("Wall: {wall_ms} ms  CPU eff: {:.0}%  Imbalance: {:.2}",
            cpu_eff * 100.0, imbalance),
        format!("Batches: {}  Alloc: {}",
            result.events.len(), format_bytes(total_alloc)),
    ];
    for (i, line) in lines.iter().enumerate() {
        let y = sy + i as u32 * LINE_HEIGHT;
        let _ = write!(svg,
            r#"<text x="12" y="{y}" fill="{METRICS_COLOR}" font-size="11">{line}</text>"#);
    }

    svg.push_str("</svg>");
    svg
}

// ── 6. Memory Analysis ─────────────────────────────────────────────────────

const SPARK_HEIGHT: u32 = 80;
const SPARK_COLOR_BYTES: &str = "#A142F4";
const SPARK_COLOR_COUNT: &str = "#24C1E0";

fn format_bytes(bytes: u64) -> String {
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

/// Render a sparkline bar chart into the SVG at the given y offset.
/// Returns the y position after the chart (for stacking).
fn render_sparkline(
    svg: &mut String,
    data: &[u64],
    y_top: u32,
    global_max: u64,
    color: &str,
    title: &str,
) -> u32 {
    let title_y = y_top + 14;
    let _ = write!(svg,
        r#"<text x="12" y="{title_y}" fill="{LABEL_COLOR}" font-size="11" font-weight="bold">{title}</text>"#);

    let chart_top = y_top + 20;
    let base_y = chart_top + SPARK_HEIGHT;
    let n = data.len().max(1);
    let col_w = BAR_WIDTH as f64 / n as f64;
    let y_max = global_max.max(1) as f64;

    // Background
    let _ = write!(svg,
        r#"<rect x="{LEFT_PAD}" y="{chart_top}" width="{BAR_WIDTH}" height="{SPARK_HEIGHT}" fill="{IDLE_COLOR}" rx="2" />"#);

    for (i, &val) in data.iter().enumerate() {
        if val == 0 { continue; }
        let h = (val as f64 / y_max * SPARK_HEIGHT as f64).max(1.0);
        let x = LEFT_PAD as f64 + i as f64 * col_w;
        let y = base_y as f64 - h;
        let _ = write!(svg,
            r#"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{h:.1}" fill="{color}" />"#,
            (col_w - 0.5).max(1.0));
    }

    base_y + 4
}

/// Render a combined memory analysis SVG with alloc bytes + alloc count sparklines
/// and a summary section.  `global_max_bytes` / `global_max_count` set the shared
/// y-axis scale across all strategies.
pub fn render_memory(
    result: &ProfiledResult,
    global_max_bytes: u64,
    global_max_count: u64,
) -> String {
    let name = result.price_result.strategy_name.trim();

    let raw_bytes: Vec<u64> = result.events.iter().map(|e| e.alloc_bytes as u64).collect();
    let raw_count: Vec<u64> = result.events.iter().map(|e| e.alloc_count as u64).collect();

    let max_bytes = raw_bytes.iter().copied().max().unwrap_or(0);
    let max_count = raw_count.iter().copied().max().unwrap_or(0);
    let total_bytes: u64 = raw_bytes.iter().sum();
    let total_count: u64 = raw_count.iter().sum();
    let n_batches = result.events.len().max(1) as u64;
    let avg_bytes = total_bytes / n_batches;
    let avg_count = total_count / n_batches;

    // Layout: header + sparkline1 + gap + sparkline2 + gap + summary text
    let spark_block = 20 + SPARK_HEIGHT + 4; // title + chart + pad
    let summary_lines: u32 = 4;
    let svg_height = TOP_PAD + HEADER_HEIGHT + spark_block * 2 + 8 + summary_lines * LINE_HEIGHT + BOTTOM_PAD;

    let mut svg = String::with_capacity(8192);
    svg.push_str(&svg_open(SVG_WIDTH, svg_height));

    let header_y = TOP_PAD + 18;
    let _ = write!(svg,
        r#"<text x="12" y="{header_y}" fill="{HEADER_COLOR}" font-size="14" font-weight="bold">{} — Memory analysis</text>"#,
        xml_escape(name));

    // Sparkline 1: alloc bytes
    let y1 = TOP_PAD + HEADER_HEIGHT;
    let bytes_title = format!("Allocation volume per batch  [max: {}]", format_bytes(max_bytes));
    let after_bytes = render_sparkline(&mut svg, &raw_bytes, y1, global_max_bytes, SPARK_COLOR_BYTES, &bytes_title);

    // Sparkline 2: alloc count
    let y2 = after_bytes + 8;
    let count_title = format!("Allocation count per batch  [max: {}]", max_count);
    let after_count = render_sparkline(&mut svg, &raw_count, y2, global_max_count, SPARK_COLOR_COUNT, &count_title);

    // Summary text
    let sy = after_count + 12;
    let lines = [
        format!("Total alloc bytes:  {}  ({} avg/batch)", format_bytes(total_bytes), format_bytes(avg_bytes)),
        format!("Total alloc count:  {}  ({} avg/batch)", total_count, avg_count),
        format!("Batches:  {}", result.events.len()),
    ];
    for (i, line) in lines.iter().enumerate() {
        let y = sy + i as u32 * LINE_HEIGHT;
        let _ = write!(svg,
            r#"<text x="12" y="{y}" fill="{METRICS_COLOR}" font-size="11">{line}</text>"#);
    }

    svg.push_str("</svg>");
    svg
}

// ── Batch export ────────────────────────────────────────────────────────────

/// Export all SVGs (timelines, histograms, thread mappings, completion order)
/// to subdirectories under `dir`.  Returns the list of written file paths.
pub fn export_all(
    results: &[ProfiledResult],
    dir: &Path,
) -> std::io::Result<Vec<PathBuf>> {
    let timeline_dir  = dir.join("timelines");
    let histogram_dir = dir.join("histograms");
    let mapping_dir   = dir.join("thread_mappings");
    let order_dir     = dir.join("completion_order");
    let memory_dir    = dir.join("memory");
    let converge_dir  = dir.join("convergence");

    std::fs::create_dir_all(&timeline_dir)?;
    std::fs::create_dir_all(&histogram_dir)?;
    std::fs::create_dir_all(&mapping_dir)?;
    std::fs::create_dir_all(&order_dir)?;
    std::fs::create_dir_all(&memory_dir)?;
    std::fs::create_dir_all(&converge_dir)?;

    let (hist_min, hist_max) = global_histogram_range(results);

    let global_max_bytes: u64 = results.iter()
        .flat_map(|r| r.events.iter().map(|e| e.alloc_bytes as u64))
        .max().unwrap_or(1);
    let global_max_count: u64 = results.iter()
        .flat_map(|r| r.events.iter().map(|e| e.alloc_count as u64))
        .max().unwrap_or(1);

    let mut paths = Vec::new();

    for result in results {
        let base = strategy_filename(&result.price_result.strategy_name);

        let p = timeline_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_timeline(result))?;
        paths.push(p);

        let p = histogram_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_histogram(result, hist_min, hist_max))?;
        paths.push(p);

        let p = mapping_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_thread_mapping(result))?;
        paths.push(p);

        let p = order_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_completion_order(result))?;
        paths.push(p);

        let p = memory_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_memory(result, global_max_bytes, global_max_count))?;
        paths.push(p);

        let p = converge_dir.join(format!("{base}.svg"));
        std::fs::write(&p, render_convergence(result))?;
        paths.push(p);
    }

    Ok(paths)
}
