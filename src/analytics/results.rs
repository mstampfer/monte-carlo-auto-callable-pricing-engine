use std::time::Duration;
use crate::engine::batch_runner::PartialResult;

/// Final pricing result for a single strategy run.
#[derive(Debug, Clone)]
pub struct PriceResult {
    pub strategy_name: String,
    pub n_paths:       usize,
    pub price:         f64,
    pub std_err:       f64,
    pub ci_lower:      f64,
    pub ci_upper:      f64,
    pub wall_time:     Duration,
}

impl PriceResult {
    pub fn from_partial(
        strategy_name: impl Into<String>,
        partial:       &PartialResult,
        wall_time:     Duration,
    ) -> Self {
        let price = partial.price();
        let se    = partial.std_err();
        let (lo, hi) = partial.confidence_interval_95();
        Self {
            strategy_name: strategy_name.into(),
            n_paths:       partial.n_paths,
            price,
            std_err: se,
            ci_lower: lo,
            ci_upper: hi,
            wall_time,
        }
    }

    pub fn speedup_vs(&self, baseline_ms: f64) -> f64 {
        baseline_ms / self.wall_time.as_secs_f64() / 1000.0
    }
}

/// Collection of results from all strategies — rendered as a benchmark table.
#[derive(Debug, Default)]
pub struct BenchmarkReport {
    pub results: Vec<PriceResult>,
}

impl BenchmarkReport {
    pub fn add(&mut self, result: PriceResult) {
        self.results.push(result);
    }

    /// Print the benchmark table to stdout.
    pub fn print_table(&self) {
        // Find baseline (S1 naive_spawn)
        let baseline_ms = self.results.first()
            .map(|r| r.wall_time.as_millis() as f64)
            .unwrap_or(1.0);

        println!("\n╔{:═<32}╦{:═<10}╦{:═<11}╦{:═<10}╦{:═<18}╦{:═<9}╗",
            "", "", "", "", "", "");
        println!("║ {:<30} ║ {:>8} ║ {:>9} ║ {:>8} ║ {:^16} ║ {:>7} ║",
            "Strategy", "Paths", "Time (ms)", "Price", "95% CI", "Speedup");
        println!("╠{:═<32}╬{:═<10}╬{:═<11}╬{:═<10}╬{:═<18}╬{:═<9}╣",
            "", "", "", "", "", "");

        for r in &self.results {
            let speedup = baseline_ms / r.wall_time.as_millis() as f64;
            println!(
                "║ {:<30} ║ {:>8} ║ {:>9.0} ║ {:>8.3} ║ [{:>6.2},{:>6.2}] ║ {:>6.1}× ║",
                r.strategy_name,
                format_paths(r.n_paths),
                r.wall_time.as_millis(),
                r.price,
                r.ci_lower,
                r.ci_upper,
                speedup,
            );
        }

        println!("╚{:═<32}╩{:═<10}╩{:═<11}╩{:═<10}╩{:═<18}╩{:═<9}╝",
            "", "", "", "", "", "");
    }
}

fn format_paths(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
