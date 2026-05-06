use std::sync::Arc;
use std::process;

use hsbc_monte_carlo_auto_callable::{
    analytics::BenchmarkReport,
    concurrency::{run_simulation, ConcurrencyStrategy},
    domain::{BlackScholes, DualTimeGrid, MarketData, AutoCallable},
    engine::MonteCarloEngine,
};

// ── Simulation parameters ────────────────────────────────────────────────────

const DEFAULT_N_PATHS:     usize = 200_000;
const N_THREADS:           usize = 8;
const GLOBAL_SEED:         u64   = 42;

// Instrument parameters
const SPOT_INITIAL:        f64   = 100.0;
const NOTIONAL:            f64   = 100.0;
const VOLATILITY:          f64   = 0.25;   // 25% annualised vol
const RISK_FREE_RATE:      f64   = 0.05;   // 5% risk-free rate
const DIVIDEND_YIELD:      f64   = 0.02;   // 2% continuous dividend yield
const BARRIER_CALL_FRAC:   f64   = 1.00;   // Autocall at 100% of S_0
const BARRIER_KI_FRAC:     f64   = 0.70;   // Knock-in at 70% of S_0
const MATURITY_YEARS:      f64   = 1.0;
const N_MONTHLY:           usize = 12;     // Monthly autocall observations
const BUSINESS_DAYS_PER_MONTH: usize = 21; // ~21 daily sub-steps per month

// ─────────────────────────────────────────────────────────────────────────────

/// All variants of the hybrid (tokio controller + rayon workers) strategy.
/// Add new tweaks here as they land; the harness will profile them all in
/// the same run for direct comparison.
const ALL_VARIANTS: &[ConcurrencyStrategy] = &[
    ConcurrencyStrategy::RayonBridgeBaseline,
];

fn parse_npaths(s: &str) -> Result<usize, String> {
    // Accept Rust-style underscore separators: "75_000_000", "75000000"
    let cleaned: String = s.chars().filter(|&c| c != '_').collect();
    cleaned.parse::<usize>()
        .map_err(|_| format!("'{}' is not a valid integer", s))
        .and_then(|n| {
            if n == 0 {
                Err("--npaths must be greater than zero".into())
            } else {
                Ok(n)
            }
        })
}

fn parse_variant(arg: &str) -> Option<ConcurrencyStrategy> {
    match arg.to_lowercase().as_str() {
        "1" | "baseline" | "rayon_bridge_baseline" =>
            Some(ConcurrencyStrategy::RayonBridgeBaseline),
        _ => None,
    }
}

fn print_usage() {
    eprintln!("Usage: monte-carlo-auto-callable [--npaths N] [VARIANT]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --npaths N    Number of Monte Carlo paths (default: {DEFAULT_N_PATHS})");
    eprintln!("                Underscores accepted: 75_000_000 or 75000000");
    eprintln!();
    eprintln!("  VARIANT  (omit to run all variants)");
    eprintln!("    baseline   rayon_bridge_baseline   (alias: 1)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  cargo run --release                             # all variants, 200_000 paths");
    eprintln!("  cargo run --release -- --npaths 500_000         # all variants, 500_000 paths");
    eprintln!("  cargo run --release -- --npaths 75_000_000 baseline   # baseline, 75M paths");
}

#[tokio::main]
async fn main() {
    // Parse arguments: [--npaths N] [VARIANT]
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = raw.as_slice();

    if args.first().map(|s| s == "--help" || s == "-h").unwrap_or(false) {
        print_usage();
        process::exit(0);
    }

    // --npaths N  (accepts "500_000" or "500000")
    let n_paths: usize = if args.first().map(|s| s == "--npaths").unwrap_or(false) {
        match args.get(1) {
            Some(val) => match parse_npaths(val) {
                Ok(n)  => { args = &args[2..]; n }
                Err(e) => { eprintln!("error: {e}"); eprintln!(); print_usage(); process::exit(1); }
            },
            None => {
                eprintln!("error: --npaths requires a value");
                eprintln!();
                print_usage();
                process::exit(1);
            }
        }
    } else {
        DEFAULT_N_PATHS
    };

    // Optional variant name/number
    let selected: Vec<ConcurrencyStrategy> = match args {
        [] => ALL_VARIANTS.to_vec(),
        [arg] => match parse_variant(arg) {
            Some(s) => vec![s],
            None => {
                eprintln!("error: unknown variant '{arg}'");
                eprintln!();
                print_usage();
                process::exit(1);
            }
        },
        _ => {
            eprintln!("error: unexpected arguments: {}", args.join(" "));
            eprintln!();
            print_usage();
            process::exit(1);
        }
    };

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║    HSBC Monte Carlo Auto-Callable Pricing Engine — Hybrid Runtime        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Architecture: tokio controller + rayon worker pool");
    println!("  Instrument  : Autocallable note, maturity = {MATURITY_YEARS}Y");
    println!("  S_0         : {SPOT_INITIAL}, σ = {:.0}%, r = {:.0}%, q = {:.0}%",
        VOLATILITY * 100.0, RISK_FREE_RATE * 100.0, DIVIDEND_YIELD * 100.0);
    println!("  Barriers    : Call = {:.0}% of S_0, KI = {:.0}% of S_0",
        BARRIER_CALL_FRAC * 100.0, BARRIER_KI_FRAC * 100.0);
    println!("  Grid        : {N_MONTHLY} monthly x {BUSINESS_DAYS_PER_MONTH} daily sub-steps");
    println!("  Paths       : {n_paths}, Threads = {N_THREADS}");
    println!("  Method      : One-Step Survival (Glasserman-Staum) + Brownian Bridge");
    println!();

    let market_data = MarketData::new(SPOT_INITIAL, VOLATILITY, RISK_FREE_RATE, DIVIDEND_YIELD);

    let coupons: Vec<f64> = (1..=N_MONTHLY)
        .map(|k| 0.05 * k as f64 / N_MONTHLY as f64)
        .collect();
    let obs_dates: Vec<f64> = (1..=N_MONTHLY)
        .map(|k| MATURITY_YEARS * k as f64 / N_MONTHLY as f64)
        .collect();

    let product = AutoCallable::new(
        SPOT_INITIAL,
        NOTIONAL,
        BARRIER_CALL_FRAC,
        BARRIER_KI_FRAC,
        coupons,
        obs_dates,
    );

    let propagator   = Arc::new(BlackScholes::new(&market_data));
    let time_grid    = DualTimeGrid::new(MATURITY_YEARS, N_MONTHLY, BUSINESS_DAYS_PER_MONTH);
    let barrier_call = SPOT_INITIAL * BARRIER_CALL_FRAC;
    let barrier_ki   = SPOT_INITIAL * BARRIER_KI_FRAC;

    let engine: Arc<MonteCarloEngine<AutoCallable, BlackScholes>> = Arc::new(
        MonteCarloEngine::new(
            product,
            propagator,
            market_data,
            time_grid,
            barrier_call,
            barrier_ki,
            N_MONTHLY,
            BUSINESS_DAYS_PER_MONTH,
        )
    );

    let mut report = BenchmarkReport::default();

    for variant in &selected {
        print!("  Running {:30} ... ", variant.name());
        let profiled = run_simulation(
            *variant,
            Arc::clone(&engine),
            n_paths,
            N_THREADS,
            N_THREADS,  // n_batches == n_threads for the benchmark binary
            GLOBAL_SEED,
        ).await;
        println!("{:.0} ms  price = {:.3}",
            profiled.price_result.wall_time.as_millis(),
            profiled.price_result.price);
        report.add(profiled.price_result);
    }

    report.print_table();

    // ── AmericanOption stub — demonstrates plug-in genericity ────────────────
    println!("\n── AmericanOption stub (Bermudan approximation, same engine) ────────────");
    price_american_option(market_data, (n_paths / 10).max(1), N_THREADS, GLOBAL_SEED).await;
}

async fn price_american_option(
    market_data: MarketData,
    n_paths:     usize,
    n_threads:   usize,
    seed:        u64,
) {
    use hsbc_monte_carlo_auto_callable::domain::AmericanOption;

    let obs_dates: Vec<f64> = (1..=N_MONTHLY)
        .map(|k| MATURITY_YEARS * k as f64 / N_MONTHLY as f64)
        .collect();

    let option = AmericanOption::new(
        SPOT_INITIAL * 1.05, // 5% OTM call, strike = 105
        true,
        1.0,  // 1 unit notional → payoff = max(S - 105, 0) per unit
        obs_dates,
    );

    let propagator = Arc::new(BlackScholes::new(&market_data));
    let time_grid  = DualTimeGrid::new(MATURITY_YEARS, N_MONTHLY, BUSINESS_DAYS_PER_MONTH);

    let engine = Arc::new(MonteCarloEngine::new(
        option,
        propagator,
        market_data,
        time_grid,
        f64::INFINITY, // no autocall barrier for plain option
        0.0,           // no knock-in
        N_MONTHLY,
        BUSINESS_DAYS_PER_MONTH,
    ));

    let profiled = run_simulation(
        ConcurrencyStrategy::RayonBridgeBaseline,
        engine,
        n_paths,
        n_threads,
        n_threads,
        seed,
    ).await;

    println!(
        "  AmericanOption (Bermudan approx): price = {:.3},  95% CI = [{:.3}, {:.3}]",
        profiled.price_result.price,
        profiled.price_result.ci_lower,
        profiled.price_result.ci_upper,
    );
}
