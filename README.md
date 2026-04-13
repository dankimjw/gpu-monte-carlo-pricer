# GPU-Accelerated Monte Carlo Option Pricing Engine

A CUDA-based Monte Carlo simulation engine for pricing financial options with GPU acceleration. Final project for EN605.617 GPU Programming (JHU).

## Features

- **Option Types** — European, Asian, Barrier (Up/Down-and-Out), Digital (Call/Put)
- **Monte Carlo Simulation** — Millions of price paths simulated in parallel on GPU
- **Black-Scholes Validation** — Analytical pricing for European options to verify accuracy
- **Greeks** — Delta, Gamma, Vega, Theta, Rho via bump-and-reprice
- **Merton Jump-Diffusion** — Model crash/black swan events with configurable jump parameters
- **CUDA Streams** — Concurrent portfolio pricing (8 options on 4 streams)
- **GPU Visualization** — Spaghetti plots and payoff histograms rendered directly on GPU (PPM output)
- **Interactive Dashboard** — ncurses-based real-time parameter adjustment with ASCII sparklines
- **CPU Baseline** — Side-by-side comparison with single-threaded CPU implementation
- **Benchmarking** — Block-size sweeps, path-count scaling, CPU vs GPU timing

## Requirements

- NVIDIA GPU (developed on RTX 3060 Ti, compute capability 8.6)
- CUDA Toolkit 11.5+
- GCC 11+
- ncurses (for dashboard mode)

## Build & Run

```bash
make all
./mc_pricer --help
```

### Quick Examples

```bash
# European call with default params
./mc_pricer

# SPY preset
./mc_pricer --preset SPY

# BTC with jump-diffusion
./mc_pricer --preset BTC --jumps btc

# Asian option
./mc_pricer --type asian

# Interactive dashboard
./mc_pricer --dashboard

# Benchmark mode
./mc_pricer --benchmark
```

## Project Structure

```
├── src/
│   ├── main.cu              # CLI parsing, entry point
│   ├── monte_carlo.cu       # MC simulation kernels (European, Asian, Barrier, Digital)
│   ├── reduction.cu         # Shared-memory parallel reduction
│   ├── black_scholes.cu     # Analytical Black-Scholes pricing
│   ├── greeks.cu            # Greeks via bump-and-reprice
│   ├── streams.cu           # Multi-stream portfolio pricing
│   ├── render_ppm.cu        # GPU spaghetti plot + histogram rendering
│   ├── cpu_baseline.cpp     # CPU reference implementation
│   └── dashboard.cu         # ncurses interactive dashboard
├── include/
│   └── option_params.h      # Option parameter structs, constants
├── benchmark/
│   └── run_benchmarks.sh    # Automated benchmark script
├── output/                  # Sample output images and benchmark CSVs
├── Makefile
└── proposal.md
```

## Performance

- ~467x speedup over CPU baseline
- < 0.05% error vs Black-Scholes analytical price
- Optimal block size: 256 threads (605 Mpaths/sec)

## Author

Daniel Kim — Johns Hopkins University, EN605.617 GPU Programming
