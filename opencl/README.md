> **DO NOT MERGE** — This branch contains a standalone homework assignment (OpenCL). It is kept separate from the final project on `main` intentionally.

# OpenCL Monte Carlo Option Pricer

An OpenCL implementation of a Monte Carlo European option pricing engine.  Prices
European call and put options under the Black-Scholes (GBM) model using massively
parallel GPU simulation, with a CPU baseline and analytical Black-Scholes price for
validation.

## OpenCL Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Platform / device discovery | `src/ocl_helpers.cpp` |
| Context and command queue creation | `src/ocl_helpers.cpp` |
| Kernel compilation from `.cl` source | `src/ocl_helpers.cpp` — `createProgram()` |
| Buffers (`clCreateBuffer`) | `src/main.cpp` — payoff buffer |
| **Sub-buffers** (`clCreateSubBuffer`) | `src/main.cpp` — splits payoff array in two halves |
| **Local (shared) memory** | `kernels/reduction.cl` — `__local float *sdata` |
| **float4 vector types** | `kernels/mc_sim.cl` — `rand_normal4()` |
| **Event-based profiling** | `src/main.cpp` — `CL_PROFILING_COMMAND_*` |
| Multiple kernels | `mc_simulate`, `sum_reduce`, `variance_diff` |
| NDRange work-groups | Configurable via `-w` flag |
| Error handling | `checkErr()` pattern throughout |

## Prerequisites

- OpenCL 1.2+ runtime and headers (e.g. `ocl-icd-opencl-dev`, `nvidia-opencl-dev`)
- GCC / G++ with C11 and C++14 support
- A GPU with OpenCL support

On Ubuntu:
```bash
sudo apt install ocl-icd-opencl-dev opencl-c-headers
```

## Build and Run

```bash
# Build
make all

# Run with defaults (European call, S=100, K=100, 1M paths)
./mc_pricer_ocl

# Or use the convenience script
./run.sh

# European put with custom parameters
./run.sh -p -S 150 -K 145 -v 0.30 -n 2000000

# Skip CPU baseline (faster)
./run.sh --no-cpu -n 5000000

# Show help
./mc_pricer_ocl -h
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-S <float>` | Spot price | 100.0 |
| `-K <float>` | Strike price | 100.0 |
| `-r <float>` | Risk-free rate | 0.05 |
| `-v <float>` | Volatility (sigma) | 0.20 |
| `-T <float>` | Time to expiry (years) | 1.0 |
| `-n <int>` | Number of simulation paths | 1,000,000 |
| `-s <int>` | Time steps per path | 252 |
| `-w <int>` | OpenCL work-group size | 256 |
| `-p` | Price a put (default: call) | — |
| `--no-cpu` | Skip CPU baseline | — |
| `-h` | Show help | — |

## File Structure

```
opencl/
├── Makefile                  Build system
├── run.sh                    Convenience build+run script
├── README.md                 This file
├── writeup.md                Assignment discussion document
├── include/
│   ├── option_params.h       Shared structs (pure C)
│   └── ocl_helpers.h         OpenCL helper declarations
├── kernels/
│   ├── mc_sim.cl             MC simulation kernel (xorshift128+ RNG, float4)
│   └── reduction.cl          Parallel sum reduction (__local memory)
└── src/
    ├── main.cpp              Host orchestration, CLI, sub-buffers, profiling
    ├── ocl_helpers.cpp       Platform/device/context/program helpers
    ├── black_scholes.c       Analytical BS pricing (pure C)
    └── cpu_baseline.c        Sequential CPU MC baseline
```

## Sample Output

```
================================================================================
  OpenCL Monte Carlo Option Pricer
================================================================================
Platform:       NVIDIA CUDA
Device:         NVIDIA GeForce RTX 3060 Ti
Compute units:  38
Global memory:  7836 MB
Local memory:   48 KB
Max work-group: 1024
--------------------------------------------------------------------------------
Option Type:    European Call
Spot (S):       100.00
Strike (K):     100.00
Paths:          1000000
Steps/path:     252
================================================================================

[Black-Scholes Analytical]
  Price:   $10.450577

[CPU Monte Carlo]  (1000000 paths, 252 steps)...
  Price:   $10.447721
  Time:    8262.50 ms

[OpenCL Monte Carlo]  (1000000 paths, 252 steps, wg=256)...
  Price:   $10.469247
  Sim kernel: 2.30 ms
  Speedup: 3586.2x over CPU
  Throughput: 434.03 M paths/sec
================================================================================
```

## Author

Daniel Kim — EN605.617 Introduction to GPU Programming, Spring 2026
