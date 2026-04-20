# OpenCL Monte Carlo Option Pricer — Discussion

**Daniel Kim**
**EN605.617 — Introduction to GPU Programming, Spring 2026**

## Goals

The goal of this assignment was to build a non-trivial OpenCL application that demonstrates
a thorough understanding of both basic and advanced OpenCL concepts.  I chose to port a
subset of my CUDA-based Monte Carlo option pricing engine to OpenCL.  Monte Carlo
simulation is an ideal workload for GPU parallelism because each of the millions of
simulated price paths is completely independent, making it embarrassingly parallel.  The
specific objectives were to:

- Implement a GBM (Geometric Brownian Motion) path simulation kernel in OpenCL
- Demonstrate buffer and sub-buffer usage for partitioning work
- Use local (shared) memory for an efficient parallel reduction
- Leverage OpenCL vector types (float4) for vectorized random number generation
- Use event-based profiling for accurate kernel timing
- Provide a fully configurable CLI and build/run scripts

## Challenges

The biggest challenge was replacing CUDA's cuRAND library, which has no direct OpenCL
equivalent.  I implemented a xorshift128+ pseudo-random number generator directly in the
kernel code, seeded uniquely per work-item using a combination of a base seed and the global
ID.  Ensuring statistical quality required careful seeding — simply using sequential seeds
produced correlated sequences, so I applied multiplicative hashing with large prime
constants and an 8-iteration warmup per generator.

Another challenge was adapting the parallel reduction from CUDA to OpenCL.  The CUDA version
used `__shared__` memory and `__syncthreads()`, which map cleanly to OpenCL's `__local`
memory and `barrier(CLK_LOCAL_MEM_FENCE)`, but the syntax for passing local memory as a
kernel argument (setting the argument to NULL with a specified size) was initially confusing
and required careful reading of the specification.

Sub-buffer creation had an alignment constraint that I initially overlooked — the offset
must be aligned to `CL_DEVICE_MEM_BASE_ADDR_ALIGN`.  Using half the payoff array as the
split point worked naturally because the buffer sizes were always multiples of large powers
of two.

## Results

The OpenCL implementation achieves a 3500x+ speedup over the sequential CPU baseline on an
NVIDIA RTX 3060 Ti, pricing 1 million paths with 252 time steps in approximately 2.3
milliseconds.  The Monte Carlo prices converge to within 0.2% of the analytical
Black-Scholes solution, confirming correctness of both the RNG and the GBM path simulation.
The use of float4 vector types for batching four normal variates at a time in the simulation
kernel contributed to efficient use of the GPU's SIMD lanes.

Overall, this project reinforced that OpenCL provides a portable and capable alternative to
CUDA for GPU-accelerated numerical computing, with the trade-off being more verbose host
code and the absence of vendor-specific convenience libraries like cuRAND.
