#include <stdio.h>
#include <curand_kernel.h>
#include "../include/option_params.h"
#include "monte_carlo.h"

/* Option parameters in constant memory — broadcast to all threads */
__constant__ OptionParams d_params;

/* Kernel: initialize cuRAND states (one per thread) */
__global__ void init_curand_states(curandState *states, unsigned long seed,
                                   int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/* Unified MC kernel: handles all option types
 * Each thread simulates one complete GBM price path */
__global__ void mc_unified_kernel(curandState *states, float *payoffs,
                                  int num_paths, int num_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    /* Load params from constant memory into registers */
    float S = d_params.S;
    float K = d_params.K;
    float r = d_params.r;
    float sigma = d_params.sigma;
    float T = d_params.T;
    float barrier = d_params.barrier;
    OptionType type = d_params.type;

    /* Merton jump-diffusion parameters */
    float jlambda = d_params.jump_lambda;   /* jump intensity (jumps/year) */
    float jmean   = d_params.jump_mean;     /* mean log-jump size */
    float jvol    = d_params.jump_vol;      /* jump size volatility */

    float dt = T / (float)num_steps;

    /* Compensated drift: subtract jump contribution to keep risk-neutral
     * k = E[e^J - 1] = exp(jmean + 0.5*jvol^2) - 1 */
    float k_comp = 0.0f;
    if (jlambda > 0.0f) {
        k_comp = expf(jmean + 0.5f * jvol * jvol) - 1.0f;
    }
    float drift = (r - 0.5f * sigma * sigma - jlambda * k_comp) * dt;
    float diffusion = sigma * sqrtf(dt);
    float jump_prob = jlambda * dt;  /* probability of jump in one timestep */

    /* Load RNG state into registers */
    curandState local_state = states[idx];

    /* Simulate path (GBM or jump-diffusion depending on jump_lambda) */
    float S_t = S;
    float sum_S = S;          /* running sum for Asian option */
    int barrier_hit = 0;      /* flag for barrier option */

    for (int step = 0; step < num_steps; step++) {
        float Z = curand_normal(&local_state);
        float jump_component = 0.0f;

        /* Jump-diffusion: Poisson jump events */
        if (jlambda > 0.0f) {
            float u = curand_uniform(&local_state);
            if (u < jump_prob) {
                /* Jump occurred — log-normal jump size */
                float Zj = curand_normal(&local_state);
                jump_component = jmean + jvol * Zj;
            }
        }

        S_t *= expf(drift + diffusion * Z + jump_component);
        sum_S += S_t;

        /* Check barrier crossing */
        if (type == OPTION_BARRIER_UP_AND_OUT_CALL && S_t >= barrier) {
            barrier_hit = 1;
        }
        if (type == OPTION_BARRIER_DOWN_AND_OUT_CALL && S_t <= barrier) {
            barrier_hit = 1;
        }
    }

    /* Write back RNG state */
    states[idx] = local_state;

    /* Compute payoff based on option type */
    float payoff = 0.0f;
    float avg_S = sum_S / (float)(num_steps + 1);

    switch (type) {
        case OPTION_EUROPEAN_CALL:
            payoff = fmaxf(S_t - K, 0.0f);
            break;
        case OPTION_EUROPEAN_PUT:
            payoff = fmaxf(K - S_t, 0.0f);
            break;
        case OPTION_ASIAN_CALL:
            payoff = fmaxf(avg_S - K, 0.0f);
            break;
        case OPTION_ASIAN_PUT:
            payoff = fmaxf(K - avg_S, 0.0f);
            break;
        case OPTION_BARRIER_UP_AND_OUT_CALL:
            payoff = barrier_hit ? 0.0f : fmaxf(S_t - K, 0.0f);
            break;
        case OPTION_BARRIER_DOWN_AND_OUT_CALL:
            payoff = barrier_hit ? 0.0f : fmaxf(S_t - K, 0.0f);
            break;
        case OPTION_DIGITAL_CALL:
            payoff = (S_t > K) ? 1.0f : 0.0f;
            break;
        case OPTION_DIGITAL_PUT:
            payoff = (S_t < K) ? 1.0f : 0.0f;
            break;
    }

    payoffs[idx] = payoff;
}

/* Host function: copy params to constant memory */
void mc_set_params(const OptionParams *params) {
    cudaMemcpyToSymbol(d_params, params, sizeof(OptionParams));
}

/* Host function: allocate and initialize cuRAND states */
curandState* mc_init_rng(int num_paths, int block_size) {
    curandState *d_states;
    cudaMalloc(&d_states, num_paths * sizeof(curandState));

    int grid_size = (num_paths + block_size - 1) / block_size;
    init_curand_states<<<grid_size, block_size>>>(d_states, time(NULL), num_paths);
    cudaDeviceSynchronize();

    return d_states;
}

/* Host function: run MC simulation, return payoffs in device memory */
float* mc_run_simulation(curandState *d_states, int num_paths,
                         int num_steps, int block_size) {
    float *d_payoffs;
    cudaMalloc(&d_payoffs, num_paths * sizeof(float));

    int grid_size = (num_paths + block_size - 1) / block_size;
    mc_unified_kernel<<<grid_size, block_size>>>(d_states, d_payoffs,
                                                 num_paths, num_steps);
    cudaDeviceSynchronize();

    return d_payoffs;
}
