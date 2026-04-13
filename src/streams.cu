#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "../include/option_params.h"
#include "streams.h"

/* Per-stream MC kernel — reads params from global memory (not constant,
 * since constant memory is shared and we need different params per stream) */
__global__ void mc_stream_kernel(curandState *states, float *payoffs,
                                 OptionParams params,
                                 int num_paths, int num_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    float S = params.S;
    float K = params.K;
    float r = params.r;
    float sigma = params.sigma;
    float T = params.T;
    float barrier = params.barrier;
    OptionType type = params.type;

    /* Jump-diffusion parameters */
    float jlambda = params.jump_lambda;
    float jmean   = params.jump_mean;
    float jvol    = params.jump_vol;

    float dt = T / (float)num_steps;
    float k_comp = 0.0f;
    if (jlambda > 0.0f)
        k_comp = expf(jmean + 0.5f * jvol * jvol) - 1.0f;
    float drift = (r - 0.5f * sigma * sigma - jlambda * k_comp) * dt;
    float diffusion = sigma * sqrtf(dt);
    float jump_prob = jlambda * dt;

    curandState local_state = states[idx];

    float S_t = S;
    float sum_S = S;
    int barrier_hit = 0;

    for (int step = 0; step < num_steps; step++) {
        float Z = curand_normal(&local_state);
        float jump_component = 0.0f;
        if (jlambda > 0.0f) {
            float u = curand_uniform(&local_state);
            if (u < jump_prob) {
                float Zj = curand_normal(&local_state);
                jump_component = jmean + jvol * Zj;
            }
        }
        S_t *= expf(drift + diffusion * Z + jump_component);
        sum_S += S_t;
        if (type == OPTION_BARRIER_UP_AND_OUT_CALL && S_t >= barrier)
            barrier_hit = 1;
        if (type == OPTION_BARRIER_DOWN_AND_OUT_CALL && S_t <= barrier)
            barrier_hit = 1;
    }

    states[idx] = local_state;

    float payoff = 0.0f;
    float avg_S = sum_S / (float)(num_steps + 1);

    switch (type) {
        case OPTION_EUROPEAN_CALL:   payoff = fmaxf(S_t - K, 0.0f); break;
        case OPTION_EUROPEAN_PUT:    payoff = fmaxf(K - S_t, 0.0f); break;
        case OPTION_ASIAN_CALL:      payoff = fmaxf(avg_S - K, 0.0f); break;
        case OPTION_ASIAN_PUT:       payoff = fmaxf(K - avg_S, 0.0f); break;
        case OPTION_BARRIER_UP_AND_OUT_CALL:
            payoff = barrier_hit ? 0.0f : fmaxf(S_t - K, 0.0f); break;
        case OPTION_BARRIER_DOWN_AND_OUT_CALL:
            payoff = barrier_hit ? 0.0f : fmaxf(S_t - K, 0.0f); break;
        case OPTION_DIGITAL_CALL:    payoff = (S_t > K) ? 1.0f : 0.0f; break;
        case OPTION_DIGITAL_PUT:     payoff = (S_t < K) ? 1.0f : 0.0f; break;
    }

    payoffs[idx] = payoff;
}

/* RNG init kernel for streams */
__global__ void init_curand_stream(curandState *states, unsigned long seed,
                                   int offset, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        curand_init(seed, idx + offset, 0, &states[idx]);
    }
}

/* Price a portfolio using concurrent CUDA streams */
void streams_price_portfolio(PortfolioEntry *portfolio, int num_options,
                             int num_paths_per_option, int num_steps,
                             int block_size, int num_streams) {

    if (num_streams > num_options) num_streams = num_options;
    if (num_streams < 1) num_streams = 1;

    printf("\n================================================================\n");
    printf("  Portfolio Pricing — %d options, %d streams\n", num_options, num_streams);
    printf("================================================================\n");

    /* Create streams */
    cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    /* Allocate per-stream device memory */
    curandState **d_states = (curandState **)malloc(num_streams * sizeof(curandState *));
    float **d_payoffs = (float **)malloc(num_streams * sizeof(float *));
    float **h_payoffs = (float **)malloc(num_streams * sizeof(float *));

    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&d_states[i], num_paths_per_option * sizeof(curandState));
        cudaMalloc(&d_payoffs[i], num_paths_per_option * sizeof(float));
        cudaMallocHost(&h_payoffs[i], num_paths_per_option * sizeof(float));
    }

    int grid_size = (num_paths_per_option + block_size - 1) / block_size;

    /* Timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* Process options in batches of num_streams */
    for (int batch_start = 0; batch_start < num_options; batch_start += num_streams) {
        int batch_end = batch_start + num_streams;
        if (batch_end > num_options) batch_end = num_options;
        int batch_size = batch_end - batch_start;

        /* Launch RNG init + MC kernel on each stream */
        for (int s = 0; s < batch_size; s++) {
            int opt_idx = batch_start + s;
            int seed_offset = opt_idx * num_paths_per_option;

            init_curand_stream<<<grid_size, block_size, 0, streams[s]>>>(
                d_states[s], time(NULL) + opt_idx, seed_offset, num_paths_per_option);

            mc_stream_kernel<<<grid_size, block_size, 0, streams[s]>>>(
                d_states[s], d_payoffs[s], portfolio[opt_idx].params,
                num_paths_per_option, num_steps);

            /* Async copy back — overlap with other stream kernels */
            cudaMemcpyAsync(h_payoffs[s], d_payoffs[s],
                           num_paths_per_option * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[s]);
        }

        /* Sync all streams in this batch */
        for (int s = 0; s < batch_size; s++) {
            cudaStreamSynchronize(streams[s]);

            int opt_idx = batch_start + s;
            float r = portfolio[opt_idx].params.r;
            float T = portfolio[opt_idx].params.T;
            float discount = expf(-r * T);

            /* Compute stats on host */
            float sum = 0.0f, sum_sq = 0.0f;
            for (int i = 0; i < num_paths_per_option; i++) {
                sum += h_payoffs[s][i];
                sum_sq += h_payoffs[s][i] * h_payoffs[s][i];
            }
            float mean = sum / (float)num_paths_per_option;
            float var = sum_sq / (float)num_paths_per_option - mean * mean;
            float std_err = sqrtf(var / (float)num_paths_per_option);

            portfolio[opt_idx].price = discount * mean;
            portfolio[opt_idx].std_err = discount * std_err;
            portfolio[opt_idx].ci_low = portfolio[opt_idx].price - 1.96f * portfolio[opt_idx].std_err;
            portfolio[opt_idx].ci_high = portfolio[opt_idx].price + 1.96f * portfolio[opt_idx].std_err;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    /* Print results */
    printf("  %-4s %-12s %8s %8s %10s  %s\n",
           "#", "Type", "Strike", "Price", "Std Err", "95% CI");
    printf("  %-4s %-12s %8s %8s %10s  %s\n",
           "----", "------------", "--------", "--------", "----------", "-------------------");

    float total_value = 0.0f;
    const char *type_names[] = {
        "EU Call", "EU Put", "Asian Call", "Asian Put",
        "Barrier UO", "Barrier DO", "Digital C", "Digital P"
    };

    for (int i = 0; i < num_options; i++) {
        const char *tn = type_names[portfolio[i].params.type];
        printf("  %-4d %-12s %8.1f %8.4f %10.4f  [%.4f, %.4f]\n",
               i + 1, tn,
               portfolio[i].params.K,
               portfolio[i].price,
               portfolio[i].std_err,
               portfolio[i].ci_low,
               portfolio[i].ci_high);
        total_value += portfolio[i].price;
    }

    printf("  %-4s %-12s %8s %8.4f\n", "", "TOTAL", "", total_value);
    printf("\n  Portfolio priced in %.2f ms (%d streams)\n", elapsed_ms, num_streams);

    /* Cleanup */
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_states[i]);
        cudaFree(d_payoffs[i]);
        cudaFreeHost(h_payoffs[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    free(d_states);
    free(d_payoffs);
    free(h_payoffs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
