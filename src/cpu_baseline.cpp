#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/option_params.h"
#include "cpu_baseline.h"

/* Box-Muller transform: generate standard normal from uniform */
static float rand_normal(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/* Simulate one GBM price path and return terminal price */
static float simulate_path(const OptionParams *params, int num_steps) {
    float dt = params->T / (float)num_steps;
    float drift = (params->r - 0.5f * params->sigma * params->sigma) * dt;
    float diffusion = params->sigma * sqrtf(dt);
    float S = params->S;

    for (int i = 0; i < num_steps; i++) {
        float Z = rand_normal();
        S *= expf(drift + diffusion * Z);
    }
    return S;
}

/* Compute payoff for a given terminal price */
static float compute_payoff(const OptionParams *params, float S_T) {
    float payoff = 0.0f;
    switch (params->type) {
        case OPTION_EUROPEAN_CALL:
            payoff = fmaxf(S_T - params->K, 0.0f);
            break;
        case OPTION_EUROPEAN_PUT:
            payoff = fmaxf(params->K - S_T, 0.0f);
            break;
        default:
            payoff = fmaxf(S_T - params->K, 0.0f);
            break;
    }
    return payoff;
}

/* CPU Monte Carlo pricer */
PricingResult cpu_monte_carlo(const OptionParams *params,
                              int num_paths, int num_steps) {
    PricingResult result;
    srand((unsigned int)time(NULL));

    float sum_payoff = 0.0f;
    float sum_payoff_sq = 0.0f;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_paths; i++) {
        float S_T = simulate_path(params, num_steps);
        float payoff = compute_payoff(params, S_T);
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    float discount = expf(-params->r * params->T);
    float mean_payoff = sum_payoff / (float)num_paths;
    float mean_sq = sum_payoff_sq / (float)num_paths;
    float variance = mean_sq - mean_payoff * mean_payoff;
    float std_err = sqrtf(variance / (float)num_paths);

    result.price = discount * mean_payoff;
    result.std_err = discount * std_err;
    result.ci_low = result.price - 1.96f * result.std_err;
    result.ci_high = result.price + 1.96f * result.std_err;

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0
                   + (end.tv_nsec - start.tv_nsec) / 1e6;
    result.cpu_time_ms = (float)elapsed;
    result.gpu_time_ms = 0.0f;
    result.speedup = 0.0f;

    return result;
}
