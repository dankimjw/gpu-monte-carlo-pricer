/*
 * cpu_baseline.c
 * Sequential CPU Monte Carlo pricer for speedup comparison.
 * Pure C — no GPU dependencies.
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/option_params.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Box-Muller: standard normal variate */
static float rand_normal(void)
{
    float u1 = ((float)rand() + 1.0f)
             / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f)
             / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1))
         * cosf(2.0f * (float)M_PI * u2);
}

/* Simulate one GBM path, return terminal price */
static float simulate_path(
    const OptionParams *p, int steps)
{
    float dt   = p->T / (float)steps;
    float drft = (p->r - 0.5f * p->sigma * p->sigma) * dt;
    float diff = p->sigma * sqrtf(dt);
    float S    = p->S;
    for (int i = 0; i < steps; i++) {
        S *= expf(drft + diff * rand_normal());
    }
    return S;
}

/* CPU Monte Carlo pricer */
PricingResult cpu_monte_carlo(
    const OptionParams *p, int paths, int steps)
{
    PricingResult res = {0};
    srand((unsigned)time(NULL));

    float sum   = 0.0f;
    float sum2  = 0.0f;
    float disc  = expf(-p->r * p->T);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < paths; i++) {
        float S_T = simulate_path(p, steps);
        float pay;
        if (p->type == 0) {
            pay = fmaxf(S_T - p->K, 0.0f);
        } else {
            pay = fmaxf(p->K - S_T, 0.0f);
        }
        sum  += pay;
        sum2 += pay * pay;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    float mean = sum / (float)paths;
    float var  = sum2 / (float)paths - mean * mean;
    float se   = sqrtf(var / (float)paths);

    res.price   = disc * mean;
    res.std_err = disc * se;
    res.ci_low  = res.price - 1.96f * res.std_err;
    res.ci_high = res.price + 1.96f * res.std_err;
    res.time_ms = (float)(
        (t1.tv_sec - t0.tv_sec) * 1000.0
        + (t1.tv_nsec - t0.tv_nsec) / 1e6);
    return res;
}
