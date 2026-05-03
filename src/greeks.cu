#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include "../include/option_params.h"
#include "greeks.h"
#include "monte_carlo.h"
#include "reduction.h"

/* Helper: run a full MC pricing with given params, return discounted price */
static float mc_price(const OptionParams *params, int num_paths,
                      int num_steps, int block_size) {
    mc_set_params(params);
    curandState *d_states = mc_init_rng(num_paths, block_size, (unsigned long)time(NULL));
    float *d_payoffs = mc_run_simulation(d_states, num_paths, num_steps, block_size);

    float mean_payoff, variance;
    reduce_payoffs(d_payoffs, num_paths, &mean_payoff, &variance);

    float discount = expf(-params->r * params->T);
    float price = discount * mean_payoff;

    cudaFree(d_states);
    cudaFree(d_payoffs);
    return price;
}

/* Compute Greeks via bump-and-reprice (finite differences) */
Greeks compute_greeks_gpu(const OptionParams *params, int num_paths,
                          int num_steps, int block_size) {
    Greeks g;
    OptionParams bumped = *params;
    float base_price = mc_price(params, num_paths, num_steps, block_size);

    /* Delta = (V(S+dS) - V(S-dS)) / (2*dS) */
    float dS = params->S * 0.01f;  /* 1% bump */
    bumped = *params;
    bumped.S = params->S + dS;
    float price_up = mc_price(&bumped, num_paths, num_steps, block_size);
    bumped.S = params->S - dS;
    float price_down = mc_price(&bumped, num_paths, num_steps, block_size);
    g.delta = (price_up - price_down) / (2.0f * dS);

    /* Gamma = (V(S+dS) - 2*V(S) + V(S-dS)) / (dS^2) */
    g.gamma = (price_up - 2.0f * base_price + price_down) / (dS * dS);

    /* Vega = (V(sigma+dsigma) - V(sigma-dsigma)) / (2*dsigma), scaled per 1% */
    float dsigma = 0.01f;
    bumped = *params;
    bumped.sigma = params->sigma + dsigma;
    float price_vup = mc_price(&bumped, num_paths, num_steps, block_size);
    bumped.sigma = params->sigma - dsigma;
    float price_vdown = mc_price(&bumped, num_paths, num_steps, block_size);
    g.vega = (price_vup - price_vdown) / (2.0f * dsigma) * 0.01f;

    /* Theta = (V(T-dT) - V(T)) / dT, scaled per day */
    float dT = 1.0f / 365.0f;  /* one day */
    bumped = *params;
    bumped.T = params->T - dT;
    if (bumped.T < 0.001f) bumped.T = 0.001f;
    float price_tdec = mc_price(&bumped, num_paths, num_steps, block_size);
    g.theta = (price_tdec - base_price) / dT / 365.0f;

    /* Rho = (V(r+dr) - V(r-dr)) / (2*dr), scaled per 1% */
    float dr = 0.001f;
    bumped = *params;
    bumped.r = params->r + dr;
    float price_rup = mc_price(&bumped, num_paths, num_steps, block_size);
    bumped.r = params->r - dr;
    float price_rdown = mc_price(&bumped, num_paths, num_steps, block_size);
    g.rho = (price_rup - price_rdown) / (2.0f * dr) * 0.01f;

    /* Restore original params to constant memory */
    mc_set_params(params);

    return g;
}
