#include <math.h>
#include <stdio.h>
#include "../include/option_params.h"

/* Standard normal CDF using erff() */
HOST_DEVICE static float norm_cdf(float x) {
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

/* Black-Scholes analytical price for European call */
float bs_call_price(const OptionParams *params) {
    float S = params->S;
    float K = params->K;
    float r = params->r;
    float sigma = params->sigma;
    float T = params->T;

    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);

    return S * norm_cdf(d1) - K * expf(-r * T) * norm_cdf(d2);
}

/* Black-Scholes analytical price for European put */
float bs_put_price(const OptionParams *params) {
    float S = params->S;
    float K = params->K;
    float r = params->r;
    float T = params->T;

    /* Put-call parity: P = C - S + K*exp(-rT) */
    return bs_call_price(params) - S + K * expf(-r * T);
}

/* Analytical Greeks for European call */
void bs_greeks(const OptionParams *params,
               float *delta, float *gamma, float *vega, float *theta) {
    float S = params->S;
    float K = params->K;
    float r = params->r;
    float sigma = params->sigma;
    float T = params->T;

    float sqrt_T = sqrtf(T);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
    float d2 = d1 - sigma * sqrt_T;

    /* Standard normal PDF */
    float pdf_d1 = expf(-0.5f * d1 * d1) / sqrtf(2.0f * M_PI);

    if (delta) *delta = norm_cdf(d1);
    if (gamma) *gamma = pdf_d1 / (S * sigma * sqrt_T);
    if (vega)  *vega  = S * pdf_d1 * sqrt_T / 100.0f;  /* per 1% vol move */
    if (theta) *theta = (-(S * pdf_d1 * sigma) / (2.0f * sqrt_T)
                         - r * K * expf(-r * T) * norm_cdf(d2)) / 365.0f; /* per day */
}