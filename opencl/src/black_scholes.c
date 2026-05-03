/*
 * black_scholes.c
 * Analytical Black-Scholes pricing for European options.
 * Pure C — no CUDA or OpenCL dependencies.
 */
#include <math.h>
#include "../include/option_params.h"

/* Standard normal CDF via erff() */
static float norm_cdf(float x)
{
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

/* European call price */
float bs_call_price(const OptionParams *p)
{
    float d1 = (logf(p->S / p->K)
        + (p->r + 0.5f * p->sigma * p->sigma) * p->T)
        / (p->sigma * sqrtf(p->T));
    float d2 = d1 - p->sigma * sqrtf(p->T);
    return p->S * norm_cdf(d1)
         - p->K * expf(-p->r * p->T) * norm_cdf(d2);
}

/* European put price (put-call parity) */
float bs_put_price(const OptionParams *p)
{
    return bs_call_price(p)
         - p->S + p->K * expf(-p->r * p->T);
}
