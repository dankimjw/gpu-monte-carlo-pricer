#ifndef GREEKS_H
#define GREEKS_H

#include "../include/option_params.h"

typedef struct {
    float delta;    /* dV/dS */
    float gamma;    /* d²V/dS² */
    float vega;     /* dV/dsigma (per 1% move) */
    float theta;    /* dV/dT (per day) */
    float rho;      /* dV/dr (per 1% move) */
} Greeks;

/* Compute Greeks via bump-and-reprice on GPU */
Greeks compute_greeks_gpu(const OptionParams *params, int num_paths,
                          int num_steps, int block_size);

#endif /* GREEKS_H */
