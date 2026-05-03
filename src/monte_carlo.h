#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>
#include "../include/option_params.h"

void mc_set_params(const OptionParams *params);
curandState* mc_init_rng(int num_paths, int block_size, unsigned long seed);
float* mc_run_simulation(curandState *d_states, int num_paths,
                         int num_steps, int block_size);

#endif /* MONTE_CARLO_H */
