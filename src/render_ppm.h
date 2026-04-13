#ifndef RENDER_PPM_H
#define RENDER_PPM_H

#include <curand_kernel.h>
#include "../include/option_params.h"

/* Render spaghetti plot of price paths to a PPM file */
void render_spaghetti_ppm(const OptionParams *params, int num_paths,
                          int num_steps, int block_size,
                          int width, int height, const char *filename);

/* Render payoff histogram to a PPM file */
void render_histogram_ppm(float *d_payoffs, int num_paths,
                          int width, int height, const char *filename);

#endif /* RENDER_PPM_H */
