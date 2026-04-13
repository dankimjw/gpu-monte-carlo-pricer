#ifndef STREAMS_H
#define STREAMS_H

#include "../include/option_params.h"

typedef struct {
    OptionParams params;
    float price;
    float std_err;
    float ci_low;
    float ci_high;
} PortfolioEntry;

/* Price a portfolio of options using N CUDA streams concurrently */
void streams_price_portfolio(PortfolioEntry *portfolio, int num_options,
                             int num_paths_per_option, int num_steps,
                             int block_size, int num_streams);

#endif /* STREAMS_H */
