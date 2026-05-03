#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include "../include/option_params.h"

float bs_call_price(const OptionParams *params);
float bs_put_price(const OptionParams *params);
void bs_greeks(const OptionParams *params, OptionType type,
               float *delta, float *gamma, float *vega, float *theta);

#endif /* BLACK_SCHOLES_H */
