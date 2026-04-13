#ifndef CPU_BASELINE_H
#define CPU_BASELINE_H

#include "../include/option_params.h"

#ifdef __cplusplus
extern "C" {
#endif

PricingResult cpu_monte_carlo(const OptionParams *params,
                              int num_paths, int num_steps);

#ifdef __cplusplus
}
#endif

#endif /* CPU_BASELINE_H */
