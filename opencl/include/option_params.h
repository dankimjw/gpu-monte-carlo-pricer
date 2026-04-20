/*
 * option_params.h
 * Shared option parameter structures for OpenCL MC pricer.
 * Pure C — no CUDA or OpenCL dependencies.
 */
#ifndef OPTION_PARAMS_H
#define OPTION_PARAMS_H

/* Option type enumeration */
typedef enum {
    OPTION_EUROPEAN_CALL = 0,
    OPTION_EUROPEAN_PUT  = 1
} OptionType;

/* Parameters passed to simulation kernels */
typedef struct {
    float S;       /* Spot price                 */
    float K;       /* Strike price               */
    float r;       /* Risk-free rate             */
    float sigma;   /* Volatility                 */
    float T;       /* Time to expiry (years)     */
    int   type;    /* 0 = call, 1 = put          */
} OptionParams;

/* Result container returned by pricing routines */
typedef struct {
    float price;       /* Option price               */
    float std_err;     /* Standard error              */
    float ci_low;      /* 95 % CI lower bound         */
    float ci_high;     /* 95 % CI upper bound         */
    float time_ms;     /* Wall-clock time (ms)        */
} PricingResult;

#endif /* OPTION_PARAMS_H */
