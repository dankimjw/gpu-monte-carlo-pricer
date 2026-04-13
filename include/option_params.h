#ifndef OPTION_PARAMS_H
#define OPTION_PARAMS_H

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

typedef enum {
    OPTION_EUROPEAN_CALL,
    OPTION_EUROPEAN_PUT,
    OPTION_ASIAN_CALL,
    OPTION_ASIAN_PUT,
    OPTION_BARRIER_UP_AND_OUT_CALL,
    OPTION_BARRIER_DOWN_AND_OUT_CALL,
    OPTION_DIGITAL_CALL,
    OPTION_DIGITAL_PUT
} OptionType;

typedef struct {
    float S;             /* Spot price */
    float K;             /* Strike price */
    float r;             /* Risk-free rate */
    float sigma;         /* Volatility */
    float T;             /* Time to expiry (years) */
    float barrier;       /* Barrier level (for barrier options) */
    OptionType type;     /* Option type */

    /* Merton jump-diffusion parameters */
    float jump_lambda;   /* Jump intensity (expected jumps/year), 0 = pure GBM */
    float jump_mean;     /* Mean log-jump size (e.g. -0.10 = avg 10% crash) */
    float jump_vol;      /* Jump size volatility (e.g. 0.15) */
} OptionParams;

typedef struct {
    float price;         /* Option price */
    float std_err;       /* Standard error */
    float ci_low;        /* 95% confidence interval lower bound */
    float ci_high;       /* 95% confidence interval upper bound */
    float cpu_time_ms;   /* CPU computation time */
    float gpu_time_ms;   /* GPU computation time */
    float speedup;       /* CPU / GPU time ratio */
} PricingResult;

#endif /* OPTION_PARAMS_H */
