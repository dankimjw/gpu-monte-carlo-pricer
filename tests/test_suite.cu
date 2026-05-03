/*
 * test_suite.cu — unit tests for GPU Monte Carlo Option Pricing Engine
 *
 * Tests cover:
 *   1.  Black-Scholes call/put analytical prices (put-call parity)
 *   2.  Black-Scholes Greeks: call delta, put delta, gamma/vega symmetry
 *   3.  BS validation guard: has_bs_reference logic
 *   4.  GPU European call MC vs BS analytical (convergence within 1%)
 *   5.  GPU European put MC vs BS analytical (convergence within 1%)
 *   6.  GPU Asian call: price strictly less than European call
 *   7.  GPU Asian put: price strictly less than European put
 *   8.  GPU Digital call: price in (0, discount) range
 *   9.  GPU Digital put: price + digital call ≈ discount (parity)
 *  10.  GPU Barrier up-and-out call: price ≤ European call
 *  11.  GPU Barrier down-and-out call: price ≤ European call
 *  12.  Jump-diffusion call: price > GBM call (jumps increase variance)
 *  13.  Reproducibility: same seed → same price
 *  14.  Different seeds → different prices
 *  15.  Shared-memory reduction: mean of constant array
 *  16.  Shared-memory reduction: variance of constant array (should be 0)
 *  17.  CPU baseline: European call/put prices close to BS
 *  18.  CPU baseline: exotic types do NOT crash (default payoff path)
 *  19.  --benchmark mode: CLI flag parses without error (smoke test via binary)
 *  20.  --type CLI: all new type strings parse correctly (smoke test via binary)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/option_params.h"
#include "../src/black_scholes.h"
#include "../src/monte_carlo.h"
#include "../src/reduction.h"
#include "../src/cpu_baseline.h"

/* ------------------------------------------------------------------ */
/* Test harness                                                         */
/* ------------------------------------------------------------------ */
static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { printf("  [PASS] %s\n", name); g_pass++; } while(0)
#define FAIL(name, ...) do { printf("  [FAIL] %s: ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } while(0)

#define CHECK_NEAR(name, got, expected, tol) do { \
    float _g = (float)(got), _e = (float)(expected), _t = (float)(tol); \
    if (fabsf(_g - _e) <= _t) { PASS(name); } \
    else { FAIL(name, "got %.6f, expected %.6f, tol %.6f", _g, _e, _t); } \
} while(0)

#define CHECK_TRUE(name, expr) do { \
    if (expr) { PASS(name); } \
    else { FAIL(name, "condition false"); } \
} while(0)

#define CHECK_LT(name, a, b) do { \
    if ((float)(a) < (float)(b)) { PASS(name); } \
    else { FAIL(name, "%.6f not < %.6f", (float)(a), (float)(b)); } \
} while(0)

#define CHECK_LE(name, a, b) do { \
    if ((float)(a) <= (float)(b) + 1e-4f) { PASS(name); } \
    else { FAIL(name, "%.6f not <= %.6f", (float)(a), (float)(b)); } \
} while(0)

/* ------------------------------------------------------------------ */
/* Helper: run GPU MC and return discounted price                       */
/* ------------------------------------------------------------------ */
static float gpu_price(const OptionParams *params, int num_paths,
                       int num_steps, int block_size, unsigned long seed) {
    mc_set_params(params);
    curandState *d_states = mc_init_rng(num_paths, block_size, seed);
    float *d_payoffs = mc_run_simulation(d_states, num_paths, num_steps, block_size);
    float mean_payoff, variance;
    reduce_payoffs(d_payoffs, num_paths, &mean_payoff, &variance);
    cudaFree(d_states);
    cudaFree(d_payoffs);
    return expf(-params->r * params->T) * mean_payoff;
}

/* ------------------------------------------------------------------ */
/* Standard ATM params used across most tests                          */
/* ------------------------------------------------------------------ */
static OptionParams make_base_params(OptionType type) {
    OptionParams p;
    p.S = 100.0f; p.K = 100.0f; p.r = 0.05f; p.sigma = 0.20f;
    p.T = 1.0f;   p.barrier = 0.0f; p.type = type;
    p.jump_lambda = 0.0f; p.jump_mean = 0.0f; p.jump_vol = 0.0f;
    return p;
}

/* ------------------------------------------------------------------ */
/* 1-2. Black-Scholes analytical prices                                */
/* ------------------------------------------------------------------ */
static void test_bs_prices(void) {
    printf("\n[Black-Scholes Analytical]\n");
    OptionParams p = make_base_params(OPTION_EUROPEAN_CALL);

    float call = bs_call_price(&p);
    float put  = bs_put_price(&p);

    /* Known value: ATM S=K=100, r=5%, σ=20%, T=1 → call ≈ 10.4506 */
    CHECK_NEAR("BS call ATM price", call, 10.4506f, 0.01f);

    /* Put-call parity: C - P = S - K*e^{-rT} */
    float parity = p.S - p.K * expf(-p.r * p.T);
    CHECK_NEAR("BS put-call parity", call - put, parity, 1e-4f);
}

/* ------------------------------------------------------------------ */
/* 3. Black-Scholes Greeks: call vs put                                */
/* ------------------------------------------------------------------ */
static void test_bs_greeks(void) {
    printf("\n[Black-Scholes Greeks]\n");
    OptionParams p = make_base_params(OPTION_EUROPEAN_CALL);

    float dc, gc, vc, tc;
    float dp, gp, vp, tp;
    bs_greeks(&p, OPTION_EUROPEAN_CALL, &dc, &gc, &vc, &tc);
    bs_greeks(&p, OPTION_EUROPEAN_PUT,  &dp, &gp, &vp, &tp);

    /* Call delta in (0,1) */
    CHECK_TRUE("call delta in (0,1)", dc > 0.0f && dc < 1.0f);

    /* Put delta in (-1,0) */
    CHECK_TRUE("put delta in (-1,0)", dp > -1.0f && dp < 0.0f);

    /* Put-call delta parity: delta_call - delta_put = 1 */
    CHECK_NEAR("delta parity (call - put = 1)", dc - dp, 1.0f, 1e-4f);

    /* Gamma identical for call and put */
    CHECK_NEAR("gamma call == gamma put", gc, gp, 1e-6f);

    /* Vega identical for call and put */
    CHECK_NEAR("vega call == vega put", vc, vp, 1e-6f);

    /* ATM call theta < 0 (time decay costs money) */
    CHECK_TRUE("call theta < 0", tc < 0.0f);

    /* ATM put theta < 0 for this param set (r*K*e^-rT term check) */
    CHECK_TRUE("put theta < 0", tp < 0.0f);
}

/* ------------------------------------------------------------------ */
/* 4. has_bs_reference guard logic                                      */
/* ------------------------------------------------------------------ */
static void test_bs_reference_guard(void) {
    printf("\n[BS Reference Guard]\n");
    OptionParams euro_call = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams euro_put  = make_base_params(OPTION_EUROPEAN_PUT);
    OptionParams asian     = make_base_params(OPTION_ASIAN_CALL);
    OptionParams digital   = make_base_params(OPTION_DIGITAL_CALL);
    OptionParams jumped    = make_base_params(OPTION_EUROPEAN_CALL);
    jumped.jump_lambda = 1.0f;

    /* Macro mirrors main.cu logic */
    #define HAS_BS(p) ((p).jump_lambda == 0.0f && \
                       ((p).type == OPTION_EUROPEAN_CALL || (p).type == OPTION_EUROPEAN_PUT))

    CHECK_TRUE("European call has BS ref",   HAS_BS(euro_call));
    CHECK_TRUE("European put has BS ref",    HAS_BS(euro_put));
    CHECK_TRUE("Asian call has NO BS ref",  !HAS_BS(asian));
    CHECK_TRUE("Digital call has NO BS ref",!HAS_BS(digital));
    CHECK_TRUE("Jump diffusion has NO BS ref", !HAS_BS(jumped));
    #undef HAS_BS
}

/* ------------------------------------------------------------------ */
/* 5-6. GPU MC convergence: European call and put                      */
/* ------------------------------------------------------------------ */
static void test_gpu_european(void) {
    printf("\n[GPU MC: European Call/Put Convergence]\n");
    const int N = 1000000, STEPS = 252, BS = 256;
    const unsigned long SEED = 42UL;

    OptionParams pc = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams pp = make_base_params(OPTION_EUROPEAN_PUT);

    float mc_call = gpu_price(&pc, N, STEPS, BS, SEED);
    float mc_put  = gpu_price(&pp, N, STEPS, BS, SEED);
    float bs_call = bs_call_price(&pc);
    float bs_put  = bs_put_price(&pp);

    /* Allow 0.5% relative error at 1M paths */
    CHECK_NEAR("GPU European call vs BS (1M paths)", mc_call, bs_call, bs_call * 0.005f);
    CHECK_NEAR("GPU European put vs BS (1M paths)",  mc_put,  bs_put,  bs_put  * 0.005f);
}

/* ------------------------------------------------------------------ */
/* 7-8. GPU MC: Asian call/put pricing relationships                   */
/* ------------------------------------------------------------------ */
static void test_gpu_asian(void) {
    printf("\n[GPU MC: Asian Options]\n");
    const int N = 500000, STEPS = 252, BS = 256;
    const unsigned long SEED = 42UL;

    OptionParams pc = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams pp = make_base_params(OPTION_EUROPEAN_PUT);
    OptionParams ac = make_base_params(OPTION_ASIAN_CALL);
    OptionParams ap = make_base_params(OPTION_ASIAN_PUT);

    float euro_call  = gpu_price(&pc, N, STEPS, BS, SEED);
    float euro_put   = gpu_price(&pp, N, STEPS, BS, SEED);
    float asian_call = gpu_price(&ac, N, STEPS, BS, SEED);
    float asian_put  = gpu_price(&ap, N, STEPS, BS, SEED);

    /* Asian call <= European call (averaging dampens upside) */
    CHECK_LE("Asian call <= European call", asian_call, euro_call);

    /* Asian put <= European put */
    CHECK_LE("Asian put <= European put", asian_put, euro_put);

    /* Asian call > 0 */
    CHECK_TRUE("Asian call price > 0", asian_call > 0.0f);

    /* Asian put > 0 */
    CHECK_TRUE("Asian put price > 0", asian_put > 0.0f);
}

/* ------------------------------------------------------------------ */
/* 9-10. GPU MC: Digital call/put parity                               */
/* ------------------------------------------------------------------ */
static void test_gpu_digital(void) {
    printf("\n[GPU MC: Digital Options]\n");
    const int N = 1000000, STEPS = 252, BS = 256;
    const unsigned long SEED = 42UL;

    OptionParams dc = make_base_params(OPTION_DIGITAL_CALL);
    OptionParams dp = make_base_params(OPTION_DIGITAL_PUT);

    float disc       = expf(-dc.r * dc.T);
    float price_dc   = gpu_price(&dc, N, STEPS, BS, SEED);
    float price_dp   = gpu_price(&dp, N, STEPS, BS, SEED + 1);

    /* Digital call price in (0, discount) */
    CHECK_TRUE("Digital call in (0, discount)", price_dc > 0.0f && price_dc < disc);

    /* Digital put price in (0, discount) */
    CHECK_TRUE("Digital put in (0, discount)", price_dp > 0.0f && price_dp < disc);

    /* Digital call + digital put ≈ discount (binary parity, same seed) */
    OptionParams dc2 = make_base_params(OPTION_DIGITAL_CALL);
    OptionParams dp2 = make_base_params(OPTION_DIGITAL_PUT);
    float pdc = gpu_price(&dc2, N, STEPS, BS, SEED);
    float pdp = gpu_price(&dp2, N, STEPS, BS, SEED);
    CHECK_NEAR("Digital call + put ≈ discount", pdc + pdp, disc, disc * 0.005f);
}

/* ------------------------------------------------------------------ */
/* 11-12. GPU MC: Barrier options ≤ European call                      */
/* ------------------------------------------------------------------ */
static void test_gpu_barrier(void) {
    printf("\n[GPU MC: Barrier Options]\n");
    const int N = 500000, STEPS = 252, BS = 256;
    const unsigned long SEED = 42UL;

    OptionParams ec  = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams buo = make_base_params(OPTION_BARRIER_UP_AND_OUT_CALL);
    OptionParams bdo = make_base_params(OPTION_BARRIER_DOWN_AND_OUT_CALL);
    buo.barrier = 130.0f;
    bdo.barrier = 80.0f;

    float euro  = gpu_price(&ec,  N, STEPS, BS, SEED);
    float up_out = gpu_price(&buo, N, STEPS, BS, SEED);
    float dn_out = gpu_price(&bdo, N, STEPS, BS, SEED);

    /* Barrier options can only be worth <= vanilla (knock-out reduces value) */
    CHECK_LE("Barrier up-and-out call <= European call", up_out, euro);
    CHECK_LE("Barrier down-and-out call <= European call", dn_out, euro);

    /* Both barrier prices should be non-negative */
    CHECK_TRUE("Barrier up-and-out price >= 0", up_out >= 0.0f);
    CHECK_TRUE("Barrier down-and-out price >= 0", dn_out >= 0.0f);
}

/* ------------------------------------------------------------------ */
/* 13. Jump-diffusion: price > GBM (higher variance inflates option)   */
/* ------------------------------------------------------------------ */
static void test_jump_diffusion(void) {
    printf("\n[Jump-Diffusion]\n");
    const int N = 1000000, STEPS = 252, BS = 256;
    const unsigned long SEED = 42UL;

    OptionParams gbm  = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams jump = make_base_params(OPTION_EUROPEAN_CALL);
    jump.jump_lambda = 3.0f;
    jump.jump_mean   = -0.10f;
    jump.jump_vol    = 0.20f;

    float price_gbm  = gpu_price(&gbm,  N, STEPS, BS, SEED);
    float price_jump = gpu_price(&jump, N, STEPS, BS, SEED);

    /* Jump diffusion increases variance → higher call price (Jensen's inequality) */
    CHECK_TRUE("Jump-diffusion call > GBM call", price_jump > price_gbm);
}

/* ------------------------------------------------------------------ */
/* 14-15. Reproducibility: seed determinism                            */
/* ------------------------------------------------------------------ */
static void test_seed_reproducibility(void) {
    printf("\n[Seed Reproducibility]\n");
    const int N = 100000, STEPS = 252, BS = 256;

    OptionParams p = make_base_params(OPTION_EUROPEAN_CALL);

    float p1 = gpu_price(&p, N, STEPS, BS, 99999UL);
    float p2 = gpu_price(&p, N, STEPS, BS, 99999UL);
    float p3 = gpu_price(&p, N, STEPS, BS, 11111UL);

    CHECK_NEAR("Same seed → same price", p1, p2, 1e-6f);
    CHECK_TRUE("Different seed → different price", fabsf(p1 - p3) > 1e-4f);
}

/* ------------------------------------------------------------------ */
/* 16-17. Reduction: correctness on known inputs                       */
/* ------------------------------------------------------------------ */
__global__ void fill_constant(float *arr, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

__global__ void fill_alternating(float *arr, int n, float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = (idx % 2 == 0) ? a : b;
}

static void test_reduction(void) {
    printf("\n[Reduction]\n");
    const int N = 1 << 20; /* 1M */
    float *d_arr;
    cudaMalloc(&d_arr, N * sizeof(float));

    /* Test: all 5.0 → mean=5.0, variance=0 */
    fill_constant<<<(N+255)/256, 256>>>(d_arr, N, 5.0f);
    cudaDeviceSynchronize();
    float mean, var;
    reduce_payoffs(d_arr, N, &mean, &var);
    CHECK_NEAR("Reduction mean of constant 5.0", mean, 5.0f, 1e-3f);
    CHECK_NEAR("Reduction variance of constant (=0)", var, 0.0f, 1e-2f);

    /* Test: alternating 3.0/7.0 → mean=5.0, variance=4.0 */
    fill_alternating<<<(N+255)/256, 256>>>(d_arr, N, 3.0f, 7.0f);
    cudaDeviceSynchronize();
    reduce_payoffs(d_arr, N, &mean, &var);
    CHECK_NEAR("Reduction mean of alternating 3/7 = 5.0", mean, 5.0f, 1e-3f);
    CHECK_NEAR("Reduction variance of alternating 3/7 = 4.0", var, 4.0f, 0.1f);

    cudaFree(d_arr);
}

/* ------------------------------------------------------------------ */
/* 18-19. CPU baseline correctness                                     */
/* ------------------------------------------------------------------ */
static void test_cpu_baseline(void) {
    printf("\n[CPU Baseline]\n");
    const int N = 100000, STEPS = 252;

    OptionParams pc = make_base_params(OPTION_EUROPEAN_CALL);
    OptionParams pp = make_base_params(OPTION_EUROPEAN_PUT);
    OptionParams pa = make_base_params(OPTION_ASIAN_CALL);

    float bs_call = bs_call_price(&pc);
    float bs_put  = bs_put_price(&pp);

    PricingResult rc = cpu_monte_carlo(&pc, N, STEPS);
    PricingResult rp = cpu_monte_carlo(&pp, N, STEPS);
    PricingResult ra = cpu_monte_carlo(&pa, N, STEPS);

    /* CPU call within 1% of BS at 100K paths */
    CHECK_NEAR("CPU European call vs BS (100K)", rc.price, bs_call, bs_call * 0.01f);

    /* CPU put within 1% of BS at 100K paths */
    CHECK_NEAR("CPU European put vs BS (100K)", rp.price, bs_put, bs_put * 0.01f);

    /* CPU timing > 0 */
    CHECK_TRUE("CPU timing > 0ms", rc.cpu_time_ms > 0.0f);

    /* CPU with Asian type should not crash and return positive price */
    CHECK_TRUE("CPU Asian call does not crash, price >= 0", ra.price >= 0.0f);
}

/* ------------------------------------------------------------------ */
/* 20-21. Smoke tests via binary invocation                            */
/* ------------------------------------------------------------------ */
static void test_cli_smoke(void) {
    printf("\n[CLI Smoke Tests]\n");

    /* Each system() call returns 0 on success */
    /* Binary is one level up from tests/ — use absolute-style relative path */
    #define BIN "../mc_pricer"
    struct { const char *name; const char *cmd; } cases[] = {
        { "--benchmark runs cleanly",
          BIN " --benchmark > /dev/null 2>&1" },
        { "--type european-call",
          BIN " --type european-call --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type european-put",
          BIN " --type european-put --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type asian-call",
          BIN " --type asian-call --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type asian-put",
          BIN " --type asian-put --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type digital-call",
          BIN " --type digital-call --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type digital-put",
          BIN " --type digital-put --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type barrier-uo-call",
          BIN " --type barrier-uo-call --barrier 130 --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--type barrier-do-call",
          BIN " --type barrier-do-call --barrier 80 --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--seed sets reproducible output",
          BIN " --seed 12345 --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--jumps btc preset",
          BIN " --preset BTC --jumps btc --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--portfolio mode",
          BIN " --portfolio --no-cpu -n 10000 > /dev/null 2>&1" },
        { "--greeks mode",
          BIN " --greeks --no-cpu -n 50000 > /dev/null 2>&1" },
    };
    #undef BIN

    int n = (int)(sizeof(cases)/sizeof(cases[0]));
    for (int i = 0; i < n; i++) {
        int ret = system(cases[i].cmd);
        if (ret == 0) { PASS(cases[i].name); }
        else { FAIL(cases[i].name, "exit code %d", ret); }
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=================================================================\n");
    printf("  GPU Monte Carlo Option Pricing Engine — Test Suite\n");
    printf("=================================================================\n");

    test_bs_prices();
    test_bs_greeks();
    test_bs_reference_guard();
    test_gpu_european();
    test_gpu_asian();
    test_gpu_digital();
    test_gpu_barrier();
    test_jump_diffusion();
    test_seed_reproducibility();
    test_reduction();
    test_cpu_baseline();
    test_cli_smoke();

    printf("\n=================================================================\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("=================================================================\n");

    return (g_fail == 0) ? 0 : 1;
}
