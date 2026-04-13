#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/option_params.h"
#include "black_scholes.h"
#include "cpu_baseline.h"
#include "monte_carlo.h"
#include "reduction.h"
#include "render_ppm.h"
#include "greeks.h"
#include "streams.h"
#include "dashboard.h"
#include "gl_dashboard.h"

/* Default option parameters (standard test case) */
#define DEFAULT_SPOT       100.0f
#define DEFAULT_STRIKE     100.0f
#define DEFAULT_RATE       0.05f
#define DEFAULT_VOL        0.20f
#define DEFAULT_EXPIRY     1.0f
#define DEFAULT_PATHS      1000000
#define DEFAULT_STEPS      252
#define DEFAULT_BLOCK_SIZE 256

/* Preset underlyings with realistic parameters */
typedef struct {
    const char *name;
    float spot;
    float strike;
    float vol;
    float barrier;  /* suggested barrier for barrier options */
} Preset;

static const Preset PRESETS[] = {
    { "BTC",  96800.0f, 100000.0f, 0.65f, 130000.0f },
    { "ETH",   1820.0f,   2000.0f, 0.75f,   2800.0f },
    { "SPY",    563.0f,    565.0f, 0.16f,    620.0f },
    { "AAPL",   198.0f,    200.0f, 0.25f,    250.0f },
    { "TSLA",   272.0f,    280.0f, 0.55f,    400.0f },
    { "NVDA",   110.0f,    115.0f, 0.45f,    160.0f },
};
#define NUM_PRESETS (sizeof(PRESETS) / sizeof(PRESETS[0]))

/* Jump-diffusion preset profiles */
typedef struct {
    const char *name;
    float lambda;  /* jumps/year */
    float mean;    /* mean log-jump size */
    float vol;     /* jump size volatility */
} JumpPreset;

static const JumpPreset JUMP_PRESETS[] = {
    { "btc",     3.0f, -0.10f, 0.20f },   /* BTC: ~3 crashes/yr, avg -10%, high variance */
    { "crypto",  4.0f, -0.12f, 0.25f },   /* Generic crypto: more frequent, deeper crashes */
    { "equity",  0.5f, -0.08f, 0.10f },   /* Stocks: rare crashes, moderate depth */
    { "crisis",  1.0f, -0.30f, 0.15f },   /* Black swan: rare but devastating */
    { "mild",    2.0f, -0.03f, 0.05f },   /* Mild: frequent small jumps */
};
#define NUM_JUMP_PRESETS (sizeof(JUMP_PRESETS) / sizeof(JUMP_PRESETS[0]))

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -S <float>   Spot price          (default: %.1f)\n", DEFAULT_SPOT);
    printf("  -K <float>   Strike price         (default: %.1f)\n", DEFAULT_STRIKE);
    printf("  -r <float>   Risk-free rate        (default: %.2f)\n", DEFAULT_RATE);
    printf("  -v <float>   Volatility (sigma)    (default: %.2f)\n", DEFAULT_VOL);
    printf("  -T <float>   Time to expiry (yrs)  (default: %.1f)\n", DEFAULT_EXPIRY);
    printf("  -n <int>     Number of paths        (default: %d)\n", DEFAULT_PATHS);
    printf("  -s <int>     Number of time steps   (default: %d)\n", DEFAULT_STEPS);
    printf("  -b <int>     CUDA block size         (default: %d)\n", DEFAULT_BLOCK_SIZE);
    printf("  -p           Price put instead of call\n");
    printf("  --type <str> Option type: european, asian, barrier-uo, barrier-do, digital\n");
    printf("  --barrier <float>  Barrier level (for barrier options)\n");
    printf("  --greeks     Compute Greeks via bump-and-reprice\n");
    printf("  --jumps <str>  Jump-diffusion preset: btc, crypto, equity, crisis, mild\n");
    printf("  --jump-lambda <float>  Jump intensity (jumps/year)\n");
    printf("  --jump-mean <float>    Mean log-jump size (e.g. -0.10)\n");
    printf("  --jump-vol <float>     Jump size volatility\n");
    printf("  --no-cpu     Skip CPU baseline\n");
    printf("  --visualize  Generate spaghetti plot + histogram PPMs\n");
    printf("  --viz-paths <int>  Paths to draw in visualization (default: 5000)\n");
    printf("  --dashboard  Launch interactive ncurses terminal dashboard\n");
    printf("  --gui        Launch OpenGL graphical dashboard\n");
    printf("  --preset <str>   Load preset: BTC, ETH, SPY, AAPL, TSLA, NVDA\n");
    printf("  --portfolio  Demo: price a sample portfolio using CUDA streams\n");
    printf("  --streams <int>  Number of CUDA streams for portfolio (default: 4)\n");
    printf("  -h           Show this help\n");
}

int main(int argc, char **argv) {
    /* Default parameters */
    OptionParams params;
    params.S       = DEFAULT_SPOT;
    params.K       = DEFAULT_STRIKE;
    params.r       = DEFAULT_RATE;
    params.sigma   = DEFAULT_VOL;
    params.T       = DEFAULT_EXPIRY;
    params.barrier = 0.0f;
    params.type    = OPTION_EUROPEAN_CALL;
    params.jump_lambda = 0.0f;
    params.jump_mean   = 0.0f;
    params.jump_vol    = 0.0f;

    int num_paths  = DEFAULT_PATHS;
    int num_steps  = DEFAULT_STEPS;
    int block_size = DEFAULT_BLOCK_SIZE;
    int run_cpu    = 1;
    int visualize  = 0;
    int viz_paths  = 5000;
    int show_greeks = 0;
    int portfolio_mode = 0;
    int num_streams = 4;
    const char *underlying_name = "Custom";
    int dashboard_mode = 0;
    int gui_mode = 0;

    /* Parse CLI arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-S") == 0 && i + 1 < argc) {
            params.S = atof(argv[++i]);
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            params.K = atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            params.r = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            params.sigma = atof(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            params.T = atof(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_paths = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            num_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0) {
            params.type = OPTION_EUROPEAN_PUT;
        } else if (strcmp(argv[i], "--no-cpu") == 0) {
            run_cpu = 0;
        } else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "european") == 0)       params.type = OPTION_EUROPEAN_CALL;
            else if (strcmp(argv[i], "asian") == 0)     params.type = OPTION_ASIAN_CALL;
            else if (strcmp(argv[i], "barrier-uo") == 0) params.type = OPTION_BARRIER_UP_AND_OUT_CALL;
            else if (strcmp(argv[i], "barrier-do") == 0) params.type = OPTION_BARRIER_DOWN_AND_OUT_CALL;
            else if (strcmp(argv[i], "digital") == 0)   params.type = OPTION_DIGITAL_CALL;
            else { fprintf(stderr, "Unknown type: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--barrier") == 0 && i + 1 < argc) {
            params.barrier = atof(argv[++i]);
        } else if (strcmp(argv[i], "--greeks") == 0) {
            show_greeks = 1;
        } else if (strcmp(argv[i], "--jumps") == 0 && i + 1 < argc) {
            i++;
            int jfound = 0;
            for (int p = 0; p < (int)NUM_JUMP_PRESETS; p++) {
                if (strcasecmp(argv[i], JUMP_PRESETS[p].name) == 0) {
                    params.jump_lambda = JUMP_PRESETS[p].lambda;
                    params.jump_mean = JUMP_PRESETS[p].mean;
                    params.jump_vol = JUMP_PRESETS[p].vol;
                    jfound = 1;
                    break;
                }
            }
            if (!jfound) {
                fprintf(stderr, "Unknown jump preset: %s\nAvailable: ", argv[i]);
                for (int p = 0; p < (int)NUM_JUMP_PRESETS; p++) fprintf(stderr, "%s ", JUMP_PRESETS[p].name);
                fprintf(stderr, "\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--jump-lambda") == 0 && i + 1 < argc) {
            params.jump_lambda = atof(argv[++i]);
        } else if (strcmp(argv[i], "--jump-mean") == 0 && i + 1 < argc) {
            params.jump_mean = atof(argv[++i]);
        } else if (strcmp(argv[i], "--jump-vol") == 0 && i + 1 < argc) {
            params.jump_vol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dashboard") == 0) {
            dashboard_mode = 1;
        } else if (strcmp(argv[i], "--gui") == 0) {
            gui_mode = 1;
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            i++;
            int found = 0;
            for (int p = 0; p < (int)NUM_PRESETS; p++) {
                if (strcasecmp(argv[i], PRESETS[p].name) == 0) {
                    params.S = PRESETS[p].spot;
                    params.K = PRESETS[p].strike;
                    params.sigma = PRESETS[p].vol;
                    if (params.barrier == 0.0f) params.barrier = PRESETS[p].barrier;
                    underlying_name = PRESETS[p].name;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "Unknown preset: %s\nAvailable: ", argv[i]);
                for (int p = 0; p < (int)NUM_PRESETS; p++) fprintf(stderr, "%s ", PRESETS[p].name);
                fprintf(stderr, "\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--portfolio") == 0) {
            portfolio_mode = 1;
        } else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            num_streams = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--visualize") == 0) {
            visualize = 1;
        } else if (strcmp(argv[i], "--viz-paths") == 0 && i + 1 < argc) {
            viz_paths = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Dashboard mode — launch immediately if requested */
    if (gui_mode) {
        launch_gl_dashboard(&params, num_paths, num_steps, block_size);
        return 0;
    }
    if (dashboard_mode) {
        launch_dashboard(&params, num_paths, num_steps, block_size);
        return 0;
    }

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("================================================================\n");
    printf("  GPU Monte Carlo Option Pricing Engine\n");
    printf("================================================================\n");
    printf("GPU:            %s\n", prop.name);
    printf("CUDA Cores:     %d SMs x %d = ~%d cores\n",
           prop.multiProcessorCount, 128, prop.multiProcessorCount * 128);
    printf("Memory:         %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("Compute:        %d.%d\n", prop.major, prop.minor);
    printf("Underlying:     %s\n", underlying_name);
    printf("----------------------------------------------------------------\n");
    const char *type_name;
    switch (params.type) {
        case OPTION_EUROPEAN_CALL:              type_name = "European Call"; break;
        case OPTION_EUROPEAN_PUT:               type_name = "European Put"; break;
        case OPTION_ASIAN_CALL:                 type_name = "Asian Call"; break;
        case OPTION_ASIAN_PUT:                  type_name = "Asian Put"; break;
        case OPTION_BARRIER_UP_AND_OUT_CALL:    type_name = "Barrier Up-and-Out Call"; break;
        case OPTION_BARRIER_DOWN_AND_OUT_CALL:  type_name = "Barrier Down-and-Out Call"; break;
        case OPTION_DIGITAL_CALL:               type_name = "Digital Call"; break;
        case OPTION_DIGITAL_PUT:                type_name = "Digital Put"; break;
        default:                               type_name = "Unknown"; break;
    }
    printf("Option Type:    %s\n", type_name);
    printf("Spot (S):       %.2f\n", params.S);
    printf("Strike (K):     %.2f\n", params.K);
    printf("Rate (r):       %.4f\n", params.r);
    printf("Volatility (σ): %.4f\n", params.sigma);
    printf("Expiry (T):     %.2f years\n", params.T);
    if (params.barrier > 0.0f)
        printf("Barrier:        %.2f\n", params.barrier);
    if (params.jump_lambda > 0.0f) {
        printf("Jump Model:     Merton Jump-Diffusion\n");
        printf("  λ (jumps/yr): %.2f\n", params.jump_lambda);
        printf("  μ_J (mean):   %.4f\n", params.jump_mean);
        printf("  σ_J (vol):    %.4f\n", params.jump_vol);
    }
    printf("Paths:          %d\n", num_paths);
    printf("Steps/path:     %d\n", num_steps);
    printf("Block size:     %d\n", block_size);
    printf("================================================================\n\n");

    /* 1. Analytical Black-Scholes price */
    float bs_price;
    if (params.type == OPTION_EUROPEAN_CALL) {
        bs_price = bs_call_price(&params);
    } else {
        bs_price = bs_put_price(&params);
    }

    float delta, gamma, vega, theta;
    bs_greeks(&params, &delta, &gamma, &vega, &theta);

    printf("[Black-Scholes Analytical]\n");
    printf("  Price:   $%.6f\n", bs_price);
    printf("  Delta:   %.6f\n", delta);
    printf("  Gamma:   %.6f\n", gamma);
    printf("  Vega:    %.6f\n", vega);
    printf("  Theta:   %.6f\n", theta);
    printf("\n");

    /* 2. CPU Monte Carlo baseline */
    PricingResult cpu_result = {0};
    if (run_cpu) {
        printf("[CPU Monte Carlo]  (%d paths, %d steps)...\n", num_paths, num_steps);
        cpu_result = cpu_monte_carlo(&params, num_paths, num_steps);
        printf("  Price:   $%.6f\n", cpu_result.price);
        printf("  Std Err: $%.6f\n", cpu_result.std_err);
        printf("  95%% CI:  [$%.6f, $%.6f]\n", cpu_result.ci_low, cpu_result.ci_high);
        printf("  Time:    %.2f ms\n", cpu_result.cpu_time_ms);
        printf("  Error vs BS: %.4f%%\n",
               fabsf(cpu_result.price - bs_price) / bs_price * 100.0f);
        printf("\n");
    }

    /* 3. GPU Monte Carlo */
    printf("[GPU Monte Carlo]  (%d paths, %d steps, block=%d)...\n",
           num_paths, num_steps, block_size);

    /* Copy params to constant memory */
    mc_set_params(&params);

    /* Initialize RNG states */
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_total);

    curandState *d_states = mc_init_rng(num_paths, block_size);

    /* Run simulation */
    cudaEventRecord(start_kernel);
    float *d_payoffs = mc_run_simulation(d_states, num_paths, num_steps, block_size);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    /* Reduce payoffs */
    float mean_payoff, variance;
    reduce_payoffs(d_payoffs, num_paths, &mean_payoff, &variance);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    /* Compute price and stats */
    float discount = expf(-params.r * params.T);
    float gpu_price = discount * mean_payoff;
    float std_err = discount * sqrtf(variance / (float)num_paths);
    float ci_low = gpu_price - 1.96f * std_err;
    float ci_high = gpu_price + 1.96f * std_err;

    float kernel_ms, total_ms;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    printf("  Price:   $%.6f\n", gpu_price);
    printf("  Std Err: $%.6f\n", std_err);
    printf("  95%% CI:  [$%.6f, $%.6f]\n", ci_low, ci_high);
    printf("  Kernel:  %.2f ms\n", kernel_ms);
    printf("  Total:   %.2f ms  (includes RNG init + reduction)\n", total_ms);
    printf("  Error vs BS: %.4f%%\n",
           fabsf(gpu_price - bs_price) / bs_price * 100.0f);

    if (run_cpu && cpu_result.cpu_time_ms > 0.0f) {
        float speedup = cpu_result.cpu_time_ms / total_ms;
        printf("  Speedup: %.1fx over CPU\n", speedup);
    }

    float paths_per_sec = (float)num_paths / (kernel_ms / 1000.0f);
    printf("  Throughput: %.2f M paths/sec\n", paths_per_sec / 1e6f);
    printf("\n");

    /* Summary comparison */
    printf("================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================\n");
    printf("  %-20s %12s %12s %12s\n", "Method", "Price", "Error vs BS", "Time (ms)");
    printf("  %-20s %12.6f %11s  %12s\n", "Black-Scholes", bs_price, "-", "-");
    if (run_cpu) {
        printf("  %-20s %12.6f %10.4f%%  %12.2f\n", "CPU Monte Carlo",
               cpu_result.price,
               fabsf(cpu_result.price - bs_price) / bs_price * 100.0f,
               cpu_result.cpu_time_ms);
    }
    printf("  %-20s %12.6f %10.4f%%  %12.2f\n", "GPU Monte Carlo",
           gpu_price,
           fabsf(gpu_price - bs_price) / bs_price * 100.0f,
           total_ms);
    printf("================================================================\n");

    /* 4. Greeks via bump-and-reprice */
    if (show_greeks) {
        printf("\n[GPU Greeks]  (bump-and-reprice, %d paths each)...\n", num_paths);
        Greeks mc_greeks = compute_greeks_gpu(&params, num_paths, num_steps, block_size);
        printf("  Delta:  %+.6f\n", mc_greeks.delta);
        printf("  Gamma:  %+.6f\n", mc_greeks.gamma);
        printf("  Vega:   %+.6f\n", mc_greeks.vega);
        printf("  Theta:  %+.6f\n", mc_greeks.theta);
        printf("  Rho:    %+.6f\n", mc_greeks.rho);

        if (params.type == OPTION_EUROPEAN_CALL || params.type == OPTION_EUROPEAN_PUT) {
            printf("\n  %-8s %12s %12s\n", "Greek", "MC GPU", "Analytical");
            printf("  %-8s %+12.6f %+12.6f\n", "Delta", mc_greeks.delta, delta);
            printf("  %-8s %+12.6f %+12.6f\n", "Gamma", mc_greeks.gamma, gamma);
            printf("  %-8s %+12.6f %+12.6f\n", "Vega", mc_greeks.vega, vega);
            printf("  %-8s %+12.6f %+12.6f\n", "Theta", mc_greeks.theta, theta);
        }
    }

    /* 5. Visualization */
    if (visualize) {
        printf("\n");
        render_spaghetti_ppm(&params, viz_paths, num_steps, block_size,
                             1920, 1080, "output/spaghetti.ppm");
        render_histogram_ppm(d_payoffs, num_paths,
                             800, 600, "output/histogram.ppm");
    }

    /* 6. Portfolio mode */
    if (portfolio_mode) {
        /* Build a sample portfolio of 8 diverse options */
        PortfolioEntry portfolio[8];

        /* EU Call ATM */
        portfolio[0].params = params;
        portfolio[0].params.type = OPTION_EUROPEAN_CALL;
        portfolio[0].params.K = 100.0f;

        /* EU Put ATM */
        portfolio[1].params = params;
        portfolio[1].params.type = OPTION_EUROPEAN_PUT;
        portfolio[1].params.K = 100.0f;

        /* EU Call OTM */
        portfolio[2].params = params;
        portfolio[2].params.type = OPTION_EUROPEAN_CALL;
        portfolio[2].params.K = 110.0f;

        /* EU Put ITM */
        portfolio[3].params = params;
        portfolio[3].params.type = OPTION_EUROPEAN_PUT;
        portfolio[3].params.K = 110.0f;

        /* Asian Call */
        portfolio[4].params = params;
        portfolio[4].params.type = OPTION_ASIAN_CALL;
        portfolio[4].params.K = 100.0f;

        /* Digital Call */
        portfolio[5].params = params;
        portfolio[5].params.type = OPTION_DIGITAL_CALL;
        portfolio[5].params.K = 100.0f;

        /* Barrier Up-and-Out Call */
        portfolio[6].params = params;
        portfolio[6].params.type = OPTION_BARRIER_UP_AND_OUT_CALL;
        portfolio[6].params.K = 100.0f;
        portfolio[6].params.barrier = 130.0f;

        /* EU Call high vol */
        portfolio[7].params = params;
        portfolio[7].params.type = OPTION_EUROPEAN_CALL;
        portfolio[7].params.K = 100.0f;
        portfolio[7].params.sigma = 0.40f;

        streams_price_portfolio(portfolio, 8, num_paths, num_steps,
                                block_size, num_streams);
    }

    /* Cleanup */
    cudaFree(d_states);
    cudaFree(d_payoffs);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
