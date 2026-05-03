/*
 * main.cpp
 * OpenCL Monte Carlo European Option Pricer
 *
 * Demonstrates:
 *   - Platform/device discovery and context creation
 *   - Kernel compilation from .cl source files
 *   - Buffers and sub-buffers
 *   - Local (shared) memory reduction
 *   - float4 vector types in kernels
 *   - Event-based kernel profiling
 *   - Configurable CLI arguments
 *   - CPU baseline comparison
 *   - Black-Scholes analytical validation
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <CL/cl.h>
#include "../include/option_params.h"
#include "../include/ocl_helpers.h"

/* Declared in black_scholes.c / cpu_baseline.c */
extern "C" {
    float bs_call_price(const OptionParams *p);
    float bs_put_price(const OptionParams *p);
    PricingResult cpu_monte_carlo(
        const OptionParams *p, int paths, int steps);
}

/* Defaults */
#define DEF_SPOT    100.0f
#define DEF_STRIKE  100.0f
#define DEF_RATE    0.05f
#define DEF_VOL     0.20f
#define DEF_EXPIRY  1.0f
#define DEF_PATHS   1000000
#define DEF_STEPS   252
#define DEF_WG_SIZE 256

/* Reduction work-group size (must match kernel) */
#define RED_WG 256

/* -------------------------------------------------- */
static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("  -S <f>  Spot price       (%.1f)\n", DEF_SPOT);
    printf("  -K <f>  Strike price     (%.1f)\n", DEF_STRIKE);
    printf("  -r <f>  Risk-free rate   (%.2f)\n", DEF_RATE);
    printf("  -v <f>  Volatility       (%.2f)\n", DEF_VOL);
    printf("  -T <f>  Expiry (years)   (%.1f)\n", DEF_EXPIRY);
    printf("  -n <i>  Number of paths  (%d)\n", DEF_PATHS);
    printf("  -s <i>  Steps per path   (%d)\n", DEF_STEPS);
    printf("  -w <i>  Work-group size  (%d)\n", DEF_WG_SIZE);
    printf("  -p      Price a put (default: call)\n");
    printf("  --no-cpu  Skip CPU baseline\n");
    printf("  -h      Show this help\n");
}

/* -------------------------------------------------- */
/*  Host-side parallel reduction via sub-buffers      */
/* -------------------------------------------------- */
static float ocl_reduce_sum(
    cl_command_queue queue,
    cl_kernel        k_reduce,
    cl_context       context,
    cl_mem           d_input,
    int              n)
{
    cl_int err;
    cl_mem d_in = d_input;
    int cur_n = n;
    int allocated = 0; /* track if d_in needs freeing */

    while (cur_n > 1) {
        int grid = (cur_n + (2 * RED_WG) - 1)
                 / (2 * RED_WG);

        cl_mem d_out = clCreateBuffer(
            context, CL_MEM_READ_WRITE,
            grid * sizeof(float), NULL, &err);
        checkErr(err, "clCreateBuffer(reduce out)");

        /* Set kernel args */
        err  = clSetKernelArg(
            k_reduce, 0, sizeof(cl_mem), &d_in);
        err |= clSetKernelArg(
            k_reduce, 1, sizeof(cl_mem), &d_out);
        err |= clSetKernelArg(
            k_reduce, 2, RED_WG * sizeof(float), NULL);
        err |= clSetKernelArg(
            k_reduce, 3, sizeof(int), &cur_n);
        checkErr(err, "clSetKernelArg(reduce)");

        size_t gws = (size_t)grid * RED_WG;
        size_t lws = RED_WG;
        err = clEnqueueNDRangeKernel(
            queue, k_reduce, 1, NULL,
            &gws, &lws, 0, NULL, NULL);
        checkErr(err, "clEnqueueNDRangeKernel(reduce)");
        clFinish(queue);

        if (allocated) clReleaseMemObject(d_in);
        d_in = d_out;
        cur_n = grid;
        allocated = 1;
    }

    float result;
    err = clEnqueueReadBuffer(
        queue, d_in, CL_TRUE,
        0, sizeof(float), &result,
        0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer(reduce)");

    if (allocated) clReleaseMemObject(d_in);
    return result;
}

/* -------------------------------------------------- */
/*  main                                              */
/* -------------------------------------------------- */
int main(int argc, char **argv)
{
    /* Parse CLI ------------------------------------ */
    OptionParams params;
    params.S     = DEF_SPOT;
    params.K     = DEF_STRIKE;
    params.r     = DEF_RATE;
    params.sigma = DEF_VOL;
    params.T     = DEF_EXPIRY;
    params.type  = 0;

    int num_paths = DEF_PATHS;
    int num_steps = DEF_STEPS;
    int wg_size   = DEF_WG_SIZE;
    int run_cpu   = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S") && i+1 < argc)
            params.S = atof(argv[++i]);
        else if (!strcmp(argv[i], "-K") && i+1 < argc)
            params.K = atof(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i+1 < argc)
            params.r = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v") && i+1 < argc)
            params.sigma = atof(argv[++i]);
        else if (!strcmp(argv[i], "-T") && i+1 < argc)
            params.T = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i+1 < argc)
            num_paths = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1 < argc)
            num_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i+1 < argc)
            wg_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p"))
            params.type = 1;
        else if (!strcmp(argv[i], "--no-cpu"))
            run_cpu = 0;
        else if (!strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n",
                    argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* OpenCL setup -------------------------------- */
    cl_platform_id platform;
    cl_device_id   device;
    cl_context ctx = createContext(&platform, &device);
    cl_command_queue queue = createCommandQueue(
        ctx, device);

    /* Print header -------------------------------- */
    printf("========================================"
           "========================================\n");
    printf("  OpenCL Monte Carlo Option Pricer\n");
    printf("========================================"
           "========================================\n");
    printDeviceInfo(platform, device);
    printf("----------------------------------------"
           "----------------------------------------\n");
    const char *tname = params.type == 0
        ? "European Call" : "European Put";
    printf("Option Type:    %s\n", tname);
    printf("Spot (S):       %.2f\n", params.S);
    printf("Strike (K):     %.2f\n", params.K);
    printf("Rate (r):       %.4f\n", params.r);
    printf("Volatility:     %.4f\n", params.sigma);
    printf("Expiry (T):     %.2f years\n", params.T);
    printf("Paths:          %d\n", num_paths);
    printf("Steps/path:     %d\n", num_steps);
    printf("Work-group:     %d\n", wg_size);
    printf("========================================"
           "========================================\n\n");

    /* 1. Black-Scholes analytical ----------------- */
    float bs_price = (params.type == 0)
        ? bs_call_price(&params)
        : bs_put_price(&params);
    printf("[Black-Scholes Analytical]\n");
    printf("  Price:   $%.6f\n\n", bs_price);

    /* 2. CPU baseline ----------------------------- */
    PricingResult cpu_res;
    memset(&cpu_res, 0, sizeof(cpu_res));
    if (run_cpu) {
        printf("[CPU Monte Carlo]  "
               "(%d paths, %d steps)...\n",
               num_paths, num_steps);
        cpu_res = cpu_monte_carlo(
            &params, num_paths, num_steps);
        printf("  Price:   $%.6f\n", cpu_res.price);
        printf("  Std Err: $%.6f\n", cpu_res.std_err);
        printf("  95%% CI:  [$%.6f, $%.6f]\n",
               cpu_res.ci_low, cpu_res.ci_high);
        printf("  Time:    %.2f ms\n", cpu_res.time_ms);
        printf("  Error vs BS: %.4f%%\n",
               fabsf(cpu_res.price - bs_price)
               / bs_price * 100.0f);
        printf("\n");
    }

    /* 3. OpenCL Monte Carlo ----------------------- */
    printf("[OpenCL Monte Carlo]  "
           "(%d paths, %d steps, wg=%d)...\n",
           num_paths, num_steps, wg_size);

    /* Build kernels */
    cl_program prog_sim = createProgram(
        ctx, device, "kernels/mc_sim.cl");
    cl_program prog_red = createProgram(
        ctx, device, "kernels/reduction.cl");

    cl_int err;
    cl_kernel k_sim = clCreateKernel(
        prog_sim, "mc_simulate", &err);
    checkErr(err, "clCreateKernel(mc_simulate)");

    cl_kernel k_reduce = clCreateKernel(
        prog_red, "sum_reduce", &err);
    checkErr(err, "clCreateKernel(sum_reduce)");

    cl_kernel k_vardiff = clCreateKernel(
        prog_red, "variance_diff", &err);
    checkErr(err, "clCreateKernel(variance_diff)");

    /* Allocate main payoff buffer */
    cl_mem d_payoffs = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE,
        num_paths * sizeof(float), NULL, &err);
    checkErr(err, "clCreateBuffer(payoffs)");

    /* Create a sub-buffer for the first half —
     * demonstrates clCreateSubBuffer usage */
    int half = num_paths / 2;
    cl_buffer_region region_lo = {
        0, (size_t)(half * sizeof(float))
    };
    cl_mem d_pay_lo = clCreateSubBuffer(
        d_payoffs, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION,
        &region_lo, &err);
    checkErr(err, "clCreateSubBuffer(lo)");

    cl_buffer_region region_hi = {
        (size_t)(half * sizeof(float)),
        (size_t)((num_paths - half) * sizeof(float))
    };
    cl_mem d_pay_hi = clCreateSubBuffer(
        d_payoffs, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION,
        &region_hi, &err);
    checkErr(err, "clCreateSubBuffer(hi)");

    /* Generate RNG seed */
    cl_ulong seed = (cl_ulong)time(NULL);

    /* --- Launch MC kernel on lower sub-buffer --- */
    int lo_paths = half;
    err  = clSetKernelArg(k_sim, 0, sizeof(cl_mem),
                          &d_pay_lo);
    err |= clSetKernelArg(k_sim, 1, sizeof(int),
                          &lo_paths);
    err |= clSetKernelArg(k_sim, 2, sizeof(int),
                          &num_steps);
    err |= clSetKernelArg(k_sim, 3, sizeof(float),
                          &params.S);
    err |= clSetKernelArg(k_sim, 4, sizeof(float),
                          &params.K);
    err |= clSetKernelArg(k_sim, 5, sizeof(float),
                          &params.r);
    err |= clSetKernelArg(k_sim, 6, sizeof(float),
                          &params.sigma);
    err |= clSetKernelArg(k_sim, 7, sizeof(float),
                          &params.T);
    err |= clSetKernelArg(k_sim, 8, sizeof(int),
                          &params.type);
    err |= clSetKernelArg(k_sim, 9, sizeof(cl_ulong),
                          &seed);
    checkErr(err, "clSetKernelArg(sim lo)");

    size_t gws_lo = ((lo_paths + wg_size - 1)
                    / wg_size) * wg_size;
    size_t lws = (size_t)wg_size;

    cl_event ev_lo;
    err = clEnqueueNDRangeKernel(
        queue, k_sim, 1, NULL,
        &gws_lo, &lws, 0, NULL, &ev_lo);
    checkErr(err, "clEnqueueNDRangeKernel(sim lo)");

    /* --- Launch MC kernel on upper sub-buffer --- */
    int hi_paths = num_paths - half;
    cl_ulong seed_hi = seed ^ 0xDEADBEEFCAFEUL;
    err  = clSetKernelArg(k_sim, 0, sizeof(cl_mem),
                          &d_pay_hi);
    err |= clSetKernelArg(k_sim, 1, sizeof(int),
                          &hi_paths);
    err |= clSetKernelArg(k_sim, 9, sizeof(cl_ulong),
                          &seed_hi);
    checkErr(err, "clSetKernelArg(sim hi)");

    size_t gws_hi = ((hi_paths + wg_size - 1)
                    / wg_size) * wg_size;

    cl_event ev_hi;
    err = clEnqueueNDRangeKernel(
        queue, k_sim, 1, NULL,
        &gws_hi, &lws, 0, NULL, &ev_hi);
    checkErr(err, "clEnqueueNDRangeKernel(sim hi)");

    /* Wait for both simulation halves */
    clWaitForEvents(1, &ev_lo);
    clWaitForEvents(1, &ev_hi);

    /* --- Profiling: measure kernel time --- */
    cl_ulong t_start_lo, t_end_lo;
    cl_ulong t_start_hi, t_end_hi;
    clGetEventProfilingInfo(
        ev_lo, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &t_start_lo, NULL);
    clGetEventProfilingInfo(
        ev_lo, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &t_end_lo, NULL);
    clGetEventProfilingInfo(
        ev_hi, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &t_start_hi, NULL);
    clGetEventProfilingInfo(
        ev_hi, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &t_end_hi, NULL);

    /* Overall wall time: earliest start → latest end */
    cl_ulong t_start = (t_start_lo < t_start_hi)
                      ? t_start_lo : t_start_hi;
    cl_ulong t_end   = (t_end_lo > t_end_hi)
                      ? t_end_lo : t_end_hi;
    float sim_ms = (float)(t_end - t_start) / 1.0e6f;

    /* --- Reduction: compute mean payoff --- */
    float sum = ocl_reduce_sum(
        queue, k_reduce, ctx, d_payoffs, num_paths);
    float mean = sum / (float)num_paths;

    /* --- Variance kernel --- */
    cl_mem d_sq = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE,
        num_paths * sizeof(float), NULL, &err);
    checkErr(err, "clCreateBuffer(sq)");

    err  = clSetKernelArg(
        k_vardiff, 0, sizeof(cl_mem), &d_payoffs);
    err |= clSetKernelArg(
        k_vardiff, 1, sizeof(cl_mem), &d_sq);
    err |= clSetKernelArg(
        k_vardiff, 2, sizeof(float), &mean);
    err |= clSetKernelArg(
        k_vardiff, 3, sizeof(int), &num_paths);
    checkErr(err, "clSetKernelArg(vardiff)");

    size_t gws_var = ((num_paths + wg_size - 1)
                     / wg_size) * wg_size;
    err = clEnqueueNDRangeKernel(
        queue, k_vardiff, 1, NULL,
        &gws_var, &lws, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel(vardiff)");
    clFinish(queue);

    float var_sum = ocl_reduce_sum(
        queue, k_reduce, ctx, d_sq, num_paths);
    float variance = var_sum / (float)num_paths;

    /* --- Compute final price and statistics --- */
    /* Payoffs are already discounted in the kernel */
    float gpu_price = mean;
    float std_err   = sqrtf(variance / (float)num_paths);
    float ci_low    = gpu_price - 1.96f * std_err;
    float ci_high   = gpu_price + 1.96f * std_err;

    printf("  Price:   $%.6f\n", gpu_price);
    printf("  Std Err: $%.6f\n", std_err);
    printf("  95%% CI:  [$%.6f, $%.6f]\n",
           ci_low, ci_high);
    printf("  Sim kernel: %.2f ms\n", sim_ms);
    printf("  Error vs BS: %.4f%%\n",
           fabsf(gpu_price - bs_price)
           / bs_price * 100.0f);

    if (run_cpu && cpu_res.time_ms > 0.0f) {
        float speedup = cpu_res.time_ms / sim_ms;
        printf("  Speedup: %.1fx over CPU\n", speedup);
    }

    float pps = (float)num_paths
              / (sim_ms / 1000.0f);
    printf("  Throughput: %.2f M paths/sec\n",
           pps / 1e6f);
    printf("\n");

    /* Summary table -------------------------------- */
    printf("========================================"
           "========================================\n");
    printf("  SUMMARY\n");
    printf("========================================"
           "========================================\n");
    printf("  %-20s %12s %12s %12s\n",
           "Method", "Price", "Err vs BS", "Time(ms)");
    printf("  %-20s %12.6f %11s  %12s\n",
           "Black-Scholes", bs_price, "-", "-");
    if (run_cpu) {
        printf("  %-20s %12.6f %10.4f%%  %12.2f\n",
               "CPU Monte Carlo",
               cpu_res.price,
               fabsf(cpu_res.price - bs_price)
               / bs_price * 100.0f,
               cpu_res.time_ms);
    }
    printf("  %-20s %12.6f %10.4f%%  %12.2f\n",
           "OpenCL Monte Carlo",
           gpu_price,
           fabsf(gpu_price - bs_price)
           / bs_price * 100.0f,
           sim_ms);
    printf("========================================"
           "========================================\n");

    /* Cleanup -------------------------------------- */
    clReleaseMemObject(d_sq);
    clReleaseMemObject(d_pay_lo);
    clReleaseMemObject(d_pay_hi);
    clReleaseMemObject(d_payoffs);
    clReleaseKernel(k_sim);
    clReleaseKernel(k_reduce);
    clReleaseKernel(k_vardiff);
    clReleaseProgram(prog_sim);
    clReleaseProgram(prog_red);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    clReleaseEvent(ev_lo);
    clReleaseEvent(ev_hi);

    return 0;
}
