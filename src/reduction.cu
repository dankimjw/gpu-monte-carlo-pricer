#include <stdio.h>
#include "../include/option_params.h"
#include "reduction.h"

#define REDUCE_BLOCK 256

/* Kernel: parallel sum reduction using shared memory
 * Each block reduces its chunk, writes partial sum to output */
__global__ void sum_reduce_kernel(const float *input, float *output, int n) {
    __shared__ float sdata[REDUCE_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    /* Load two elements per thread into shared memory */
    float val = 0.0f;
    if (idx < n) val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    /* Reduction in shared memory */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* Kernel: compute (x - mean)^2 for each element, store in output */
__global__ void variance_kernel(const float *input, float *output,
                                float mean, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = input[idx] - mean;
        output[idx] = diff * diff;
    }
}

/* Host helper: fully reduce a device array to a single sum */
static float device_sum(float *d_input, int n) {
    float *d_in = d_input;
    int current_n = n;
    int allocated_temp = 0;

    while (current_n > 1) {
        int grid = (current_n + (2 * REDUCE_BLOCK) - 1) / (2 * REDUCE_BLOCK);
        float *d_out;
        cudaMalloc(&d_out, grid * sizeof(float));

        sum_reduce_kernel<<<grid, REDUCE_BLOCK>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        /* Free intermediate buffer (not the original input) */
        if (allocated_temp) cudaFree(d_in);

        d_in = d_out;
        current_n = grid;
        allocated_temp = 1;
    }

    float result;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    if (allocated_temp) cudaFree(d_in);

    return result;
}

/* Reduce payoffs array: compute mean and variance */
void reduce_payoffs(float *d_payoffs, int num_paths,
                    float *mean_out, float *variance_out) {
    /* Step 1: compute sum → mean */
    float sum = device_sum(d_payoffs, num_paths);
    float mean = sum / (float)num_paths;

    /* Step 2: compute variance = mean((x - mean)^2) */
    float *d_sq_diff;
    cudaMalloc(&d_sq_diff, num_paths * sizeof(float));

    int grid = (num_paths + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    variance_kernel<<<grid, REDUCE_BLOCK>>>(d_payoffs, d_sq_diff, mean, num_paths);
    cudaDeviceSynchronize();

    float var_sum = device_sum(d_sq_diff, num_paths);
    float variance = var_sum / (float)num_paths;

    cudaFree(d_sq_diff);

    *mean_out = mean;
    *variance_out = variance;
}
