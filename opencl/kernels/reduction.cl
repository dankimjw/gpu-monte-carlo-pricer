/*
 * reduction.cl
 * Parallel sum reduction using local (shared) memory.
 * Demonstrates: __local memory, work-group barriers,
 *               two-element-per-thread loading pattern.
 *
 * Also contains a variance helper kernel.
 */

/* --------------------------------------------------------
 * sum_reduce — each work-group reduces its portion of the
 * input array.  Each thread loads two elements.
 * The partial sums are written to the output array
 * (one element per work-group).
 * ------------------------------------------------------ */
__kernel void sum_reduce(
    __global const float *input,
    __global       float *output,
    __local        float *sdata,
    const int             n)
{
    unsigned int tid = get_local_id(0);
    unsigned int bsz = get_local_size(0);
    unsigned int idx = get_group_id(0) * (2 * bsz) + tid;

    /* Load two elements per thread */
    float val = 0.0f;
    if (idx < n)       val += input[idx];
    if (idx + bsz < n) val += input[idx + bsz];
    sdata[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Tree reduction in local memory */
    for (unsigned int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Work-group leader writes result */
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

/* --------------------------------------------------------
 * variance_diff — compute (x - mean)^2 for each element
 * ------------------------------------------------------ */
__kernel void variance_diff(
    __global const float *input,
    __global       float *output,
    const float           mean,
    const int             n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float d = input[gid] - mean;
        output[gid] = d * d;
    }
}
