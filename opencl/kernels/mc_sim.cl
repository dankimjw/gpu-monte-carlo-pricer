/*
 * mc_sim.cl
 * Monte Carlo GBM path simulation kernel.
 * Uses xorshift128+ PRNG and Box-Muller transform.
 * Demonstrates: global/local IDs, buffers, float4 vectors.
 */

/* --------------------------------------------------------
 * xorshift128+ state — two 64-bit words per generator
 * Produces high-quality pseudo-random numbers suitable
 * for Monte Carlo work without requiring cuRAND.
 * ------------------------------------------------------ */
typedef struct {
    ulong s0;
    ulong s1;
} rng_state_t;

/* Advance xorshift128+ and return a 64-bit value */
static ulong xorshift128plus(rng_state_t *st)
{
    ulong s1 = st->s0;
    ulong s0 = st->s1;
    st->s0 = s0;
    s1 ^= (s1 << 23);
    s1 ^= (s1 >> 17);
    s1 ^= s0;
    s1 ^= (s0 >> 26);
    st->s1 = s1;
    return st->s0 + st->s1;
}

/* Convert 64-bit integer to uniform float in (0,1] */
static float rand_uniform(rng_state_t *st)
{
    ulong v = xorshift128plus(st);
    /* Use upper 24 bits for float precision */
    return (float)(v >> 40) / 16777216.0f + 1.0e-12f;
}

/* Box-Muller: generate a standard normal variate */
static float rand_normal(rng_state_t *st)
{
    float u1 = rand_uniform(st);
    float u2 = rand_uniform(st);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

/* --------------------------------------------------------
 * Generate four normal variates at once using float4
 * vector type.  Demonstrates OpenCL vector operations.
 * ------------------------------------------------------ */
static float4 rand_normal4(rng_state_t *st)
{
    float u1 = rand_uniform(st);
    float u2 = rand_uniform(st);
    float u3 = rand_uniform(st);
    float u4 = rand_uniform(st);

    float r1 = sqrt(-2.0f * log(u1));
    float r2 = sqrt(-2.0f * log(u3));
    float t1 = 2.0f * M_PI_F * u2;
    float t2 = 2.0f * M_PI_F * u4;

    return (float4)(
        r1 * cos(t1),
        r1 * sin(t1),
        r2 * cos(t2),
        r2 * sin(t2));
}

/* --------------------------------------------------------
 * MC simulation kernel
 * Each work-item simulates one full GBM price path and
 * writes the discounted payoff to the output buffer.
 *
 * Arguments:
 *   payoffs   — output buffer (one float per path)
 *   num_paths — total number of paths
 *   num_steps — time steps per path
 *   S, K, r, sigma, T — option parameters
 *   opt_type  — 0 = call, 1 = put
 *   seed      — base seed for RNG
 * ------------------------------------------------------ */
__kernel void mc_simulate(
    __global float *payoffs,
    const int       num_paths,
    const int       num_steps,
    const float     S,
    const float     K,
    const float     r,
    const float     sigma,
    const float     T,
    const int       opt_type,
    const ulong     seed)
{
    int gid = get_global_id(0);
    if (gid >= num_paths) return;

    /* Seed RNG: mix global id with base seed */
    rng_state_t rng;
    rng.s0 = seed ^ ((ulong)gid * 6364136223846793005UL);
    rng.s1 = seed ^ ((ulong)(gid + 1) * 1442695040888963407UL);
    /* Warm up RNG */
    for (int w = 0; w < 8; w++) xorshift128plus(&rng);

    float dt        = T / (float)num_steps;
    float drift     = (r - 0.5f * sigma * sigma) * dt;
    float diffusion = sigma * sqrt(dt);
    float discount  = exp(-r * T);

    /* Simulate path using float4 vectorised normals
     * Process 4 steps at a time where possible */
    float S_t = S;
    int step = 0;
    int vec_steps = (num_steps / 4) * 4;

    for (; step < vec_steps; step += 4) {
        float4 Z = rand_normal4(&rng);
        S_t *= exp(drift + diffusion * Z.x);
        S_t *= exp(drift + diffusion * Z.y);
        S_t *= exp(drift + diffusion * Z.z);
        S_t *= exp(drift + diffusion * Z.w);
    }
    /* Remaining steps */
    for (; step < num_steps; step++) {
        float Z = rand_normal(&rng);
        S_t *= exp(drift + diffusion * Z);
    }

    /* Compute payoff */
    float payoff;
    if (opt_type == 0) {
        payoff = fmax(S_t - K, 0.0f);
    } else {
        payoff = fmax(K - S_t, 0.0f);
    }

    payoffs[gid] = discount * payoff;
}
