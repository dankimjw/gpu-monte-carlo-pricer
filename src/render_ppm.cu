#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include "../include/option_params.h"
#include "render_ppm.h"
#include "monte_carlo.h"

/* RGB pixel */
typedef struct { unsigned char r, g, b; } Pixel;

/* ---------- Minimal 5x7 bitmap font (ASCII 32-90) ---------- */
static const unsigned char FONT5X7[][7] = {
 /* ' ' */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
 /* '!' */ {0x04,0x04,0x04,0x04,0x00,0x04,0x00},
 /* '"' */ {0x0A,0x0A,0x00,0x00,0x00,0x00,0x00},
 /* '#' */ {0x0A,0x1F,0x0A,0x0A,0x1F,0x0A,0x00},
 /* '$' */ {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04},
 /* '%' */ {0x18,0x19,0x02,0x04,0x08,0x13,0x03},
 /* '&' */ {0x0C,0x12,0x14,0x08,0x15,0x12,0x0D},
 /* '\'' */ {0x04,0x04,0x00,0x00,0x00,0x00,0x00},
 /* '(' */ {0x02,0x04,0x08,0x08,0x08,0x04,0x02},
 /* ')' */ {0x08,0x04,0x02,0x02,0x02,0x04,0x08},
 /* '*' */ {0x00,0x04,0x15,0x0E,0x15,0x04,0x00},
 /* '+' */ {0x00,0x04,0x04,0x1F,0x04,0x04,0x00},
 /* ',' */ {0x00,0x00,0x00,0x00,0x06,0x04,0x08},
 /* '-' */ {0x00,0x00,0x00,0x1F,0x00,0x00,0x00},
 /* '.' */ {0x00,0x00,0x00,0x00,0x00,0x06,0x06},
 /* '/' */ {0x00,0x01,0x02,0x04,0x08,0x10,0x00},
 /* '0' */ {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
 /* '1' */ {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
 /* '2' */ {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F},
 /* '3' */ {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E},
 /* '4' */ {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
 /* '5' */ {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
 /* '6' */ {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E},
 /* '7' */ {0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
 /* '8' */ {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
 /* '9' */ {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
 /* ':' */ {0x00,0x06,0x06,0x00,0x06,0x06,0x00},
 /* ';' */ {0x00,0x06,0x06,0x00,0x06,0x04,0x08},
 /* '<' */ {0x02,0x04,0x08,0x10,0x08,0x04,0x02},
 /* '=' */ {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00},
 /* '>' */ {0x08,0x04,0x02,0x01,0x02,0x04,0x08},
 /* '?' */ {0x0E,0x11,0x01,0x06,0x04,0x00,0x04},
 /* '@' */ {0x0E,0x11,0x01,0x0D,0x15,0x15,0x0E},
 /* 'A' */ {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11},
 /* 'B' */ {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E},
 /* 'C' */ {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E},
 /* 'D' */ {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C},
 /* 'E' */ {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F},
 /* 'F' */ {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10},
 /* 'G' */ {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F},
 /* 'H' */ {0x11,0x11,0x11,0x1F,0x11,0x11,0x11},
 /* 'I' */ {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E},
 /* 'J' */ {0x07,0x02,0x02,0x02,0x02,0x12,0x0C},
 /* 'K' */ {0x11,0x12,0x14,0x18,0x14,0x12,0x11},
 /* 'L' */ {0x10,0x10,0x10,0x10,0x10,0x10,0x1F},
 /* 'M' */ {0x11,0x1B,0x15,0x15,0x11,0x11,0x11},
 /* 'N' */ {0x11,0x19,0x15,0x13,0x11,0x11,0x11},
 /* 'O' */ {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E},
 /* 'P' */ {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10},
 /* 'Q' */ {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D},
 /* 'R' */ {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11},
 /* 'S' */ {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E},
 /* 'T' */ {0x1F,0x04,0x04,0x04,0x04,0x04,0x04},
 /* 'U' */ {0x11,0x11,0x11,0x11,0x11,0x11,0x0E},
 /* 'V' */ {0x11,0x11,0x11,0x11,0x11,0x0A,0x04},
 /* 'W' */ {0x11,0x11,0x15,0x15,0x15,0x15,0x0A},
 /* 'X' */ {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11},
 /* 'Y' */ {0x11,0x11,0x0A,0x04,0x04,0x04,0x04},
 /* 'Z' */ {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F},
};

/* Draw a single character at (px, py) with given scale and color */
static void draw_char(Pixel *fb, int fw, int fh, int px, int py,
                      char c, int scale,
                      unsigned char cr, unsigned char cg, unsigned char cb) {
    int idx;
    if (c < 32 || c > 90) c = 32; /* clamp to font range; lowercase treated as space */
    const unsigned char *glyph = FONT5X7[(unsigned char)(c - 32)];
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[row] & (0x10 >> col)) {
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int x = px + col * scale + sx;
                        int y = py + row * scale + sy;
                        if (x >= 0 && x < fw && y >= 0 && y < fh) {
                            idx = y * fw + x;
                            fb[idx].r = cr; fb[idx].g = cg; fb[idx].b = cb;
                        }
                    }
                }
            }
        }
    }
}

/* Draw a string — uppercased automatically to match font range */
static void draw_text(Pixel *fb, int fw, int fh, int px, int py,
                      const char *str, int scale,
                      unsigned char cr, unsigned char cg, unsigned char cb) {
    int x = px;
    for (int i = 0; str[i]; i++) {
        char c = str[i];
        if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
        draw_char(fb, fw, fh, x, py, c, scale, cr, cg, cb);
        x += (5 + 1) * scale;
    }
}

/* Fill a solid rectangle */
static void fill_rect(Pixel *fb, int fw, int fh,
                      int x0, int y0, int w, int h,
                      unsigned char cr, unsigned char cg, unsigned char cb) {
    for (int y = y0; y < y0 + h && y < fh; y++)
        for (int x = x0; x < x0 + w && x < fw; x++)
            { fb[y*fw+x].r=cr; fb[y*fw+x].g=cg; fb[y*fw+x].b=cb; }
}

/* Kernel: simulate paths and draw them onto the framebuffer
 * Each thread simulates one path and plots it pixel by pixel */
__global__ void spaghetti_kernel(curandState *states, Pixel *framebuffer,
                                 int num_paths, int num_steps,
                                 int width, int height,
                                 float S0, float K, float r, float sigma, float T,
                                 float y_min, float y_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    float dt = T / (float)num_steps;
    float drift = (r - 0.5f * sigma * sigma) * dt;
    float diffusion = sigma * sqrtf(dt);

    curandState local_state = states[idx];

    float S_prev = S0;
    int x_prev = 0;
    int y_prev = (int)((S_prev - y_min) / (y_max - y_min) * (height - 1));
    y_prev = max(0, min(height - 1, y_prev));

    float S_t = S0;

    for (int step = 1; step <= num_steps; step++) {
        float Z = curand_normal(&local_state);
        S_t *= expf(drift + diffusion * Z);

        /* Map step to x pixel */
        int x = (int)((float)step / (float)num_steps * (width - 1));

        /* Map price to y pixel (inverted: top = high price) */
        int y = height - 1 - (int)((S_t - y_min) / (y_max - y_min) * (height - 1));
        y = max(0, min(height - 1, y));

        /* Determine color: green if currently above strike, red if below */
        unsigned char cr, cg, cb;
        if (S_t >= K) {
            cr = 0; cg = 180; cb = 80;    /* green - ITM for call */
        } else {
            cr = 200; cg = 50; cb = 50;   /* red - OTM for call */
        }

        /* Draw line from (x_prev, y_prev) to (x, y) using Bresenham-ish */
        int dx = abs(x - x_prev);
        int dy = abs(y - y_prev);
        int sx = (x_prev < x) ? 1 : -1;
        int sy = (y_prev < y) ? 1 : -1;
        int err = dx - dy;
        int cx = x_prev, cy = y_prev;

        for (int iter = 0; iter < dx + dy + 1; iter++) {
            if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                int pidx = cy * width + cx;
                /* Direct write: last-writer-wins (races are benign for visualization) */
                framebuffer[pidx].r = cr;
                framebuffer[pidx].g = cg;
                framebuffer[pidx].b = cb;
            }
            if (cx == x && cy == y) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; cx += sx; }
            if (e2 <  dx) { err += dx; cy += sy; }
        }

        x_prev = x;
        y_prev = y;
    }

    states[idx] = local_state;
}

/* Kernel: draw the strike price line */
__global__ void draw_strike_line(Pixel *framebuffer, int width, int height,
                                 float K, float y_min, float y_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int y = height - 1 - (int)((K - y_min) / (y_max - y_min) * (height - 1));
    if (y >= 0 && y < height) {
        /* Dashed white line */
        if ((x / 8) % 2 == 0) {
            int pidx = y * width + x;
            framebuffer[pidx].r = 255;
            framebuffer[pidx].g = 255;
            framebuffer[pidx].b = 255;
        }
    }
}

/* Kernel: draw grid lines for reference */
__global__ void draw_grid(Pixel *framebuffer, int width, int height,
                          float y_min, float y_max, float grid_step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    for (float price = ceilf(y_min / grid_step) * grid_step; price < y_max; price += grid_step) {
        int y = height - 1 - (int)((price - y_min) / (y_max - y_min) * (height - 1));
        if (y >= 0 && y < height && (x % 4 == 0)) {
            int pidx = y * width + x;
            framebuffer[pidx].r = 60;
            framebuffer[pidx].g = 60;
            framebuffer[pidx].b = 60;
        }
    }
}

/* Write framebuffer to PPM file */
static void write_ppm(const char *filename, Pixel *pixels, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(pixels, sizeof(Pixel), width * height, fp);
    fclose(fp);
}

/* Render spaghetti plot */
void render_spaghetti_ppm(const OptionParams *params, int num_paths,
                          int num_steps, int block_size,
                          int width, int height, const char *filename) {
    printf("[Spaghetti Plot]  %d paths, %dx%d px...\n", num_paths, width, height);

    /* Set price range: center on spot, ±3 sigma */
    float range = params->S * params->sigma * sqrtf(params->T) * 3.0f;
    float y_min = params->S - range;
    float y_max = params->S + range;
    if (y_min < 0.0f) y_min = 0.0f;

    /* Allocate framebuffer on GPU */
    int num_pixels = width * height;
    Pixel *d_fb;
    cudaMalloc(&d_fb, num_pixels * sizeof(Pixel));

    /* Fill with dark background */
    Pixel *h_fb = (Pixel *)calloc(num_pixels, sizeof(Pixel));
    for (int i = 0; i < num_pixels; i++) {
        h_fb[i].r = 15; h_fb[i].g = 15; h_fb[i].b = 25;
    }
    cudaMemcpy(d_fb, h_fb, num_pixels * sizeof(Pixel), cudaMemcpyHostToDevice);

    /* Draw grid */
    float grid_step = 10.0f;
    if (range > 100.0f) grid_step = 25.0f;
    int grid_blocks = (width + 255) / 256;
    draw_grid<<<grid_blocks, 256>>>(d_fb, width, height, y_min, y_max, grid_step);
    cudaDeviceSynchronize();

    /* Initialize RNG */
    mc_set_params(params);
    curandState *d_states = mc_init_rng(num_paths, block_size, (unsigned long)time(NULL));

    /* Draw paths */
    int path_grid = (num_paths + block_size - 1) / block_size;
    spaghetti_kernel<<<path_grid, block_size>>>(d_states, d_fb, num_paths, num_steps,
                                                  width, height,
                                                  params->S, params->K, params->r,
                                                  params->sigma, params->T,
                                                  y_min, y_max);
    cudaDeviceSynchronize();

    /* Draw strike line on top */
    draw_strike_line<<<grid_blocks, 256>>>(d_fb, width, height,
                                            params->K, y_min, y_max);
    cudaDeviceSynchronize();

    /* Copy back */
    cudaMemcpy(h_fb, d_fb, num_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);

    /* --- CPU-side annotation: title bar at top --- */
    {
        int bar_h = 28;
        fill_rect(h_fb, width, height, 0, 0, width, bar_h, 20, 20, 35);
        /* Title */
        char title[128];
        snprintf(title, sizeof(title),
                 "GPU MONTE CARLO  S=%.0f  K=%.0f  VOL=%.0f%%  T=%.1FYR  PATHS=%d",
                 params->S, params->K, params->sigma * 100.0f, params->T, num_paths);
        draw_text(h_fb, width, height, 10, 6, title, 2, 220, 220, 220);
        /* Legend: green square = ITM, red square = OTM */
        int lx = width - 340;
        fill_rect(h_fb, width, height, lx,      8, 12, 12,  0, 180,  80);
        draw_text(h_fb, width, height, lx + 16, 7, "ITM", 2, 0, 180, 80);
        fill_rect(h_fb, width, height, lx + 80, 8, 12, 12, 200,  50,  50);
        draw_text(h_fb, width, height, lx + 96, 7, "OTM", 2, 200, 50, 50);
        /* Strike label near dashed line */
        int ky = height - 1 - (int)((params->K - y_min) / (y_max - y_min) * (height - 1));
        if (ky > bar_h + 10 && ky < height - 10) {
            char klabel[32];
            snprintf(klabel, sizeof(klabel), "K=%.0f", params->K);
            draw_text(h_fb, width, height, 6, ky - 8, klabel, 2, 255, 255, 200);
        }
    }

    write_ppm(filename, h_fb, width, height);
    printf("  Saved %s\n", filename);

    cudaFree(d_fb);
    cudaFree(d_states);
    free(h_fb);
}

/* Histogram kernel: each thread atomically increments a bin */
__global__ void histogram_kernel(const float *payoffs, int *bins,
                                 int num_paths, int num_bins,
                                 float bin_min, float bin_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    float val = payoffs[idx];
    int bin = (int)((val - bin_min) / bin_width);
    if (bin < 0) bin = 0;
    if (bin >= num_bins) bin = num_bins - 1;
    atomicAdd(&bins[bin], 1);
}

/* Render payoff histogram */
void render_histogram_ppm(float *d_payoffs, int num_paths,
                          int width, int height, const char *filename) {
    printf("[Histogram]  %d paths, %dx%d px...\n", num_paths, width, height);

    /* Copy all payoffs to host for flexible binning */
    float *h_payoffs = (float *)malloc(num_paths * sizeof(float));
    cudaMemcpy(h_payoffs, d_payoffs, num_paths * sizeof(float),
               cudaMemcpyDeviceToHost);

    /* Separate ITM payoffs and find range */
    float max_payoff = 0.0f;
    int itm_count = 0;
    int otm_count = 0;
    for (int i = 0; i < num_paths; i++) {
        if (h_payoffs[i] > 0.001f) {
            itm_count++;
            if (h_payoffs[i] > max_payoff) max_payoff = h_payoffs[i];
        } else {
            otm_count++;
        }
    }

    if (max_payoff < 1.0f) max_payoff = 1.0f;

    int margin_left = 60;
    int margin_right = 20;
    int margin_top = 40;
    int margin_bottom = 40;
    int plot_w = width - margin_left - margin_right;
    int plot_h = height - margin_top - margin_bottom;

    int num_bins = (plot_w < 200) ? plot_w : 200;
    float bin_width = max_payoff / (float)num_bins;

    /* Build histogram of ITM payoffs only */
    int *bins = (int *)calloc(num_bins, sizeof(int));
    for (int i = 0; i < num_paths; i++) {
        if (h_payoffs[i] > 0.001f) {
            int b = (int)(h_payoffs[i] / bin_width);
            if (b >= num_bins) b = num_bins - 1;
            bins[b]++;
        }
    }

    int max_count = 0;
    for (int i = 0; i < num_bins; i++) {
        if (bins[i] > max_count) max_count = bins[i];
    }

    /* Render to framebuffer */
    Pixel *h_fb = (Pixel *)calloc(width * height, sizeof(Pixel));
    for (int i = 0; i < width * height; i++) {
        h_fb[i].r = 20; h_fb[i].g = 20; h_fb[i].b = 30;
    }

    /* Draw bars */
    int bar_w = plot_w / num_bins;
    if (bar_w < 1) bar_w = 1;

    for (int b = 0; b < num_bins; b++) {
        float frac = (max_count > 0) ? (float)bins[b] / (float)max_count : 0.0f;
        int bar_h = (int)(frac * plot_h);

        /* Color: gradient from cyan (low payoff) to green (high payoff) */
        float t = (float)b / (float)num_bins;
        unsigned char cr = (unsigned char)(30 + 20 * t);
        unsigned char cg = (unsigned char)(180 + 60 * t);
        unsigned char cb = (unsigned char)(220 * (1.0f - t * 0.8f));

        int x_start = margin_left + b * bar_w;
        for (int bw = 0; bw < bar_w - 1 && bw < bar_w; bw++) {
            int x = x_start + bw;
            if (x >= width) break;
            for (int dy = 0; dy < bar_h; dy++) {
                int y = height - margin_bottom - 1 - dy;
                if (y >= 0 && y < height) {
                    /* Slight gradient within each bar (lighter at top) */
                    float vy = (float)dy / (float)(bar_h > 0 ? bar_h : 1);
                    h_fb[y * width + x].r = (unsigned char)(cr + (int)(30 * vy));
                    h_fb[y * width + x].g = (unsigned char)fminf(255, cg + 20 * vy);
                    h_fb[y * width + x].b = (unsigned char)fminf(255, cb + 10 * vy);
                }
            }
        }
    }

    /* Draw axes */
    for (int x = margin_left; x < width - margin_right; x++) {
        int y = height - margin_bottom;
        if (y < height) { h_fb[y * width + x].r = 100; h_fb[y * width + x].g = 100; h_fb[y * width + x].b = 100; }
    }
    for (int y = margin_top; y < height - margin_bottom; y++) {
        int x = margin_left;
        h_fb[y * width + x].r = 100; h_fb[y * width + x].g = 100; h_fb[y * width + x].b = 100;
    }

    /* --- CPU-side annotation: title bar + axis labels --- */
    {
        int bar_h = 28;
        fill_rect(h_fb, width, height, 0, 0, width, bar_h, 20, 20, 35);
        char title[128];
        snprintf(title, sizeof(title),
                 "PAYOFF DISTRIBUTION  %d PATHS  ITM=%d (%.1f%%)  MAX=$%.1f",
                 num_paths, itm_count,
                 100.0f * itm_count / (itm_count + otm_count > 0 ? itm_count + otm_count : 1),
                 max_payoff);
        draw_text(h_fb, width, height, 10, 6, title, 2, 220, 220, 220);
        /* X-axis label */
        draw_text(h_fb, width, height, width/2 - 80, height - 14,
                  "PAYOFF ($)", 2, 160, 160, 160);
        /* Y-axis label (rotated not possible without more code — just print at top-left) */
        draw_text(h_fb, width, height, 2, margin_top + 2,
                  "COUNT", 2, 160, 160, 160);
        /* X-axis tick: $0 and max */
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "$0");
        draw_text(h_fb, width, height, margin_left, height - margin_bottom + 4, tmp, 2, 130, 130, 130);
        snprintf(tmp, sizeof(tmp), "$%.0F", max_payoff);
        draw_text(h_fb, width, height, width - margin_right - 60, height - margin_bottom + 4, tmp, 2, 130, 130, 130);
    }

    write_ppm(filename, h_fb, width, height);
    printf("  Saved %s  (ITM: %d, OTM: %d, max payoff: $%.1f)\n",
           filename, itm_count, otm_count, max_payoff);

    free(h_payoffs);
    free(bins);
    free(h_fb);
}
