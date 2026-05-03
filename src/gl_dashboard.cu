#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/option_params.h"
#include "gl_dashboard.h"
#include "monte_carlo.h"
#include "reduction.h"
#include "black_scholes.h"
#include "greeks.h"

/* ─── Configuration ─── */
#define WIN_W 1400
#define WIN_H 900
#define MAX_HISTORY 500
#define MAX_HIST_BINS 120

/* ─── Colors ─── */
static void set_color(float r, float g, float b, float a) { glColor4f(r, g, b, a); }

/* ─── Bitmap font rendering (no external deps) ─── */
/* Minimal 5x7 pixel font for digits, letters, and symbols */
static const unsigned char FONT_5X7[][7] = {
    /* ' ' */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    /* '!' */ {0x04,0x04,0x04,0x04,0x04,0x00,0x04},
    /* '"' */ {0x0A,0x0A,0x00,0x00,0x00,0x00,0x00},
    /* '#' */ {0x0A,0x1F,0x0A,0x0A,0x1F,0x0A,0x00},
    /* '$' */ {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04},
    /* '%' */ {0x19,0x19,0x02,0x04,0x08,0x13,0x13},
    /* '&' */ {0x08,0x14,0x14,0x08,0x15,0x12,0x0D},
    /* ''' */ {0x04,0x04,0x00,0x00,0x00,0x00,0x00},
    /* '(' */ {0x02,0x04,0x08,0x08,0x08,0x04,0x02},
    /* ')' */ {0x08,0x04,0x02,0x02,0x02,0x04,0x08},
    /* '*' */ {0x00,0x04,0x15,0x0E,0x15,0x04,0x00},
    /* '+' */ {0x00,0x04,0x04,0x1F,0x04,0x04,0x00},
    /* ',' */ {0x00,0x00,0x00,0x00,0x00,0x04,0x08},
    /* '-' */ {0x00,0x00,0x00,0x1F,0x00,0x00,0x00},
    /* '.' */ {0x00,0x00,0x00,0x00,0x00,0x00,0x04},
    /* '/' */ {0x01,0x01,0x02,0x04,0x08,0x10,0x10},
    /* '0' */ {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
    /* '1' */ {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
    /* '2' */ {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F},
    /* '3' */ {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E},
    /* '4' */ {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
    /* '5' */ {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
    /* '6' */ {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E},
    /* '7' */ {0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
    /* '8' */ {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
    /* '9' */ {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
    /* ':' */ {0x00,0x00,0x04,0x00,0x04,0x00,0x00},
    /* ';' */ {0x00,0x00,0x04,0x00,0x04,0x04,0x08},
    /* '<' */ {0x02,0x04,0x08,0x10,0x08,0x04,0x02},
    /* '=' */ {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00},
    /* '>' */ {0x08,0x04,0x02,0x01,0x02,0x04,0x08},
    /* '?' */ {0x0E,0x11,0x01,0x02,0x04,0x00,0x04},
    /* '@' */ {0x0E,0x11,0x17,0x15,0x17,0x10,0x0E},
    /* A-Z */
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, /* A */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, /* B */
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, /* C */
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, /* D */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, /* E */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, /* F */
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, /* G */
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, /* H */
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, /* I */
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, /* J */
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, /* K */
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, /* L */
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, /* M */
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, /* N */
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* O */
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, /* P */
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, /* Q */
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, /* R */
    {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E}, /* S */
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, /* T */
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, /* U */
    {0x11,0x11,0x11,0x11,0x0A,0x0A,0x04}, /* V */
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, /* W */
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, /* X */
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, /* Y */
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, /* Z */
};

static int font_index(char c) {
    if (c >= ' ' && c <= '@') return c - ' ';
    if (c >= 'A' && c <= 'Z') return c - 'A' + ('@' - ' ' + 1);
    if (c >= 'a' && c <= 'z') return c - 'a' + ('@' - ' ' + 1); /* lowercase = uppercase */
    return 0;
}

static void draw_bitmap_text(float x, float y, const char *str, float sz) {
    float cx = x;
    for (int i = 0; str[i]; i++) {
        if (str[i] == '\n') { y -= sz * 9; cx = x; continue; }
        int idx = font_index(str[i]);
        const unsigned char *glyph = FONT_5X7[idx];
        for (int row = 0; row < 7; row++) {
            for (int col = 0; col < 5; col++) {
                if (glyph[row] & (0x10 >> col)) {
                    float px = cx + col * sz;
                    float py = y - row * sz;
                    glBegin(GL_QUADS);
                    glVertex2f(px, py);
                    glVertex2f(px + sz, py);
                    glVertex2f(px + sz, py - sz);
                    glVertex2f(px, py - sz);
                    glEnd();
                }
            }
        }
        cx += sz * 6;
    }
}

/* ─── Draw primitives ─── */
static void draw_rect(float x, float y, float w, float h) {
    glBegin(GL_QUADS);
    glVertex2f(x, y); glVertex2f(x + w, y);
    glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void draw_rect_outline(float x, float y, float w, float h) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y); glVertex2f(x + w, y);
    glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void draw_line(float x1, float y1, float x2, float y2) {
    glBegin(GL_LINES);
    glVertex2f(x1, y1); glVertex2f(x2, y2);
    glEnd();
}

/* ─── State ─── */
typedef struct {
    OptionParams params;
    int num_paths;
    int num_steps;
    int block_size;

    /* Price history */
    float price_history[MAX_HISTORY];
    float time_history[MAX_HISTORY];
    int hist_count;

    /* Histogram bins */
    float hist_bins[MAX_HIST_BINS];
    int hist_bin_count;
    float hist_max_payoff;

    /* Latest results */
    float mc_price;
    float bs_price;
    float std_err;
    float ci_low, ci_high;
    float kernel_ms;
    float total_ms;
    float throughput;
    int iteration;
    float avg_price;
    float cum_time_ms;

    /* Greeks */
    float delta, gamma_val, vega, theta, rho;
    int greeks_computed;
} DashState;

static DashState g_state;

/* ─── MC pricing helper ─── */
static void run_mc_tick(DashState *st) {
    cudaEvent_t start, stop, ks, ke;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&ks); cudaEventCreate(&ke);

    cudaEventRecord(start);
    mc_set_params(&st->params);
    curandState *d_states = mc_init_rng(st->num_paths, st->block_size, (unsigned long)time(NULL));

    cudaEventRecord(ks);
    float *d_payoffs = mc_run_simulation(d_states, st->num_paths, st->num_steps, st->block_size);
    cudaEventRecord(ke);
    cudaEventSynchronize(ke);

    float mean_payoff, variance;
    reduce_payoffs(d_payoffs, st->num_paths, &mean_payoff, &variance);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float discount = expf(-st->params.r * st->params.T);
    st->mc_price = discount * mean_payoff;
    st->std_err = discount * sqrtf(variance / (float)st->num_paths);
    st->ci_low = st->mc_price - 1.96f * st->std_err;
    st->ci_high = st->mc_price + 1.96f * st->std_err;

    cudaEventElapsedTime(&st->kernel_ms, ks, ke);
    cudaEventElapsedTime(&st->total_ms, start, stop);
    st->throughput = (float)st->num_paths / (st->kernel_ms / 1000.0f) / 1e6f;
    st->iteration++;
    st->cum_time_ms += st->total_ms;

    /* Update history */
    if (st->hist_count < MAX_HISTORY) {
        st->price_history[st->hist_count] = st->mc_price;
        st->time_history[st->hist_count] = st->total_ms;
        st->hist_count++;
    } else {
        memmove(st->price_history, st->price_history + 1, (MAX_HISTORY - 1) * sizeof(float));
        memmove(st->time_history, st->time_history + 1, (MAX_HISTORY - 1) * sizeof(float));
        st->price_history[MAX_HISTORY - 1] = st->mc_price;
        st->time_history[MAX_HISTORY - 1] = st->total_ms;
    }

    /* Running average */
    float sum = 0.0f;
    for (int i = 0; i < st->hist_count; i++) sum += st->price_history[i];
    st->avg_price = sum / (float)st->hist_count;

    /* Build histogram from payoffs */
    float *h_payoffs = (float *)malloc(st->num_paths * sizeof(float));
    cudaMemcpy(h_payoffs, d_payoffs, st->num_paths * sizeof(float), cudaMemcpyDeviceToHost);

    float max_pay = 0.0f;
    for (int i = 0; i < st->num_paths; i++) {
        if (h_payoffs[i] > max_pay) max_pay = h_payoffs[i];
    }
    st->hist_max_payoff = max_pay;
    st->hist_bin_count = MAX_HIST_BINS;

    memset(st->hist_bins, 0, sizeof(st->hist_bins));
    if (max_pay > 0.0f) {
        for (int i = 0; i < st->num_paths; i++) {
            if (h_payoffs[i] > 0.0f) {
                int bin = (int)(h_payoffs[i] / max_pay * (MAX_HIST_BINS - 1));
                if (bin >= MAX_HIST_BINS) bin = MAX_HIST_BINS - 1;
                st->hist_bins[bin] += 1.0f;
            }
        }
        /* Normalize */
        float mx = 0.0f;
        for (int i = 0; i < MAX_HIST_BINS; i++)
            if (st->hist_bins[i] > mx) mx = st->hist_bins[i];
        if (mx > 0.0f)
            for (int i = 0; i < MAX_HIST_BINS; i++)
                st->hist_bins[i] /= mx;
    }

    free(h_payoffs);
    cudaFree(d_states);
    cudaFree(d_payoffs);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(ks); cudaEventDestroy(ke);

    /* BS reference */
    if (st->params.type == OPTION_EUROPEAN_CALL)
        st->bs_price = bs_call_price(&st->params);
    else if (st->params.type == OPTION_EUROPEAN_PUT)
        st->bs_price = bs_put_price(&st->params);
    else
        st->bs_price = 0.0f;

    /* Greeks — compute every 10 iterations using reduced path count */
    if (st->iteration % 10 == 1) {
        int greek_paths = st->num_paths / 4;
        if (greek_paths < 10000) greek_paths = 10000;
        Greeks g = compute_greeks_gpu(&st->params, greek_paths, st->num_steps, st->block_size);
        st->delta       = g.delta;
        st->gamma_val   = g.gamma;
        st->vega        = g.vega;
        st->theta       = g.theta;
        st->rho         = g.rho;
        st->greeks_computed = 1;
        /* Restore params after Greeks bumping */
        mc_set_params(&st->params);
    }
}

/* ─── Draw the panels ─── */

static void draw_panel_bg(float x, float y, float w, float h, const char *title) {
    /* Panel background */
    set_color(0.12f, 0.14f, 0.18f, 0.95f);
    draw_rect(x, y, w, h);
    /* Border */
    set_color(0.3f, 0.5f, 0.7f, 0.8f);
    glLineWidth(1.5f);
    draw_rect_outline(x, y, w, h);
    /* Title bar */
    set_color(0.15f, 0.35f, 0.55f, 1.0f);
    draw_rect(x, y + h - 22, w, 22);
    set_color(1.0f, 1.0f, 1.0f, 1.0f);
    draw_bitmap_text(x + 6, y + h - 6, title, 2.0f);
}

static void draw_price_chart(float px, float py, float pw, float ph, DashState *st) {
    draw_panel_bg(px, py, pw, ph, "PRICE HISTORY");

    float cx = px + 10, cy = py + 10;
    float cw = pw - 20, ch = ph - 40;

    if (st->hist_count < 2) return;

    float mn = st->price_history[0], mx = st->price_history[0];
    for (int i = 1; i < st->hist_count; i++) {
        if (st->price_history[i] < mn) mn = st->price_history[i];
        if (st->price_history[i] > mx) mx = st->price_history[i];
    }
    float pad = (mx - mn) * 0.1f;
    if (pad < 0.01f) pad = 1.0f;
    mn -= pad; mx += pad;

    /* Grid lines */
    set_color(0.25f, 0.28f, 0.32f, 0.5f);
    for (int i = 0; i <= 4; i++) {
        float gy = cy + ch * i / 4.0f;
        draw_line(cx, gy, cx + cw, gy);
    }

    /* BS reference line */
    if (st->bs_price > 0.0f && st->bs_price >= mn && st->bs_price <= mx) {
        float by = cy + (st->bs_price - mn) / (mx - mn) * ch;
        set_color(1.0f, 0.8f, 0.0f, 0.6f);
        glLineWidth(1.0f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0xAAAA);
        draw_line(cx, by, cx + cw, by);
        glDisable(GL_LINE_STIPPLE);
        set_color(1.0f, 0.8f, 0.0f, 0.8f);
        draw_bitmap_text(cx + cw - 25 * 2.0f * 6, by + 2, "BS ANALYTICAL", 1.5f);
    }

    /* Price line */
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < st->hist_count; i++) {
        float t = (float)i / (float)(MAX_HISTORY - 1);
        float v = (st->price_history[i] - mn) / (mx - mn);
        /* Color gradient: green if above avg, red if below */
        if (st->price_history[i] >= st->avg_price)
            set_color(0.2f, 0.9f, 0.4f, 1.0f);
        else
            set_color(0.9f, 0.3f, 0.3f, 1.0f);
        glVertex2f(cx + t * cw, cy + v * ch);
    }
    glEnd();

    /* Confidence interval band */
    if (st->ci_low >= mn && st->ci_high <= mx) {
        float y_lo = cy + (st->ci_low - mn) / (mx - mn) * ch;
        float y_hi = cy + (st->ci_high - mn) / (mx - mn) * ch;
        set_color(0.2f, 0.6f, 1.0f, 0.12f);
        draw_rect(cx + cw * 0.9f, y_lo, cw * 0.1f, y_hi - y_lo);
    }

    /* Y-axis labels */
    char buf[64];
    set_color(0.7f, 0.7f, 0.7f, 1.0f);
    snprintf(buf, sizeof(buf), "$%.2f", mx);
    draw_bitmap_text(cx + 2, cy + ch - 2, buf, 1.5f);
    snprintf(buf, sizeof(buf), "$%.2f", mn);
    draw_bitmap_text(cx + 2, cy + 12, buf, 1.5f);

    /* Legend box — bottom right of chart */
    float lx = cx + cw - 180, ly = cy + 8;
    float lw = 175, lh = 68;
    set_color(0.08f, 0.10f, 0.14f, 0.85f);
    draw_rect(lx, ly, lw, lh);
    set_color(0.3f, 0.4f, 0.5f, 0.6f);
    draw_rect_outline(lx, ly, lw, lh);

    float row = ly + lh - 14;
    float swx = lx + 6, tsx = lx + 22;
    float fs = 1.5f;

    /* Green swatch */
    set_color(0.2f, 0.9f, 0.4f, 1.0f);
    draw_rect(swx, row, 12, 8);
    set_color(0.85f, 0.85f, 0.85f, 1.0f);
    draw_bitmap_text(tsx, row + 6, "MC PRICE > AVG", fs);
    row -= 16;

    /* Red swatch */
    set_color(0.9f, 0.3f, 0.3f, 1.0f);
    draw_rect(swx, row, 12, 8);
    set_color(0.85f, 0.85f, 0.85f, 1.0f);
    draw_bitmap_text(tsx, row + 6, "MC PRICE < AVG", fs);
    row -= 16;

    /* Yellow swatch */
    set_color(1.0f, 0.8f, 0.0f, 0.8f);
    draw_rect(swx, row, 12, 3);
    set_color(0.85f, 0.85f, 0.85f, 1.0f);
    draw_bitmap_text(tsx, row + 6, "BLACK-SCHOLES", fs);
    row -= 16;

    /* Blue swatch */
    set_color(0.2f, 0.6f, 1.0f, 0.5f);
    draw_rect(swx, row, 12, 8);
    set_color(0.85f, 0.85f, 0.85f, 1.0f);
    draw_bitmap_text(tsx, row + 6, "95% CONF. INTERVAL", fs);
}

static void draw_histogram(float px, float py, float pw, float ph, DashState *st) {
    draw_panel_bg(px, py, pw, ph, "PAYOFF DISTRIBUTION");

    float cx = px + 10, cy = py + 10;
    float cw = pw - 20, ch = ph - 40;
    float bar_w = cw / (float)MAX_HIST_BINS;

    for (int i = 0; i < MAX_HIST_BINS; i++) {
        float h = st->hist_bins[i] * ch;
        if (h < 1.0f) continue;

        /* Color gradient: cyan → blue */
        float t = (float)i / (float)MAX_HIST_BINS;
        set_color(0.0f + t * 0.2f, 0.7f - t * 0.3f, 0.9f, 0.9f);
        draw_rect(cx + i * bar_w, cy, bar_w - 1, h);
    }

    /* Axis */
    set_color(0.5f, 0.5f, 0.5f, 0.8f);
    draw_line(cx, cy, cx + cw, cy);
    draw_line(cx, cy, cx, cy + ch);
}

static void draw_params_panel(float px, float py, float pw, float ph, DashState *st) {
    draw_panel_bg(px, py, pw, ph, "PARAMETERS");

    float tx = px + 10, ty = py + ph - 35;
    float fs = 1.8f;
    float lh = 16.0f;
    char buf[128];

    const char *type_names[] = {
        "EU CALL", "EU PUT", "ASIAN CALL", "ASIAN PUT",
        "BARRIER UO", "BARRIER DO", "DIGITAL C", "DIGITAL P"
    };

    set_color(0.6f, 0.85f, 1.0f, 1.0f);
    snprintf(buf, sizeof(buf), "TYPE: %s", type_names[st->params.type]);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;

    set_color(0.9f, 0.9f, 0.9f, 1.0f);
    snprintf(buf, sizeof(buf), "SPOT:  $%.2f", st->params.S);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "STRIKE:$%.2f", st->params.K);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "VOL:   %.2f%%", st->params.sigma * 100.0f);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "RATE:  %.2f%%", st->params.r * 100.0f);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "EXPIRY:%.2f YR", st->params.T);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;

    if (st->params.jump_lambda > 0.0f) {
        ty -= 4;
        set_color(1.0f, 0.6f, 0.3f, 1.0f);
        snprintf(buf, sizeof(buf), "JUMPS: L=%.1f", st->params.jump_lambda);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        snprintf(buf, sizeof(buf), "  MEAN=%.2f", st->params.jump_mean);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    }

    ty -= 4;
    set_color(0.5f, 0.7f, 0.5f, 1.0f);
    snprintf(buf, sizeof(buf), "PATHS: %d", st->num_paths);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "STEPS: %d", st->num_steps);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
}

static void draw_results_panel(float px, float py, float pw, float ph, DashState *st) {
    draw_panel_bg(px, py, pw, ph, "RESULTS");

    float tx = px + 10, ty = py + ph - 35;
    float fs = 1.8f;
    float lh = 16.0f;
    char buf[128];

    set_color(0.3f, 1.0f, 0.5f, 1.0f);
    snprintf(buf, sizeof(buf), "MC PRICE: $%.2f", st->mc_price);
    draw_bitmap_text(tx, ty, buf, fs * 1.1f); ty -= lh * 1.2f;

    set_color(1.0f, 0.9f, 0.3f, 1.0f);
    snprintf(buf, sizeof(buf), "AVG:      $%.2f", st->avg_price);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;

    set_color(0.8f, 0.8f, 0.8f, 1.0f);
    snprintf(buf, sizeof(buf), "STD ERR:  $%.4f", st->std_err);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "95CI: [%.1f,%.1f]", st->ci_low, st->ci_high);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;

    if (st->bs_price > 0.0f) {
        set_color(0.3f, 0.7f, 1.0f, 1.0f);
        snprintf(buf, sizeof(buf), "BS:       $%.2f", st->bs_price);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        float err = fabsf(st->avg_price - st->bs_price) / st->bs_price * 100.0f;
        if (err < 1.0f) set_color(0.3f, 1.0f, 0.5f, 1.0f);
        else set_color(1.0f, 0.4f, 0.3f, 1.0f);
        snprintf(buf, sizeof(buf), "ERROR:    %.4f%%", err);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    }

    ty -= 6;
    set_color(0.5f, 0.8f, 0.9f, 1.0f);
    snprintf(buf, sizeof(buf), "KERNEL:   %.2f MS", st->kernel_ms);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "TOTAL:    %.2f MS", st->total_ms);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "THRU: %.0f MPPS", st->throughput);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
    snprintf(buf, sizeof(buf), "ITER: %d", st->iteration);
    draw_bitmap_text(tx, ty, buf, fs); ty -= lh;

    if (st->greeks_computed) {
        ty -= 4;
        set_color(0.4f, 0.7f, 0.4f, 1.0f);
        draw_bitmap_text(tx, ty, "-- GREEKS --", fs); ty -= lh;
        set_color(0.85f, 0.85f, 0.85f, 1.0f);
        snprintf(buf, sizeof(buf), "DELTA: %+.4f", st->delta);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        snprintf(buf, sizeof(buf), "GAMMA: %+.4f", st->gamma_val);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        snprintf(buf, sizeof(buf), "VEGA:  %+.4f", st->vega);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        snprintf(buf, sizeof(buf), "THETA: %+.4f", st->theta);
        draw_bitmap_text(tx, ty, buf, fs); ty -= lh;
        snprintf(buf, sizeof(buf), "RHO:   %+.4f", st->rho);
        draw_bitmap_text(tx, ty, buf, fs);
    } else {
        ty -= 4;
        set_color(0.4f, 0.4f, 0.4f, 0.8f);
        draw_bitmap_text(tx, ty, "GREEKS: COMPUTING...", fs);
    }
}

static void draw_help_bar(float w) {
    set_color(0.1f, 0.2f, 0.35f, 0.95f);
    draw_rect(0, 0, w, 24);
    set_color(0.7f, 0.8f, 0.9f, 1.0f);
    draw_bitmap_text(6, 14, "Q:QUIT  +/-:VOL  </>:SPOT  C/P:CALL/PUT  A:ASIAN  D:DIGITAL  J:TOGGLE JUMPS  1-9:PATHS", 1.5f);
}

/* ─── GLFW callbacks ─── */
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    DashState *st = &g_state;

    int reset = 0;
    switch (key) {
        case GLFW_KEY_Q: case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_EQUAL: /* + */
            st->params.sigma += 0.05f; reset = 1; break;
        case GLFW_KEY_MINUS:
            st->params.sigma = fmaxf(0.01f, st->params.sigma - 0.05f); reset = 1; break;
        case GLFW_KEY_PERIOD: /* > */
        case GLFW_KEY_RIGHT_BRACKET:
            st->params.S *= 1.02f; reset = 1; break;
        case GLFW_KEY_COMMA: /* < */
        case GLFW_KEY_LEFT_BRACKET:
            st->params.S *= 0.98f; reset = 1; break;
        case GLFW_KEY_C:
            st->params.type = OPTION_EUROPEAN_CALL; reset = 1; break;
        case GLFW_KEY_P:
            st->params.type = OPTION_EUROPEAN_PUT; reset = 1; break;
        case GLFW_KEY_A:
            st->params.type = OPTION_ASIAN_CALL; reset = 1; break;
        case GLFW_KEY_E:
            st->params.type = OPTION_EUROPEAN_CALL; reset = 1; break;
        case GLFW_KEY_D:
            st->params.type = OPTION_DIGITAL_CALL; reset = 1; break;
        case GLFW_KEY_J:
            if (st->params.jump_lambda > 0.0f) {
                st->params.jump_lambda = 0.0f;
                st->params.jump_mean = 0.0f;
                st->params.jump_vol = 0.0f;
            } else {
                st->params.jump_lambda = 3.0f;
                st->params.jump_mean = -0.10f;
                st->params.jump_vol = 0.20f;
            }
            reset = 1; break;
        case GLFW_KEY_1: st->num_paths = 10000; reset = 1; break;
        case GLFW_KEY_2: st->num_paths = 50000; reset = 1; break;
        case GLFW_KEY_3: st->num_paths = 100000; reset = 1; break;
        case GLFW_KEY_4: st->num_paths = 250000; reset = 1; break;
        case GLFW_KEY_5: st->num_paths = 500000; reset = 1; break;
        case GLFW_KEY_6: st->num_paths = 1000000; reset = 1; break;
        case GLFW_KEY_7: st->num_paths = 2000000; reset = 1; break;
        case GLFW_KEY_8: st->num_paths = 5000000; reset = 1; break;
        case GLFW_KEY_9: st->num_paths = 10000000; reset = 1; break;
        default: break;
    }

    if (reset) {
        st->hist_count = 0;
        st->iteration = 0;
        st->cum_time_ms = 0.0f;
    }
}

/* ─── Main dashboard loop ─── */
void launch_gl_dashboard(OptionParams *initial_params, int num_paths,
                         int num_steps, int block_size) {
    /* Initialize state */
    memset(&g_state, 0, sizeof(g_state));
    g_state.params = *initial_params;
    g_state.num_paths = num_paths;
    g_state.num_steps = num_steps;
    g_state.block_size = block_size;

    /* Init GLFW */
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow *window = glfwCreateWindow(WIN_W, WIN_H,
        "GPU Monte Carlo Option Pricing Dashboard", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  /* no vsync — run as fast as possible */
    glfwSetKeyCallback(window, key_callback);

    /* Init GLEW */
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW error: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    printf("OpenGL dashboard launched. Press Q or ESC to close.\n");

    /* Main loop */
    while (!glfwWindowShouldClose(window)) {
        /* Run MC simulation */
        run_mc_tick(&g_state);

        /* Get framebuffer size */
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);

        /* Setup 2D projection */
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, fb_w, 0, fb_h, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        /* Clear */
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        float w = (float)fb_w;
        float h = (float)fb_h;

        /* Layout:
         * ┌──────────────────────┬───────────┐
         * │   PRICE CHART        │ PARAMS    │
         * │                      │           │
         * ├──────────────────────┤ RESULTS   │
         * │   HISTOGRAM          │           │
         * └──────────────────────┴───────────┘
         * │   HELP BAR                       │
         */
        float header_h = 28.0f;
        float sidebar_w = 360;
        float chart_w = w - sidebar_w - 20;
        float usable_h = h - 24 - header_h - 15;
        float chart_h = usable_h * 0.58f;
        float hist_h  = usable_h * 0.42f;
        float bar_y   = 0;
        float hist_y  = bar_y + 24 + 5;
        float chart_y = hist_y + hist_h + 5;
        float sidebar_x = chart_w + 15;
        float params_h  = usable_h * 0.32f;
        float results_h = usable_h * 0.68f;
        float params_y  = hist_y + results_h + 5;

        /* Top header bar */
        set_color(0.05f, 0.12f, 0.25f, 1.0f);
        draw_rect(0, h - header_h, w, header_h);
        set_color(0.3f, 0.6f, 1.0f, 0.6f);
        draw_line(0, h - header_h, w, h - header_h);
        set_color(0.9f, 0.95f, 1.0f, 1.0f);
        draw_bitmap_text(8, h - 8, "GPU MONTE CARLO OPTION PRICING ENGINE", 2.2f);
        set_color(0.5f, 0.7f, 0.9f, 0.8f);
        draw_bitmap_text(w - 340, h - 8, "EN605.617  |  RTX 3060 Ti", 1.8f);

        /* Draw panels */
        draw_price_chart(5, chart_y, chart_w, chart_h, &g_state);
        draw_histogram(5, hist_y, chart_w, hist_h, &g_state);
        draw_params_panel(sidebar_x, params_y, sidebar_w, params_h, &g_state);
        draw_results_panel(sidebar_x, hist_y, sidebar_w, results_h, &g_state);
        draw_help_bar(w);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    printf("\nDashboard closed. Final avg price: $%.4f (%d iterations)\n",
           g_state.avg_price, g_state.iteration);

    glfwDestroyWindow(window);
    glfwTerminate();
}
