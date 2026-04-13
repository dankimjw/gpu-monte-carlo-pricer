#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ncurses.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/option_params.h"
#include "dashboard.h"
#include "monte_carlo.h"
#include "reduction.h"
#include "black_scholes.h"
#include "greeks.h"

/* Colors */
#define CLR_TITLE    1
#define CLR_GREEN    2
#define CLR_RED      3
#define CLR_CYAN     4
#define CLR_YELLOW   5
#define CLR_WHITE    6
#define CLR_HEADER   7

/* Helper: run MC once and return price + stats */
typedef struct {
    float price;
    float std_err;
    float ci_low;
    float ci_high;
    float kernel_ms;
    float total_ms;
    float throughput;
} MCResult;

static MCResult run_mc_once(const OptionParams *params, int num_paths,
                            int num_steps, int block_size) {
    MCResult res;
    cudaEvent_t start, stop, ks, ke;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&ks); cudaEventCreate(&ke);

    cudaEventRecord(start);
    mc_set_params(params);
    curandState *d_states = mc_init_rng(num_paths, block_size);

    cudaEventRecord(ks);
    float *d_payoffs = mc_run_simulation(d_states, num_paths, num_steps, block_size);
    cudaEventRecord(ke);
    cudaEventSynchronize(ke);

    float mean_payoff, variance;
    reduce_payoffs(d_payoffs, num_paths, &mean_payoff, &variance);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float discount = expf(-params->r * params->T);
    res.price = discount * mean_payoff;
    res.std_err = discount * sqrtf(variance / (float)num_paths);
    res.ci_low = res.price - 1.96f * res.std_err;
    res.ci_high = res.price + 1.96f * res.std_err;

    cudaEventElapsedTime(&res.kernel_ms, ks, ke);
    cudaEventElapsedTime(&res.total_ms, start, stop);
    res.throughput = (float)num_paths / (res.kernel_ms / 1000.0f) / 1e6f;

    cudaFree(d_states);
    cudaFree(d_payoffs);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(ks); cudaEventDestroy(ke);
    return res;
}

static const char* type_str(OptionType t) {
    switch (t) {
        case OPTION_EUROPEAN_CALL: return "European Call";
        case OPTION_EUROPEAN_PUT: return "European Put";
        case OPTION_ASIAN_CALL: return "Asian Call";
        case OPTION_ASIAN_PUT: return "Asian Put";
        case OPTION_BARRIER_UP_AND_OUT_CALL: return "Barrier U&O Call";
        case OPTION_BARRIER_DOWN_AND_OUT_CALL: return "Barrier D&O Call";
        case OPTION_DIGITAL_CALL: return "Digital Call";
        case OPTION_DIGITAL_PUT: return "Digital Put";
        default: return "Unknown";
    }
}

/* Draw the sparkline-style price history */
static void draw_sparkline(int row, int col, int width, float *history, int count, int color) {
    if (count < 2) return;
    float mn = history[0], mx = history[0];
    for (int i = 1; i < count; i++) {
        if (history[i] < mn) mn = history[i];
        if (history[i] > mx) mx = history[i];
    }
    float range = mx - mn;
    if (range < 0.001f) range = 1.0f;

    /* ASCII-safe sparkline characters (works in all terminals) */
    const char blocks[] = {' ', '_', '.', '-', '~', '=', '#', '#', '#'};

    attron(COLOR_PAIR(color));
    int start = (count > width) ? count - width : 0;
    for (int i = start; i < count && (i - start) < width; i++) {
        int level = (int)((history[i] - mn) / range * 7.0f);
        if (level < 0) level = 0;
        if (level > 7) level = 7;
        mvaddch(row, col + (i - start), blocks[level + 1]);
    }
    attroff(COLOR_PAIR(color));
}

void launch_dashboard(OptionParams *params, int num_paths,
                      int num_steps, int block_size) {
    /* Initialize ncurses */
    initscr();
    start_color();
    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    timeout(100);  /* non-blocking getch, 100ms */

    init_pair(CLR_TITLE, COLOR_WHITE, COLOR_BLUE);
    init_pair(CLR_GREEN, COLOR_GREEN, COLOR_BLACK);
    init_pair(CLR_RED, COLOR_RED, COLOR_BLACK);
    init_pair(CLR_CYAN, COLOR_CYAN, COLOR_BLACK);
    init_pair(CLR_YELLOW, COLOR_YELLOW, COLOR_BLACK);
    init_pair(CLR_WHITE, COLOR_WHITE, COLOR_BLACK);
    init_pair(CLR_HEADER, COLOR_BLACK, COLOR_CYAN);

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    /* Price history for sparkline */
    #define MAX_HISTORY 200
    float price_history[MAX_HISTORY];
    float time_history[MAX_HISTORY];
    int hist_count = 0;

    /* Analytical reference */
    float bs_price = 0.0f;
    if (params->type == OPTION_EUROPEAN_CALL)
        bs_price = bs_call_price(params);
    else if (params->type == OPTION_EUROPEAN_PUT)
        bs_price = bs_put_price(params);

    int running = 1;
    int iteration = 0;
    float cum_time_ms = 0.0f;
    float best_price = 0.0f;

    while (running) {
        /* Run MC simulation */
        MCResult res = run_mc_once(params, num_paths, num_steps, block_size);
        iteration++;
        cum_time_ms += res.total_ms;

        /* Update history */
        if (hist_count < MAX_HISTORY) {
            price_history[hist_count] = res.price;
            time_history[hist_count] = res.total_ms;
            hist_count++;
        } else {
            memmove(price_history, price_history + 1, (MAX_HISTORY - 1) * sizeof(float));
            memmove(time_history, time_history + 1, (MAX_HISTORY - 1) * sizeof(float));
            price_history[MAX_HISTORY - 1] = res.price;
            time_history[MAX_HISTORY - 1] = res.total_ms;
        }

        /* Running average */
        float avg_price = 0.0f;
        for (int i = 0; i < hist_count; i++) avg_price += price_history[i];
        avg_price /= (float)hist_count;
        best_price = avg_price;

        /* --- DRAW --- */
        erase();
        int rows, cols;
        getmaxyx(stdscr, rows, cols);
        int w = (cols < 80) ? cols : cols;

        /* Title bar */
        attron(COLOR_PAIR(CLR_TITLE) | A_BOLD);
        mvhline(0, 0, ' ', w);
        mvprintw(0, 1, " GPU Monte Carlo Option Pricing Dashboard ");
        mvprintw(0, w - 25, " [q]uit [+/-] vol [<>] S ");
        attroff(COLOR_PAIR(CLR_TITLE) | A_BOLD);

        /* GPU info line */
        attron(COLOR_PAIR(CLR_CYAN));
        mvprintw(2, 1, "GPU: %s  |  %d SMs  |  Compute %d.%d",
                 prop.name, prop.multiProcessorCount, prop.major, prop.minor);
        attroff(COLOR_PAIR(CLR_CYAN));

        /* Parameters panel */
        int row = 4;
        attron(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        mvhline(row, 1, ' ', 38);
        mvprintw(row, 1, " Option Parameters ");
        attroff(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        row++;

        attron(COLOR_PAIR(CLR_WHITE));
        mvprintw(row++, 2, "Type:    %-20s", type_str(params->type));
        mvprintw(row++, 2, "Spot:    $%12.2f", params->S);
        mvprintw(row++, 2, "Strike:  $%12.2f", params->K);
        mvprintw(row++, 2, "Rate:    %12.4f", params->r);
        mvprintw(row++, 2, "Vol (σ): %12.4f", params->sigma);
        mvprintw(row++, 2, "Expiry:  %12.2f yr", params->T);
        if (params->barrier > 0.0f)
            mvprintw(row++, 2, "Barrier: $%12.2f", params->barrier);
        mvprintw(row++, 2, "Paths:   %12d", num_paths);
        mvprintw(row++, 2, "Steps:   %12d", num_steps);
        attroff(COLOR_PAIR(CLR_WHITE));

        /* Results panel */
        int res_col = 42;
        int rrow = 4;
        attron(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        mvhline(rrow, res_col, ' ', w - res_col - 1);
        mvprintw(rrow, res_col, " Pricing Results ");
        attroff(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        rrow++;

        attron(COLOR_PAIR(CLR_GREEN) | A_BOLD);
        mvprintw(rrow++, res_col + 1, "MC Price:   $%.4f", res.price);
        attroff(COLOR_PAIR(CLR_GREEN) | A_BOLD);

        attron(COLOR_PAIR(CLR_YELLOW));
        mvprintw(rrow++, res_col + 1, "Avg Price:  $%.4f  (n=%d)", avg_price, hist_count);
        attroff(COLOR_PAIR(CLR_YELLOW));

        attron(COLOR_PAIR(CLR_WHITE));
        mvprintw(rrow++, res_col + 1, "Std Error:  $%.4f", res.std_err);
        mvprintw(rrow++, res_col + 1, "95%% CI:    [$%.2f, $%.2f]", res.ci_low, res.ci_high);
        attroff(COLOR_PAIR(CLR_WHITE));

        if (bs_price > 0.0f) {
            float err_pct = fabsf(avg_price - bs_price) / bs_price * 100.0f;
            int clr = (err_pct < 1.0f) ? CLR_GREEN : CLR_RED;
            attron(COLOR_PAIR(CLR_CYAN));
            mvprintw(rrow++, res_col + 1, "BS Price:   $%.4f", bs_price);
            attroff(COLOR_PAIR(CLR_CYAN));
            attron(COLOR_PAIR(clr));
            mvprintw(rrow++, res_col + 1, "Error:      %.4f%%", err_pct);
            attroff(COLOR_PAIR(clr));
        }

        rrow++;
        attron(COLOR_PAIR(CLR_CYAN));
        mvprintw(rrow++, res_col + 1, "Kernel:     %.2f ms", res.kernel_ms);
        mvprintw(rrow++, res_col + 1, "Total:      %.2f ms", res.total_ms);
        mvprintw(rrow++, res_col + 1, "Throughput: %.1f M paths/sec", res.throughput);
        mvprintw(rrow++, res_col + 1, "Iteration:  %d  (%.1f ms cum.)", iteration, cum_time_ms);
        attroff(COLOR_PAIR(CLR_CYAN));

        /* Price sparkline */
        int spark_row = (row > rrow) ? row + 1 : rrow + 1;
        spark_row++;
        attron(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        mvhline(spark_row, 1, ' ', w - 2);
        mvprintw(spark_row, 1, " Price History (last %d iterations) ", hist_count);
        attroff(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        spark_row++;

        int spark_w = w - 4;
        if (spark_w > MAX_HISTORY) spark_w = MAX_HISTORY;
        draw_sparkline(spark_row, 2, spark_w, price_history, hist_count, CLR_GREEN);

        /* Price range labels */
        if (hist_count > 1) {
            float mn = price_history[0], mx = price_history[0];
            for (int i = 1; i < hist_count; i++) {
                if (price_history[i] < mn) mn = price_history[i];
                if (price_history[i] > mx) mx = price_history[i];
            }
            attron(COLOR_PAIR(CLR_WHITE));
            mvprintw(spark_row + 1, 2, "Low: $%.2f", mn);
            mvprintw(spark_row + 1, spark_w / 2, "Avg: $%.2f", avg_price);
            mvprintw(spark_row + 1, spark_w - 14, "High: $%.2f", mx);
            attroff(COLOR_PAIR(CLR_WHITE));
        }

        /* Timing sparkline */
        int time_row = spark_row + 3;
        attron(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        mvhline(time_row, 1, ' ', w - 2);
        mvprintw(time_row, 1, " Kernel Time (ms) ");
        attroff(COLOR_PAIR(CLR_HEADER) | A_BOLD);
        time_row++;
        draw_sparkline(time_row, 2, spark_w, time_history, hist_count, CLR_YELLOW);

        /* Help bar at bottom */
        attron(COLOR_PAIR(CLR_TITLE));
        mvhline(rows - 1, 0, ' ', w);
        mvprintw(rows - 1, 1,
                 " q:quit  +/-:vol  </> or [/]:spot  c/p:call/put  a:asian  d:digital  1-9:paths ");
        attroff(COLOR_PAIR(CLR_TITLE));

        refresh();

        /* Handle input */
        int ch = getch();
        switch (ch) {
            case 'q': case 'Q': running = 0; break;
            case '+': case '=': params->sigma += 0.05f; break;
            case '-': params->sigma = fmaxf(0.01f, params->sigma - 0.05f); break;
            case '>': case ']': params->S *= 1.02f; break;
            case '<': case '[': params->S *= 0.98f; break;
            case 'c': case 'C': params->type = OPTION_EUROPEAN_CALL; break;
            case 'p': case 'P': params->type = OPTION_EUROPEAN_PUT; break;
            case 'a': case 'A': params->type = OPTION_ASIAN_CALL; break;
            case 'e': case 'E': params->type = OPTION_EUROPEAN_CALL; break;
            case 'd': case 'D': params->type = OPTION_DIGITAL_CALL; break;
            case '1': num_paths = 10000; break;
            case '2': num_paths = 50000; break;
            case '3': num_paths = 100000; break;
            case '4': num_paths = 250000; break;
            case '5': num_paths = 500000; break;
            case '6': num_paths = 1000000; break;
            case '7': num_paths = 2000000; break;
            case '8': num_paths = 5000000; break;
            case '9': num_paths = 10000000; break;
            default: break;
        }

        /* Recalculate BS reference if type changed */
        if (ch == 'c' || ch == 'C' || ch == 'p' || ch == 'P' ||
            ch == '<' || ch == '>' || ch == '[' || ch == ']' ||
            ch == '+' || ch == '-' || ch == '=' ||
            ch == 'a' || ch == 'A' || ch == 'e' || ch == 'E' ||
            ch == 'd' || ch == 'D') {
            if (params->type == OPTION_EUROPEAN_CALL)
                bs_price = bs_call_price(params);
            else if (params->type == OPTION_EUROPEAN_PUT)
                bs_price = bs_put_price(params);
            else
                bs_price = 0.0f;
            hist_count = 0;
            iteration = 0;
            cum_time_ms = 0.0f;
        }
    }

    endwin();
    printf("\nDashboard closed. Final average price: $%.4f (%d iterations)\n",
           best_price, iteration);
}
