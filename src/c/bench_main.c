/* Benchmarks compute_velocity_metrics() in isolation to validate the
 * "< 1ms per transaction" claim for the C hot path specifically (the
 * Ruby DSL and Python ML layers are benchmarked separately in Python --
 * see tests/python/test_latency.py -- since the < 1ms budget in the brief
 * is about this module's own computation, not the whole pipeline). */
#include "fraud_core.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    const long iterations = 2000000L;

    TransactionContext ctx = {0};
    ctx.current_lat = -18.47;
    ctx.current_lon = -70.30;
    ctx.prev_lat = -33.45;
    ctx.prev_lon = -70.66;
    ctx.has_prev = 1;
    ctx.seconds_since_prev = 60.0;
    ctx.amount_clp = 500.0;
    ctx.hist_mean_amount = 10000.0;
    ctx.hist_std_amount = 3000.0;
    ctx.txn_count_last_1h = 4;
    ctx.txn_count_last_24h = 6;

    VelocityMetrics out;

    /* Warm up (page faults, branch predictor, etc.) before timing. */
    for (long i = 0; i < 10000; i++) {
        compute_velocity_metrics(&ctx, &out);
    }

    clock_t start = clock();
    for (long i = 0; i < iterations; i++) {
        /* Vary the input slightly so the compiler can't hoist/constant-fold
         * the whole loop away. */
        ctx.seconds_since_prev = 60.0 + (double)(i % 1000);
        compute_velocity_metrics(&ctx, &out);
    }
    clock_t end = clock();

    double total_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    double ns_per_call = (total_seconds / (double)iterations) * 1e9;
    double us_per_call = ns_per_call / 1000.0;

    printf("Iterations:        %ld\n", iterations);
    printf("Total time:        %.4f s\n", total_seconds);
    printf("Time per call:      %.1f ns (%.4f microseconds)\n", ns_per_call, us_per_call);
    printf("Budget (< 1000 us): %s\n", us_per_call < 1000.0 ? "PASS" : "FAIL");
    printf("Last velocity_score computed: %.2f (sanity check, not a benchmark artifact)\n", out.velocity_score);

    return us_per_call < 1000.0 ? 0 : 1;
}
