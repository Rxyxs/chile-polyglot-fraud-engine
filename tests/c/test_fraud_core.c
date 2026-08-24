/* Minimal assert-based test harness for fraud_core.c -- run via
 * `make test-c` (root Makefile) / `make test` (src/c/Makefile), no
 * external test framework dependency. Lives under tests/c/ (not src/c/)
 * per the repo's language-agnostic tests/ convention; fraud_core.h is
 * pulled in via an explicit relative path rather than an -I flag. */
#include "../../src/c/fraud_core.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static int check(const char *name, int condition) {
    printf("%s ... %s\n", name, condition ? "OK" : "FAIL");
    return condition;
}

static int approx(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

int main(void) {
    int all_ok = 1;

    /* Santiago -> Valparaiso is approximately 100-110 km in a straight line. */
    double d = haversine_km(-33.4489, -70.6693, -33.0472, -71.6127);
    all_ok &= check("haversine_santiago_valparaiso_in_range", d > 90.0 && d < 120.0);

    /* Same point -> zero distance. */
    d = haversine_km(-33.45, -70.66, -33.45, -70.66);
    all_ok &= check("haversine_same_point_is_zero", approx(d, 0.0, 1e-9));

    /* First transaction (no prior point): everything should be neutral/zero. */
    {
        TransactionContext ctx = {0};
        ctx.current_lat = -33.45;
        ctx.current_lon = -70.66;
        ctx.has_prev = 0;
        ctx.seconds_since_prev = FRAUD_NO_HISTORY_SECONDS;
        ctx.amount_clp = 10000.0;
        ctx.hist_mean_amount = 0.0;
        ctx.hist_std_amount = 0.0;
        ctx.txn_count_last_1h = 0;
        ctx.txn_count_last_24h = 0;

        VelocityMetrics out;
        compute_velocity_metrics(&ctx, &out);

        all_ok &= check("first_txn_zero_distance", approx(out.distance_from_prev_km, 0.0, 1e-9));
        all_ok &= check("first_txn_zero_speed", approx(out.implied_speed_kmh, 0.0, 1e-9));
        all_ok &= check("first_txn_not_impossible", out.is_impossible_travel == 0);
        all_ok &= check("first_txn_zero_zscore", approx(out.amount_zscore, 0.0, 1e-9));
    }

    /* Impossible travel: ~1600km jump (central Chile -> northern Chile) in
     * 60 seconds implies a speed far beyond any commercial flight. */
    {
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
        compute_velocity_metrics(&ctx, &out);

        all_ok &= check("impossible_travel_distance_over_1000km", out.distance_from_prev_km > 1000.0);
        all_ok &= check("impossible_travel_flagged", out.is_impossible_travel == 1);
        all_ok &= check("impossible_travel_speed_capped", out.implied_speed_kmh <= FRAUD_SPEED_CAP_KMH);
        all_ok &= check("impossible_travel_high_velocity_score", out.velocity_score > 50.0);
    }

    /* Amount z-score clipping: an absurd deviation must saturate at the cap,
     * not blow up to a non-informative extreme (see build_features.py's
     * Python counterpart for why this matters). */
    {
        TransactionContext ctx = {0};
        ctx.has_prev = 0;
        ctx.seconds_since_prev = FRAUD_NO_HISTORY_SECONDS;
        ctx.amount_clp = 50000000.0;
        ctx.hist_mean_amount = 10000.0;
        ctx.hist_std_amount = 100.0; /* tiny std -> would blow up uncapped */
        ctx.txn_count_last_1h = 0;
        ctx.txn_count_last_24h = 0;

        VelocityMetrics out;
        compute_velocity_metrics(&ctx, &out);

        all_ok &= check("zscore_capped_at_limit", approx(out.amount_zscore, FRAUD_ZSCORE_CAP, 1e-9));
    }

    printf(all_ok ? "\nALL TESTS PASSED\n" : "\nSOME TESTS FAILED\n");
    return all_ok ? 0 : 1;
}
