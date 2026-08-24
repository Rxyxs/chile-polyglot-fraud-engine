/* fraud_core.c -- ultra-low-latency transaction velocity/geo metrics.
 *
 * This is the C11 hot-path module: given a transaction and a small amount of
 * per-customer rolling state (supplied by the caller -- see the docstring
 * in src/python/bridge.py for why no history lookup happens in here), it
 * computes the same spatio-temporal features as the pure-Python reference
 * implementation, compiled to native code for the sub-millisecond budget.
 *
 * Build: see src/c/Makefile (invokes MSVC via vcvars64.bat on Windows).
 */
#include "fraud_core.h"
#include <math.h>

#define EARTH_RADIUS_KM 6371.0
#define DEG2RAD(deg) ((deg) * 3.14159265358979323846 / 180.0)

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double phi1 = DEG2RAD(lat1);
    double phi2 = DEG2RAD(lat2);
    double dphi = DEG2RAD(lat2 - lat1);
    double dlambda = DEG2RAD(lon2 - lon1);

    double a = sin(dphi / 2.0) * sin(dphi / 2.0)
             + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) * sin(dlambda / 2.0);
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;

    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a));
}

static double clip(double value, double lo, double hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

void compute_velocity_metrics(const TransactionContext *ctx, VelocityMetrics *out) {
    double distance_km = 0.0;
    double speed_kmh = 0.0;
    int impossible = 0;
    double zscore = 0.0;

    if (ctx->has_prev) {
        distance_km = haversine_km(ctx->current_lat, ctx->current_lon,
                                    ctx->prev_lat, ctx->prev_lon);

        if (ctx->seconds_since_prev > 0.0) {
            speed_kmh = distance_km / (ctx->seconds_since_prev / 3600.0);
        } else if (distance_km > 0.0) {
            /* Simultaneous timestamps, different places: maximally strong
             * impossible-travel signal, not a missing value. */
            speed_kmh = FRAUD_IMPOSSIBLE_TRAVEL_KMH * 10.0;
        } else {
            speed_kmh = 0.0;
        }
        speed_kmh = clip(speed_kmh, 0.0, FRAUD_SPEED_CAP_KMH);
        impossible = speed_kmh > FRAUD_IMPOSSIBLE_TRAVEL_KMH ? 1 : 0;
    }

    if (ctx->hist_std_amount > 0.0) {
        zscore = (ctx->amount_clp - ctx->hist_mean_amount) / ctx->hist_std_amount;
        zscore = clip(zscore, -FRAUD_ZSCORE_CAP, FRAUD_ZSCORE_CAP);
    }

    /* Composite 0-100 score blending the three signals this module owns
     * (velocity counts + impossible travel + amount deviation). This is a
     * fast, explainable pre-score for the C layer specifically -- it is NOT
     * the final fraud decision, which the Python ML layer and the Ruby
     * rules engine each contribute to independently (see run_pipeline.py /
     * src/python/api.py for how the three verdicts are combined). */
    double velocity_score = 0.0;
    velocity_score += clip((double)ctx->txn_count_last_1h * 8.0, 0.0, 40.0);
    velocity_score += impossible ? 40.0 : 0.0;
    velocity_score += clip(fabs(zscore) * 2.0, 0.0, 20.0);
    velocity_score = clip(velocity_score, 0.0, 100.0);

    out->distance_from_prev_km = distance_km;
    out->implied_speed_kmh = speed_kmh;
    out->is_impossible_travel = impossible;
    out->amount_zscore = zscore;
    out->velocity_score = velocity_score;
}
