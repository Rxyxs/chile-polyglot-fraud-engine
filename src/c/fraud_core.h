/* fraud_core.h -- ultra-low-latency transaction velocity/geo metrics.
 *
 * This header defines the ABI shared by fraud_core.c, its C test harness
 * (test_fraud_core.c), its benchmark (bench_main.c), and the Python ctypes
 * bridge (src/python/bridge.py), which mirrors these structs field-for-field
 * with ctypes.Structure. Changing field order/types here requires updating
 * bridge.py's mirrored layout too.
 */
#ifndef FRAUD_CORE_H
#define FRAUD_CORE_H

#ifdef _WIN32
#define FRAUD_API __declspec(dllexport)
#else
#define FRAUD_API
#endif

/* Sentinel for "no previous transaction" / "no history yet" -- 30 days in
 * seconds. Keeps every downstream computation well-defined (finite) without
 * special-casing infinity across the FFI boundary. */
#define FRAUD_NO_HISTORY_SECONDS (30.0 * 24.0 * 3600.0)

/* A commercial flight cruises around 850-900 km/h; faster than this between
 * two consecutive purchases is physically impossible for one person. */
#define FRAUD_IMPOSSIBLE_TRAVEL_KMH 900.0

/* Winsorizing caps -- mirror src/python/generate_data.py / bridge.py.
 * Prevents a near-zero denominator (a customer with almost no history)
 * from blowing a ratio up to a non-informative extreme. */
#define FRAUD_ZSCORE_CAP 30.0
#define FRAUD_SPEED_CAP_KMH 5000.0

typedef struct {
    double current_lat;
    double current_lon;
    double prev_lat;
    double prev_lon;
    int    has_prev;              /* 0 if this is the customer's first transaction */
    double seconds_since_prev;    /* FRAUD_NO_HISTORY_SECONDS if has_prev == 0 */
    double amount_clp;
    double hist_mean_amount;
    double hist_std_amount;       /* 0 if fewer than 2 prior transactions */
    long   txn_count_last_1h;
    long   txn_count_last_24h;
} TransactionContext;

typedef struct {
    double distance_from_prev_km;
    double implied_speed_kmh;
    int    is_impossible_travel;
    double amount_zscore;
    double velocity_score;        /* composite 0-100, see fraud_core.c */
} VelocityMetrics;

/* Great-circle distance between two lat/lon points, in kilometers. */
FRAUD_API double haversine_km(double lat1, double lon1, double lat2, double lon2);

/* Fills *out from *ctx. Pure function, no I/O, no allocation -- this is the
 * single call Python makes per transaction across the ctypes boundary. */
FRAUD_API void compute_velocity_metrics(const TransactionContext *ctx, VelocityMetrics *out);

#endif /* FRAUD_CORE_H */
