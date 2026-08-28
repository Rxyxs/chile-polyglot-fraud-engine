"""Stress test: measures real request/second throughput of the three IPC/FFI
mechanisms in this repo, side by side, from the same Python process:

    1. CVelocityEngine   -- in-process ctypes DLL call (no IPC at all).
    2. MmapFeatureStoreClient -- shared-memory IPC, busy-wait handshake,
       against the separate feature_store_server.exe process.
    3. RubyRulesEngine   -- subprocess pipe (stdin/stdout, JSON), against
       the persistent `ruby rules_engine.rb --server` process.

Every number below is measured on this run, not assumed from theory --
run it yourself with `make -C . bench-throughput` (starts
feature_store_server.exe first) or directly:

    .venv/Scripts/python.exe -m src.python.benchmark_throughput [--iterations N]
"""
from __future__ import annotations

import argparse
import statistics
import time

from .bridge import CVelocityEngine, RubyRulesEngine

TEST_TRANSACTION = dict(
    current_lat=-33.45,
    current_lon=-70.66,
    amount_clp=50_000.0,
    prev_lat=-18.47,
    prev_lon=-70.30,
    seconds_since_prev=120.0,
    hist_mean_amount=40_000.0,
    hist_std_amount=5_000.0,
    txn_count_last_1h=2,
    txn_count_last_24h=5,
)

RUBY_TEST_TRANSACTION = {
    "amount_clp": 50_000.0,
    "merchant_category": "supermercado",
    "hour_of_day": 14,
    "is_foreign_country": False,
    "velocity_score": 25.0,
    "distance_from_prev_km": 10.0,
    "is_impossible_travel": False,
}


def _run_throughput(name: str, call_once, iterations: int) -> dict:
    # Warm-up: page faults, branch predictor, (for the mmap client) the
    # server's first few Sleep(0) spins -- excluded from the timed loop so
    # steady-state throughput isn't diluted by one-time costs.
    for _ in range(min(1000, iterations // 10 or 1)):
        call_once()

    latencies_us = []
    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        call_once()
        latencies_us.append((time.perf_counter() - t0) * 1e6)
    elapsed = time.perf_counter() - start

    req_per_sec = iterations / elapsed
    latencies_us.sort()
    p50 = latencies_us[len(latencies_us) // 2]
    p99 = latencies_us[int(len(latencies_us) * 0.99)]

    return {
        "name": name,
        "iterations": iterations,
        "elapsed_s": elapsed,
        "req_per_sec": req_per_sec,
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": statistics.mean(latencies_us),
    }


def _print_result(r: dict) -> None:
    print(f"{r['name']}:")
    print(f"  {r['iterations']:,} requests in {r['elapsed_s']:.3f}s")
    print(f"  Throughput: {r['req_per_sec']:,.0f} req/s")
    print(f"  Latency:    p50={r['p50_us']:.1f}us  p99={r['p99_us']:.1f}us  mean={r['mean_us']:.1f}us")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--skip-mmap", action="store_true", help="Omite el benchmark mmap (requiere feature_store_server.exe corriendo)")
    parser.add_argument("--skip-ruby", action="store_true", help="Omite el benchmark de Ruby (mas lento, tarda mas)")
    args = parser.parse_args()

    results = []

    print(f"=== Stress test: {args.iterations:,} iteraciones por mecanismo ===\n")

    ctypes_engine = CVelocityEngine()
    r = _run_throughput(
        "1. CVelocityEngine (ctypes DLL, in-process, sin IPC)",
        lambda: ctypes_engine.compute(**TEST_TRANSACTION),
        args.iterations,
    )
    results.append(r)
    _print_result(r)

    if not args.skip_mmap:
        try:
            from .mmap_feature_store_client import MmapFeatureStoreClient

            with MmapFeatureStoreClient() as mmap_client:
                r = _run_throughput(
                    "2. MmapFeatureStoreClient (memoria compartida, busy-wait)",
                    lambda: mmap_client.compute(**TEST_TRANSACTION),
                    args.iterations,
                )
                results.append(r)
                _print_result(r)
        except ConnectionError as exc:
            print(f"2. MmapFeatureStoreClient: OMITIDO ({exc})\n")

    if not args.skip_ruby:
        ruby_iterations = min(args.iterations, 5_000)  # el pipe JSON es mucho mas lento; evita una corrida de minutos
        with RubyRulesEngine() as ruby_engine:
            r = _run_throughput(
                "3. RubyRulesEngine (subprocess pipe, JSON sobre stdin/stdout)",
                lambda: ruby_engine.evaluate(RUBY_TEST_TRANSACTION),
                ruby_iterations,
            )
            results.append(r)
            _print_result(r)

    print("=== Resumen ===")
    for r in results:
        budget = "PASS" if r["req_per_sec"] >= 10_000 else "por debajo de 10.000 req/s"
        print(f"  {r['name'].split('.')[0].strip()}. {r['req_per_sec']:>12,.0f} req/s  ({budget})")


if __name__ == "__main__":
    main()
