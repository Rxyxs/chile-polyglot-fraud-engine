"""Validates the layered latency budget: the C module's own compute must
stay under the brief's < 1ms target, and the full three-language pipeline
(as a regression guard, not the headline claim -- see README's latency
section for the honest breakdown of where the milliseconds actually go)
must stay comfortably within a real-time-serving budget.
"""
import time

import pytest
from fastapi.testclient import TestClient

from src.python.api import app
from src.python.bridge import CVelocityEngine
from tests.python.test_api import LEGIT_PAYLOAD

N_REQUESTS = 200
C_MODULE_BUDGET_MS = 1.0
FULL_PIPELINE_BUDGET_MS = 50.0


def _percentile(values, pct):
    return sorted(values)[int(pct * (len(values) - 1))]


def test_c_module_p95_under_1ms():
    engine = CVelocityEngine()
    for _ in range(50):  # warm-up
        engine.compute(current_lat=-33.45, current_lon=-70.66, amount_clp=20000)

    latencies = []
    for _ in range(2000):
        start = time.perf_counter()
        engine.compute(
            current_lat=-18.47, current_lon=-70.30, amount_clp=500,
            prev_lat=-33.45, prev_lon=-70.66, seconds_since_prev=60,
            hist_mean_amount=10000, hist_std_amount=3000,
            txn_count_last_1h=4, txn_count_last_24h=6,
        )
        latencies.append((time.perf_counter() - start) * 1000)

    p95 = _percentile(latencies, 0.95)
    print(f"C module (ctypes) p50={_percentile(latencies, 0.5):.5f}ms p95={p95:.5f}ms")
    assert p95 < C_MODULE_BUDGET_MS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_full_pipeline_p95_under_budget(client):
    for i in range(20):  # warm-up (Ruby subprocess, IsolationForest, LightGBM)
        payload = {**LEGIT_PAYLOAD, "transaction_id": f"WARMUP_{i}"}
        client.post("/detect-fraud", json=payload)

    latencies = []
    for i in range(N_REQUESTS):
        payload = {**LEGIT_PAYLOAD, "transaction_id": f"TXN_LAT_{i}"}
        start = time.perf_counter()
        r = client.post("/detect-fraud", json=payload)
        latencies.append((time.perf_counter() - start) * 1000)
        assert r.status_code == 200

    p50, p95 = _percentile(latencies, 0.5), _percentile(latencies, 0.95)
    print(f"Full pipeline p50={p50:.3f}ms p95={p95:.3f}ms max={max(latencies):.3f}ms")
    assert p95 < FULL_PIPELINE_BUDGET_MS
