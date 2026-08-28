import pytest
from fastapi.testclient import TestClient

from src.python.api import app

LEGIT_PAYLOAD = {
    "transaction_id": "TXN_TEST_LEGIT",
    "customer_id": 1,
    "amount_clp": 20000,
    "merchant_id": "MER_00001",
    "country_code": "CL",
    "latitude": -33.45,
    "longitude": -70.66,
    "hour_of_day": 14,
    "day_of_week": 2,
    "customer_state": {
        "last_latitude": -33.44,
        "last_longitude": -70.65,
        "last_timestamp_seconds_ago": 3600,
        "hist_mean_amount": 18000,
        "hist_std_amount": 4000,
        "txn_count_last_1h": 0,
        "txn_count_last_24h": 1,
    },
}


def _fraud_payload():
    payload = {**LEGIT_PAYLOAD, "transaction_id": "TXN_TEST_FRAUD", "merchant_id": "MER_00666", "amount_clp": 1000}
    return payload


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_fraud_legit_transaction(client):
    r = client.post("/detect-fraud", json=LEGIT_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is False
    assert body["ruby_layer"]["flagged"] is False


def test_detect_fraud_blacklisted_merchant(client):
    r = client.post("/detect-fraud", json=_fraud_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["ruby_layer"]["flagged"] is True
    assert "comercio_en_lista_negra" in body["ruby_layer"]["triggered_rules"]


def test_detect_fraud_impossible_travel(client):
    payload = {
        **LEGIT_PAYLOAD,
        "transaction_id": "TXN_TEST_TRAVEL",
        "latitude": -18.47,
        "longitude": -70.30,
        "customer_state": {**LEGIT_PAYLOAD["customer_state"], "last_timestamp_seconds_ago": 60},
    }
    r = client.post("/detect-fraud", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["c_layer"]["is_impossible_travel"] is True


def test_detect_fraud_rejects_nonpositive_amount(client):
    bad = {**LEGIT_PAYLOAD, "amount_clp": -100}
    r = client.post("/detect-fraud", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint_reflects_real_requests(client):
    # No solo "el endpoint responde 200" -- verifica que un request real a
    # /detect-fraud efectivamente incremento los contadores/histogramas de
    # Prometheus, no solo que existen declarados en el codigo.
    before = client.get("/metrics").text
    client.post("/detect-fraud", json=LEGIT_PAYLOAD)
    after = client.get("/metrics").text

    assert "fraud_requests_total" in after
    assert "fraud_c_layer_latency_seconds" in after
    assert "fraud_rules_layer_latency_seconds" in after
    assert "fraud_ml_layer_latency_seconds" in after
    assert "fraud_probability_score" in after
    assert before != after
