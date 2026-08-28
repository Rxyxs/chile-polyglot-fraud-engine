"""FastAPI orchestration layer: combines the C velocity engine, the Ruby
rules engine, and the Python ML layer (IsolationForest + LightGBM) into one
``/detect-fraud`` verdict.

Like ``customer_state`` in the request schema, this endpoint does not look
up transaction history itself -- see ``TransactionRequest`` below for why:
a real-time scorer can't afford a historical database join per request, so
an online feature store is assumed to maintain each customer's rolling
state and hand it in as O(1) context.
"""
from __future__ import annotations

import json
import pathlib
import time
from contextlib import asynccontextmanager

import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.python.bridge import CVelocityEngine, RubyRulesEngine
from src.python.generate_data import BLACKLISTED_MERCHANT_IDS, HIGH_RISK_COUNTRY_CODES
from src.python.train_model import ISO_FOREST_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "models"

# Observabilidad Prometheus: cada etapa polyglot del pipeline (C, Ruby,
# Python/LightGBM) queda instrumentada por separado -- no solo la latencia
# total -- para que un dashboard de Grafana pueda mostrar cual capa domina
# la latencia p99 en vivo, la misma pregunta que este repo ya respondio una
# vez a mano (ver README: el cuello de botella real es IsolationForest.
# score_samples, no las llamadas cruzadas de lenguaje) pero ahora medible
# de forma continua en produccion, no solo en un benchmark puntual.
REQUESTS_TOTAL = Counter(
    "fraud_requests_total", "Transacciones procesadas por /detect-fraud", ["result"]
)
TOTAL_LATENCY = Histogram(
    "fraud_detect_latency_seconds", "Latencia total de /detect-fraud (todas las capas)"
)
C_LAYER_LATENCY = Histogram(
    "fraud_c_layer_latency_seconds", "Latencia de compute_velocity_metrics (capa C)"
)
RUBY_LAYER_LATENCY = Histogram(
    "fraud_rules_layer_latency_seconds", "Latencia del motor de reglas Ruby (pipe subprocess)"
)
ML_LAYER_LATENCY = Histogram(
    "fraud_ml_layer_latency_seconds", "Latencia de IsolationForest + LightGBM"
)
FRAUD_PROBABILITY = Histogram(
    "fraud_probability_score",
    "Distribucion de fraud_probability emitida",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

_state: dict = {}


class CustomerState(BaseModel):
    last_latitude: float | None = None
    last_longitude: float | None = None
    last_timestamp_seconds_ago: float | None = Field(
        None, description="Seconds elapsed since this customer's previous transaction"
    )
    hist_mean_amount: float = 0.0
    hist_std_amount: float = 0.0
    txn_count_last_1h: int = 0
    txn_count_last_24h: int = 0


class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: int
    amount_clp: float = Field(..., gt=0)
    merchant_id: str
    country_code: str
    latitude: float
    longitude: float
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    customer_state: CustomerState = Field(default_factory=CustomerState)


class FraudDetectionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision_threshold: float
    c_layer: dict
    ruby_layer: dict
    total_latency_ms: float


def _load_artifacts() -> dict:
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_fraud.txt"))
    c_engine = CVelocityEngine()
    rules_engine = RubyRulesEngine()
    return {
        "metadata": metadata,
        "iso_forest": iso_forest,
        "booster": booster,
        "c_engine": c_engine,
        "rules_engine": rules_engine,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.update(_load_artifacts())
    yield
    _state["rules_engine"].close()
    _state.clear()


app = FastAPI(
    title="Chile Polyglot Fraud Engine",
    description="C velocity core + Ruby rules DSL + Python ML, combined into one verdict.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok" if _state else "loading"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/detect-fraud", response_model=FraudDetectionResponse)
def detect_fraud(req: TransactionRequest) -> FraudDetectionResponse:
    start = time.perf_counter()
    cs = req.customer_state

    has_prev = cs.last_latitude is not None and cs.last_longitude is not None
    c_start = time.perf_counter()
    c_metrics = _state["c_engine"].compute(
        current_lat=req.latitude,
        current_lon=req.longitude,
        amount_clp=req.amount_clp,
        prev_lat=cs.last_latitude if has_prev else None,
        prev_lon=cs.last_longitude if has_prev else None,
        seconds_since_prev=cs.last_timestamp_seconds_ago,
        hist_mean_amount=cs.hist_mean_amount,
        hist_std_amount=cs.hist_std_amount,
        txn_count_last_1h=cs.txn_count_last_1h,
        txn_count_last_24h=cs.txn_count_last_24h,
    )
    C_LAYER_LATENCY.observe(time.perf_counter() - c_start)

    ruby_start = time.perf_counter()
    ruby_verdict = _state["rules_engine"].evaluate({
        "amount_clp": req.amount_clp,
        "merchant_id": req.merchant_id,
        "country_code": req.country_code,
        "txn_count_last_1h": cs.txn_count_last_1h,
        "txn_count_last_24h": cs.txn_count_last_24h,
        "is_impossible_travel": c_metrics["is_impossible_travel"],
    })
    RUBY_LAYER_LATENCY.observe(time.perf_counter() - ruby_start)

    feature_row = {
        "amount_clp": req.amount_clp,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "distance_from_prev_km": c_metrics["distance_from_prev_km"],
        "implied_speed_kmh": c_metrics["implied_speed_kmh"],
        "is_impossible_travel": int(c_metrics["is_impossible_travel"]),
        "amount_zscore": c_metrics["amount_zscore"],
        "velocity_score": c_metrics["velocity_score"],
        "rules_risk_score": ruby_verdict["risk_score"],
        "rules_flagged": int(ruby_verdict["flagged"]),
        "is_blacklisted_merchant": int(req.merchant_id in BLACKLISTED_MERCHANT_IDS),
        "is_high_risk_country": int(req.country_code in HIGH_RISK_COUNTRY_CODES),
        "txn_count_last_1h": cs.txn_count_last_1h,
        "txn_count_last_24h": cs.txn_count_last_24h,
    }
    ml_start = time.perf_counter()
    numeric_vector = np.array([[feature_row[c] for c in NUMERIC_FEATURE_COLUMNS]], dtype=np.float64)
    iso_vector = np.array([[feature_row[c] for c in ISO_FOREST_FEATURE_COLUMNS]], dtype=np.float64)
    iso_score = -_state["iso_forest"].score_samples(iso_vector)[0]
    full_vector = np.concatenate([numeric_vector, np.array([[iso_score]])], axis=1)

    fraud_probability = float(_state["booster"].predict(full_vector)[0])
    ML_LAYER_LATENCY.observe(time.perf_counter() - ml_start)
    threshold = _state["metadata"]["decision_threshold"]
    is_fraud = fraud_probability >= threshold

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    TOTAL_LATENCY.observe(elapsed_ms / 1000.0)
    FRAUD_PROBABILITY.observe(fraud_probability)
    REQUESTS_TOTAL.labels(result="fraud" if is_fraud else "legit").inc()

    return FraudDetectionResponse(
        transaction_id=req.transaction_id,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        decision_threshold=threshold,
        c_layer=c_metrics,
        ruby_layer=ruby_verdict,
        total_latency_ms=elapsed_ms,
    )
