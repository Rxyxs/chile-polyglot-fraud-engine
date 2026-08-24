import numpy as np
import pandas as pd
import pytest

from src.python.bridge import CVelocityEngine, RubyRulesEngine
from src.python.train_model import (
    NUMERIC_FEATURE_COLUMNS,
    build_features,
    evaluate,
    find_best_f1_threshold,
    time_based_split,
)


def test_time_based_split_is_chronological_and_proportional():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
        "value": range(100),
    })
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle first
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_find_best_f1_threshold_recovers_a_clean_separation():
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.array([0.1] * 90 + [0.9] * 10)
    threshold, f1 = find_best_f1_threshold(y_true, y_proba)
    assert 0.1 < threshold < 0.9
    assert f1 == pytest.approx(1.0)


def test_evaluate_computes_expected_confusion_counts():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.1])  # 1 FN (idx 1), 1 FP (idx 2)
    metrics = evaluate(y_true, y_proba, threshold=0.5)
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1


@pytest.fixture(scope="module")
def engines():
    c_engine = CVelocityEngine()
    with RubyRulesEngine() as rules_engine:
        yield c_engine, rules_engine


def test_build_features_adds_all_expected_columns_and_no_lookahead(engines):
    c_engine, rules_engine = engines
    df = pd.DataFrame({
        "customer_id": [1, 1, 1],
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:01", "2026-01-01 10:02"]),
        "amount_clp": [10000.0, 500.0, 500.0],
        "merchant_id": ["MER_00001", "MER_00666", "MER_00001"],
        "country_code": ["CL", "CL", "CL"],
        "latitude": [-33.45, -18.47, -18.47],
        "longitude": [-70.66, -70.30, -70.30],
        "is_fraud": [0, 1, 1],
    })
    out = build_features(df, c_engine, rules_engine)

    for col in NUMERIC_FEATURE_COLUMNS:
        assert col in out.columns

    # First row: no prior transaction -> zero distance/speed, not flagged.
    assert out.loc[0, "distance_from_prev_km"] == 0.0
    assert out.loc[0, "is_blacklisted_merchant"] == 0

    # Second row: ~1600km jump in 60s -> impossible travel; blacklisted merchant.
    assert out.loc[1, "is_impossible_travel"] == 1
    assert out.loc[1, "is_blacklisted_merchant"] == 1
    assert out.loc[1, "rules_flagged"] == 1
