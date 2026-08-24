import pytest

from src.python.bridge import CVelocityEngine, RubyRulesEngine


@pytest.fixture(scope="module")
def c_engine():
    return CVelocityEngine()


@pytest.fixture(scope="module")
def rules_engine():
    with RubyRulesEngine() as engine:
        yield engine


def test_first_transaction_has_no_prior_signal(c_engine):
    out = c_engine.compute(current_lat=-33.45, current_lon=-70.66, amount_clp=10000)
    assert out["distance_from_prev_km"] == 0.0
    assert out["implied_speed_kmh"] == 0.0
    assert out["is_impossible_travel"] is False
    assert out["amount_zscore"] == 0.0


def test_impossible_travel_is_flagged(c_engine):
    # Santiago -> northern Chile (~1600km) in 60 seconds.
    out = c_engine.compute(
        current_lat=-18.47, current_lon=-70.30, amount_clp=500,
        prev_lat=-33.45, prev_lon=-70.66, seconds_since_prev=60,
        hist_mean_amount=10000, hist_std_amount=3000,
        txn_count_last_1h=4, txn_count_last_24h=6,
    )
    assert out["distance_from_prev_km"] > 1000
    assert out["is_impossible_travel"] is True
    assert out["velocity_score"] > 50


def test_normal_local_transaction_is_not_flagged(c_engine):
    out = c_engine.compute(
        current_lat=-33.45, current_lon=-70.66, amount_clp=20000,
        prev_lat=-33.44, prev_lon=-70.65, seconds_since_prev=3600,
        hist_mean_amount=18000, hist_std_amount=4000,
        txn_count_last_1h=0, txn_count_last_24h=1,
    )
    assert out["is_impossible_travel"] is False
    assert out["velocity_score"] < 20


def test_amount_zscore_is_capped(c_engine):
    out = c_engine.compute(
        current_lat=-33.45, current_lon=-70.66, amount_clp=50_000_000,
        hist_mean_amount=10000, hist_std_amount=100,  # tiny std -> would blow up uncapped
    )
    assert out["amount_zscore"] == pytest.approx(30.0)


def test_ruby_rules_engine_flags_blacklisted_merchant(rules_engine):
    verdict = rules_engine.evaluate({
        "amount_clp": 1000, "merchant_id": "MER_00666", "country_code": "CL",
        "txn_count_last_1h": 0, "txn_count_last_24h": 1, "is_impossible_travel": False,
    })
    assert verdict["flagged"] is True
    assert "comercio_en_lista_negra" in verdict["triggered_rules"]


def test_ruby_rules_engine_does_not_flag_ordinary_transaction(rules_engine):
    verdict = rules_engine.evaluate({
        "amount_clp": 20000, "merchant_id": "MER_00001", "country_code": "CL",
        "txn_count_last_1h": 0, "txn_count_last_24h": 1, "is_impossible_travel": False,
    })
    assert verdict["flagged"] is False
    assert verdict["triggered_rules"] == []


def test_ruby_rules_engine_survives_many_sequential_calls(rules_engine):
    # Regression guard for the persistent-subprocess design: the process
    # must not die or desync after repeated round trips.
    for i in range(50):
        verdict = rules_engine.evaluate({
            "amount_clp": 1000 + i, "merchant_id": "MER_00001", "country_code": "CL",
            "txn_count_last_1h": 0, "txn_count_last_24h": 1, "is_impossible_travel": False,
        })
        assert verdict["flagged"] is False
