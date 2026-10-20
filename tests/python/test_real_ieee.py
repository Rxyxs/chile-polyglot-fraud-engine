"""Schema mapping for the IEEE-CIS real-data run (no download needed)."""
import pandas as pd

from src.python.real_ieee import CLP_PER_USD, REFERENCE_DATE, to_engine_schema


def test_ieee_rows_map_to_the_engine_schema():
    raw = pd.DataFrame({
        "TransactionID": [1, 2], "TransactionDT": [86400, 90000], "TransactionAmt": [10.0, 2.5],
        "card1": [111, 111], "addr1": [325.0, None], "addr2": [87.0, None],
        "ProductCD": ["W", "C"], "isFraud": [0, 1],
    })
    out = to_engine_schema(raw)
    assert list(out["customer_id"]) == ["111_325", "111_-1"]
    assert out["timestamp"].iloc[0] == REFERENCE_DATE + pd.Timedelta(days=1)
    assert out["amount_clp"].iloc[1] == 2.5 * CLP_PER_USD
    assert list(out["country_code"]) == ["87", "-1"]
    # No location in IEEE-CIS: impossible travel cannot fire.
    assert (out[["latitude", "longitude"]] == 0).all().all()
    assert list(out["is_fraud"]) == [0, 1]
