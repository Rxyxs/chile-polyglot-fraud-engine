import pathlib

import duckdb
import pandas as pd
import pytest


def test_duckdb_round_trips_comparison_table(tmp_path: pathlib.Path):
    db_path = tmp_path / "model_comparison.duckdb"
    df = pd.DataFrame({
        "model": ["logistic_regression", "lightgbm_isoforest_ensemble"],
        "f1": [0.85, 1.00],
        "roc_auc": [0.99, 1.00],
    })

    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE model_metrics AS SELECT * FROM df")
    con.close()

    assert db_path.exists()

    con = duckdb.connect(str(db_path))
    result = con.execute("SELECT model, f1 FROM model_metrics ORDER BY f1 DESC").fetchall()
    con.close()

    assert result[0][0] == "lightgbm_isoforest_ensemble"
    assert result[0][1] == pytest.approx(1.00)
