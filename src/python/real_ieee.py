"""Runs the three-layer engine on real fraud: IEEE-CIS (Vesta e-commerce, Kaggle competition
`ieee-fraud-detection`, 590,540 card-not-present transactions, 3.5% fraud).

The synthetic dataset gives a perfect 1.000 on every metric because each of its four fraud
archetypes was built with a clean signature in one layer. This script asks what the same
layers do on real fraud. The mapping to the engine's schema is partial on purpose, and the
gaps are part of the result:

| Engine column | IEEE-CIS source | Note |
|---|---|---|
| `customer_id` | `card1` + `addr1` | the usual proxy for a card holder; IEEE-CIS has no customer id |
| `timestamp` | `TransactionDT` seconds after 2017-12-01 | the reference date is the common assumption |
| `amount_clp` | `TransactionAmt` x 650 | USD to CLP at roughly the Dec 2017 rate, so the UF rules see pesos |
| `merchant_id` | `ProductCD` | five anonymized product codes, none on the synthetic blacklist |
| `country_code` | `addr2` | an anonymized country code, none on the synthetic high-risk list |
| `latitude`, `longitude` | none (0, 0) | no location: impossible travel cannot fire |

Three LightGBM models on the same chronological 70/15/15 split as `train_model.py`:
(1) the engine's own features (C velocity + Ruby rules + Isolation Forest score),
(2) IEEE-CIS's native features, (3) both. Thresholds are chosen on validation (max F1).

    python -m src.python.real_ieee --zip data/raw/ieee/ieee-fraud-detection.zip
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.python.bridge import CVelocityEngine, RubyRulesEngine
from src.python.train_model import (
    ISO_FOREST_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_features,
    find_best_f1_threshold,
    time_based_split,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "outputs" / "reports" / "real_ieee_metrics.json"
CLP_PER_USD = 650.0
REFERENCE_DATE = pd.Timestamp("2017-12-01")
SEED = 42


def load_ieee(zip_path: pathlib.Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("train_transaction.csv") as f:
            return pd.read_csv(f)


def to_engine_schema(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": raw["TransactionID"].astype(str),
        "customer_id": raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str),
        "timestamp": REFERENCE_DATE + pd.to_timedelta(raw["TransactionDT"], unit="s"),
        "amount_clp": raw["TransactionAmt"] * CLP_PER_USD,
        "merchant_id": raw["ProductCD"].astype(str),
        "country_code": raw["addr2"].fillna(-1).astype(int).astype(str),
        "latitude": 0.0,
        "longitude": 0.0,
        "is_fraud": raw["isFraud"].astype(int),
    })


def native_features(raw: pd.DataFrame) -> pd.DataFrame:
    X = raw.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")
    return X


def fit_eval(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    model = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               is_unbalance=True, random_state=SEED, verbosity=-1)
    model.fit(X_train, y_train)
    threshold, _ = find_best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "threshold": round(float(threshold), 3),
        "false_positives": int(((pred == 1) & (y_test == 0)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=pathlib.Path, default=ROOT / "data" / "raw" / "ieee" / "ieee-fraud-detection.zip")
    args = parser.parse_args()

    start = time.perf_counter()
    raw = load_ieee(args.zip)
    print(f"IEEE-CIS: {len(raw):,} transactions, fraud rate {raw['isFraud'].mean():.3%}")
    engine_df = to_engine_schema(raw)

    print("Computing C velocity + Ruby rule features (the real engine, row by row)...")
    t0 = time.perf_counter()
    c_engine = CVelocityEngine()
    with RubyRulesEngine() as rules:
        feats = build_features(engine_df, c_engine, rules)
    feature_seconds = time.perf_counter() - t0
    print(f"  {len(feats):,} rows in {feature_seconds:.0f} s")

    # Native features aligned to the engine rows (build_features re-sorts by customer/time).
    native = native_features(raw).set_index(raw["TransactionID"].astype(str))
    feats = feats.set_index("transaction_id")
    native = native.loc[feats.index]

    train, val, test = time_based_split(feats.reset_index())
    idx = {name: part["transaction_id"].to_numpy() for name, part in (("train", train), ("val", val), ("test", test))}

    iso = IsolationForest(n_estimators=50, contamination=0.035, random_state=SEED, n_jobs=-1)
    iso.fit(train[ISO_FOREST_FEATURE_COLUMNS].to_numpy())
    for part in (train, val, test):
        part["isolation_forest_score"] = -iso.score_samples(part[ISO_FOREST_FEATURE_COLUMNS].to_numpy())
    engine_cols = NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]
    y = {k: feats.loc[v, "is_fraud"].to_numpy() for k, v in idx.items()}

    engine_X = {"train": train[engine_cols], "val": val[engine_cols], "test": test[engine_cols]}
    native_X = {k: native.loc[v].reset_index(drop=True) for k, v in idx.items()}
    both_X = {k: pd.concat([native_X[k], engine_X[k].reset_index(drop=True)], axis=1) for k in idx}

    results = {}
    for name, X in (("engine_features", engine_X), ("ieee_native_features", native_X), ("native_plus_engine", both_X)):
        print(f"LightGBM on {name}...")
        results[name] = fit_eval(X["train"], y["train"], X["val"], y["val"], X["test"], y["test"])
        print("  ", results[name])

    fraud = feats["is_fraud"] == 1
    layer_activity = {
        "rules_flagged_share_fraud": round(float(feats.loc[fraud, "rules_flagged"].mean()), 4),
        "rules_flagged_share_legit": round(float(feats.loc[~fraud, "rules_flagged"].mean()), 4),
        "impossible_travel_share": round(float(feats["is_impossible_travel"].mean()), 4),
        "blacklisted_merchant_share": round(float(feats["is_blacklisted_merchant"].mean()), 4),
        "high_risk_country_share": round(float(feats["is_high_risk_country"].mean()), 4),
        "median_txn_last_24h_fraud": float(feats.loc[fraud, "txn_count_last_24h"].median()),
        "median_txn_last_24h_legit": float(feats.loc[~fraud, "txn_count_last_24h"].median()),
    }
    summary = {
        "dataset": "ieee-fraud-detection (train_transaction.csv)",
        "n_transactions": int(len(feats)),
        "fraud_rate": round(float(feats["is_fraud"].mean()), 4),
        "test_rows": int(len(test)),
        "test_frauds": int(y["test"].sum()),
        "feature_seconds": round(feature_seconds, 1),
        "models": results,
        "layer_activity": layer_activity,
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
