"""Synthetic Chilean bank-transaction generator for the polyglot fraud
engine demo.

Unlike a single-signal fraud dataset, this generator deliberately injects
FOUR distinct fraud archetypes, each aimed at a different layer of the
architecture, so the three-language design has a genuine reason to exist
rather than being decorative:

- ``velocity_burst``  -- rapid-fire transactions + a geographic jump  -> the
  C module's velocity/distance metrics are what catches this.
- ``blacklisted_merchant`` -- a single transaction at a known-bad merchant
  -> the Ruby rules engine's blacklist rule catches this; the C/ML layers
  see nothing unusual about the transaction itself.
- ``high_risk_country`` -- a transaction tagged with a high-risk country
  code -> also a Ruby rules-engine signal.
- ``structuring`` -- several transactions just under the UF 450 reporting
  threshold in the same day -> a Ruby rules-engine pattern rule.

No single layer catches every archetype -- that's the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHILEAN_CITIES = {
    "Santiago": (-33.4489, -70.6693),
    "Valparaiso": (-33.0472, -71.6127),
    "Concepcion": (-36.8201, -73.0444),
    "La Serena": (-29.9027, -71.2519),
    "Antofagasta": (-23.6509, -70.3975),
    "Temuco": (-38.7359, -72.5904),
}

TRANSACTION_TYPES = [
    "transferencia", "retiro_cajero", "pago_tarjeta", "deposito",
    "pago_servicios", "compra_online",
]

# Must match src/ruby/rules_engine.rb's BLACKLISTED_MERCHANT_IDS /
# HIGH_RISK_COUNTRY_CODES exactly -- these are the cross-language contract.
# All three IDs are deliberately chosen outside the legit merchant pool
# (MER_00001..MER_00199, see N_MERCHANTS below) -- a real bug was caught
# here empirically: MER_00013 fell inside that range, so ~76% of its
# occurrences were ordinary legit transactions that happened to land on it
# by chance, contaminating the "blacklisted merchant" signal and dragging
# that fraud archetype's recall down to 44% (vs. 100% for the other three
# archetypes). Fixed by moving it outside the legit range, like the other two.
BLACKLISTED_MERCHANT_IDS = ["MER_00666", "MER_00777", "MER_00999"]
HIGH_RISK_COUNTRY_CODES = ["XX", "YY", "ZZ"]
NORMAL_COUNTRY_CODES = ["CL", "AR", "US", "BR", "PE"]
NORMAL_COUNTRY_WEIGHTS = [0.90, 0.04, 0.03, 0.02, 0.01]

UF_TO_CLP = 39_000.0
UF_REPORT_THRESHOLD_CLP = 450 * UF_TO_CLP

N_MERCHANTS = 200
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _build_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    city_names = list(CHILEAN_CITIES.keys())
    weights = np.array([0.45, 0.14, 0.12, 0.10, 0.10, 0.09])
    home_city = rng.choice(city_names, size=n_customers, p=weights)

    home_lat = np.empty(n_customers)
    home_lon = np.empty(n_customers)
    for i, city in enumerate(home_city):
        base_lat, base_lon = CHILEAN_CITIES[city]
        home_lat[i] = base_lat + rng.normal(0, 0.06)
        home_lon[i] = base_lon + rng.normal(0, 0.06)

    avg_amount = rng.lognormal(mean=10.5, sigma=0.6, size=n_customers)
    amount_std = avg_amount * rng.uniform(0.15, 0.4, size=n_customers)

    return pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "home_lat": home_lat,
        "home_lon": home_lon,
        "avg_amount_clp": avg_amount,
        "amount_std_clp": amount_std,
    })


def _legit_rows(customers: pd.DataFrame, n: int, start_ts, end_ts, rng) -> pd.DataFrame:
    n_customers = len(customers)
    activity = rng.lognormal(0, 0.8, size=n_customers)
    activity /= activity.sum()
    idx = rng.choice(n_customers, size=n, p=activity)
    cust = customers.iloc[idx].reset_index(drop=True)

    span = (end_ts - start_ts).total_seconds()
    timestamps = start_ts + pd.to_timedelta(rng.uniform(0, span, size=n), unit="s")

    amounts = cust["avg_amount_clp"].to_numpy() * rng.lognormal(0, 0.3, size=n)
    lat = cust["home_lat"].to_numpy() + rng.normal(0, 0.03, size=n)
    lon = cust["home_lon"].to_numpy() + rng.normal(0, 0.03, size=n)

    merchant_ids = [f"MER_{i:05d}" for i in rng.integers(1, N_MERCHANTS, size=n)]
    countries = rng.choice(NORMAL_COUNTRY_CODES, size=n, p=NORMAL_COUNTRY_WEIGHTS)
    txn_types = rng.choice(TRANSACTION_TYPES, size=n)

    return pd.DataFrame({
        "customer_id": cust["customer_id"].to_numpy(),
        "timestamp": timestamps,
        "amount_clp": np.round(amounts, 0),
        "transaction_type": txn_types,
        "merchant_id": merchant_ids,
        "country_code": countries,
        "latitude": lat,
        "longitude": lon,
        "is_fraud": 0,
        "fraud_type": "none",
    })


def _velocity_burst_rows(customers, n_events, start_ts, end_ts, rng) -> pd.DataFrame:
    span = (end_ts - start_ts).total_seconds()
    city_names = list(CHILEAN_CITIES.keys())
    rows = []
    victims = rng.choice(len(customers), size=n_events, replace=False)
    for ci in victims:
        cust = customers.iloc[ci]
        event_start = start_ts + pd.to_timedelta(rng.uniform(0, span), unit="s")
        fraud_city = rng.choice(city_names)
        base_lat, base_lon = CHILEAN_CITIES[fraud_city]
        f_lat, f_lon = base_lat + rng.normal(0, 0.05), base_lon + rng.normal(0, 0.05)

        burst = rng.integers(2, 6)
        gaps = np.concatenate([[0], rng.uniform(20, 200, size=burst - 1)])
        offsets = np.cumsum(gaps)
        amounts = rng.uniform(500, 3000, size=burst)

        for j in range(burst):
            rows.append({
                "customer_id": cust["customer_id"],
                "timestamp": event_start + pd.to_timedelta(offsets[j], unit="s"),
                "amount_clp": round(float(amounts[j]), 0),
                "transaction_type": rng.choice(TRANSACTION_TYPES),
                "merchant_id": f"MER_{rng.integers(1, N_MERCHANTS):05d}",
                "country_code": "CL",
                "latitude": f_lat + rng.normal(0, 0.01),
                "longitude": f_lon + rng.normal(0, 0.01),
                "is_fraud": 1,
                "fraud_type": "velocity_burst",
            })
    return pd.DataFrame(rows)


def _blacklisted_merchant_rows(customers, n, rng) -> pd.DataFrame:
    idx = rng.choice(len(customers), size=n, replace=False)
    cust = customers.iloc[idx].reset_index(drop=True)
    return pd.DataFrame({
        "customer_id": cust["customer_id"].to_numpy(),
        "timestamp": pd.NaT,  # filled in by caller with a random time in-range
        "amount_clp": np.round(cust["avg_amount_clp"].to_numpy() * rng.uniform(0.8, 1.5, size=n), 0),
        "transaction_type": rng.choice(TRANSACTION_TYPES, size=n),
        "merchant_id": rng.choice(BLACKLISTED_MERCHANT_IDS, size=n),
        "country_code": "CL",
        "latitude": cust["home_lat"].to_numpy() + rng.normal(0, 0.03, size=n),
        "longitude": cust["home_lon"].to_numpy() + rng.normal(0, 0.03, size=n),
        "is_fraud": 1,
        "fraud_type": "blacklisted_merchant",
    })


def _high_risk_country_rows(customers, n, rng) -> pd.DataFrame:
    idx = rng.choice(len(customers), size=n, replace=False)
    cust = customers.iloc[idx].reset_index(drop=True)
    return pd.DataFrame({
        "customer_id": cust["customer_id"].to_numpy(),
        "timestamp": pd.NaT,
        "amount_clp": np.round(cust["avg_amount_clp"].to_numpy() * rng.uniform(1.0, 3.0, size=n), 0),
        "transaction_type": "compra_online",
        "merchant_id": [f"MER_{i:05d}" for i in rng.integers(1, N_MERCHANTS, size=n)],
        "country_code": rng.choice(HIGH_RISK_COUNTRY_CODES, size=n),
        "latitude": cust["home_lat"].to_numpy(),
        "longitude": cust["home_lon"].to_numpy(),
        "is_fraud": 1,
        "fraud_type": "high_risk_country",
    })


def _structuring_rows(customers, n_events, start_ts, end_ts, rng) -> pd.DataFrame:
    span = (end_ts - start_ts).total_seconds()
    rows = []
    victims = rng.choice(len(customers), size=n_events, replace=False)
    for ci in victims:
        cust = customers.iloc[ci]
        event_start = start_ts + pd.to_timedelta(rng.uniform(0, span - 86400), unit="s")
        n_parts = rng.integers(3, 5)
        # Each part just under the UF 450 threshold -- the classic
        # "structuring" / smurfing signature.
        part_amount = UF_REPORT_THRESHOLD_CLP * rng.uniform(0.86, 0.97)
        for k in range(n_parts):
            rows.append({
                "customer_id": cust["customer_id"],
                "timestamp": event_start + pd.to_timedelta(k * rng.uniform(3600, 5 * 3600), unit="s"),
                "amount_clp": round(float(part_amount * rng.uniform(0.97, 1.03)), 0),
                "transaction_type": "transferencia",
                "merchant_id": f"MER_{rng.integers(1, N_MERCHANTS):05d}",
                "country_code": "CL",
                "latitude": cust["home_lat"],
                "longitude": cust["home_lon"],
                "is_fraud": 1,
                "fraud_type": "structuring",
            })
    return pd.DataFrame(rows)


def generate_dataset(
    n_transactions: int = 50_000,
    fraud_rate: float = 0.015,
    n_customers: int = 3_000,
    start_date: str = "2026-03-01",
    n_days: int = 45,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(start_date) + pd.Timedelta(days=n_days)
    customers = _build_customers(n_customers, rng)

    n_fraud_total = max(4, int(n_transactions * fraud_rate))
    # Split the fraud budget across the four archetypes.
    n_burst_events = max(1, int(n_fraud_total * 0.35 / 3.5))
    n_blacklist = max(1, int(n_fraud_total * 0.25))
    n_high_risk = max(1, int(n_fraud_total * 0.20))
    n_structuring_events = max(1, int(n_fraud_total * 0.20 / 3.5))

    burst_df = _velocity_burst_rows(customers, n_burst_events, start_ts, end_ts, rng)
    blacklist_df = _blacklisted_merchant_rows(customers, n_blacklist, rng)
    blacklist_df["timestamp"] = start_ts + pd.to_timedelta(
        rng.uniform(0, (end_ts - start_ts).total_seconds(), size=len(blacklist_df)), unit="s"
    )
    high_risk_df = _high_risk_country_rows(customers, n_high_risk, rng)
    high_risk_df["timestamp"] = start_ts + pd.to_timedelta(
        rng.uniform(0, (end_ts - start_ts).total_seconds(), size=len(high_risk_df)), unit="s"
    )
    structuring_df = _structuring_rows(customers, n_structuring_events, start_ts, end_ts, rng)

    fraud_df = pd.concat([burst_df, blacklist_df, high_risk_df, structuring_df], ignore_index=True)
    n_legit = max(0, n_transactions - len(fraud_df))
    legit_df = _legit_rows(customers, n_legit, start_ts, end_ts, rng)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i:07d}" for i in range(len(df))]
    df = df[[
        "transaction_id", "customer_id", "timestamp", "amount_clp", "transaction_type",
        "merchant_id", "country_code", "latitude", "longitude", "is_fraud", "fraud_type",
    ]]
    return df


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_parquet(out_dir / "transactions.parquet", index=False)
    df.to_csv(out_dir / "transactions.csv", index=False)

    print(f"Generated {len(df):,} transactions -> {out_dir}")
    print(f"Fraud rate: {100 * df['is_fraud'].mean():.3f}%")
    print(df.loc[df['is_fraud'] == 1, 'fraud_type'].value_counts())
