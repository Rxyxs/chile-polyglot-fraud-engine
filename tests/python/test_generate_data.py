from src.python.generate_data import (
    BLACKLISTED_MERCHANT_IDS,
    HIGH_RISK_COUNTRY_CODES,
    N_MERCHANTS,
    generate_dataset,
)


def test_fraud_rate_is_extreme_minority():
    df = generate_dataset(n_transactions=5000, fraud_rate=0.015, n_customers=400, seed=1)
    rate = df["is_fraud"].mean()
    assert 0.005 < rate < 0.05


def test_all_four_fraud_archetypes_present():
    df = generate_dataset(n_transactions=8000, fraud_rate=0.02, n_customers=600, seed=2)
    fraud_types = set(df.loc[df["is_fraud"] == 1, "fraud_type"].unique())
    assert fraud_types == {"velocity_burst", "blacklisted_merchant", "high_risk_country", "structuring"}


def test_blacklisted_merchant_ids_never_collide_with_legit_pool():
    """Regression test for a real bug: MER_00013 used to fall inside the
    legit merchant pool (MER_00001..MER_00199), so ~76% of its occurrences
    were ordinary legit transactions -- see the comment above
    BLACKLISTED_MERCHANT_IDS in generate_data.py. Every blacklisted ID's
    numeric suffix must stay outside the legit pool's range.
    """
    for merchant_id in BLACKLISTED_MERCHANT_IDS:
        suffix = int(merchant_id.split("_")[1])
        assert suffix >= N_MERCHANTS, f"{merchant_id} collides with the legit merchant pool"


def test_blacklisted_merchant_rows_are_all_fraud():
    df = generate_dataset(n_transactions=8000, fraud_rate=0.02, n_customers=600, seed=3)
    for merchant_id in BLACKLISTED_MERCHANT_IDS:
        subset = df[df["merchant_id"] == merchant_id]
        if len(subset) > 0:
            assert (subset["is_fraud"] == 1).all()


def test_high_risk_country_rows_are_all_fraud():
    df = generate_dataset(n_transactions=8000, fraud_rate=0.02, n_customers=600, seed=4)
    for code in HIGH_RISK_COUNTRY_CODES:
        subset = df[df["country_code"] == code]
        if len(subset) > 0:
            assert (subset["is_fraud"] == 1).all()


def test_generation_is_deterministic_given_a_seed():
    df1 = generate_dataset(n_transactions=1000, n_customers=200, seed=99)
    df2 = generate_dataset(n_transactions=1000, n_customers=200, seed=99)
    assert df1["is_fraud"].sum() == df2["is_fraud"].sum()
    assert df1["amount_clp"].sum() == df2["amount_clp"].sum()
