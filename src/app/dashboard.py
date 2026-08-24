"""Streamlit monitoring dashboard for the polyglot fraud engine.

Replays the held-out test split in chronological order, scored with the
same artifacts the API serves (IsolationForest + LightGBM), and breaks each
decision down by which layer (C / Ruby / Python ML) actually contributed the
signal -- since no single layer catches every fraud archetype by design
(see README's architecture section).

Run with: streamlit run src/app/dashboard.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.python.train_model import NUMERIC_FEATURE_COLUMNS, time_based_split  # noqa: E402

MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
PROCESSED_PATH = ROOT / "data" / "processed" / "features.parquet"

st.set_page_config(page_title="Motor Poliglota de Deteccion de Fraude", page_icon="🧩", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_fraud.txt"))
    return metadata, iso_forest, booster


@st.cache_data
def load_scored_test_split():
    features = pd.read_parquet(PROCESSED_PATH)
    _, _, test_df = time_based_split(features)
    test_df = test_df.reset_index(drop=True)

    metadata, iso_forest, booster = load_artifacts()
    X = test_df[NUMERIC_FEATURE_COLUMNS].to_numpy()
    test_df["isolation_forest_score"] = -iso_forest.score_samples(X)
    X_full = test_df[metadata["lgb_feature_columns"]].to_numpy()
    test_df["fraud_probability"] = booster.predict(X_full)

    return test_df, metadata["decision_threshold"]


def kpi_row(df: pd.DataFrame, threshold: float):
    scored = df.copy()
    scored["alert"] = scored["fraud_probability"] >= threshold
    n_total, n_alerts = len(scored), int(scored["alert"].sum())
    n_true_fraud = int(scored["is_fraud"].sum())
    n_correct = int(((scored["alert"]) & (scored["is_fraud"] == 1)).sum())
    precision = n_correct / n_alerts if n_alerts else 0.0
    recall = n_correct / n_true_fraud if n_true_fraud else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transacciones evaluadas", f"{n_total:,}")
    c2.metric("Alertas emitidas", f"{n_alerts:,}")
    c3.metric("Precision", f"{precision:.1%}")
    c4.metric("Recall", f"{recall:.1%}")


def main():
    st.title("🧩 Motor Poliglota de Deteccion de Fraude — C + Ruby + Python")
    st.caption(
        "Nucleo de velocidad en C (< 1ms), motor de reglas DSL en Ruby (blacklist / "
        "pais de riesgo / estructuracion), y ensemble IsolationForest + LightGBM en "
        "Python, combinados en un solo veredicto."
    )

    test_df, threshold = load_scored_test_split()

    st.sidebar.header("Configuracion")
    threshold = st.sidebar.slider("Umbral de decision", 0.0, 1.0, float(threshold), step=0.005)

    kpi_row(test_df, threshold)

    tab_layers, tab_live, tab_map, tab_model = st.tabs(
        ["🧬 Contribucion por capa", "📡 Feed en vivo", "🗺️ Geolocalizacion", "📊 Rendimiento"]
    )

    with tab_layers:
        st.subheader("Que capa detecta cada arquetipo de fraude")
        fraud_only = test_df[test_df["is_fraud"] == 1].copy()
        fraud_only["alert"] = fraud_only["fraud_probability"] >= threshold
        by_type = fraud_only.groupby("fraud_type").agg(
            total=("alert", "count"),
            detected=("alert", "sum"),
            avg_rules_risk_score=("rules_risk_score", "mean"),
            avg_velocity_score=("velocity_score", "mean"),
        ).reset_index()
        by_type["recall"] = (by_type["detected"] / by_type["total"]).round(3)
        st.dataframe(by_type, width="stretch")

        fig = px.bar(
            by_type, x="fraud_type", y=["avg_rules_risk_score", "avg_velocity_score"],
            barmode="group", title="Señal promedio por capa y arquetipo de fraude",
            labels={"value": "Puntaje promedio", "fraud_type": "Tipo de fraude", "variable": "Capa"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_live:
        st.subheader("Simulacion de transacciones en tiempo real")
        n_replay = st.slider("Transacciones a simular", 20, 300, 100, step=10)
        speed = st.select_slider("Velocidad", options=["Lenta", "Normal", "Rapida"], value="Normal")
        delay = {"Lenta": 0.15, "Normal": 0.05, "Rapida": 0.01}[speed]

        if st.button("▶ Iniciar simulacion"):
            replay = test_df.sort_values("timestamp").head(n_replay).reset_index(drop=True)
            table_slot, alert_slot, progress = st.empty(), st.empty(), st.progress(0.0)
            log_rows = []
            for i, row in replay.iterrows():
                is_alert = row["fraud_probability"] >= threshold
                log_rows.append({
                    "hora": row["timestamp"],
                    "cliente": row["customer_id"],
                    "monto_clp": f"${row['amount_clp']:,.0f}",
                    "prob_fraude": f"{row['fraud_probability']:.3f}",
                    "riesgo_ruby": int(row["rules_risk_score"]),
                    "velocity_score_c": round(float(row["velocity_score"]), 1),
                    "alerta": "🚨 FRAUDE" if is_alert else "✅ OK",
                })
                table_slot.dataframe(pd.DataFrame(log_rows[-15:]), width="stretch")
                if is_alert:
                    alert_slot.error(
                        f"🚨 {row['transaction_id']} — cliente {row['customer_id']} — "
                        f"${row['amount_clp']:,.0f} CLP — prob. {row['fraud_probability']:.1%}"
                    )
                progress.progress((i + 1) / len(replay))
                time.sleep(delay)
            st.success(f"Simulacion completa sobre {len(replay)} transacciones.")

    with tab_map:
        st.subheader("Ubicacion geografica de transacciones")
        sample = test_df.sample(min(3000, len(test_df)), random_state=42)
        sample = sample.copy()
        sample["estado"] = np.where(sample["is_fraud"] == 1, "Fraude real", "Legitima")
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude", color="estado",
            color_discrete_map={"Fraude real": "#c0392b", "Legitima": "#2c3e50"},
            size="amount_clp", size_max=15, zoom=3.2, height=600,
            hover_data=["transaction_id", "amount_clp", "fraud_probability"],
            center={"lat": -35.5, "lon": -71.5},
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    with tab_model:
        st.subheader("Metricas del modelo (conjunto de prueba)")
        col1, col2, col3 = st.columns(3)
        for col, name in zip(
            (col1, col2, col3),
            ("precision_recall_curve.png", "confusion_matrix.png", "feature_importance.png"),
        ):
            path = PLOTS_DIR / name
            if path.exists():
                col.image(str(path), width="stretch")

        report_path = ROOT / "outputs" / "reports" / "training_report.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                st.json(json.load(f)["metrics"])


if __name__ == "__main__":
    main()
