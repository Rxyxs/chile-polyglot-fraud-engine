"""Two additional, complementary modeling approaches over the *same*
features.parquet / time-based split produced by ``train_model.py``:

1. **Logistic Regression** -- an interpretable statistical baseline. Its
   only job is to answer "how much of the separation does a two-stage
   IsolationForest+LightGBM ensemble actually buy over the simplest
   possible linear model on the same engineered features?"
2. **PyTorch MLP with Focal Loss**, trained three times with a different
   hidden-layer activation each run (ReLU, GELU, Swish/SiLU) so the
   activation choice is an empirical comparison, not an assumption. Focal
   Loss (Lin et al., 2017) is used instead of plain BCE because the fraud
   rate here is 1.5% -- focal loss's ``(1-p_t)^gamma`` term down-weights
   the easy, already-well-classified legit majority so gradient signal
   isn't dominated by them the way it is under uniform BCE.

Both consume the exact same feature matrix LightGBM sees (including
``isolation_forest_score`` from the already-fitted IsolationForest), so all
three supervised approaches are compared on identical inputs. Results are
persisted to DuckDB (``outputs/reports/model_comparison.duckdb``) alongside
the existing JSON report convention, and two comparison plots are written
to ``outputs/plots/`` in the same style as ``train_model.py``.

Run after ``train_model.py`` (needs its parquet + isolation_forest.joblib +
metadata.json):

    python -m src.python.train_extra_models
"""
from __future__ import annotations

import json
import pathlib

import duckdb
import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.python.train_model import (
    ISO_FOREST_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    PROCESSED_PATH,
    find_best_f1_threshold,
    time_based_split,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"
DUCKDB_PATH = REPORTS_DIR / "model_comparison.duckdb"

SEED = 42
ACTIVATIONS = ["relu", "gelu", "swish"]


def _activation_module(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "swish":
        return torch.nn.SiLU()  # SiLU == Swish (x * sigmoid(x))
    raise ValueError(f"unknown activation: {name}")


class FraudMLP(torch.nn.Module):
    """Small MLP: input -> 64 -> 32 -> 1 (logit), configurable activation."""

    def __init__(self, n_features: int, activation: str):
        super().__init__()
        act = _activation_module(activation)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64), act,
            torch.nn.Linear(64, 32), _activation_module(activation),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(torch.nn.Module):
    """Binary focal loss (Lin et al., 2017): down-weights easy examples so
    the 1.5%-fraud class imbalance doesn't get drowned out by legit-majority
    gradient under plain BCE."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def _train_mlp(activation: str, X_train, y_train, X_val, y_val, n_epochs=40, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    model = FraudMLP(X_train.shape[1], activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    batch_size = 512
    n = X_train_t.shape[0]
    best_state, best_pr_auc = None, -1.0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(X_val_t)).numpy()
        pr_auc = average_precision_score(y_val, val_proba)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def _predict_mlp(model: FraudMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def _metrics_row(name: str, y_true, y_proba, threshold) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def main():
    print("[1/6] Loading processed features + prior artifacts from train_model.py...")
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} not found. Run `python -m src.python.train_model` first.")
    features = pd.read_parquet(PROCESSED_PATH)

    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found. Run `python -m src.python.train_model` first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lgb_feature_columns = metadata["lgb_feature_columns"]

    iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    lgb_booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_fraud.txt"))

    train_df, val_df, test_df = time_based_split(features)

    print("[2/6] Recomputing isolation_forest_score for the extra-models feature matrix...")
    for split_df in (train_df, val_df, test_df):
        X_iso = split_df[ISO_FOREST_FEATURE_COLUMNS].to_numpy()
        split_df["isolation_forest_score"] = -iso_forest.score_samples(X_iso)

    X_train_raw = train_df[lgb_feature_columns].to_numpy()
    X_val_raw = val_df[lgb_feature_columns].to_numpy()
    X_test_raw = test_df[lgb_feature_columns].to_numpy()
    y_train = train_df["is_fraud"].to_numpy()
    y_val = val_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    scaler = StandardScaler().fit(X_train_raw)
    X_train, X_val, X_test = scaler.transform(X_train_raw), scaler.transform(X_val_raw), scaler.transform(X_test_raw)

    print("[3/6] Training Logistic Regression baseline (interpretable statistical model)...")
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    logreg.fit(X_train, y_train)
    logreg_val_proba = logreg.predict_proba(X_val)[:, 1]
    logreg_threshold, _ = find_best_f1_threshold(y_val, logreg_val_proba)
    logreg_test_proba = logreg.predict_proba(X_test)[:, 1]

    print("[4/6] Training PyTorch MLP + Focal Loss, one run per activation (ReLU / GELU / Swish)...")
    activation_results = {}
    for activation in ACTIVATIONS:
        model = _train_mlp(activation, X_train, y_train, X_val, y_val, seed=SEED)
        val_proba = _predict_mlp(model, X_val)
        threshold, _ = find_best_f1_threshold(y_val, val_proba)
        test_proba = _predict_mlp(model, X_test)
        activation_results[activation] = {
            "model": model,
            "threshold": threshold,
            "val_metrics": _metrics_row(f"pytorch_mlp_{activation}", y_val, val_proba, threshold),
            "test_metrics": _metrics_row(f"pytorch_mlp_{activation}", y_test, test_proba, threshold),
            "test_proba": test_proba,
        }
        torch.save(model.state_dict(), MODELS_DIR / f"pytorch_mlp_{activation}.pt")
        print(f"  {activation:>6}: val F1={activation_results[activation]['val_metrics']['f1']:.4f} "
              f"PR-AUC={activation_results[activation]['val_metrics']['pr_auc']:.4f}")

    best_activation = max(ACTIVATIONS, key=lambda a: activation_results[a]["val_metrics"]["pr_auc"])
    print(f"  best activation on val PR-AUC: {best_activation}")

    print("[5/6] Scoring existing LightGBM booster on the identical feature matrix for comparison...")
    lgb_test_proba = lgb_booster.predict(X_test_raw)
    lgb_threshold = metadata["decision_threshold"]

    print("[6/6] Assembling comparison table, saving artifacts, DuckDB, and plots...")
    comparison_rows = [
        _metrics_row("logistic_regression", y_test, logreg_test_proba, logreg_threshold),
        *[activation_results[a]["test_metrics"] for a in ACTIVATIONS],
        _metrics_row("lightgbm_isoforest_ensemble", y_test, lgb_test_proba, lgb_threshold),
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "extra_models_scaler.joblib")

    comparison_report = {
        "best_activation": best_activation,
        "test_split_comparison": comparison_rows,
        "activation_val_comparison": [activation_results[a]["val_metrics"] for a in ACTIVATIONS],
        "feature_columns": lgb_feature_columns,
        "seed": SEED,
    }
    with open(REPORTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    # --- DuckDB persistence: comparative metrics + per-row predictions ---
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("DROP TABLE IF EXISTS model_metrics")
    con.execute("CREATE TABLE model_metrics AS SELECT * FROM comparison_df")

    predictions_df = pd.DataFrame({
        "row_id": np.arange(len(y_test)),
        "y_true": y_test,
        "logistic_regression_proba": logreg_test_proba,
        "pytorch_mlp_relu_proba": activation_results["relu"]["test_proba"],
        "pytorch_mlp_gelu_proba": activation_results["gelu"]["test_proba"],
        "pytorch_mlp_swish_proba": activation_results["swish"]["test_proba"],
        "lightgbm_isoforest_proba": lgb_test_proba,
    })
    con.execute("DROP TABLE IF EXISTS test_predictions")
    con.execute("CREATE TABLE test_predictions AS SELECT * FROM predictions_df")
    con.close()
    print(f"  DuckDB comparison persisted to {DUCKDB_PATH}")

    # --- Plot 1: ROC curves, all four supervised approaches ---
    plt.figure(figsize=(6.5, 5.5))
    for name, proba, color in [
        ("Logistic Regression", logreg_test_proba, "#7f8c8d"),
        (f"PyTorch MLP ({best_activation})", activation_results[best_activation]["test_proba"], "#8e44ad"),
        ("IsoForest + LightGBM", lgb_test_proba, "#c0392b"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#bdc3c7", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Test Set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison_roc.png", dpi=150)
    plt.close()

    # --- Plot 2: activation comparison bar chart (val PR-AUC + F1) ---
    x = np.arange(len(ACTIVATIONS))
    width = 0.35
    pr_aucs = [activation_results[a]["val_metrics"]["pr_auc"] for a in ACTIVATIONS]
    f1s = [activation_results[a]["val_metrics"]["f1"] for a in ACTIVATIONS]
    plt.figure(figsize=(6.5, 5))
    plt.bar(x - width / 2, pr_aucs, width, label="PR-AUC", color="#2c3e50")
    plt.bar(x + width / 2, f1s, width, label="F1", color="#c0392b")
    plt.xticks(x, [a.upper() if a != "swish" else "Swish (SiLU)" for a in ACTIVATIONS])
    plt.ylabel("Score (validation split)")
    plt.title("PyTorch MLP + Focal Loss: Activation Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "activation_comparison.png", dpi=150)
    plt.close()

    print(f"Artifacts saved under {MODELS_DIR}, {PLOTS_DIR}, {REPORTS_DIR}")


if __name__ == "__main__":
    main()
