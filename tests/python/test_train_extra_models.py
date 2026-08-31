import numpy as np
import pytest
import torch

from src.python.train_extra_models import (
    ACTIVATIONS,
    FocalLoss,
    FraudMLP,
    _metrics_row,
    _predict_mlp,
    _train_mlp,
)


def test_activation_module_matches_requested_kind():
    relu_model = FraudMLP(n_features=5, activation="relu")
    gelu_model = FraudMLP(n_features=5, activation="gelu")
    swish_model = FraudMLP(n_features=5, activation="swish")
    assert isinstance(relu_model.net[1], torch.nn.ReLU)
    assert isinstance(gelu_model.net[1], torch.nn.GELU)
    assert isinstance(swish_model.net[1], torch.nn.SiLU)


def test_fraud_mlp_forward_shape():
    model = FraudMLP(n_features=6, activation="relu")
    x = torch.randn(8, 6)
    out = model(x)
    assert out.shape == (8,)


def test_focal_loss_is_lower_for_confident_correct_predictions():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.tensor([1.0, 0.0])
    confident_correct_logits = torch.tensor([8.0, -8.0])
    confident_wrong_logits = torch.tensor([-8.0, 8.0])
    loss_correct = criterion(confident_correct_logits, targets)
    loss_wrong = criterion(confident_wrong_logits, targets)
    assert loss_correct.item() < loss_wrong.item()


def test_focal_loss_downweights_easy_examples_vs_bce():
    """The whole point of focal loss over plain BCE: an easy, already
    well-classified example should contribute much less to the loss."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    easy_correct = torch.tensor([6.0])
    target = torch.tensor([1.0])
    focal = criterion(easy_correct, target).item()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(easy_correct, target).item()
    assert focal < bce


def test_metrics_row_contains_expected_keys_and_ranges():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    row = _metrics_row("dummy_model", y_true, y_proba, threshold=0.5)
    assert row["model"] == "dummy_model"
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= row[key] <= 1.0


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_train_mlp_learns_a_linearly_separable_toy_problem(activation):
    rng = np.random.RandomState(0)
    n = 400
    X_pos = rng.normal(loc=3.0, scale=0.5, size=(n // 2, 4))
    X_neg = rng.normal(loc=-3.0, scale=0.5, size=(n // 2, 4))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    X_train, X_val = X[:300], X[300:]
    y_train, y_val = y[:300], y[300:]

    model = _train_mlp(activation, X_train, y_train, X_val, y_val, n_epochs=15, seed=0)
    val_proba = _predict_mlp(model, X_val)

    val_pred = (val_proba >= 0.5).astype(int)
    accuracy = (val_pred == y_val).mean()
    assert accuracy > 0.9
