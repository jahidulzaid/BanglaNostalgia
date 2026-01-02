import numpy as np
from sklearn.metrics import f1_score

from src.evaluation import bootstrap_ci, compute_classification_metrics


def test_metrics_and_ci_shapes():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.2, 0.8, 0.4, 0.3])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "f1_macro" in metrics
    ci = bootstrap_ci(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro"), n_bootstrap=10, seed=0)
    assert len(ci) == 2
