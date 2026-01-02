from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from .utils import ensure_dir


MetricFunc = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ModelResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    metrics: dict
    f1_ci: tuple[float, float]
    train_time: float
    inference_time_per_1k: float
    params: int | float | str
    best_params: Optional[dict] = None
    notes: Optional[str] = None


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFunc,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the provided metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = float(np.percentile(scores, 100 * (alpha / 2)))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Optional[Iterable[float]] = None,
) -> dict:
    """Compute benchmark metrics."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    metrics_dict = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision_macro": metrics.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None:
        y_prob = np.asarray(list(y_prob))
        metrics_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
        metrics_dict["pr_auc"] = metrics.auc(recall, precision)
    else:
        metrics_dict["roc_auc"] = math.nan
        metrics_dict["pr_auc"] = math.nan

    return metrics_dict


def plot_roc_curves(results: List[ModelResult], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(8, 6))
    for res in results:
        if res.y_prob is None:
            continue
        fpr, tpr, _ = metrics.roc_curve(res.y_true, res.y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{res.name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr_curves(results: List[ModelResult], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(8, 6))
    for res in results:
        if res.y_prob is None:
            continue
        precision, recall, _ = metrics.precision_recall_curve(res.y_true, res.y_prob)
        pr_auc = metrics.auc(recall, precision)
        plt.plot(recall, precision, label=f"{res.name} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(res: ModelResult, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    cm = metrics.confusion_matrix(res.y_true, res.y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {res.name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def format_interval(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def seconds_to_str(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {sec:.1f}s"
    return f"{sec:.2f}s"


def write_results_csv(results: List[ModelResult], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    rows = []
    for res in results:
        row = {
            "model": res.name,
            **res.metrics,
            "f1_ci_lower": res.f1_ci[0],
            "f1_ci_upper": res.f1_ci[1],
            "train_time_sec": res.train_time,
            "inference_time_per_1k_sec": res.inference_time_per_1k,
            "params": res.params,
            "best_params": res.best_params,
            "notes": res.notes,
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_results_md(results: List[ModelResult], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    sorted_results = sorted(results, key=lambda r: r.metrics.get("f1_macro", 0), reverse=True)
    best_name = sorted_results[0].name if sorted_results else ""

    lines = ["## Model Performance Comparison", "", "| Model | Accuracy | F1 (Macro) | 95% CI | Precision | Recall | ROC-AUC | Params | Train Time |", "|-------|----------|------------|--------|-----------|--------|---------|--------|------------|"]
    for res in sorted_results:
        name = f"**{res.name}**" if res.name == best_name else res.name
        m = res.metrics
        params_display = res.params
        if isinstance(params_display, float) and math.isnan(params_display):
            params_display = "-"
        line = f"| {name} | {m.get('accuracy', float('nan')):.3f} | {m.get('f1_macro', float('nan')):.3f} | {format_interval(res.f1_ci)} | {m.get('precision_macro', float('nan')):.3f} | {m.get('recall_macro', float('nan')):.3f} | {m.get('roc_auc', float('nan')):.3f} | {params_display} | {seconds_to_str(res.train_time)} |"
        lines.append(line)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ModelResult",
    "bootstrap_ci",
    "compute_classification_metrics",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_matrix",
    "write_results_csv",
    "write_results_md",
    "seconds_to_str",
]
