from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ..evaluation import ModelResult, bootstrap_ci, compute_classification_metrics
from ..utils import ensure_dir


def _build_vectorizer(choice: str, config: dict | None = None) -> TfidfVectorizer:
    cfg = (config or {}).get(choice, {})
    if choice == "word":
        return TfidfVectorizer(
            analyzer=cfg.get("analyzer", "word"),
            ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
            max_features=cfg.get("max_features", 50000),
            min_df=cfg.get("min_df", 2),
        )
    if choice == "char":
        return TfidfVectorizer(
            analyzer=cfg.get("analyzer", "char"),
            ngram_range=tuple(cfg.get("ngram_range", (3, 5))),
            max_features=cfg.get("max_features", 50000),
            min_df=cfg.get("min_df", 2),
        )
    raise ValueError(f"Unknown vectorizer choice: {choice}")


def select_vectorizer(texts: Iterable[str], labels: Iterable[int], seed: int, logger=None, vectorizer_config: dict | None = None) -> str:
    """Pick the better vectorizer (word vs char) using macro F1 on train+val."""
    scoring = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores: Dict[str, float] = {}
    for choice in ("word", "char"):
        pipeline = Pipeline(
            [
                ("vectorizer", _build_vectorizer(choice, vectorizer_config)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=seed)),
            ]
        )
        cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring=scoring, n_jobs=-1)
        scores[choice] = float(np.mean(cv_scores))
        if logger:
            logger.info(f"Vectorizer {choice}: macro F1={scores[choice]:.4f}")

    best_choice = max(scores, key=scores.get)
    if logger:
        logger.info(f"Selected vectorizer: {best_choice}")
    return best_choice


def _param_count(clf) -> int | float | str:
    """Best-effort parameter counting for classical models."""
    try:
        if hasattr(clf, "coef_"):
            return int(np.prod(clf.coef_.shape) + clf.intercept_.size)
        if hasattr(clf, "estimators_"):
            return int(sum(getattr(est.tree_, "node_count", 0) for est in clf.estimators_))
        if hasattr(clf, "tree_"):
            return int(getattr(clf.tree_, "node_count", math.nan))
        if hasattr(clf, "n_neighbors"):
            return clf.n_neighbors
    except Exception:
        return math.nan
    return math.nan


def train_classical_models(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    seed: int,
    output_dir: str | Path,
    figures_dir: str | Path,
    logger=None,
    vectorizer_config: dict | None = None,
    cv_folds: int = 5,
) -> List[ModelResult]:
    """Train and evaluate the specified classical ML models."""
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    train_val_texts = list(train_texts) + list(val_texts)
    train_val_labels = list(train_labels) + list(val_labels)

    vectorizer_choice = select_vectorizer(
        train_val_texts, train_val_labels, seed=seed, logger=logger, vectorizer_config=vectorizer_config
    )

    base_vectorizer = _build_vectorizer(vectorizer_choice, vectorizer_config)
    scoring = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grids = {
        "logistic_regression": {
            "clf": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=seed),
            "params": {"clf__solver": ["liblinear", "saga"], "clf__C": [0.1, 1, 10]},
        },
        "linear_svc": {
            "clf": LinearSVC(class_weight="balanced", random_state=seed),
            "params": {"clf__C": [0.1, 1, 10], "clf__loss": ["hinge", "squared_hinge"]},
        },
        "decision_tree": {
            "clf": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            "params": {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_split": [2, 5, 10]},
        },
        "random_forest": {
            "clf": RandomForestClassifier(random_state=seed),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [10, 20],
                "clf__class_weight": ["balanced", None],
            },
        },
        "knn": {
            "clf": KNeighborsClassifier(),
            "params": {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
        },
    }

    results: List[ModelResult] = []

    for model_name, spec in grids.items():
        if logger:
            logger.info(f"Searching hyperparameters for {model_name}...")
        pipeline = Pipeline([("vectorizer", base_vectorizer), ("clf", spec["clf"])])

        search = GridSearchCV(
            pipeline,
            param_grid=spec["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )

        t_start = time.time()
        search.fit(train_val_texts, train_val_labels)
        train_time = time.time() - t_start

        best_params = search.best_params_
        if logger:
            logger.info(f"{model_name} best params: {best_params}")

        # For LinearSVC, calibrate probabilities after selecting hyperparameters.
        if model_name == "linear_svc":
            base_svc = LinearSVC(
                C=search.best_params_.get("clf__C"),
                loss=search.best_params_.get("clf__loss"),
                class_weight="balanced",
                random_state=seed,
            )
            calibrated = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)
            final_pipeline = Pipeline(
                [("vectorizer", _build_vectorizer(vectorizer_choice, vectorizer_config)), ("clf", calibrated)]
            )
            final_pipeline.fit(train_val_texts, train_val_labels)
        else:
            final_pipeline = search.best_estimator_
            final_pipeline.fit(train_val_texts, train_val_labels)

        y_pred = final_pipeline.predict(test_texts)
        if hasattr(final_pipeline.named_steps["clf"], "predict_proba"):
            y_prob = final_pipeline.predict_proba(test_texts)[:, 1]
        elif hasattr(final_pipeline.named_steps["clf"], "decision_function"):
            y_prob = final_pipeline.decision_function(test_texts)
        else:
            y_prob = None

        metrics_dict = compute_classification_metrics(test_labels, y_pred, y_prob)
        f1_ci = bootstrap_ci(np.array(test_labels), np.array(y_pred), lambda y_t, y_p: f1_score(y_t, y_p, average="macro"), seed=seed)

        # Inference time per 1k
        n_infer = min(1000, len(test_texts))
        infer_start = time.time()
        _ = final_pipeline.predict(test_texts[:n_infer])
        infer_time = time.time() - infer_start
        inference_time_per_1k = infer_time * (1000 / n_infer)

        param_count = _param_count(final_pipeline.named_steps["clf"])

        # Save predictions
        preds_df = pd.DataFrame(
            {
                "text": test_texts,
                "gold_label": test_labels,
                "prediction": y_pred,
                "probability": y_prob if y_prob is not None else np.nan,
            }
        )
        preds_path = Path(output_dir) / f"predictions_{model_name}.csv"
        preds_df.to_csv(preds_path, index=False)

        results.append(
            ModelResult(
                name=model_name,
                y_true=np.array(test_labels),
                y_pred=np.array(y_pred),
                y_prob=np.array(y_prob) if y_prob is not None else None,
                metrics=metrics_dict,
                f1_ci=f1_ci,
                train_time=train_time,
                inference_time_per_1k=inference_time_per_1k,
                params=param_count,
                best_params=best_params,
            )
        )

    return results


__all__ = ["train_classical_models", "select_vectorizer"]
