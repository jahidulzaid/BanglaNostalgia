from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import yaml

from src.data_loader import load_dataset, save_split_indices, stratified_split
from src.evaluation import (
    ModelResult,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    write_results_csv,
    write_results_md,
)
from src.models.classical_ml import train_classical_models
from src.models.deep_learning import train_deep_models
from src.models.transformer import train_transformer_model
from src.utils import ensure_dir, get_logger, set_global_seed


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(data_path: str, config_path: str, output_dir: str, models_to_run: List[str]) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))

    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    models_dir = ensure_dir(output_dir / "models")
    logs_dir = ensure_dir(output_dir / "logs")

    logger = get_logger("benchmark", logs_dir / "run.log")
    set_global_seed(seed)
    logger.info(f"Running experiment with seed={seed}")

    df = load_dataset(data_path, logger=logger)
    train_df, val_df, test_df = stratified_split(
        df,
        seed=seed,
        train_size=config.get("splits", {}).get("train", 0.7),
        val_size=config.get("splits", {}).get("val", 0.15),
        test_size=config.get("splits", {}).get("test", 0.15),
    )

    save_split_indices(train_df, val_df, test_df, Path(output_dir) / "splits.json", seed=seed)

    train_texts, train_labels = train_df["text"].tolist(), train_df["label"].tolist()
    val_texts, val_labels = val_df["text"].tolist(), val_df["label"].tolist()
    test_texts, test_labels = test_df["text"].tolist(), test_df["label"].tolist()

    all_results: List[ModelResult] = []

    if "classical" in models_to_run:
        logger.info("=== Classical ML models ===")
        classical_results = train_classical_models(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            seed=seed,
            output_dir=output_dir,
            figures_dir=figures_dir,
            logger=logger,
            vectorizer_config=config.get("vectorizers", {}),
            cv_folds=config.get("classical", {}).get("cv_folds", 5),
        )
        all_results.extend(classical_results)

    if "deep" in models_to_run:
        logger.info("=== Deep learning models ===")
        deep_results = train_deep_models(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            config=config.get("deep_learning", {}),
            output_dir=models_dir,
            figures_dir=figures_dir,
            seed=seed,
            logger=logger,
            predictions_dir=output_dir,
        )
        all_results.extend(deep_results)

    if "transformer" in models_to_run:
        logger.info("=== Transformer model(s) ===")
        # Support both old format (single "transformer" key) and new format ("transformers" list)
        transformer_config = config.get("transformers") or config.get("transformer", {})
        transformer_results = train_transformer_model(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            config={"transformers": transformer_config} if isinstance(transformer_config, list) else transformer_config,
            output_dir=models_dir,
            figures_dir=figures_dir,
            seed=seed,
            logger=logger,
            predictions_dir=output_dir,
        )
        all_results.extend(transformer_results)

    if not all_results:
        logger.warning("No models were run. Exiting.")
        return

    plot_roc_curves(all_results, figures_dir / "roc_curves.png")
    plot_pr_curves(all_results, figures_dir / "pr_curves.png")
    for res in all_results:
        plot_confusion_matrix(res, figures_dir / f"confusion_{res.name}.png")

    write_results_csv(all_results, Path(output_dir) / "results.csv")
    write_results_md(all_results, Path(output_dir) / "results.md")

    ranked = sorted(all_results, key=lambda r: r.metrics.get("f1_macro", 0), reverse=True)
    logger.info("=== Model ranking by macro F1 ===")
    for res in ranked:
        logger.info(
            f"{res.name}: F1={res.metrics.get('f1_macro', float('nan')):.3f} (CI {res.f1_ci}), "
            f"Accuracy={res.metrics.get('accuracy', float('nan')):.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Bangla Nostalgia Detection Benchmark")
    parser.add_argument("--data_path", type=str, required=True, help="Path to bengali_nostalgia_labeled.csv")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        # default=["classical", "deep", "transformer"],
        default=["transformer"],

        help="Which model families to run: classical deep transformer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.data_path, args.config, args.output_dir, args.models)
