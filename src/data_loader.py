from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import preprocess_text_series
from .utils import ensure_dir


def load_dataset(path: str | Path, logger=None) -> pd.DataFrame:
    """Load, clean, and validate the nostalgia dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].copy()
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"] = preprocess_text_series(df["text"].astype(str))
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]

    if logger:
        logger.info(f"Loaded {len(df)} rows after cleaning.")
        label_counts = df["label"].value_counts().to_dict()
        logger.info(f"Class distribution: {label_counts}")

    return df


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("train/val/test sizes must sum to 1.")

    stratify = df["label"]
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_size), stratify=stratify, random_state=seed
    )

    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        stratify=temp_df["label"],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_indices(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_path: str | Path, seed: int = 42) -> None:
    """Persist split indices and class distributions to JSON."""
    ensure_dir(Path(out_path).parent)
    payload = {
        "seed": seed,
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "indices": {
            "train": train_df.index.tolist(),
            "val": val_df.index.tolist(),
            "test": test_df.index.tolist(),
        },
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["load_dataset", "stratified_split", "save_split_indices"]
