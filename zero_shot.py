"""
Zero-shot labeling for the nostalgia dataset with periodic checkpoints.

Reads:  dataset/bengali_nostalgia_dataset.csv
Writes: dataset/bengali_nostalgia_labeled.csv
Saves progress every 100 rows so interruptions keep prior work.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from transformers import pipeline

INPUT_PATH = Path("dataset/bengali_nostalgia_dataset_gpu.csv")
OUTPUT_PATH = Path("dataset/bengali_nostalgia_labeled_raw.csv")
CHECKPOINT_EVERY = 100
CSV_FIELDS = ["id", "text", "label", "clean_text", "reference_time"]


def load_checkpoint(path: Path) -> int:
    """Return how many rows have already been written (excluding header)."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        existing = pd.read_csv(path)
        return len(existing)
    except Exception:
        return 0


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    labels = ["nostalgic", "not nostalgic"]

    clf = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=0,  # set to -1 for CPU
    )

    already_done = load_checkpoint(OUTPUT_PATH)
    if already_done >= len(df):
        print(f"All {len(df)} rows already labeled in {OUTPUT_PATH}")
        return

    mode = "a" if already_done else "w"
    with OUTPUT_PATH.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if already_done == 0:
            writer.writeheader()

        processed_since_ckpt = 0
        for idx, row in df.iloc[already_done:].iterrows():
            text = row.get("text")  #if isinstance(row.get("text"), str) else row.get("clean_text", "")
            text = text if isinstance(text, str) else ""
            out = clf(text, labels, hypothesis_template="This text is {}.")
            label_val = 1 if out["labels"][0] == "nostalgic" else 0

            writer.writerow(
                {
                    "id": row.get("id"),
                    "text": row.get("text"),
                    "label": label_val,
                    "init_label": row.get("init_label"),
                    "clean_text": row.get("clean_text"),
                    "reference_time": row.get("reference_time"),
                }
            )
            processed_since_ckpt += 1

            if processed_since_ckpt >= CHECKPOINT_EVERY:
                f.flush()
                processed_since_ckpt = 0
                print(f"Checkpoint: labeled up to row {idx + 1} / {len(df)}")

        f.flush()

    print(f"Finished labeling. Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
