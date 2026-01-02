"""
Utility to split the generated nostalgia dataset CSV into smaller chunks.

Default behavior:
- Reads the main dataset at CONFIG.out_csv_path (data/bengali_nostalgia_dataset.csv)
- Writes chunks of 1,000 rows (including header) to data/splits/*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import CONFIG


def split_csv(
    input_path: Path,
    output_dir: Path,
    chunk_size: int = 1_000,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(f"Input file not found or empty: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []

    current_file = None
    current_writer = None
    rows_in_chunk = 0
    chunk_index = 0

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV missing header / fieldnames.")

        def start_new_chunk():
            nonlocal current_file, current_writer, rows_in_chunk, chunk_index
            if current_file:
                current_file.close()
            chunk_index += 1
            rows_in_chunk = 0
            chunk_path = output_dir / f"{input_path.stem}_part{chunk_index:03d}.csv"
            chunk_paths.append(chunk_path)
            current_file = chunk_path.open("w", encoding="utf-8", newline="")
            current_writer = csv.DictWriter(current_file, fieldnames=fieldnames)
            current_writer.writeheader()
            return chunk_path

        start_new_chunk()

        for row in reader:
            if rows_in_chunk >= chunk_size:
                start_new_chunk()
            current_writer.writerow(row)  # type: ignore[arg-type]
            rows_in_chunk += 1

    if current_file:
        current_file.close()
    if chunk_paths and rows_in_chunk == 0:
        chunk_paths[-1].unlink(missing_ok=True)
        chunk_paths.pop()
    return chunk_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the nostalgia dataset CSV into chunks.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CONFIG.out_csv_path,
        help=f"Input CSV path (default: {CONFIG.out_csv_path})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "splits",
        help="Directory to write chunked CSVs (default: data/splits)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000,
        help="Rows per chunk (excluding header) (default: 1000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = split_csv(args.input, args.output_dir, args.chunk_size)
    print(f"Wrote {len(paths)} chunk files to {args.output_dir}")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
