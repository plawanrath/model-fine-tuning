#!/usr/bin/env python3
"""
parquet_to_jsonl.py
-------------------
Recursively walks an input directory, finds *.parquet files,
extracts `instruction`, `input`, `output` columns (or skips the file),
and writes all valid rows to a single Alpaca-style JSON-Lines file.

Usage
-----
python parquet_to_jsonl.py \
       --input-dir path/to/parquet_dir \
       --output-file path/to/data.jsonl \
       [--debug]

Options
-------
--debug   Print column names and the first 2 rows for every skipped file
          (helpful for seeing what the parquet actually contains).
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd  # pip install pandas pyarrow


def main() -> None:
    # --------------- CLI ---------------
    parser = argparse.ArgumentParser(
        description="Combine Parquet files into a single Alpaca-format JSONL dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory to search (recursively) for *.parquet files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write the combined .jsonl file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample rows/columns for parquet files that are skipped",
    )
    args = parser.parse_args()
    # -----------------------------------

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_path = Path(args.output_file).expanduser().resolve()

    if not in_dir.is_dir():
        logging.error(f"[ERR] Input directory '{in_dir}' does not exist or is not a directory.")
        return

    REQUIRED_COLS = {"instruction", "input", "output"}
    total_records: list[dict] = []
    total_count = 0

    for parquet_file in in_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            logging.error(f"[ERR] Cannot read '{parquet_file.name}': {e}")
            continue

        if not REQUIRED_COLS.issubset(df.columns):
            missing = REQUIRED_COLS - set(df.columns)
            logging.info(f"Skipping '{parquet_file.name}' – missing {missing}")
            if args.debug:
                logging.info(f"  columns: {list(df.columns)}")
                sample = df.head(2).to_dict(orient="records")
                logging.info("  sample rows:\n" + json.dumps(sample, indent=2, ensure_ascii=False))
            continue

        # Drop NaNs in required columns
        df = df.dropna(subset=list(REQUIRED_COLS))
        # Strip whitespace
        for col in REQUIRED_COLS:
            df[col] = df[col].astype(str).str.strip()

        # Remove rows with empty instruction or output
        df = df[(df["instruction"] != "") & (df["output"] != "")]
        if df.empty:
            logging.info(f"Skipping '{parquet_file.name}' after filtering (0 valid rows).")
            continue

        records = df[["instruction", "input", "output"]].to_dict(orient="records")
        total_records.extend(records)
        count = len(records)
        total_count += count
        logging.info(f"Loaded {count:>5} examples from {parquet_file.name}")

    # Write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in total_records:
            json.dump(rec, f_out, ensure_ascii=False)
            f_out.write("\n")

    logging.info(f"\n✅  Finished. Total examples written: {total_count}")
    logging.info(f"→ JSONL saved to {out_path}")


if __name__ == "__main__":
    main()
