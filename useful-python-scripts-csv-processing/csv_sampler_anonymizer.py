#!/usr/bin/env python3
"""
csv_sampler_anonymizer.py

Take a random, uniform sample of rows from a large CSV (via reservoir
sampling, so the whole file never needs to be loaded into memory) and
mask any columns flagged as sensitive using a consistent keyed hash, so
the same input value always produces the same masked output within a
run and relationships between rows are preserved.

Usage:
    python 05_csv_sampler_anonymizer.py input.csv output.csv \
        --sample-size 500 --mask email,phone,name --salt "some-secret"
"""

import argparse
import csv
import hashlib
import random


def reservoir_sample(reader, sample_size, rng):
    """Return a uniform random sample of up to sample_size rows from an iterator."""
    sample = []
    for i, row in enumerate(reader):
        if i < sample_size:
            sample.append(row)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                sample[j] = row
    return sample


def mask_value(value, column_name, salt):
    """Deterministically mask a value using a keyed hash, keeping output readable."""
    if value == "":
        return value
    digest = hashlib.sha256((salt + column_name + value).encode("utf-8")).hexdigest()[:8]

    if "email" in column_name.lower():
        return f"user_{digest}@example.com"
    if "phone" in column_name.lower():
        return f"555-{digest[:3]}-{digest[3:7]}"
    return f"{column_name}_{digest}"


def run(input_path, output_path, sample_size, mask_columns, salt, seed=None):
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in mask_columns:
            if col not in fieldnames:
                raise ValueError(f"mask column '{col}' not found in input file")
        sample = reservoir_sample(reader, sample_size, rng)

    for row in sample:
        for col in mask_columns:
            row[col] = mask_value(row[col], col, salt)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)

    return len(sample)


def main():
    parser = argparse.ArgumentParser(
        description="Sample rows from a CSV and mask sensitive columns consistently."
    )
    parser.add_argument("input_file", help="Path to the input CSV")
    parser.add_argument("output_file", help="Path to write the sampled/masked CSV")
    parser.add_argument("--sample-size", type=int, default=100,
                         help="Number of rows to sample (default: 100)")
    parser.add_argument("--mask", default="",
                         help="Comma-separated list of column names to mask")
    parser.add_argument("--salt", default="change-this-salt",
                         help="Salt used for the masking hash (keep consistent across runs "
                              "if you need the same value to mask the same way each time)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional random seed for reproducible sampling")
    args = parser.parse_args()

    mask_columns = [c.strip() for c in args.mask.split(",") if c.strip()]

    row_count = run(
        args.input_file, args.output_file, args.sample_size, mask_columns, args.salt, args.seed
    )

    print(f"Sampled {row_count} row(s) from {args.input_file}")
    if mask_columns:
        print(f"Masked columns: {mask_columns}")
    else:
        print("No columns were masked (use --mask to specify sensitive columns)")
    print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
  
