#!/usr/bin/env python3
"""
csv_diff_tool.py

Compare two CSV files by a key column (or combination of key columns)
and report added rows, removed rows, and field-level changes for rows
present in both files. Unchanged rows are skipped entirely.

Usage:
    python 02_csv_diff_tool.py old.csv new.csv --key id --output diff_report.csv
    python 02_csv_diff_tool.py old.csv new.csv --key region,sku --output diff_report.csv
"""

import argparse
import csv
import sys


def load_rows(path, key_columns):
    """Load a CSV into a dict keyed on the given column(s)."""
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for missing in key_columns:
            if missing not in fieldnames:
                raise ValueError(f"key column '{missing}' not found in {path}")

        for row in reader:
            key = tuple(row[k] for k in key_columns)
            if key in rows:
                print(f"warning: duplicate key {key} in {path}, keeping last occurrence",
                      file=sys.stderr)
            rows[key] = row
    return rows, fieldnames


def diff(old_rows, new_rows, key_columns):
    old_keys = set(old_rows)
    new_keys = set(new_rows)

    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    changes = []

    for key in removed:
        changes.append({
            "change_type": "removed",
            "key": format_key(key, key_columns),
            "column": "", "old_value": "", "new_value": ""
        })

    for key in added:
        changes.append({
            "change_type": "added",
            "key": format_key(key, key_columns),
            "column": "", "old_value": "", "new_value": ""
        })

    for key in common:
        old_row = old_rows[key]
        new_row = new_rows[key]
        all_columns = set(old_row) | set(new_row)
        for col in all_columns - set(key_columns):
            old_val = old_row.get(col, "")
            new_val = new_row.get(col, "")
            if old_val != new_val:
                changes.append({
                    "change_type": "changed",
                    "key": format_key(key, key_columns),
                    "column": col,
                    "old_value": old_val,
                    "new_value": new_val
                })

    return changes


def format_key(key_tuple, key_columns):
    return "|".join(f"{col}={val}" for col, val in zip(key_columns, key_tuple))


def write_report(changes, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["change_type", "key", "column", "old_value", "new_value"]
        )
        writer.writeheader()
        writer.writerows(changes)


def main():
    parser = argparse.ArgumentParser(description="Diff two CSV files by key column(s).")
    parser.add_argument("old_file", help="Path to the baseline/old CSV file")
    parser.add_argument("new_file", help="Path to the updated/new CSV file")
    parser.add_argument("--key", required=True,
                         help="Comma-separated column name(s) to use as the row identifier")
    parser.add_argument("--output", default="diff_report.csv",
                         help="Path to write the diff report (default: diff_report.csv)")
    args = parser.parse_args()

    key_columns = [k.strip() for k in args.key.split(",")]

    old_rows, _ = load_rows(args.old_file, key_columns)
    new_rows, _ = load_rows(args.new_file, key_columns)

    changes = diff(old_rows, new_rows, key_columns)

    added = sum(1 for c in changes if c["change_type"] == "added")
    removed = sum(1 for c in changes if c["change_type"] == "removed")
    changed_fields = sum(1 for c in changes if c["change_type"] == "changed")

    print(f"Added rows:     {added}")
    print(f"Removed rows:   {removed}")
    print(f"Changed fields: {changed_fields}")

    write_report(changes, args.output)
    print(f"Diff report written to {args.output}")


if __name__ == "__main__":
    main()


