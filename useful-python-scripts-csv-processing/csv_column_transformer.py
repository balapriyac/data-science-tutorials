#!/usr/bin/env python3
"""
csv_column_transformer.py

Apply a sequence of column operations (rename, drop, reorder, derive) to
a CSV file, driven entirely by a JSON config, so the same reshape can be
repeated consistently across many files without editing code each time.

Usage:
    python 04_csv_column_transformer.py input.csv output.csv config.json

Config file format (JSON), operations applied in order:
[
  {"op": "rename", "mapping": {"fname": "first_name", "lname": "last_name"}},
  {"op": "derive", "new_column": "full_name", "template": "{first_name} {last_name}"},
  {"op": "derive", "new_column": "price", "template": "{price_str}", "convert": "strip_currency"},
  {"op": "drop", "columns": ["fname", "lname", "price_str"]},
  {"op": "reorder", "columns": ["full_name", "price", "email"]}
]

Available convert functions: to_int, to_float, strip_currency, upper, lower, title
"""

import argparse
import csv
import json
import re


def to_int(value):
    return str(int(float(value))) if value.strip() else ""


def to_float(value):
    return str(float(value)) if value.strip() else ""


def strip_currency(value):
    cleaned = re.sub(r"[^\d.\-]", "", value)
    return cleaned if cleaned else ""


CONVERTERS = {
    "to_int": to_int,
    "to_float": to_float,
    "strip_currency": strip_currency,
    "upper": str.upper,
    "lower": str.lower,
    "title": str.title,
}


def apply_rename(rows, columns, mapping):
    new_columns = [mapping.get(c, c) for c in columns]
    new_rows = [{mapping.get(k, k): v for k, v in row.items()} for row in rows]
    return new_rows, new_columns


def apply_drop(rows, columns, drop_cols):
    new_columns = [c for c in columns if c not in drop_cols]
    new_rows = [{k: v for k, v in row.items() if k not in drop_cols} for row in rows]
    return new_rows, new_columns


def apply_derive(rows, columns, new_column, template, convert=None):
    converter = CONVERTERS.get(convert) if convert else None
    new_rows = []
    for row in rows:
        try:
            value = template.format(**row)
        except KeyError as e:
            raise ValueError(f"derive template references unknown column {e}")
        if converter:
            try:
                value = converter(value)
            except (ValueError, TypeError):
                value = ""
        row = dict(row)
        row[new_column] = value
        new_rows.append(row)
    new_columns = columns + [new_column] if new_column not in columns else columns
    return new_rows, new_columns


def apply_reorder(rows, columns, order):
    missing = [c for c in order if c not in columns]
    if missing:
        raise ValueError(f"reorder references columns not present: {missing}")
    remaining = [c for c in columns if c not in order]
    new_columns = order + remaining
    return rows, new_columns


def transform(input_path, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        operations = json.load(f)

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    for op_spec in operations:
        op = op_spec["op"]
        if op == "rename":
            rows, columns = apply_rename(rows, columns, op_spec["mapping"])
        elif op == "drop":
            rows, columns = apply_drop(rows, columns, op_spec["columns"])
        elif op == "derive":
            rows, columns = apply_derive(
                rows, columns, op_spec["new_column"], op_spec["template"], op_spec.get("convert")
            )
        elif op == "reorder":
            rows, columns = apply_reorder(rows, columns, op_spec["columns"])
        else:
            raise ValueError(f"unknown operation '{op}'")

    return rows, columns


def write_output(rows, columns, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Apply configurable column operations to a CSV.")
    parser.add_argument("input_file", help="Path to the input CSV")
    parser.add_argument("output_file", help="Path to write the transformed CSV")
    parser.add_argument("config_file", help="Path to the JSON config listing operations")
    args = parser.parse_args()

    rows, columns = transform(args.input_file, args.config_file)
    write_output(rows, columns, args.output_file)

    print(f"Applied {len(json.load(open(args.config_file)))} operation(s) to {len(rows)} row(s)")
    print(f"Final columns: {columns}")
    print(f"Transformed file saved to {args.output_file}")


if __name__ == "__main__":
    main()

