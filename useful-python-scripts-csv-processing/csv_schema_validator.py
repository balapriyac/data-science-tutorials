#!/usr/bin/env python3
"""
csv_schema_validator.py

Validate a CSV file against a JSON schema definition. Streams the file
row by row so it works on large files, and produces a row-level error
report rather than a simple pass/fail result.

Usage:
    python 01_csv_schema_validator.py data.csv schema.json --report errors.csv

Schema file format (JSON):
{
  "columns": {
    "email": {"type": "email", "required": true},
    "age":   {"type": "int", "required": false, "min": 0, "max": 120},
    "name":  {"type": "string", "required": true, "pattern": "^[A-Za-z ]+$"},
    "signup_date": {"type": "date", "required": true, "date_format": "%Y-%m-%d"}
  }
}

Exits with status code 1 if any validation errors are found, 0 otherwise.
This makes it easy to use as a gate in a larger pipeline, e.g.:
    python 01_csv_schema_validator.py data.csv schema.json || exit 1
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def check_type(value, rules):
    """Return an error string if value fails its type check, else None."""
    col_type = rules.get("type", "string")

    if value == "" or value is None:
        return None  # emptiness is handled separately via 'required'

    if col_type == "int":
        try:
            ival = int(value)
        except ValueError:
            return f"expected int, got '{value}'"
        if "min" in rules and ival < rules["min"]:
            return f"{ival} is below minimum {rules['min']}"
        if "max" in rules and ival > rules["max"]:
            return f"{ival} is above maximum {rules['max']}"

    elif col_type == "float":
        try:
            fval = float(value)
        except ValueError:
            return f"expected float, got '{value}'"
        if "min" in rules and fval < rules["min"]:
            return f"{fval} is below minimum {rules['min']}"
        if "max" in rules and fval > rules["max"]:
            return f"{fval} is above maximum {rules['max']}"

    elif col_type == "date":
        fmt = rules.get("date_format", "%Y-%m-%d")
        try:
            datetime.strptime(value, fmt)
        except ValueError:
            return f"expected date matching '{fmt}', got '{value}'"

    elif col_type == "email":
        if not EMAIL_PATTERN.match(value):
            return f"'{value}' is not a valid email address"

    elif col_type == "string":
        pattern = rules.get("pattern")
        if pattern and not re.match(pattern, value):
            return f"'{value}' does not match pattern '{pattern}'"

    return None


def validate(csv_path, schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    column_rules = schema.get("columns", {})

    errors = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        missing_columns = [c for c in column_rules if c not in header]
        for col in missing_columns:
            errors.append({
                "row": 0, "column": col, "value": "",
                "error": "required column missing from file header"
            })

        for row_num, row in enumerate(reader, start=2):  # header is row 1
            for col, rules in column_rules.items():
                if col not in row:
                    continue
                value = (row.get(col) or "").strip()

                if rules.get("required") and value == "":
                    errors.append({
                        "row": row_num, "column": col, "value": value,
                        "error": "required value is missing"
                    })
                    continue

                err = check_type(value, rules)
                if err:
                    errors.append({
                        "row": row_num, "column": col, "value": value,
                        "error": err
                    })

    return errors


def write_report(errors, report_path):
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "column", "value", "error"])
        writer.writeheader()
        writer.writerows(errors)


def main():
    parser = argparse.ArgumentParser(description="Validate a CSV file against a JSON schema.")
    parser.add_argument("csv_file", help="Path to the CSV file to validate")
    parser.add_argument("schema_file", help="Path to the JSON schema file")
    parser.add_argument("--report", default=None, help="Optional path to write an error report CSV")
    args = parser.parse_args()

    errors = validate(args.csv_file, args.schema_file)

    if not errors:
        print(f"OK: {args.csv_file} passed validation against {args.schema_file}")
        sys.exit(0)

    print(f"FAILED: {len(errors)} validation error(s) found in {args.csv_file}")
    for e in errors[:20]:
        print(f"  row {e['row']}, column '{e['column']}': {e['error']}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

    if args.report:
        write_report(errors, args.report)
        print(f"Full error report written to {args.report}")

    sys.exit(1)


if __name__ == "__main__":
    main()
  
