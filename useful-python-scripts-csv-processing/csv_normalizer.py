#!/usr/bin/env python3
"""
csv_normalizer.py

Detect the character encoding and delimiter of a messy CSV file, then
rewrite it as clean, UTF-8, comma-delimited CSV with normalized line
endings and no byte-order mark.

Usage:
    python 03_csv_normalizer.py messy_export.csv clean_output.csv
"""

import argparse
import csv
import io

CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]


def detect_encoding(raw_bytes):
    """Try a shortlist of common encodings and return the first that decodes cleanly."""
    for encoding in CANDIDATE_ENCODINGS:
        try:
            raw_bytes.decode(encoding)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: decode with replacement characters rather than failing outright
    return "latin-1"


def detect_delimiter(sample_text):
    """Use csv.Sniffer to guess the delimiter, falling back to a manual heuristic."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(CANDIDATE_DELIMITERS))
        return dialect.delimiter
    except csv.Error:
        # Fallback: count occurrences of each candidate in the first line
        first_line = sample_text.splitlines()[0] if sample_text else ""
        counts = {d: first_line.count(d) for d in CANDIDATE_DELIMITERS}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def normalize(input_path, output_path, sample_bytes=65536):
    with open(input_path, "rb") as f:
        raw = f.read()

    encoding = detect_encoding(raw[:sample_bytes])
    text = raw.decode(encoding, errors="replace")

    # Strip a leading BOM character if decoding left one in place
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    delimiter = detect_delimiter(text[:sample_bytes])

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)  # default dialect: comma-delimited, \r\n handled by newline=""
        writer.writerows(rows)

    return encoding, delimiter, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Detect encoding/delimiter and rewrite a CSV as clean UTF-8 comma-separated."
    )
    parser.add_argument("input_file", help="Path to the messy input CSV")
    parser.add_argument("output_file", help="Path to write the normalized CSV")
    args = parser.parse_args()

    encoding, delimiter, row_count = normalize(args.input_file, args.output_file)

    delimiter_name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(
        delimiter, repr(delimiter)
    )

    print(f"Detected encoding:  {encoding}")
    print(f"Detected delimiter: {delimiter_name}")
    print(f"Rows written:       {row_count}")
    print(f"Normalized file saved to {args.output_file}")


if __name__ == "__main__":
    main()
  
