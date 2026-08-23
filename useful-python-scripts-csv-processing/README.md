# CSV Automation Scripts

Five self-contained Python scripts for the CSV chores that come up constantly in data work: validating schema, diffing two versions of a file, normalizing encoding/delimiter mess, reshaping columns, and sampling/anonymizing data for safe sharing.

## Scripts

| # | Script | What it does |
|---|--------|---------------|
| 1 | `csv_schema_validator.py` | Validates a CSV against a JSON schema (types, required fields, patterns) and outputs a row-level error report. |
| 2 | `csv_diff_tool.py` | Compares two CSV files by a key column and reports added, removed, and field-level changed rows. |
| 3 | `csv_normalizer.py` | Detects encoding and delimiter, rewrites the file as clean UTF-8 comma-separated CSV. |
| 4 | `csv_column_transformer.py` | Renames, drops, reorders, and derives columns based on a JSON config. |
| 5 | `csv_sampler_anonymizer.py` | Randomly samples rows (reservoir sampling) and masks sensitive columns with a consistent keyed hash. |

