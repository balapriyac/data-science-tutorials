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

## Quick Start

```bash
# 1. Validate a CSV against a schema
python scripts/01_csv_schema_validator.py data.csv schema.json --report errors.csv

# 2. Diff two CSV snapshots by an id column
python scripts/02_csv_diff_tool.py old.csv new.csv --key id --output diff_report.csv

# 3. Normalize a messy export to clean UTF-8 / comma-delimited CSV
python scripts/03_csv_normalizer.py messy_export.csv clean_output.csv

# 4. Reshape columns using a config file
python scripts/04_csv_column_transformer.py input.csv output.csv config.json

# 5. Sample 500 rows and mask sensitive columns
python scripts/05_csv_sampler_anonymizer.py input.csv output.csv \
    --sample-size 500 --mask email,phone,name --salt "some-secret"
```
## Config File Formats

### Schema (`schema.json`, used by script 1)

```json
{
  "columns": {
    "email": {"type": "email", "required": true},
    "age":   {"type": "int", "required": false, "min": 0, "max": 120},
    "name":  {"type": "string", "required": true, "pattern": "^[A-Za-z ]+$"},
    "signup_date": {"type": "date", "required": true, "date_format": "%Y-%m-%d"}
  }
}
```

Supported types: `int`, `float`, `date`, `string`, `email`.

### Column operations (`config.json`, used by script 4)

```json
[
  {"op": "rename", "mapping": {"fname": "first_name", "lname": "last_name"}},
  {"op": "derive", "new_column": "full_name", "template": "{first_name} {last_name}"},
  {"op": "derive", "new_column": "price", "template": "{price_str}", "convert": "strip_currency"},
  {"op": "drop", "columns": ["fname", "lname", "price_str"]},
  {"op": "reorder", "columns": ["full_name", "price", "email"]}
]
```

Operations run in order. Available `convert` functions: `to_int`, `to_float`, `strip_currency`, `upper`, `lower`, `title`.


Each script also supports `-h` / `--help` for the full list of options.

