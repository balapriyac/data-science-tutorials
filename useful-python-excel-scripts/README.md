# 5 Python Scripts to Automate Boring Excel Tasks

A collection of standalone Python scripts that handle the Excel tasks you keep putting off.

---

## Scripts

| Script | What it does |
|---|---|
| `excel_merger.py` | Merge multiple `.xlsx` / `.csv` files into one |
| `duplicate_finder.py` | Find exact and near-duplicate rows with color highlighting |
| `data_cleaner.py` | Standardize dates, casing, phones, whitespace — with a change log |
| `sheet_splitter.py` | Split one sheet into separate files by column value |
| `pivot_report.py` | Generate a formatted multi-tab summary report with charts |

---

## Dependencies

| Script | Packages |
|---|---|
| `excel_merger.py` | `pandas`, `openpyxl` |
| `duplicate_finder.py` | `pandas`, `openpyxl`, `rapidfuzz` |
| `data_cleaner.py` | `pandas`, `openpyxl` |
| `sheet_splitter.py` | `pandas`, `openpyxl` |
| `pivot_report.py` | `pandas`, `openpyxl`, `matplotlib` |

### Install all at once

```bash
pip install pandas openpyxl rapidfuzz matplotlib
```

Requires **Python 3.10+** (uses `str | None` union syntax).

---


## Quick Start

### 1. excel_merger.py

Merge all `.xlsx` and `.csv` files in a folder into one output file.

```bash
# Basic merge
python excel_merger.py --input ./reports --output merged.xlsx

# Add a column showing which file each row came from
python excel_merger.py --input ./reports --output merged.xlsx --source-column "Source File"

# Read from a specific sheet name in each file
python excel_merger.py --input ./reports --output merged.xlsx --sheet "Data"

# Skip the summary tab
python excel_merger.py --input ./reports --output merged.xlsx --no-summary
```

**Output:** `merged.xlsx` with a `Merged Data` tab and a `Summary` tab (row counts per file).

---

### 2. duplicate_finder.py

Flag exact and fuzzy/near-duplicate rows based on key columns you define.

```bash
# Check for duplicates on Email column
python duplicate_finder.py --input customers.xlsx --key-cols "Email"

# Check multiple columns together as a composite key
python duplicate_finder.py --input customers.xlsx --key-cols "First Name" "Last Name" "Phone"

# Adjust fuzzy sensitivity (default 85; lower = more matches)
python duplicate_finder.py --input customers.xlsx --key-cols "Name" --fuzzy-threshold 75

# Custom output path
python duplicate_finder.py --input data.xlsx --key-cols "Email" --output flagged_output.xlsx
```

**Output:** Annotated Excel file with `_dup_group`, `_dup_type`, `_dup_score` columns.
- Orange rows = exact duplicates
- Yellow rows = near-duplicates (fuzzy match)

---

### 3. data_cleaner.py

Apply cleaning rules to a messy export. Outputs a cleaned file and a full change log.

```bash
# Run with default rules (strip whitespace, remove blank rows)
python data_cleaner.py --input export.xlsx

# Use a JSON config file to define column-specific rules
python data_cleaner.py --input export.xlsx --output cleaned.xlsx --config rules.json

# Read from a CSV
python data_cleaner.py --input export.csv --output cleaned.xlsx
```

**Example `rules.json`:**
```json
{
    "strip_whitespace":  ["Name", "Email", "Address"],
    "title_case":        ["Name", "City"],
    "lower_case":        ["Email"],
    "upper_case":        ["Country Code"],
    "date_format":       {"columns": ["Date", "DOB"], "output_format": "%Y-%m-%d"},
    "phone_normalize":   ["Phone", "Mobile"],
    "remove_blank_rows": true,
    "remove_duplicates": ["Email"]
}
```

**Output:** `Cleaned Data` tab + `Change Log` tab showing every cell that was modified.

---

### 4. sheet_splitter.py

Split a master sheet into one file per unique value in a column.

```bash
# Split by Region column
python sheet_splitter.py --input master.xlsx --split-col "Region"

# Custom output folder and filename template
python sheet_splitter.py --input sales.xlsx --split-col "Department" \
    --output-dir ./monthly_splits \
    --name-template "Sales_{value}_{date}.xlsx"

# Send each file as an email attachment (requires SMTP config in script)
python sheet_splitter.py --input data.xlsx --split-col "Region" \
    --email-map emails.csv \
    --smtp-user you@company.com --smtp-pass yourpassword
```

**Filename template placeholders:** `{value}`, `{date}`, `{datetime}`

**`emails.csv` format:**
```
Value,Email
North,north@company.com
South,south@company.com
```

**Output:** One `.xlsx` per unique column value + `_manifest.csv` listing all output files.

---

### 5. pivot_report.py

Generate a formatted summary report from raw transaction or data rows.

```bash
# Basic usage — group by Category, sum Amount, parse Date
python pivot_report.py --input transactions.xlsx \
    --date-col "Date" --value-col "Amount" --group-cols "Category"

# Multiple grouping columns
python pivot_report.py --input sales.xlsx \
    --date-col "Sale Date" --value-col "Revenue" --group-cols "Product" "Region"

# Show top 20 instead of default top 10
python pivot_report.py --input data.xlsx \
    --date-col "Date" --value-col "Amount" --group-cols "Category" --top-n 20
```

**Output tabs:**
- **Overview** — key stats (total, count, average, date range)
- **By Group** — total/count/avg/min/max per group + % of total + bar chart
- **Monthly Trend** — month-by-month totals + MoM % change + line chart
- **Top N** — top N groups by total value
- **Year-over-Year** — yearly comparison matrix

---
