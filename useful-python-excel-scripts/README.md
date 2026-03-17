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
