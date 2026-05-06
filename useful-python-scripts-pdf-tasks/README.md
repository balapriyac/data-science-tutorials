# 5 Python Scripts to Automate PDF Tasks

A collection of standalone Python scripts for common PDF automation tasks.

---

## Scripts

| Script | What it does |
|---|---|
| `pdf_merge_split.py` | Merge multiple PDFs into one, or split a PDF by page range or chunk size |
| `pdf_extractor.py` | Extract text and tables from PDFs into text files and Excel/CSV |
| `pdf_stamper.py` | Add watermarks, stamps, or page numbers to PDFs in batch |
| `pdf_redactor.py` | Permanently redact text matching patterns from PDF files |
| `pdf_inventory.py` | Scan a folder of PDFs and export a metadata inventory |

---

## Dependencies

| Script | Packages |
|---|---|
| `pdf_merge_split.py` | `pypdf` |
| `pdf_extractor.py` | `pypdf`, `pdfplumber`, `pandas`, `openpyxl` |
| `pdf_stamper.py` | `pypdf`, `reportlab` |
| `pdf_redactor.py` | `pymupdf` |
| `pdf_inventory.py` | `pypdf`, `pdfplumber`, `pandas`, `openpyxl` |

### Install all at once

```bash
pip install pypdf pdfplumber pymupdf reportlab pandas openpyxl
```

---
## Quick Start

### 1. pdf_merge_split.py

Merge a folder of PDFs into one file, or split a PDF into parts.

```bash
# Merge all PDFs in a folder (sorted alphabetically)
python pdf_merge_split.py merge --input ./reports --output merged.pdf

# Merge in a specific order defined in a text file
python pdf_merge_split.py merge --input ./chapters --output book.pdf --order order.txt

# Split every 10 pages
python pdf_merge_split.py split --input large_report.pdf --output-dir ./splits --every 10

# Split by specific page ranges
python pdf_merge_split.py split --input report.pdf --output-dir ./splits --ranges "1-5,6-20,21-"

# Split before specific page numbers
python pdf_merge_split.py split --input report.pdf --output-dir ./splits --on-pages 10 25 40
```

**`order.txt` format (merge mode):** One filename per line in the desired order.

**`--ranges` format:** Comma-separated ranges. Use `-` as the end of a range to mean "to the last page" (e.g. `21-`).

**Output:** Single merged PDF, or numbered split files in the output directory.

---

### 2. pdf_extractor.py

Extract text and tables from PDFs into usable output files.

```bash
# Extract both text and tables from a single file
python pdf_extractor.py --input report.pdf

# Extract tables only, write to CSV
python pdf_extractor.py --input report.pdf --mode tables --table-format csv

# Process an entire folder, output as markdown and Excel
python pdf_extractor.py --input ./pdfs --mode both --text-format md --table-format xlsx

# Extract text only
python pdf_extractor.py --input statement.pdf --mode text --output-dir ./text_output
```

**`--mode` options:** `text` | `tables` | `both`

**Output:**
- Text → `{filename}_text.txt` or `.md` per file
- Tables → `{filename}_tables.xlsx` (one sheet per table) or individual CSVs
- `_summary.csv` listing pages, tables found, and any extraction warnings per file

**Note:** Scanned PDFs (image-only) will not yield extractable text. The summary flags these.

---

### 3. pdf_stamper.py

Add a watermark, stamp, or page numbers to one or more PDFs.

```bash
# Diagonal watermark (default)
python pdf_stamper.py --input report.pdf --text "CONFIDENTIAL"

# Full folder, custom opacity and angle
python pdf_stamper.py --input ./pdfs --text "DRAFT" --angle 45 --opacity 0.12

# Horizontal stamp at the top
python pdf_stamper.py --input contract.pdf --mode stamp \
    --text "APPROVED" --position top-center --angle 0

# Add page numbers at the bottom
python pdf_stamper.py --input report.pdf --mode page-numbers \
    --position bottom-center --page-num-fmt "Page {n} of {total}"
```

**`--mode` options:** `watermark` | `stamp` | `page-numbers`

**`--position` options:** `center`, `top-left`, `top-center`, `top-right`, `bottom-left`, `bottom-center`, `bottom-right`

**Output:** One output PDF per input file, named `{original}_watermark.pdf` / `_stamp.pdf` / `_page-numbers.pdf`. Originals are never modified.

---

### 4. pdf_redactor.py

Permanently remove sensitive text from PDFs using pattern matching.

```bash
# Redact specific strings
python pdf_redactor.py --input document.pdf --patterns "John Smith" "Project Alpha"

# Redact using regex patterns
python pdf_redactor.py --input document.pdf --patterns "ACC-\d{6}" "REF:\s?\d+"

# Redact using built-in categories
python pdf_redactor.py --input document.pdf --categories email phone

# Combine patterns and categories
python pdf_redactor.py --input ./pdfs --patterns "CONFIDENTIAL-\d+" --categories email ssn

# Match whole words only
python pdf_redactor.py --input document.pdf --patterns "Smith" --whole-word
```

**Built-in `--categories`:** `email`, `phone`, `ssn`, `credit`, `postcode`, `date`

**Output:** `{filename}_redacted.pdf` per input file. A `_redaction_log.csv` lists every redaction made (page, pattern, matched text — recorded before redaction).

⚠️ **Always verify the output before distributing.** Test on a copy before processing originals.

---

### 5. pdf_inventory.py

Generate a metadata inventory of a folder of PDF files.

```bash
# Basic inventory
python pdf_inventory.py --input ./documents

# Custom output and deeper text sampling
python pdf_inventory.py --input ./documents --output my_inventory.xlsx --sample-pages 5

# Include subdirectories
python pdf_inventory.py --input ./archive --recursive
```

**Output columns:** `filename`, `path`, `size_kb`, `pages`, `encrypted`, `scanned`, `title`, `author`, `creator`, `producer`, `created`, `modified`, `pdf_version`, `error`

**Row colors in Excel:**
- Yellow = scanned/image PDF (no extractable text)
- Red = encrypted file that could not be opened
- Blue alternating = standard files



