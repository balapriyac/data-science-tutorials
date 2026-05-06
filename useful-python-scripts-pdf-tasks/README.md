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

