"""
pdf_extractor.py
Extract text and tables from PDF files into structured output files.

Dependencies: pypdf, pdfplumber, pandas, openpyxl
Install:      pip install pypdf pdfplumber pandas openpyxl

Usage:
    python pdf_extractor.py --input report.pdf
    python pdf_extractor.py --input report.pdf --mode tables --output-dir ./extracted
    python pdf_extractor.py --input ./pdfs --mode both --format xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pdfplumber
from pypdf import PdfReader

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH   = "input.pdf"      # Single PDF or folder of PDFs
OUTPUT_DIR   = "./extracted"
MODE         = "both"           # text | tables | both
TEXT_FORMAT  = "txt"            # txt | md  (for text output)
TABLE_FORMAT = "xlsx"           # csv | xlsx (for table output)
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_pypdf(pdf_path: Path) -> str:
    """Fallback text extraction using pypdf."""
    reader = PdfReader(pdf_path)
    parts  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(parts)


def extract_text_pdfplumber(pdf_path: Path) -> tuple[str, int, int]:
    """Extract text with pdfplumber; returns (text, page_count, empty_page_count)."""
    parts       = []
    empty_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if text.strip():
                parts.append(f"--- Page {i + 1} ---\n{text.strip()}")
            else:
                empty_pages += 1

    return "\n\n".join(parts), page_count, empty_pages


def extract_tables(pdf_path: Path) -> list[dict]:
    """Extract all tables from a PDF. Returns list of {page, table_index, df}."""
    results = []

    table_settings = {
        "vertical_strategy":   "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance":      3,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Try structured line-based detection first
            tables = page.extract_tables(table_settings)
            if not tables:
                # Fall back to text-based detection
                tables = page.extract_tables()

            for t_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                # Use first row as header if it looks like one
                header = table[0]
                data   = table[1:]

                # Clean up: replace None with empty string
                header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(header)]
                rows   = [[str(c).strip() if c is not None else "" for c in row] for row in data]

                # Remove fully empty rows
                rows = [r for r in rows if any(r)]
                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=header)
                results.append({
                    "page":        page_num,
                    "table_index": t_idx + 1,
                    "df":          df,
                })

    return results


def write_text(text: str, out_path: Path, fmt: str) -> None:
    if fmt == "md":
        # Wrap page separators as markdown headers
        text = text.replace("--- Page ", "## Page ").replace(" ---", "")
        out_path = out_path.with_suffix(".md")
    out_path.write_text(text, encoding="utf-8")


def write_tables(tables: list[dict], out_path: Path, fmt: str) -> None:
    if not tables:
        return

    if fmt == "xlsx":
        with pd.ExcelWriter(out_path.with_suffix(".xlsx"), engine="openpyxl") as writer:
            for t in tables:
                sheet = f"P{t['page']}_T{t['table_index']}"[:31]
                t["df"].to_excel(writer, sheet_name=sheet, index=False)
    else:
        # One CSV per table
        for t in tables:
            csv_path = out_path.parent / f"{out_path.stem}_p{t['page']}_t{t['table_index']}.csv"
            t["df"].to_csv(csv_path, index=False, encoding="utf-8-sig")


def process_file(pdf_path: Path, out_dir: Path, mode: str,
                 text_fmt: str, table_fmt: str) -> dict:
    stem   = pdf_path.stem
    result = {
        "file":        pdf_path.name,
        "pages":       0,
        "empty_pages": 0,
        "tables":      0,
        "text_chars":  0,
        "error":       "",
    }

    try:
        if mode in ("text", "both"):
            text, page_count, empty_pages = extract_text_pdfplumber(pdf_path)
            if not text.strip():
                # Fall back to pypdf
                text        = extract_text_pypdf(pdf_path)
                empty_pages = 0

            result["pages"]       = page_count
            result["empty_pages"] = empty_pages
            result["text_chars"]  = len(text)

            if text.strip():
                write_text(text, out_dir / f"{stem}_text.txt", text_fmt)
            else:
                result["error"] += "No text extracted (possibly scanned). "

        if mode in ("tables", "both"):
            tables = extract_tables(pdf_path)
            result["tables"] = len(tables)
            if tables:
                write_tables(tables, out_dir / f"{stem}_tables", table_fmt)

    except Exception as e:
        result["error"] += str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract text and tables from PDF files.")
    parser.add_argument("--input",        default=INPUT_PATH,
                        help="PDF file or folder of PDFs")
    parser.add_argument("--output-dir",   default=OUTPUT_DIR)
    parser.add_argument("--mode",         default=MODE,
                        choices=["text", "tables", "both"])
    parser.add_argument("--text-format",  default=TEXT_FORMAT,  choices=["txt", "md"])
    parser.add_argument("--table-format", default=TABLE_FORMAT, choices=["csv", "xlsx"])
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] Not found: {src}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    if not pdfs:
        sys.exit(f"[ERROR] No PDF files found in: {src}")

    print(f"Processing {len(pdfs)} file(s) | Mode: {args.mode}\n")

    summary = []
    for pdf_path in pdfs:
        print(f"  {pdf_path.name}")
        result = process_file(pdf_path, out_dir, args.mode,
                              args.text_format, args.table_format)
        summary.append(result)

        if result["error"]:
            print(f"    [WARN] {result['error']}")
        else:
            parts = []
            if args.mode in ("text", "both"):
                parts.append(f"{result['pages']} pages, {result['text_chars']:,} chars")
            if args.mode in ("tables", "both"):
                parts.append(f"{result['tables']} table(s)")
            print(f"    {' | '.join(parts)}")

    summary_path = out_dir / "_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nOutput directory : {out_dir.resolve()}")
    print(f"Summary          : {summary_path.name}")


if __name__ == "__main__":
    main()
