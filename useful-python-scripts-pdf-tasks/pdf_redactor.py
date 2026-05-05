"""
pdf_redactor.py
Permanently redact text matching patterns from PDF files.

Dependencies: pymupdf
Install:      pip install pymupdf

Usage:
    python pdf_redactor.py --input document.pdf --patterns "John Smith" "ACC-\d+"
    python pdf_redactor.py --input document.pdf --categories email phone
    python pdf_redactor.py --input ./pdfs --patterns "CONFIDENTIAL-\d+" --categories email

Built-in categories (--categories):
    email     Email addresses
    phone     Phone numbers (various formats)
    ssn       US Social Security numbers
    credit    Credit card numbers
    postcode  UK postcodes
    date      Common date formats

NOTE: Always verify redaction output before distributing.
      Test on a copy before processing originals.
"""

import argparse
import re
import sys
from pathlib import Path

import fitz  # pymupdf

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "input.pdf"
OUTPUT_DIR  = "./redacted"
PATTERNS    = []           # List of regex patterns or exact strings
CATEGORIES  = []           # Built-in pattern categories
REDACT_COLOR = (0, 0, 0)  # RGB fill color (black)
WHOLE_WORD  = False        # Match whole words only for exact string patterns
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_PATTERNS = {
    "email":    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone":    r"(\+?\d[\d\s\-().]{7,}\d)",
    "ssn":      r"\b\d{3}-\d{2}-\d{4}\b",
    "credit":   r"\b(?:\d[ -]?){13,16}\b",
    "postcode": r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
    "date":     r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
}


def build_patterns(raw_patterns: list[str], categories: list[str],
                   whole_word: bool) -> list[tuple[str, re.Pattern]]:
    compiled = []

    for cat in categories:
        if cat not in BUILTIN_PATTERNS:
            print(f"  [WARN] Unknown category: '{cat}'. Available: {list(BUILTIN_PATTERNS)}")
            continue
        compiled.append((f"[{cat}]", re.compile(BUILTIN_PATTERNS[cat], re.IGNORECASE)))

    for pat in raw_patterns:
        try:
            # Check if it's a valid regex; if not, escape it as a literal
            re.compile(pat)
            if whole_word:
                pat = rf"\b{re.escape(pat)}\b"
            compiled.append((pat, re.compile(pat, re.IGNORECASE)))
        except re.error:
            escaped = re.escape(pat)
            compiled.append((pat, re.compile(escaped, re.IGNORECASE)))

    return compiled


def redact_pdf(pdf_path: Path, out_path: Path,
               patterns: list[tuple[str, re.Pattern]]) -> dict:
    result = {
        "file":       pdf_path.name,
        "pages":      0,
        "redactions": 0,
        "log":        [],
        "error":      "",
    }

    try:
        doc = fitz.open(pdf_path)
        result["pages"] = len(doc)

        for page_num, page in enumerate(doc, 1):
            page_text   = page.get_text()
            page_redact = 0

            for label, pattern in patterns:
                for match in pattern.finditer(page_text):
                    matched_text = match.group()
                    # Search for all instances of the matched text on the page
                    areas = page.search_for(matched_text, quads=False)
                    for rect in areas:
                        page.add_redact_annot(rect, fill=REDACT_COLOR)
                        page_redact += 1
                        result["log"].append({
                            "page":    page_num,
                            "pattern": label,
                            "matched": matched_text,
                        })

            if page_redact > 0:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                result["redactions"] += page_redact

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_path, garbage=4, deflate=True, clean=True)
        doc.close()

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Permanently redact text from PDF files.")
    parser.add_argument("--input",       default=INPUT_PATH,
                        help="PDF file or folder of PDFs")
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--patterns",    nargs="*", default=PATTERNS,
                        help="Regex patterns or exact strings to redact")
    parser.add_argument("--categories",  nargs="*", default=CATEGORIES,
                        choices=list(BUILTIN_PATTERNS.keys()),
                        help="Built-in pattern categories")
    parser.add_argument("--whole-word",  action="store_true", default=WHOLE_WORD,
                        help="Match whole words only for string patterns")
    args = parser.parse_args()

    if not args.patterns and not args.categories:
        sys.exit("[ERROR] Specify at least one --patterns value or --categories option.")

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] Not found: {src}")

    pdfs    = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = build_patterns(args.patterns or [], args.categories or [], args.whole_word)
    print(f"Patterns  : {[label for label, _ in patterns]}")
    print(f"Files     : {len(pdfs)}\n")
    print("NOTE: Verify all output files before distributing.\n")

    all_logs = []

    for pdf_path in pdfs:
        out_path = out_dir / f"{pdf_path.stem}_redacted.pdf"
        result   = redact_pdf(pdf_path, out_path, patterns)

        if result["error"]:
            print(f"  ✗ {pdf_path.name:50s} ERROR — {result['error']}")
        else:
            print(f"  ✓ {pdf_path.name:50s} {result['redactions']:>4} redaction(s)  →  {out_path.name}")

        for entry in result["log"]:
            entry["file"] = pdf_path.name
            all_logs.append(entry)

    # Write redaction log
    if all_logs:
        import pandas as pd
        log_path = out_dir / "_redaction_log.csv"
        pd.DataFrame(all_logs).to_csv(log_path, index=False)
        print(f"\nRedaction log : {log_path.resolve()}")

    print(f"Output dir    : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
  
