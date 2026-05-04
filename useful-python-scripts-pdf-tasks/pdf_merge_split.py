"""
pdf_merge_split.py
Merge multiple PDF files into one, or split a PDF by page ranges or chunk size.

Dependencies: pypdf
Install:      pip install pypdf

Usage — merge:
    python pdf_merge_split.py merge --input ./pdfs --output merged.pdf
    python pdf_merge_split.py merge --input ./pdfs --output merged.pdf --order order.txt

Usage — split:
    python pdf_merge_split.py split --input report.pdf --output-dir ./splits --every 10
    python pdf_merge_split.py split --input report.pdf --output-dir ./splits --ranges "1-5,6-12,13-"
    python pdf_merge_split.py split --input report.pdf --output-dir ./splits --on-pages 10 20 35

order.txt format (merge mode):
    One filename per line, in the order they should be merged:
        chapter1.pdf
        chapter2.pdf
        appendix.pdf
"""

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FOLDER = "./pdfs"
OUTPUT_FILE  = "merged.pdf"
INPUT_FILE   = "input.pdf"
OUTPUT_DIR   = "./splits"
CHUNK_SIZE   = 10           # Pages per split file (--every mode)
ORDER_FILE   = None         # Text file with filenames in merge order
# ─────────────────────────────────────────────────────────────────────────────


def merge(input_folder: str, output_file: str, order_file: str | None) -> None:
    folder = Path(input_folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Input folder not found: {folder}")

    all_pdfs = sorted(folder.glob("*.pdf"))
    if not all_pdfs:
        sys.exit(f"[ERROR] No PDF files found in: {folder}")

    # Apply custom order if provided
    if order_file:
        opath = Path(order_file)
        if not opath.exists():
            sys.exit(f"[ERROR] Order file not found: {opath}")
        names = [line.strip() for line in opath.read_text().splitlines() if line.strip()]
        ordered = []
        for name in names:
            match = folder / name
            if match.exists():
                ordered.append(match)
            else:
                print(f"  [WARN] File in order list not found: {name}")
        # Append any files not in order list at the end
        listed = set(ordered)
        for p in all_pdfs:
            if p not in listed:
                ordered.append(p)
                print(f"  [WARN] '{p.name}' not in order file — appending at end")
        all_pdfs = ordered

    print(f"Merging {len(all_pdfs)} file(s) into: {output_file}\n")

    writer = PdfWriter()
    total_pages = 0

    for pdf_path in all_pdfs:
        try:
            reader = PdfReader(pdf_path)
            pages  = len(reader.pages)
            for page in reader.pages:
                writer.add_page(page)
            total_pages += pages
            print(f"  ✓ {pdf_path.name:50s} {pages:>4} pages")
        except Exception as e:
            print(f"  ✗ {pdf_path.name:50s} FAILED — {e}")

    # Copy metadata from first readable file
    try:
        first_reader = PdfReader(all_pdfs[0])
        if first_reader.metadata:
            writer.add_metadata(dict(first_reader.metadata))
    except Exception:
        pass

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        writer.write(fh)

    print(f"\nTotal pages : {total_pages:,}")
    print(f"Output      : {out.resolve()}")


def parse_ranges(ranges_str: str, total_pages: int) -> list[tuple[int, int]]:
    """Parse '1-5,6-12,13-' into list of (start, end) tuples (0-based, exclusive end)."""
    segments = []
    for part in ranges_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip()) - 1 if left.strip() else 0
            end   = int(right.strip()) if right.strip() else total_pages
        else:
            start = int(part) - 1
            end   = int(part)
        segments.append((max(0, start), min(total_pages, end)))
    return segments


def split(input_file: str, output_dir: str, every: int | None,
          ranges: str | None, on_pages: list[int] | None) -> None:
    src = Path(input_file)
    if not src.exists():
        sys.exit(f"[ERROR] File not found: {src}")

    reader     = PdfReader(src)
    total      = len(reader.pages)
    stem       = src.stem
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Splitting: {src} ({total} pages)\n")

    # Build list of (start, end) segments — 0-based, exclusive end
    if ranges:
        segments = parse_ranges(ranges, total)
    elif on_pages:
        # Split ON these page numbers (1-based): break before each listed page
        breaks = sorted(set([0] + [p - 1 for p in on_pages] + [total]))
        segments = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
    elif every:
        segments = [(i, min(i + every, total)) for i in range(0, total, every)]
    else:
        sys.exit("[ERROR] Specify --every, --ranges, or --on-pages.")

    print(f"Segments: {len(segments)}\n")

    for idx, (start, end) in enumerate(segments, 1):
        if start >= end:
            continue
        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        out_name = f"{stem}_part{idx:03d}_pages{start+1}-{end}.pdf"
        out_path = out_dir / out_name
        with open(out_path, "wb") as fh:
            writer.write(fh)
        print(f"  Part {idx:03d}: pages {start+1}–{end}  →  {out_name}")

    print(f"\nOutput directory: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Merge or split PDF files.")
    sub    = parser.add_subparsers(dest="command", required=True)

    # Merge subcommand
    mp = sub.add_parser("merge", help="Merge multiple PDFs into one")
    mp.add_argument("--input",   default=INPUT_FOLDER, help="Folder containing PDFs")
    mp.add_argument("--output",  default=OUTPUT_FILE,  help="Output PDF path")
    mp.add_argument("--order",   default=ORDER_FILE,   help="Text file with filenames in order")

    # Split subcommand
    sp = sub.add_parser("split", help="Split a PDF into parts")
    sp.add_argument("--input",      default=INPUT_FILE, help="PDF file to split")
    sp.add_argument("--output-dir", default=OUTPUT_DIR)
    sp.add_argument("--every",      type=int, help="Split every N pages")
    sp.add_argument("--ranges",     help="Page ranges, e.g. '1-5,6-12,13-'")
    sp.add_argument("--on-pages",   type=int, nargs="+",
                    help="Split before these page numbers (1-based)")

    args = parser.parse_args()

    if args.command == "merge":
        merge(args.input, args.output, args.order)
    else:
        split(args.input, args.output_dir, args.every, args.ranges, args.on_pages)


if __name__ == "__main__":
    main()
