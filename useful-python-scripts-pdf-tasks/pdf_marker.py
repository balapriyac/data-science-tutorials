"""
pdf_stamper.py
Add text watermarks, stamps, or page numbers to PDF files in batch.

Dependencies: pypdf, reportlab
Install:      pip install pypdf reportlab

Usage:
    python pdf_stamper.py --input report.pdf --text "CONFIDENTIAL"
    python pdf_stamper.py --input ./pdfs --text "DRAFT" --angle 45 --opacity 0.15
    python pdf_stamper.py --input report.pdf --mode page-numbers --position bottom-center
    python pdf_stamper.py --input report.pdf --text "INTERNAL USE ONLY" --position top-center --angle 0
"""

import argparse
import io
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH   = "input.pdf"      # Single PDF or folder of PDFs
OUTPUT_DIR   = "./stamped"
MODE         = "watermark"      # watermark | stamp | page-numbers
STAMP_TEXT   = "CONFIDENTIAL"
POSITION     = "center"         # center | top-left | top-center | top-right |
                                # bottom-left | bottom-center | bottom-right
ANGLE        = 45               # Rotation angle in degrees
OPACITY      = 0.12             # 0.0 (invisible) to 1.0 (fully opaque)
FONT_NAME    = "Helvetica-Bold"
FONT_SIZE    = 48               # For watermark/stamp
PAGE_NUM_SIZE = 10              # Font size for page numbers
COLOR        = "#CCCCCC"        # Hex color for watermark text
STAMP_COLOR  = "#CC0000"        # Hex color for stamp text
PAGE_NUM_FMT = "Page {n} of {total}"  # Page number format string
# ─────────────────────────────────────────────────────────────────────────────


def hex_to_color(hex_str: str, opacity: float) -> Color:
    base = HexColor(hex_str)
    return Color(base.red, base.green, base.blue, alpha=opacity)


def get_position_coords(page_width: float, page_height: float,
                        position: str, font_size: int) -> tuple[float, float]:
    margin = 20
    cx, cy = page_width / 2, page_height / 2

    positions = {
        "center":        (cx, cy),
        "top-left":      (margin, page_height - margin - font_size),
        "top-center":    (cx, page_height - margin - font_size),
        "top-right":     (page_width - margin, page_height - margin - font_size),
        "bottom-left":   (margin, margin),
        "bottom-center": (cx, margin),
        "bottom-right":  (page_width - margin, margin),
    }
    return positions.get(position, positions["center"])


def make_text_stamp(text: str, page_width: float, page_height: float,
                    position: str, angle: float, opacity: float,
                    font: str, font_size: int, color_hex: str) -> bytes:
    """Generate a single-page PDF stamp in memory."""
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFillColor(hex_to_color(color_hex, opacity))
    c.setFont(font, font_size)

    x, y = get_position_coords(page_width, page_height, position, font_size)

    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    c.drawCentredString(0, 0, text)
    c.restoreState()
    c.save()

    buf.seek(0)
    return buf.read()


def make_page_number_stamp(n: int, total: int, page_width: float,
                           page_height: float, position: str,
                           fmt: str, font_size: int) -> bytes:
    text = fmt.format(n=n, total=total)
    buf  = io.BytesIO()
    c    = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFillColor(Color(0, 0, 0, alpha=0.7))
    c.setFont("Helvetica", font_size)
    x, y = get_position_coords(page_width, page_height, position, font_size)
    c.drawCentredString(x, y, text)
    c.save()
    buf.seek(0)
    return buf.read()


def stamp_pdf(pdf_path: Path, out_path: Path, mode: str, text: str,
              position: str, angle: float, opacity: float,
              font: str, font_size: int, color_hex: str,
              page_num_fmt: str, page_num_size: int) -> dict:
    result = {"file": pdf_path.name, "pages": 0, "error": ""}
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        total  = len(reader.pages)
        result["pages"] = total

        for i, page in enumerate(reader.pages):
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)

            if mode == "page-numbers":
                stamp_bytes = make_page_number_stamp(
                    i + 1, total, w, h, position, page_num_fmt, page_num_size
                )
                stamp_color = "#000000"
            else:
                stamp_color = STAMP_COLOR if mode == "stamp" else color_hex
                stamp_angle = 0 if mode == "stamp" else angle
                stamp_bytes = make_text_stamp(
                    text, w, h, position, stamp_angle, opacity,
                    font, font_size, stamp_color
                )

            stamp_reader = PdfReader(io.BytesIO(stamp_bytes))
            stamp_page   = stamp_reader.pages[0]

            page.merge_page(stamp_page)
            writer.add_page(page)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            writer.write(fh)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Add watermarks, stamps, or page numbers to PDFs.")
    parser.add_argument("--input",         default=INPUT_PATH,
                        help="PDF file or folder of PDFs")
    parser.add_argument("--output-dir",    default=OUTPUT_DIR)
    parser.add_argument("--mode",          default=MODE,
                        choices=["watermark", "stamp", "page-numbers"])
    parser.add_argument("--text",          default=STAMP_TEXT,
                        help="Text to stamp or watermark")
    parser.add_argument("--position",      default=POSITION,
                        choices=["center", "top-left", "top-center", "top-right",
                                 "bottom-left", "bottom-center", "bottom-right"])
    parser.add_argument("--angle",         type=float, default=ANGLE)
    parser.add_argument("--opacity",       type=float, default=OPACITY,
                        help="Text opacity 0.0–1.0")
    parser.add_argument("--font-size",     type=int,   default=FONT_SIZE)
    parser.add_argument("--color",         default=COLOR,
                        help="Hex color for watermark text")
    parser.add_argument("--page-num-fmt",  default=PAGE_NUM_FMT,
                        help="Page number format. Use {n} and {total}")
    parser.add_argument("--page-num-size", type=int, default=PAGE_NUM_SIZE)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] Not found: {src}")

    pdfs    = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode     : {args.mode}")
    if args.mode != "page-numbers":
        print(f"Text     : {args.text}")
    print(f"Position : {args.position}")
    print(f"Files    : {len(pdfs)}\n")

    for pdf_path in pdfs:
        out_path = out_dir / f"{pdf_path.stem}_{args.mode}.pdf"
        result   = stamp_pdf(
            pdf_path, out_path, args.mode, args.text,
            args.position, args.angle, args.opacity,
            FONT_NAME, args.font_size, args.color,
            args.page_num_fmt, args.page_num_size,
        )
        if result["error"]:
            print(f"  ✗ {pdf_path.name:50s} ERROR — {result['error']}")
        else:
            print(f"  ✓ {pdf_path.name:50s} {result['pages']} pages  →  {out_path.name}")

    print(f"\nOutput directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
  
