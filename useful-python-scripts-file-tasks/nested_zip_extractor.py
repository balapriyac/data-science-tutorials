"""
Nested Zip Extractor
--------------------
Recursively extracts zip archives — no matter how deeply nested —
and places all extracted files into a single clean output directory.
Handles duplicate filenames, skips already-processed archives, and
generates a manifest of everything it extracted.

Usage:
    python nested_zip_extractor.py /path/to/input
    python nested_zip_extractor.py /path/to/input --output /path/to/output
    python nested_zip_extractor.py /path/to/input --dry-run
    python nested_zip_extractor.py /path/to/input --preserve-structure

Dependencies:
    None (standard library only)
"""

import os
import sys
import zipfile
import argparse
import shutil
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max nesting depth to prevent infinite loops from recursive/bomb zips
MAX_NESTING_DEPTH = 10

# Max total extracted size (in bytes) as a safety limit — 5 GB default
MAX_TOTAL_SIZE = 5 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def file_hash(path: str) -> str:
    """Return MD5 hash of a file (used to detect already-processed zips)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_name_conflict(target_path: Path) -> Path:
    """
    If target_path already exists, append a number to the stem
    until we find a unique name. E.g. report.txt -> report_1.txt -> report_2.txt
    """
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def estimate_zip_size(zip_path: str) -> int:
    """Sum of uncompressed sizes reported in the zip manifest."""
    total = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                total += info.file_size
    except (zipfile.BadZipFile, OSError):
        pass
    return total


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

class ExtractionLog:
    """Tracks everything that happens during extraction."""
    def __init__(self):
        self.entries = []        # (source_zip, extracted_file, status)
        self.zips_processed = set()
        self.errors = []
        self.total_extracted_size = 0

    def add(self, source: str, dest: str, status: str = "extracted"):
        self.entries.append({
            "source_zip": source,
            "extracted_to": dest,
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def add_error(self, source: str, error: str):
        self.errors.append({"source": source, "error": error})


def extract_zip(
    zip_path: str,
    output_dir: Path,
    log: ExtractionLog,
    depth: int = 0,
    preserve_structure: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Extract a single zip file. If extracted contents contain more zips,
    recursively extract those too.
    """
    zip_hash = file_hash(zip_path)

    # Skip if we've already processed this exact zip (by content hash)
    if zip_hash in log.zips_processed:
        return
    log.zips_processed.add(zip_hash)

    # Safety: max nesting depth
    if depth > MAX_NESTING_DEPTH:
        log.add_error(zip_path, f"Max nesting depth ({MAX_NESTING_DEPTH}) reached — skipped")
        return

    # Safety: check total extracted size won't exceed limit
    estimated = estimate_zip_size(zip_path)
    if log.total_extracted_size + estimated > MAX_TOTAL_SIZE:
        log.add_error(zip_path, f"Extraction would exceed size limit ({format_size(MAX_TOTAL_SIZE)}) — skipped")
        return

    # Validate the zip
    if not zipfile.is_zipfile(zip_path):
        log.add_error(zip_path, "Not a valid zip file")
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            nested_zips = []

            for member in zf.infolist():
                # Skip directories
                if member.is_dir():
                    continue

                member_name = member.filename
                file_size = member.file_size

                if preserve_structure:
                    # Keep the original directory structure under a folder named after the zip
                    zip_stem = Path(zip_path).stem
                    target = output_dir / zip_stem / member_name
                else:
                    # Flatten: just use the filename
                    target = output_dir / Path(member_name).name

                # Resolve any naming conflicts
                target = resolve_name_conflict(target)

                if dry_run:
                    log.add(zip_path, str(target), "would extract")
                    log.total_extracted_size += file_size
                    print(f"   [DRY RUN] {member_name} → {target}")
                else:
                    # Create parent directories if needed
                    target.parent.mkdir(parents=True, exist_ok=True)

                    # Extract to a temp file first, then move (safer)
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(zf.read(member.filename))
                        tmp_path = tmp.name

                    shutil.move(tmp_path, str(target))
                    log.add(zip_path, str(target), "extracted")
                    log.total_extracted_size += file_size
                    print(f"   ✓ {member_name} → {target.name}")

                # Track nested zips for recursive extraction
                if member_name.lower().endswith(".zip"):
                    nested_zips.append(str(target))

            # Recursively extract any nested zips we just pulled out
            for nested in nested_zips:
                if os.path.exists(nested):
                    print(f"\n   📦 Found nested archive: {Path(nested).name} (depth {depth + 1})")
                    extract_zip(nested, output_dir, log, depth + 1, preserve_structure, dry_run)

    except zipfile.BadZipFile:
        log.add_error(zip_path, "Corrupted or invalid zip file")
    except Exception as e:
        log.add_error(zip_path, str(e))


def write_manifest(log: ExtractionLog, output_dir: Path) -> str:
    """Write a manifest file listing everything that was extracted."""
    manifest_path = output_dir / "_extraction_manifest.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("  EXTRACTION MANIFEST")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total files extracted: {len(log.entries)}")
    lines.append(f"  Total size: {format_size(log.total_extracted_size)}")
    lines.append(f"  Zips processed: {len(log.zips_processed)}")
    lines.append("=" * 70)

    if log.entries:
        lines.append("\n📄 EXTRACTED FILES\n")
        for entry in log.entries:
            lines.append(f"  [{entry['status'].upper()}]")
            lines.append(f"    From:  {entry['source_zip']}")
            lines.append(f"    To:    {entry['extracted_to']}")
            lines.append(f"    Time:  {entry['timestamp']}")
            lines.append("")

    if log.errors:
        lines.append("\n⚠️  ERRORS\n")
        for err in log.errors:
            lines.append(f"  Source: {err['source']}")
            lines.append(f"  Error:  {err['error']}")
            lines.append("")

    manifest_text = "\n".join(lines)

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_text)

    return str(manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recursively extract nested zip archives into a clean output folder."
    )
    parser.add_argument(
        "input", type=str,
        help="Input directory to scan for zip files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for extracted files (default: 'extracted' folder next to input)"
    )
    parser.add_argument(
        "--preserve-structure", action="store_true",
        help="Keep original directory structures inside each zip (default: flatten)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be extracted without extracting anything"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Set up output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_extracted"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    log = ExtractionLog()

    # Find all top-level zip files in the input directory
    zip_files = list(input_dir.rglob("*.zip"))

    if not zip_files:
        print(f"\n📁 No zip files found in {input_dir}")
        return

    print(f"\n📦 Found {len(zip_files)} zip file(s) in {input_dir}")
    print(f"📁 Output: {output_dir}")
    if args.dry_run:
        print("🛑 DRY RUN — nothing will be extracted\n")
    print()

    # Process each zip
    for zip_path in sorted(zip_files):
        print(f"\n📦 Processing: {zip_path.name}")
        extract_zip(
            str(zip_path),
            output_dir,
            log,
            depth=0,
            preserve_structure=args.preserve_structure,
            dry_run=args.dry_run,
        )

    # Write manifest
    if not args.dry_run:
        manifest = write_manifest(log, output_dir)
        print(f"\n📋 Manifest written: {manifest}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  ✅ Extraction complete")
    print(f"  Files extracted:  {len(log.entries)}")
    print(f"  Total size:       {format_size(log.total_extracted_size)}")
    print(f"  Zips processed:   {len(log.zips_processed)}")
    if log.errors:
        print(f"  Errors:           {len(log.errors)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()


