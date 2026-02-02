"""
Empty & Stale Folder Purger
----------------------------
Scans a directory tree and identifies two categories of folders:
    1. Completely empty directories
    2. Directories where every file is older than a configurable threshold

Presents a detailed report grouped by category before any deletion occurs.
Supports dry-run mode, a protected paths list, and bottom-up traversal
so nested empty folders are caught correctly.

Usage:
    python stale_folder_purger.py /path/to/scan
    python stale_folder_purger.py /path/to/scan --days 90
    python stale_folder_purger.py /path/to/scan --dry-run
    python stale_folder_purger.py /path/to/scan --protect /path/to/keep --protect /other/path

Dependencies:
    None (standard library only)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories that should NEVER be touched, regardless of user input.
# These are system or critical paths that the purger will always skip.
SYSTEM_PROTECTED_PATHS = {
    # Linux / macOS system directories
    "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64",
    "/proc", "/root", "/sbin", "/sys", "/usr", "/var",
    "/usr/bin", "/usr/lib", "/usr/local", "/usr/sbin",
    "/var/lib", "/var/log", "/var/run", "/var/spool",
    # macOS specific
    "/Applications", "/Library", "/System", "/Users",
    # Windows (as strings for cross-platform safety)
    "C:\\Windows", "C:\\Windows\\System32", "C:\\Program Files",
    "C:\\Program Files (x86)", "C:\\Users",
}


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


def is_protected(path: Path, protected_paths: set[str]) -> bool:
    """Check if a path is in or under a protected directory."""
    resolved = str(path.resolve())
    for protected in protected_paths:
        if resolved == protected or resolved.startswith(protected + os.sep):
            return True
    return False


def get_directory_age(dir_path: Path) -> tuple[int, int, float]:
    """
    Analyze a directory's contents.
    Returns (file_count, oldest_file_days, total_size_bytes).
    oldest_file_days is the age of the NEWEST file in the directory
    (i.e., how long since anything was last modified).
    """
    file_count = 0
    newest_mtime = 0
    total_size = 0
    now = datetime.now().timestamp()

    try:
        for entry in dir_path.iterdir():
            if entry.is_file():
                stat = entry.stat()
                file_count += 1
                total_size += stat.st_size
                if stat.st_mtime > newest_mtime:
                    newest_mtime = stat.st_mtime
    except PermissionError:
        pass

    if newest_mtime == 0:
        age_days = 0
    else:
        age_days = int((now - newest_mtime) / 86400)

    return file_count, age_days, total_size


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

class FolderCandidate:
    """Represents a directory flagged for potential removal."""
    def __init__(self, path: str, category: str, file_count: int, age_days: int, total_size: int):
        self.path = path
        self.category = category      # "empty" or "stale"
        self.file_count = file_count
        self.age_days = age_days
        self.total_size = total_size


def scan_for_targets(
    root_path: Path,
    stale_threshold_days: int,
    protected_paths: set[str],
) -> list[FolderCandidate]:
    """
    Walk the directory tree bottom-up and identify empty or stale directories.
    Bottom-up traversal ensures that if a parent directory only contained
    empty child directories (which we've now flagged), it too gets flagged.
    """
    candidates = []
    empty_dirs = set()  # Track dirs we've flagged as empty for parent analysis

    # os.walk with topdown=False gives us bottom-up traversal
    for dirpath, dirnames, filenames in os.walk(str(root_path), topdown=False):
        dir_path = Path(dirpath)

        # Never process the root path itself
        if dir_path.resolve() == root_path.resolve():
            continue

        # Skip protected paths
        if is_protected(dir_path, protected_paths):
            continue

        # Skip hidden directories (starting with .)
        if any(part.startswith(".") for part in dir_path.relative_to(root_path).parts):
            continue

        try:
            # Check if directory is truly empty (no files, no non-empty subdirs)
            contents = list(dir_path.iterdir())
            real_contents = [
                item for item in contents
                if item.is_file() or (item.is_dir() and str(item.resolve()) not in empty_dirs)
            ]

            if len(real_contents) == 0:
                # Directory is empty (or only contains other empty dirs we've already flagged)
                candidates.append(FolderCandidate(
                    path=str(dir_path),
                    category="empty",
                    file_count=0,
                    age_days=0,
                    total_size=0,
                ))
                empty_dirs.add(str(dir_path.resolve()))
                continue

            # Check if all files are stale (only direct children, not recursive)
            file_count, newest_age, total_size = get_directory_age(dir_path)

            if file_count > 0 and newest_age >= stale_threshold_days:
                # Also check that there are no active subdirectories
                has_active_subdirs = False
                for item in dir_path.iterdir():
                    if item.is_dir() and str(item.resolve()) not in empty_dirs:
                        # This subdir isn't flagged as empty — check if it's stale too
                        sub_files, sub_age, _ = get_directory_age(item)
                        if sub_files > 0 and sub_age < stale_threshold_days:
                            has_active_subdirs = True
                            break

                if not has_active_subdirs:
                    candidates.append(FolderCandidate(
                        path=str(dir_path),
                        category="stale",
                        file_count=file_count,
                        age_days=newest_age,
                        total_size=total_size,
                    ))

        except PermissionError:
            continue

    return candidates


def generate_report(candidates: list[FolderCandidate]) -> str:
    """Build a formatted report of all flagged directories."""
    lines = []
    lines.append("=" * 72)
    lines.append("  EMPTY & STALE FOLDER REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    empty = [c for c in candidates if c.category == "empty"]
    stale = [c for c in candidates if c.category == "stale"]

    # Empty directories
    if empty:
        lines.append(f"\n📭 EMPTY DIRECTORIES ({len(empty)})\n")
        lines.append("-" * 72)
        for c in sorted(empty, key=lambda x: x.path):
            lines.append(f"   {c.path}")
        lines.append("")

    # Stale directories
    if stale:
        total_stale_size = sum(c.total_size for c in stale)
        total_stale_files = sum(c.file_count for c in stale)
        lines.append(f"\n📦 STALE DIRECTORIES ({len(stale)})")
        lines.append(f"   Contains {total_stale_files} files totaling {format_size(total_stale_size)}\n")
        lines.append("-" * 72)

        for c in sorted(stale, key=lambda x: x.total_size, reverse=True):
            lines.append(
                f"   {format_size(c.total_size):>10}  |  "
                f"{c.file_count:>4} files  |  "
                f"newest: {c.age_days} days old  |  "
                f"{c.path}"
            )
        lines.append("")

    # Summary
    total_size = sum(c.total_size for c in candidates)
    lines.append("=" * 72)
    lines.append(f"  Total directories flagged: {len(candidates)}")
    lines.append(f"    Empty:  {len(empty)}")
    lines.append(f"    Stale:  {len(stale)}")
    lines.append(f"  Total recoverable space: {format_size(total_size)}")
    lines.append("=" * 72)

    return "\n".join(lines)


def purge_directories(candidates: list[FolderCandidate], log_path: str) -> tuple[int, int]:
    """
    Delete all flagged directories. Returns (success_count, fail_count).
    Processes empty directories first, then stale ones.
    """
    success = 0
    failures = 0
    log_lines = []

    # Sort: empty dirs first (so nested empties are removed cleanly), then stale
    ordered = sorted(candidates, key=lambda c: (c.category != "empty", c.path), reverse=True)

    for candidate in ordered:
        path = Path(candidate.path)
        try:
            if candidate.category == "empty":
                # For empty dirs, use rmdir (only works if truly empty)
                # If it fails, fall back to rmtree (for dirs containing only empty subdirs)
                try:
                    path.rmdir()
                except OSError:
                    shutil.rmtree(str(path))
            else:
                # Stale dirs may contain files — use rmtree
                shutil.rmtree(str(path))

            log_lines.append(f"[DELETED] [{candidate.category.upper()}] {candidate.path}")
            success += 1

        except (PermissionError, OSError) as e:
            log_lines.append(f"[FAILED]  [{candidate.category.upper()}] {candidate.path} — {e}")
            failures += 1

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Purge log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 72 + "\n\n")
        f.write("\n".join(log_lines))

    return success, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find and remove empty directories and directories containing only old files."
    )
    parser.add_argument(
        "root", type=str,
        help="Root directory to scan"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Mark directories as stale if all files are older than this many days (default: 90)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without removing anything"
    )
    parser.add_argument(
        "--auto-confirm", action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--protect", action="append", default=None,
        help="Additional paths to protect from deletion. Can be used multiple times."
    )
    parser.add_argument(
        "--log", type=str, default="purge_log.txt",
        help="Path for the deletion log (default: purge_log.txt)"
    )
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    if not root_path.exists():
        print(f"❌ Directory not found: {root_path}")
        sys.exit(1)

    # Build protected paths set
    protected = set(SYSTEM_PROTECTED_PATHS)
    if args.protect:
        for p in args.protect:
            protected.add(str(Path(p).resolve()))

    print(f"\n🔍 Scanning: {root_path}")
    print(f"⏱️  Stale threshold: {args.days} days")
    print(f"🔒 Protected paths: {len(protected)}")
    if args.dry_run:
        print("🛑 DRY RUN — nothing will be deleted")
    print()

    # Scan
    candidates = scan_for_targets(root_path, args.days, protected)

    # Report
    report = generate_report(candidates)
    print(report)

    if not candidates:
        print("\n✅ Nothing to purge. Everything looks clean!")
        return

    # Dry run — stop here
    if args.dry_run:
        print("\n🛑 Dry run complete — no directories were removed.")
        return

    # Confirm
    if not args.auto_confirm:
        print()
        choice = input("⚠️  Delete all flagged directories? (yes/no): ").strip().lower()
        if choice not in ("yes", "y"):
            print("Cancelled. Nothing was deleted.")
            return

    # Purge
    print(f"\n🗑️  Purging {len(candidates)} directories...")
    success, failures = purge_directories(candidates, args.log)

    # Summary
    print(f"\n✅ Done!")
    print(f"   Removed:   {success} directories")
    if failures:
        print(f"   Failed:    {failures} directories (check {args.log})")
    print(f"   Log saved: {args.log}")


if __name__ == "__main__":
    main()
