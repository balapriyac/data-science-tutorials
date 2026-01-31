"""
Stale Temp & Cache Cleaner
--------------------------
Scans system and app-specific temp/cache directories, flags files
that haven't been accessed within a configurable number of days,
and removes them after user confirmation. Generates a full report
before any deletion occurs.

Usage:
    python temp_cache_cleaner.py
    python temp_cache_cleaner.py --days 14
    python temp_cache_cleaner.py --dry-run
    python temp_cache_cleaner.py --auto-confirm --days 30

Dependencies:
    None (standard library only)
"""

import os
import sys
import shutil
import argparse
import time
import platform
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_default_temp_paths() -> list[str]:
    """Return platform-appropriate temp and cache directory paths."""
    system = platform.system()

    if system == "Windows":
        user = os.environ.get("USERPROFILE", "C:\\Users\\Default")
        app_data = os.environ.get("APPDATA", os.path.join(user, "AppData", "Roaming"))
        local_app = os.environ.get("LOCALAPPDATA", os.path.join(user, "AppData", "Local"))
        return [
            os.environ.get("TEMP", os.path.join(user, "AppData", "Local", "Temp")),
            os.path.join(local_app, "Temp"),
            os.path.join(local_app, "Microsoft", "Windows", "InetCache"),
            os.path.join(app_data, "Mozilla", "Firefox", "Profiles"),  # Firefox cache
            os.path.join(local_app, "Google", "Chrome", "User Data", "Default", "Cache"),
        ]

    elif system == "Darwin":  # macOS
        user = os.path.expanduser("~")
        return [
            "/tmp",
            os.path.join(user, "Library", "Caches"),
            os.path.join(user, "Library", "Logs"),
            "/var/folders",  # macOS app temp files
        ]

    else:  # Linux
        user = os.path.expanduser("~")
        return [
            "/tmp",
            os.path.join(user, ".cache"),
            os.path.join(user, ".local", "share", "Trash", "files"),
            "/var/tmp",
        ]


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

class FileInfo:
    """Holds metadata about a single scanned file."""
    def __init__(self, path: str, size: int, last_access: datetime, last_modified: datetime):
        self.path = path
        self.size = size
        self.last_access = last_access
        self.last_modified = last_modified

    def age_days(self, reference: datetime) -> int:
        """Days since the file was last accessed or modified (whichever is later)."""
        latest = max(self.last_access, self.last_modified)
        return (reference - latest).days


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def scan_directory(root_path: str, max_age_days: int, cutoff: datetime) -> list[FileInfo]:
    """
    Walk a directory tree and collect info on files older than max_age_days.
    Skips files that can't be accessed (permissions, locks, etc.).
    """
    stale_files = []
    root = Path(root_path)

    if not root.exists():
        return stale_files

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            stat = file_path.stat()
            last_access = datetime.fromtimestamp(stat.st_atime)
            last_modified = datetime.fromtimestamp(stat.st_mtime)

            info = FileInfo(
                path=str(file_path),
                size=stat.st_size,
                last_access=last_access,
                last_modified=last_modified,
            )

            if info.age_days(cutoff) >= max_age_days:
                stale_files.append(info)

        except (PermissionError, OSError):
            # Can't access this file — skip silently
            continue

    return stale_files


def generate_report(results: dict[str, list[FileInfo]]) -> str:
    """Build a formatted text report from scan results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  STALE TEMP & CACHE REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    grand_total_files = 0
    grand_total_size = 0

    for directory, files in results.items():
        if not files:
            continue

        dir_size = sum(f.size for f in files)
        grand_total_files += len(files)
        grand_total_size += dir_size

        lines.append(f"\n📁 {directory}")
        lines.append(f"   Files: {len(files)}  |  Total size: {format_size(dir_size)}")
        lines.append("-" * 70)

        # Show up to 10 files per directory in the report
        for f in sorted(files, key=lambda x: x.size, reverse=True)[:10]:
            age = f.age_days(datetime.now())
            lines.append(f"   {format_size(f.size):>10}  |  {age:>4} days old  |  {f.path}")

        if len(files) > 10:
            lines.append(f"   ... and {len(files) - 10} more files")

    lines.append("\n" + "=" * 70)
    lines.append(f"  TOTAL: {grand_total_files} files  |  {format_size(grand_total_size)} of recoverable space")
    lines.append("=" * 70)

    return "\n".join(lines)


def delete_files(results: dict[str, list[FileInfo]], log_path: str) -> tuple[int, int]:
    """
    Delete all flagged files. Returns (success_count, fail_count).
    Logs every action to a file.
    """
    success = 0
    failures = 0
    log_lines = []

    for directory, files in results.items():
        for f in files:
            try:
                os.remove(f.path)
                log_lines.append(f"[DELETED] {f.path}")
                success += 1
            except (PermissionError, OSError) as e:
                log_lines.append(f"[FAILED]  {f.path} — {e}")
                failures += 1

    # Write log
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Cleanup log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 70 + "\n")
        log_file.write("\n".join(log_lines))

    return success, failures


def remove_empty_directories(paths: list[str]) -> int:
    """Remove empty directories left behind after file deletion."""
    removed = 0
    for root_path in paths:
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
                    removed += 1
            except (PermissionError, OSError):
                continue
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find and remove stale temp and cache files."
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Remove files not accessed in this many days (default: 30)"
    )
    parser.add_argument(
        "--paths", nargs="+", default=None,
        help="Custom directories to scan (default: system temp/cache paths)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting anything"
    )
    parser.add_argument(
        "--auto-confirm", action="store_true",
        help="Skip confirmation prompt and delete immediately"
    )
    parser.add_argument(
        "--log", type=str, default="cleanup_log.txt",
        help="Path for the deletion log file (default: cleanup_log.txt)"
    )
    args = parser.parse_args()

    scan_paths = args.paths if args.paths else get_default_temp_paths()
    cutoff = datetime.now()

    print(f"\n🔍 Scanning {len(scan_paths)} directories (threshold: {args.days} days)...\n")

    # Scan
    results = {}
    for path in scan_paths:
        if os.path.exists(path):
            print(f"   Scanning: {path}")
            results[path] = scan_directory(path, args.days, cutoff)
        else:
            print(f"   Skipping (not found): {path}")

    # Report
    report = generate_report(results)
    print("\n" + report)

    total_files = sum(len(files) for files in results.values())

    if total_files == 0:
        print("\n✅ Nothing to clean up. You're good!")
        return

    # Dry run — stop here
    if args.dry_run:
        print("\n🛑 Dry run mode — no files were deleted.")
        return

    # Confirm before deleting
    if not args.auto_confirm:
        print()
        choice = input("⚠️  Delete all flagged files? (yes/no): ").strip().lower()
        if choice not in ("yes", "y"):
            print("Cancelled. No files were deleted.")
            return

    # Delete
    print(f"\n🗑️  Deleting {total_files} files...")
    success, failures = delete_files(results, args.log)

    # Clean up empty directories
    empty_removed = remove_empty_directories(scan_paths)

    # Summary
    print(f"\n✅ Done!")
    print(f"   Deleted:        {success} files")
    if failures:
        print(f"   Failed:         {failures} files (check {args.log} for details)")
    if empty_removed:
        print(f"   Empty dirs removed: {empty_removed}")
    print(f"   Log saved to:   {args.log}")


if __name__ == "__main__":
    main()
  
