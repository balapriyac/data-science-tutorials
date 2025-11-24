"""
Duplicate File Finder
Finds and manages duplicate files across directories using hash comparison.
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

class DuplicateFinder:
    def __init__(self, directories, min_size=0):
        """
        Initialize duplicate finder.
        
        Args:
            directories: List of directories to scan
            min_size: Minimum file size in bytes to consider (skip tiny files)
        """
        self.directories = [Path(d) for d in directories]
        self.min_size = min_size
        self.duplicates = defaultdict(list)
        self.file_count = 0
        self.scanned_size = 0
    
    def _calculate_hash(self, file_path, chunk_size=8192):
        """Calculate MD5 hash of a file."""
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except (PermissionError, OSError) as e:
            print(f"  Skipped (error): {file_path.name} - {str(e)}")
            return None
    
    def scan(self, show_progress=True):
        """Scan directories for duplicate files."""
        print("Scanning for duplicate files...")
        print(f"Minimum file size: {self.min_size} bytes\n")
        
        # First pass: group files by size (faster than hashing everything)
        size_map = defaultdict(list)
        
        for directory in self.directories:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist. Skipping.")
                continue
            
            print(f"Scanning: {directory}")
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        
                        if size >= self.min_size:
                            size_map[size].append(file_path)
                            self.file_count += 1
                            self.scanned_size += size
                            
                            if show_progress and self.file_count % 100 == 0:
                                print(f"  Scanned {self.file_count} files...", end='\r')
                    
                    except (PermissionError, OSError):
                        continue
        
        print(f"\nScanned {self.file_count} files totaling {self.scanned_size / (1024**3):.2f} GB")
        
        # Second pass: hash files with matching sizes
        print("\nIdentifying duplicates...")
        hash_map = defaultdict(list)
        files_to_hash = []
        
        # Only hash files that have the same size as another file
        for size, files in size_map.items():
            if len(files) > 1:
                files_to_hash.extend(files)
        
        print(f"Hashing {len(files_to_hash)} potentially duplicate files...")
        
        for idx, file_path in enumerate(files_to_hash, 1):
            if show_progress and idx % 50 == 0:
                print(f"  Hashed {idx}/{len(files_to_hash)} files...", end='\r')
            
            file_hash = self._calculate_hash(file_path)
            if file_hash:
                hash_map[file_hash].append(file_path)
        
        # Store only actual duplicates (hash appears more than once)
        for file_hash, files in hash_map.items():
            if len(files) > 1:
                self.duplicates[file_hash] = files
        
        print(f"\n\n✓ Found {len(self.duplicates)} sets of duplicate files")
        
        return self.duplicates
    
    def get_duplicate_stats(self):
        """Get statistics about found duplicates."""
        if not self.duplicates:
            return None
        
        total_files = sum(len(files) for files in self.duplicates.values())
        
        # Calculate wasted space (keep one copy of each duplicate set)
        wasted_space = 0
        for files in self.duplicates.values():
            file_size = files[0].stat().st_size
            # Wasted space = size × (number of copies - 1)
            wasted_space += file_size * (len(files) - 1)
        
        return {
            'duplicate_sets': len(self.duplicates),
            'total_duplicate_files': total_files,
            'wasted_space_bytes': wasted_space,
            'wasted_space_mb': wasted_space / (1024**2),
            'wasted_space_gb': wasted_space / (1024**3)
        }
    
    def display_duplicates(self, max_sets=None):
        """Display found duplicates."""
        if not self.duplicates:
            print("No duplicates found.")
            return
        
        stats = self.get_duplicate_stats()
        
        print("\nDuplicate Files Summary:")
        print("=" * 80)
        print(f"Duplicate sets: {stats['duplicate_sets']}")
        print(f"Total duplicate files: {stats['total_duplicate_files']}")
        print(f"Wasted space: {stats['wasted_space_gb']:.2f} GB ({stats['wasted_space_mb']:.2f} MB)")
        print("\n" + "=" * 80)
        
        sets_to_show = list(self.duplicates.items())
        if max_sets:
            sets_to_show = sets_to_show[:max_sets]
        
        for idx, (file_hash, files) in enumerate(sets_to_show, 1):
            size = files[0].stat().st_size
            size_mb = size / (1024**2)
            
            print(f"\nSet {idx}: {len(files)} copies ({size_mb:.2f} MB each)")
            print(f"Hash: {file_hash[:16]}...")
            
            for file_path in files:
                print(f"  - {file_path}")
        
        if max_sets and len(self.duplicates) > max_sets:
            print(f"\n... and {len(self.duplicates) - max_sets} more duplicate sets")
    
    def export_report(self, output_file):
        """Export duplicate report to JSON file."""
        if not self.duplicates:
            print("No duplicates to export.")
            return
        
        stats = self.get_duplicate_stats()
        
        report = {
            'scan_date': str(Path.cwd()),
            'directories_scanned': [str(d) for d in self.directories],
            'statistics': stats,
            'duplicates': {}
        }
        
        for file_hash, files in self.duplicates.items():
            size = files[0].stat().st_size
            report['duplicates'][file_hash] = {
                'file_size_bytes': size,
                'file_count': len(files),
                'files': [str(f) for f in files]
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report exported to: {output_file}")
    
    def delete_duplicates(self, keep='first', dry_run=True):
        """
        Delete duplicate files, keeping one copy.
        
        Args:
            keep: Which file to keep ('first', 'last', 'smallest_path', 'newest')
            dry_run: If True, only show what would be deleted
        """
        if not self.duplicates:
            print("No duplicates to delete.")
            return
        
        deleted_count = 0
        freed_space = 0
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Deleting duplicates (keeping {keep} copy)...")
        
        for file_hash, files in self.duplicates.items():
            # Determine which file to keep
            if keep == 'first':
                file_to_keep = files[0]
            elif keep == 'last':
                file_to_keep = files[-1]
            elif keep == 'smallest_path':
                file_to_keep = min(files, key=lambda f: len(str(f)))
            elif keep == 'newest':
                file_to_keep = max(files, key=lambda f: f.stat().st_mtime)
            else:
                file_to_keep = files[0]
            
            # Delete other files
            for file_path in files:
                if file_path != file_to_keep:
                    size = file_path.stat().st_size
                    
                    if not dry_run:
                        try:
                            file_path.unlink()
                            print(f"  Deleted: {file_path}")
                        except Exception as e:
                            print(f"  Error deleting {file_path}: {e}")
                            continue
                    else:
                        print(f"  Would delete: {file_path}")
                    
                    deleted_count += 1
                    freed_space += size
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"Files {'would be ' if dry_run else ''}deleted: {deleted_count}")
        print(f"Space {'would be ' if dry_run else ''}freed: {freed_space / (1024**3):.2f} GB")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python duplicate_finder.py <directory1> [directory2] ...")
        print("\nExample:")
        print("  python duplicate_finder.py ~/Downloads ~/Documents")
        sys.exit(1)
    
    directories = sys.argv[1:]
    min_size = 1024  # Skip files smaller than 1KB
    
    finder = DuplicateFinder(directories, min_size=min_size)
    
    # Scan for duplicates
    finder.scan(show_progress=True)
    
    if not finder.duplicates:
        print("\nNo duplicates found!")
        sys.exit(0)
    
    # Interactive menu
    while True:
        print("\n" + "=" * 80)
        print("Options:")
        print("1. Show duplicate summary")
        print("2. Show all duplicates")
        print("3. Export report to JSON")
        print("4. Delete duplicates (dry run)")
        print("5. Delete duplicates (PERMANENT)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            stats = finder.get_duplicate_stats()
            print(f"\nDuplicate sets: {stats['duplicate_sets']}")
            print(f"Total duplicate files: {stats['total_duplicate_files']}")
            print(f"Wasted space: {stats['wasted_space_gb']:.2f} GB")
            
        elif choice == "2":
            finder.display_duplicates()
            
        elif choice == "3":
            output_file = input("Enter output filename (default: duplicates_report.json): ").strip()
            if not output_file:
                output_file = "duplicates_report.json"
            finder.export_report(output_file)
            
        elif choice == "4":
            keep = input("Keep which copy? (first/last/smallest_path/newest, default: first): ").strip()
            if not keep:
                keep = "first"
            finder.delete_duplicates(keep=keep, dry_run=True)
            
        elif choice == "5":
            print("\n⚠️  WARNING: This will permanently delete files!")
            keep = input("Keep which copy? (first/last/smallest_path/newest, default: first): ").strip()
            if not keep:
                keep = "first"
            
            confirm = input(f"\nDelete all duplicates keeping {keep} copy? (type 'DELETE' to confirm): ")
            if confirm == 'DELETE':
                finder.delete_duplicates(keep=keep, dry_run=False)
            else:
                print("Deletion cancelled.")
            
        elif choice == "6":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


