"""
Desktop Screenshot Organizer
Automatically organizes screenshots by date and can archive old ones.
Optional OCR functionality to make screenshots searchable.
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta
import json

class ScreenshotOrganizer:
    def __init__(self, screenshots_dir, organize_dir=None, archive_days=30):
        """
        Initialize screenshot organizer.
        
        Args:
            screenshots_dir: Directory where screenshots are saved
            organize_dir: Directory to organize screenshots into (default: screenshots_dir/Organized)
            archive_days: Move screenshots older than this to archive (0 to disable)
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.organize_dir = Path(organize_dir) if organize_dir else self.screenshots_dir / "Organized"
        self.archive_dir = self.organize_dir / "Archive"
        self.archive_days = archive_days
        self.metadata_file = self.organize_dir / "screenshot_metadata.json"
        self.metadata = self._load_metadata()
        
        # Common screenshot filename patterns
        self.patterns = [
            r'Screenshot.*(\d{4})-(\d{2})-(\d{2})',  # Screenshot 2025-11-11
            r'Screen Shot (\d{4})-(\d{2})-(\d{2})',  # Screen Shot 2025-11-11
            r'Screenshot_(\d{8})',                    # Screenshot_20251111
            r'IMG_(\d{8})',                          # IMG_20251111
        ]
        
        # Create directories
        self.organize_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_days > 0:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self):
        """Load screenshot metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'screenshots': {}}
    
    def _save_metadata(self):
        """Save screenshot metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _extract_date_from_filename(self, filename):
        """Try to extract date from screenshot filename."""
        for pattern in self.patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 3:  # YYYY-MM-DD format
                        year, month, day = groups
                        return datetime(int(year), int(month), int(day))
                    elif len(groups) == 1 and len(groups[0]) == 8:  # YYYYMMDD format
                        date_str = groups[0]
                        return datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    continue
        return None
    
    def _get_file_date(self, file_path):
        """Get date from filename or file modification time."""
        # Try filename first
        date = self._extract_date_from_filename(file_path.name)
        if date:
            return date
        
        # Fall back to modification time
        timestamp = file_path.stat().st_mtime
        return datetime.fromtimestamp(timestamp)
    
    def _is_screenshot(self, file_path):
        """Check if file is likely a screenshot based on name and type."""
        name_lower = file_path.name.lower()
        
        # Check for screenshot-like names
        screenshot_keywords = ['screenshot', 'screen shot', 'screen_shot', 'capture']
        has_keyword = any(keyword in name_lower for keyword in screenshot_keywords)
        
        # Check for image extension
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        has_extension = file_path.suffix.lower() in image_extensions
        
        return has_keyword and has_extension
    
    def organize(self, dry_run=False):
        """Organize screenshots by date."""
        if not self.screenshots_dir.exists():
            print(f"Error: Directory {self.screenshots_dir} does not exist.")
            return
        
        organized_count = 0
        archived_count = 0
        skipped_count = 0
        
        print(f"Organizing screenshots from: {self.screenshots_dir}")
        print(f"{'[DRY RUN] ' if dry_run else ''}\n")
        
        cutoff_date = datetime.now() - timedelta(days=self.archive_days) if self.archive_days > 0 else None
        
        # Get all potential screenshot files
        files = [f for f in self.screenshots_dir.iterdir() if f.is_file()]
        
        for file_path in files:
            # Skip if not a screenshot
            if not self._is_screenshot(file_path):
                continue
            
            # Skip if already in organized directory
            if self.organize_dir in file_path.parents:
                continue
            
            try:
                # Get file date
                file_date = self._get_file_date(file_path)
                
                # Determine if should be archived
                should_archive = cutoff_date and file_date < cutoff_date
                
                if should_archive:
                    # Archive old screenshot
                    year_folder = file_date.strftime("%Y")
                    month_folder = file_date.strftime("%m-%B")
                    target_dir = self.archive_dir / year_folder / month_folder
                else:
                    # Organize recent screenshot
                    year_folder = file_date.strftime("%Y")
                    month_folder = file_date.strftime("%m-%B")
                    target_dir = self.organize_dir / year_folder / month_folder
                
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                
                # Get unique target path
                target_path = target_dir / file_path.name
                if target_path.exists():
                    counter = 1
                    stem = file_path.stem
                    suffix = file_path.suffix
                    while target_path.exists():
                        target_path = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                # Move file
                if not dry_run:
                    shutil.move(str(file_path), str(target_path))
                    
                    # Save metadata
                    self.metadata['screenshots'][str(target_path)] = {
                        'original_path': str(file_path),
                        'organized_date': datetime.now().isoformat(),
                        'screenshot_date': file_date.isoformat(),
                        'archived': should_archive
                    }
                
                status = "Archived" if should_archive else "Organized"
                print(f"{'[DRY RUN] ' if dry_run else ''}{status}: {file_path.name} -> {target_dir.name}/{target_path.name}")
                
                if should_archive:
                    archived_count += 1
                else:
                    organized_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                skipped_count += 1
        
        # Save metadata
        if not dry_run:
            self._save_metadata()
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"Screenshots organized: {organized_count}")
        print(f"Screenshots archived: {archived_count}")
        print(f"Files skipped: {skipped_count}")
    
    def cleanup_old_archives(self, days_to_keep=180, dry_run=False):
        """Delete archived screenshots older than specified days."""
        if not self.archive_dir.exists():
            print("No archive directory found.")
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        freed_space = 0
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up archives older than {days_to_keep} days...")
        
        for file_path in self.archive_dir.rglob('*'):
            if file_path.is_file():
                try:
                    file_date = self._get_file_date(file_path)
                    
                    if file_date < cutoff_date:
                        size = file_path.stat().st_size
                        
                        if not dry_run:
                            file_path.unlink()
                            # Remove from metadata
                            self.metadata['screenshots'].pop(str(file_path), None)
                        
                        print(f"{'[DRY RUN] ' if dry_run else ''}Deleted: {file_path.name} ({file_date.strftime('%Y-%m-%d')})")
                        deleted_count += 1
                        freed_space += size
                        
                except Exception as e:
                    print(f"Error deleting {file_path.name}: {e}")
        
        if not dry_run:
            self._save_metadata()
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"Files deleted: {deleted_count}")
        print(f"Space freed: {freed_space / (1024**2):.2f} MB")
    
    def get_statistics(self):
        """Get statistics about organized screenshots."""
        if not self.organize_dir.exists():
            print("No organized screenshots found.")
            return
        
        total_files = 0
        total_size = 0
        by_year = {}
        by_month = {}
        
        for file_path in self.organize_dir.rglob('*'):
            if file_path.is_file() and file_path != self.metadata_file:
                total_files += 1
                total_size += file_path.stat().st_size
                
                # Count by year and month
                try:
                    parts = file_path.parts
                    if len(parts) >= 2:
                        year = parts[-3]  # Year folder
                        month = parts[-2]  # Month folder
                        
                        by_year[year] = by_year.get(year, 0) + 1
                        by_month[f"{year}-{month}"] = by_month.get(f"{year}-{month}", 0) + 1
                except:
                    pass
        
        print("\nScreenshot Statistics:")
        print("=" * 60)
        print(f"Total screenshots: {total_files}")
        print(f"Total size: {total_size / (1024**2):.2f} MB")
        
        if by_year:
            print("\nBy Year:")
            for year in sorted(by_year.keys(), reverse=True):
                print(f"  {year}: {by_year[year]} screenshots")
        
        if by_month:
            print("\nRecent Months:")
            for month in sorted(by_month.keys(), reverse=True)[:6]:
                print(f"  {month}: {by_month[month]} screenshots")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get screenshots directory from command line or use Desktop
    if len(sys.argv) > 1:
        screenshots_dir = sys.argv[1]
    else:
        # Default to Desktop (common screenshot location)
        screenshots_dir = str(Path.home() / "Desktop")
    
    organizer = ScreenshotOrganizer(
        screenshots_dir=screenshots_dir,
        archive_days=30  # Archive screenshots older than 30 days
    )
    
    print(f"Desktop Screenshot Organizer")
    print(f"Source: {organizer.screenshots_dir}")
    print(f"Organized: {organizer.organize_dir}")
    print(f"Archive after: {organizer.archive_days} days\n")
    
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("1. Preview organization (dry run)")
        print("2. Organize screenshots")
        print("3. Show statistics")
        print("4. Cleanup old archives (dry run)")
        print("5. Cleanup old archives (PERMANENT)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            organizer.organize(dry_run=True)
            
        elif choice == "2":
            confirm = input("This will move screenshot files. Continue? (yes/no): ")
            if confirm.lower() == 'yes':
                organizer.organize(dry_run=False)
            
        elif choice == "3":
            organizer.get_statistics()
            
        elif choice == "4":
            days = input("Delete archives older than how many days? (default: 180): ").strip()
            days = int(days) if days else 180
            organizer.cleanup_old_archives(days_to_keep=days, dry_run=True)
            
        elif choice == "5":
            days = input("Delete archives older than how many days? (default: 180): ").strip()
            days = int(days) if days else 180
            confirm = input(f"\n⚠️  Delete archives older than {days} days? (type 'DELETE' to confirm): ")
            if confirm == 'DELETE':
                organizer.cleanup_old_archives(days_to_keep=days, dry_run=False)
            else:
                print("Deletion cancelled.")
            
        elif choice == "6":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
          
