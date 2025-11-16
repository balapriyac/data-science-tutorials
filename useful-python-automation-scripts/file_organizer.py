"""
Automatic File Organizer
Organizes files in a directory by type and date into structured folders.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class FileOrganizer:
    def __init__(self, source_dir, create_date_folders=False):
        self.source_dir = Path(source_dir)
        self.create_date_folders = create_date_folders
        self.log_file = self.source_dir / "organization_log.json"
        self.log = []
        
        # File type categories
        self.categories = {
            'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'],
            'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
            'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.json', '.xml'],
            'Executables': ['.exe', '.dmg', '.pkg', '.deb', '.rpm', '.msi'],
            'Others': []
        }
    
    def get_category(self, file_path):
        """Determine file category based on extension."""
        ext = file_path.suffix.lower()
        for category, extensions in self.categories.items():
            if ext in extensions:
                return category
        return 'Others'
    
    def get_unique_filename(self, target_path):
        """Generate unique filename if file already exists."""
        if not target_path.exists():
            return target_path
        
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        counter = 1
        
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    def organize(self, dry_run=False):
        """Organize files in the source directory."""
        if not self.source_dir.exists():
            print(f"Error: Directory {self.source_dir} does not exist.")
            return
        
        files_moved = 0
        errors = []
        
        print(f"Organizing files in: {self.source_dir}")
        print(f"Dry run: {dry_run}\n")
        
        # Get all files in source directory (not subdirectories)
        files = [f for f in self.source_dir.iterdir() if f.is_file() and f.name != "organization_log.json"]
        
        for file_path in files:
            try:
                # Get category
                category = self.get_category(file_path)
                
                # Create target directory
                if self.create_date_folders:
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    date_folder = mod_time.strftime("%Y-%m")
                    target_dir = self.source_dir / category / date_folder
                else:
                    target_dir = self.source_dir / category
                
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                
                # Get unique target path
                target_path = target_dir / file_path.name
                if target_path.exists():
                    target_path = self.get_unique_filename(target_path)
                
                # Move file
                if not dry_run:
                    shutil.move(str(file_path), str(target_path))
                    
                    # Log the move
                    self.log.append({
                        'timestamp': datetime.now().isoformat(),
                        'original': str(file_path),
                        'new_location': str(target_path),
                        'category': category
                    })
                
                print(f"{'[DRY RUN] ' if dry_run else ''}Moved: {file_path.name} -> {category}/{target_path.name}")
                files_moved += 1
                
            except Exception as e:
                error_msg = f"Error moving {file_path.name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        # Save log
        if not dry_run and self.log:
            with open(self.log_file, 'w') as f:
                json.dump(self.log, f, indent=2)
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"Files organized: {files_moved}")
        print(f"Errors: {len(errors)}")
        if not dry_run and files_moved > 0:
            print(f"Log saved to: {self.log_file}")
    
    def show_statistics(self):
        """Show statistics about files in directory."""
        category_counts = {cat: 0 for cat in self.categories.keys()}
        total_size = 0
        
        files = [f for f in self.source_dir.iterdir() if f.is_file()]
        
        for file_path in files:
            category = self.get_category(file_path)
            category_counts[category] += 1
            total_size += file_path.stat().st_size
        
        print(f"\nDirectory Statistics: {self.source_dir}")
        print("=" * 50)
        print(f"Total files: {len(files)}")
        print(f"Total size: {total_size / (1024**2):.2f} MB\n")
        
        print("Files by category:")
        for category, count in category_counts.items():
            if count > 0:
                print(f"  {category}: {count}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get directory from command line or use Downloads folder
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to Downloads folder
        directory = str(Path.home() / "Downloads")
    
    organizer = FileOrganizer(directory, create_date_folders=False)
    
    # Show current statistics
    organizer.show_statistics()
    
    # Ask user for confirmation
    print("\nOptions:")
    print("1. Run dry-run (preview changes)")
    print("2. Organize files")
    print("3. Organize files with date folders")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        organizer.organize(dry_run=True)
    elif choice == "2":
        confirm = input("This will move files. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            organizer.organize(dry_run=False)
    elif choice == "3":
        confirm = input("This will move files into date folders. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            organizer.create_date_folders = True
            organizer.organize(dry_run=False)
    else:
        print("Exiting...")


