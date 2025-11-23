"""
Batch File Renamer
Rename multiple files with flexible patterns and rules.
"""

import os
import re
from pathlib import Path
from datetime import datetime

class BatchRenamer:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.rename_map = {}
    
    def get_files(self, pattern="*.*", include_subdirs=False):
        """Get list of files matching pattern."""
        if include_subdirs:
            return list(self.directory.rglob(pattern))
        else:
            return list(self.directory.glob(pattern))
    
    def add_prefix(self, files, prefix):
        """Add prefix to filenames."""
        for file_path in files:
            new_name = f"{prefix}{file_path.name}"
            self.rename_map[file_path] = file_path.parent / new_name
    
    def add_suffix(self, files, suffix):
        """Add suffix before file extension."""
        for file_path in files:
            stem = file_path.stem
            ext = file_path.suffix
            new_name = f"{stem}{suffix}{ext}"
            self.rename_map[file_path] = file_path.parent / new_name
    
    def replace_text(self, files, old_text, new_text, case_sensitive=True):
        """Replace text in filenames."""
        for file_path in files:
            if case_sensitive:
                new_name = file_path.name.replace(old_text, new_text)
            else:
                new_name = re.sub(re.escape(old_text), new_text, file_path.name, flags=re.IGNORECASE)
            self.rename_map[file_path] = file_path.parent / new_name
    
    def add_sequential_number(self, files, start=1, padding=3, position='suffix'):
        """Add sequential numbers to filenames."""
        sorted_files = sorted(files, key=lambda x: x.name)
        
        for idx, file_path in enumerate(sorted_files, start=start):
            number = str(idx).zfill(padding)
            stem = file_path.stem
            ext = file_path.suffix
            
            if position == 'prefix':
                new_name = f"{number}_{stem}{ext}"
            elif position == 'suffix':
                new_name = f"{stem}_{number}{ext}"
            else:  # replace
                new_name = f"{number}{ext}"
            
            self.rename_map[file_path] = file_path.parent / new_name
    
    def add_date(self, files, format='%Y%m%d', position='prefix', use_modified_date=True):
        """Add date to filenames based on file modification or creation date."""
        for file_path in files:
            if use_modified_date:
                timestamp = file_path.stat().st_mtime
            else:
                timestamp = file_path.stat().st_ctime
            
            date_str = datetime.fromtimestamp(timestamp).strftime(format)
            stem = file_path.stem
            ext = file_path.suffix
            
            if position == 'prefix':
                new_name = f"{date_str}_{stem}{ext}"
            else:  # suffix
                new_name = f"{stem}_{date_str}{ext}"
            
            self.rename_map[file_path] = file_path.parent / new_name
    
    def change_case(self, files, case_type='lower'):
        """Change filename case (lower, upper, title, capitalize)."""
        for file_path in files:
            stem = file_path.stem
            ext = file_path.suffix
            
            if case_type == 'lower':
                new_stem = stem.lower()
            elif case_type == 'upper':
                new_stem = stem.upper()
            elif case_type == 'title':
                new_stem = stem.title()
            elif case_type == 'capitalize':
                new_stem = stem.capitalize()
            else:
                new_stem = stem
            
            new_name = f"{new_stem}{ext}"
            self.rename_map[file_path] = file_path.parent / new_name
    
    def remove_pattern(self, files, pattern):
        """Remove pattern from filenames using regex."""
        for file_path in files:
            new_name = re.sub(pattern, '', file_path.name)
            # Clean up any double underscores or spaces
            new_name = re.sub(r'[_\s]+', '_', new_name)
            new_name = new_name.strip('_')
            self.rename_map[file_path] = file_path.parent / new_name
    
    def preview_changes(self):
        """Show preview of changes."""
        if not self.rename_map:
            print("No changes to preview.")
            return
        
        print(f"\nPreview of {len(self.rename_map)} file renames:")
        print("=" * 80)
        
        for old_path, new_path in self.rename_map.items():
            print(f"OLD: {old_path.name}")
            print(f"NEW: {new_path.name}")
            print("-" * 80)
    
    def check_conflicts(self):
        """Check for naming conflicts."""
        conflicts = []
        new_names = {}
        
        for old_path, new_path in self.rename_map.items():
            # Check if new name already exists
            if new_path.exists() and new_path not in self.rename_map:
                conflicts.append(f"File already exists: {new_path.name}")
            
            # Check for duplicate new names
            if new_path.name in new_names:
                conflicts.append(f"Duplicate new name: {new_path.name}")
            else:
                new_names[new_path.name] = old_path
        
        return conflicts
    
    def execute_rename(self, backup_originals=False):
        """Execute the rename operations."""
        conflicts = self.check_conflicts()
        
        if conflicts:
            print("\nConflicts detected:")
            for conflict in conflicts:
                print(f"  - {conflict}")
            print("\nRename aborted. Please resolve conflicts first.")
            return False
        
        success_count = 0
        error_count = 0
        
        print(f"\nRenaming {len(self.rename_map)} files...")
        
        for old_path, new_path in self.rename_map.items():
            try:
                if backup_originals:
                    backup_path = old_path.parent / f"{old_path.name}.backup"
                    old_path.rename(backup_path)
                    backup_path.rename(new_path)
                else:
                    old_path.rename(new_path)
                
                print(f"✓ Renamed: {old_path.name} -> {new_path.name}")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Error renaming {old_path.name}: {str(e)}")
                error_count += 1
        
        print(f"\nComplete! Success: {success_count}, Errors: {error_count}")
        
        # Clear the rename map after execution
        self.rename_map.clear()
        
        return error_count == 0
    
    def clear_map(self):
        """Clear the rename map."""
        self.rename_map.clear()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get directory from command line or use current directory
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    renamer = BatchRenamer(directory)
    
    print(f"Batch File Renamer")
    print(f"Directory: {renamer.directory.absolute()}\n")
    
    # Interactive menu
    while True:
        print("\nOptions:")
        print("1. Select files (by pattern)")
        print("2. Add prefix")
        print("3. Add suffix")
        print("4. Replace text")
        print("5. Add sequential numbers")
        print("6. Add date")
        print("7. Change case")
        print("8. Remove pattern (regex)")
        print("9. Preview changes")
        print("10. Execute rename")
        print("11. Clear and start over")
        print("12. Exit")
        
        choice = input("\nEnter choice (1-12): ").strip()
        
        if choice == "1":
            pattern = input("Enter file pattern (e.g., *.jpg, *.*, IMG_*): ").strip()
            files = renamer.get_files(pattern)
            print(f"Found {len(files)} files matching '{pattern}'")
            
        elif choice == "2":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            prefix = input("Enter prefix: ").strip()
            renamer.add_prefix(files, prefix)
            print(f"Added prefix '{prefix}' to {len(files)} files")
            
        elif choice == "3":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            suffix = input("Enter suffix: ").strip()
            renamer.add_suffix(files, suffix)
            print(f"Added suffix '{suffix}' to {len(files)} files")
            
        elif choice == "4":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            old_text = input("Text to replace: ").strip()
            new_text = input("Replace with: ").strip()
            renamer.replace_text(files, old_text, new_text)
            print(f"Replaced '{old_text}' with '{new_text}' in {len(files)} files")
            
        elif choice == "5":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            start = int(input("Starting number (default 1): ").strip() or "1")
            padding = int(input("Number padding (default 3): ").strip() or "3")
            position = input("Position (prefix/suffix/replace, default suffix): ").strip() or "suffix"
            renamer.add_sequential_number(files, start, padding, position)
            print(f"Added sequential numbers to {len(files)} files")
            
        elif choice == "6":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            date_format = input("Date format (default %Y%m%d): ").strip() or "%Y%m%d"
            position = input("Position (prefix/suffix, default prefix): ").strip() or "prefix"
            renamer.add_date(files, date_format, position)
            print(f"Added dates to {len(files)} files")
            
        elif choice == "7":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            case_type = input("Case type (lower/upper/title/capitalize): ").strip()
            renamer.change_case(files, case_type)
            print(f"Changed case to '{case_type}' for {len(files)} files")
            
        elif choice == "8":
            if 'files' not in locals():
                print("Please select files first (option 1)")
                continue
            pattern = input("Pattern to remove (regex): ").strip()
            renamer.remove_pattern(files, pattern)
            print(f"Removed pattern from {len(files)} files")
            
        elif choice == "9":
            renamer.preview_changes()
            
        elif choice == "10":
            if not renamer.rename_map:
                print("No changes to execute. Please configure rename operations first.")
                continue
            confirm = input("\nExecute rename? This cannot be undone! (yes/no): ")
            if confirm.lower() == 'yes':
                renamer.execute_rename()
            
        elif choice == "11":
            renamer.clear_map()
            print("Cleared all pending changes")
            
        elif choice == "12":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


