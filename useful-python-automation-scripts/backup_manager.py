"""
Smart Backup Manager
Creates intelligent incremental backups with compression and version management.
"""

import os
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import json

class BackupManager:
    def __init__(self, source_dir, backup_dir, max_backups=10):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata = self._load_metadata()
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self):
        """Load backup metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'backups': [], 'file_hashes': {}}
    
    def _save_metadata(self):
        """Save backup metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _calculate_hash(self, file_path):
        """Calculate MD5 hash of a file."""
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None
    
    def _get_files_to_backup(self):
        """Identify files that need to be backed up (new or modified)."""
        files_to_backup = []
        unchanged_count = 0
        
        if not self.source_dir.exists():
            print(f"Error: Source directory {self.source_dir} does not exist.")
            return []
        
        for file_path in self.source_dir.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(self.source_dir))
                current_hash = self._calculate_hash(file_path)
                
                if current_hash is None:
                    continue
                
                # Check if file is new or modified
                stored_hash = self.metadata['file_hashes'].get(relative_path)
                
                if stored_hash != current_hash:
                    files_to_backup.append((file_path, relative_path, current_hash))
                else:
                    unchanged_count += 1
        
        print(f"Files to backup: {len(files_to_backup)}")
        print(f"Unchanged files: {unchanged_count}")
        
        return files_to_backup
    
    def create_backup(self, description=""):
        """Create a new backup with only changed files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name
        
        files_to_backup = self._get_files_to_backup()
        
        if not files_to_backup:
            print("No files need backing up. All files are unchanged.")
            return None
        
        print(f"\nCreating backup: {backup_name}")
        print(f"Backing up {len(files_to_backup)} files...")
        
        # Create zip file
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, relative_path, file_hash in files_to_backup:
                zipf.write(file_path, relative_path)
                # Update hash in metadata
                self.metadata['file_hashes'][relative_path] = file_hash
        
        # Calculate backup size
        backup_size = backup_path.stat().st_size
        
        # Add backup info to metadata
        backup_info = {
            'filename': backup_name,
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'description': description,
            'files_count': len(files_to_backup),
            'size_bytes': backup_size,
            'size_mb': round(backup_size / (1024 * 1024), 2)
        }
        
        self.metadata['backups'].append(backup_info)
        self._save_metadata()
        
        print(f"✓ Backup created successfully: {backup_name}")
        print(f"  Size: {backup_info['size_mb']} MB")
        print(f"  Files: {backup_info['files_count']}")
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        return backup_path
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups limit."""
        if len(self.metadata['backups']) <= self.max_backups:
            return
        
        # Sort backups by date
        sorted_backups = sorted(self.metadata['backups'], 
                               key=lambda x: x['timestamp'], 
                               reverse=True)
        
        # Remove oldest backups
        backups_to_remove = sorted_backups[self.max_backups:]
        
        for backup in backups_to_remove:
            backup_path = self.backup_dir / backup['filename']
            if backup_path.exists():
                backup_path.unlink()
                print(f"Removed old backup: {backup['filename']}")
            
            self.metadata['backups'].remove(backup)
        
        self._save_metadata()
    
    def list_backups(self):
        """List all available backups."""
        if not self.metadata['backups']:
            print("No backups found.")
            return
        
        print(f"\nAvailable backups ({len(self.metadata['backups'])}):")
        print("=" * 80)
        
        for idx, backup in enumerate(reversed(self.metadata['backups']), 1):
            print(f"\n{idx}. {backup['filename']}")
            print(f"   Date: {backup['date']}")
            print(f"   Files: {backup['files_count']}")
            print(f"   Size: {backup['size_mb']} MB")
            if backup['description']:
                print(f"   Description: {backup['description']}")
    
    def restore_backup(self, backup_index=-1, restore_dir=None):
        """Restore files from a backup."""
        if not self.metadata['backups']:
            print("No backups available to restore.")
            return False
        
        # Get backup (default to most recent)
        backup = self.metadata['backups'][backup_index]
        backup_path = self.backup_dir / backup['filename']
        
        if not backup_path.exists():
            print(f"Error: Backup file {backup['filename']} not found.")
            return False
        
        # Set restore directory
        if restore_dir is None:
            restore_dir = self.source_dir
        else:
            restore_dir = Path(restore_dir)
        
        restore_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRestoring backup: {backup['filename']}")
        print(f"To directory: {restore_dir}")
        
        # Extract files
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            zipf.extractall(restore_dir)
        
        print(f"✓ Backup restored successfully!")
        print(f"  Files restored: {backup['files_count']}")
        
        return True
    
    def get_backup_statistics(self):
        """Get statistics about backups."""
        if not self.metadata['backups']:
            print("No backup statistics available.")
            return
        
        total_size = sum(b['size_bytes'] for b in self.metadata['backups'])
        total_files = sum(b['files_count'] for b in self.metadata['backups'])
        
        oldest = min(self.metadata['backups'], key=lambda x: x['timestamp'])
        newest = max(self.metadata['backups'], key=lambda x: x['timestamp'])
        
        print("\nBackup Statistics:")
        print("=" * 50)
        print(f"Total backups: {len(self.metadata['backups'])}")
        print(f"Total size: {total_size / (1024**2):.2f} MB")
        print(f"Total files backed up: {total_files}")
        print(f"Oldest backup: {oldest['date']}")
        print(f"Newest backup: {newest['date']}")
        print(f"Unique files tracked: {len(self.metadata['file_hashes'])}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python backup_manager.py <source_dir> <backup_dir>")
        print("\nExample:")
        print("  python backup_manager.py ~/Documents ~/Backups/Documents")
        sys.exit(1)
    
    source = sys.argv[1]
    backup = sys.argv[2]
    
    manager = BackupManager(source, backup, max_backups=10)
    
    print(f"Smart Backup Manager")
    print(f"Source: {manager.source_dir}")
    print(f"Backup: {manager.backup_dir}\n")
    
    while True:
        print("\nOptions:")
        print("1. Create new backup")
        print("2. List backups")
        print("3. Restore backup")
        print("4. Show statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            description = input("Enter backup description (optional): ").strip()
            manager.create_backup(description)
            
        elif choice == "2":
            manager.list_backups()
            
        elif choice == "3":
            manager.list_backups()
            if manager.metadata['backups']:
                try:
                    idx = input("\nEnter backup number to restore (or press Enter for most recent): ").strip()
                    if idx:
                        idx = -int(idx)
                    else:
                        idx = -1
                    
                    restore_path = input("Restore to different location? (press Enter for original): ").strip()
                    restore_path = restore_path if restore_path else None
                    
                    confirm = input("This will overwrite existing files. Continue? (yes/no): ")
                    if confirm.lower() == 'yes':
                        manager.restore_backup(idx, restore_path)
                except (ValueError, IndexError):
                    print("Invalid backup number.")
            
        elif choice == "4":
            manager.get_backup_statistics()
            
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


