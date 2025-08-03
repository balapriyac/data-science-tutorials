import pandas as pd
import os
import shutil
from datetime import datetime
import hashlib

class DataVersionManager:
    def __init__(self, project_name):
        self.project_name = project_name
        self.backup_dir = f"data_backups/{project_name}"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.log_file = f"{self.backup_dir}/version_log.txt"
    
    def get_file_hash(self, filepath):
        """Generate MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_version(self, df, description="Auto-save"):
        """Save a versioned copy of the DataFrame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.project_name}_v{timestamp}.csv"
        filepath = os.path.join(self.backup_dir, filename)
        
        # Save the dataframe
        df.to_csv(filepath, index=False)
        
        # Calculate file hash
        file_hash = self.get_file_hash(filepath)
        
        # Log the version
        log_entry = f"{timestamp},{filename},{len(df)},{df.shape[1]},{file_hash},{description}\n"
        
        # Create log header if file doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,filename,rows,columns,hash,description\n")
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(f"✓ Version saved: {filename}")
        print(f"✓ Rows: {len(df):,}, Columns: {df.shape[1]}")
        return filepath
    
    def list_versions(self):
        """Display all saved versions"""
        if not os.path.exists(self.log_file):
            print("No versions found.")
            return
        
        log_df = pd.read_csv(self.log_file)
        print(f"\nVersion History for '{self.project_name}':")
        print("-" * 80)
        
        for _, row in log_df.iterrows():
            print(f"Date: {row['timestamp']}")
            print(f"File: {row['filename']}")
            print(f"Size: {row['rows']:,} rows × {row['columns']} columns")
            print(f"Description: {row['description']}")
            print("-" * 40)
    
    def load_version(self, version_timestamp):
        """Load a specific version by timestamp"""
        filename = f"{self.project_name}_v{version_timestamp}.csv"
        filepath = os.path.join(self.backup_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✓ Loaded version: {filename}")
            return df
        else:
            print(f"Version {version_timestamp} not found.")
            return None
    
    def cleanup_old_versions(self, keep_last_n=5):
        """Keep only the last N versions to save space"""
        if not os.path.exists(self.log_file):
            return
        
        log_df = pd.read_csv(self.log_file)
        if len(log_df) <= keep_last_n:
            print("No cleanup needed.")
            return
        
        # Remove old files
        old_versions = log_df.iloc[:-keep_last_n]
        for _, row in old_versions.iterrows():
            old_file = os.path.join(self.backup_dir, row['filename'])
            if os.path.exists(old_file):
                os.remove(old_file)
        
        # Update log
        log_df.tail(keep_last_n).to_csv(self.log_file, index=False)
        print(f"✓ Cleaned up {len(old_versions)} old versions")

# Usage example:
# vm = DataVersionManager("sales_analysis")
# df = pd.read_csv('sales_data.csv')
# 
# # Save current version
# vm.save_version(df, "Initial data load")
# 
# # After making changes...
# df_cleaned = df.dropna()
# vm.save_version(df_cleaned, "Removed missing values")
# 
# # View all versions
# vm.list_versions()
