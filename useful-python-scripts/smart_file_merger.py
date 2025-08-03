import pandas as pd
import os
from pathlib import Path

def smart_file_merger(folder_path, output_filename="merged_data.csv"):
    """
    Merge all data files in a folder into one DataFrame
    Supports CSV, Excel (.xlsx, .xls), and JSON files
    """
    folder = Path(folder_path)
    all_dataframes = []
    processed_files = []
    
    # Supported file extensions
    supported_formats = {'.csv', '.xlsx', '.xls', '.json'}
    
    print("Scanning for data files...")
    
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in supported_formats:
            print(f"Processing: {file_path.name}")
            
            try:
                # Read based on file extension
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif file_path.suffix.lower() == '.json':
                    df = pd.read_json(file_path)
                
                # Add source file column
                df['source_file'] = file_path.name
                all_dataframes.append(df)
                processed_files.append(file_path.name)
                
            except Exception as e:
                print(f"Error reading {file_path.name}: {str(e)}")
                continue
    
    if not all_dataframes:
        print("No compatible data files found!")
        return None
    
    # Merge all dataframes
    print(f"\nMerging {len(all_dataframes)} files...")
    merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Save merged file
    output_path = folder / output_filename
    merged_df.to_csv(output_path, index=False)
    
    print(f"✓ Successfully merged {len(processed_files)} files")
    print(f"✓ Total rows: {len(merged_df)}")
    print(f"✓ Output saved: {output_path}")
    
    return merged_df

# Usage example:
# merged_data = smart_file_merger('/path/to/your/data/folder')
