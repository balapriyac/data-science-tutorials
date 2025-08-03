# Essential Data Scripts for Busy Professionals

Five Python scripts useful for data science tasks. Stop wrestling with repetitive tasks and focus on actual analysis.

## Usage Examples

### Quick Data Quality Check
```python
from data_quality_checker import data_quality_report
import pandas as pd

df = pd.read_csv('your_data.csv')
data_quality_report(df)  # Generates comprehensive quality report
```

### Merge All Files in a Folder
```python
from smart_file_merger import smart_file_merger

# Combines all CSV, Excel, and JSON files automatically
merged_df = smart_file_merger('/path/to/data/folder')
```

### Instant Dataset Profile
```python
from dataset_profiler import quick_profile

df = pd.read_csv('sales_data.csv')
quick_profile(df)  # Shows stats, correlations, and insights
```

### Version Control for Data
```python
from data_version_manager import DataVersionManager

vm = DataVersionManager("my_project")
vm.save_version(df, "Initial clean dataset")
# Make changes...
vm.save_version(df_processed, "After removing outliers")
vm.list_versions()  # See all saved versions
```

### Export to Multiple Formats
```python
from multi_format_exporter import DataExporter

exporter = DataExporter(df, "final_analysis")
exporter.export_all()  # Creates Excel, JSON, CSV, and SQLite files
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl

## Installation

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```
