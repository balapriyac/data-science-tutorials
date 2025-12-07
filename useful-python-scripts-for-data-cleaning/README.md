# 5 Useful Python Scripts for Data Cleaning

Automate the most tedious data cleaning tasks with these 5 Python scripts.

## Scripts Included

1. **Missing Value Handler** - Intelligent imputation based on data type and patterns
2. **Duplicate Detector and Resolver** - Finds exact and fuzzy duplicates
3. **Data Type Fixer and Standardizer** - Auto-converts to proper types
4. **Outlier Detector and Treatment** - Multiple detection methods with treatment options
5. **Text Data Cleaner and Normalizer** - Cleans and standardizes text fields

---

## Dependencies

```bash
pip install pandas numpy scikit-learn
```


## Quick Start Examples

### 1. Missing Value Handler
```python
from missing_value_handler import MissingValueHandler
import pandas as pd

df = pd.read_csv('data.csv')

# Initialize handler
handler = MissingValueHandler(df)

# Analyze missing patterns
analysis = handler.analyze()
print(analysis)

# Handle missing values automatically
cleaned_df = handler.handle(
    default_numeric='median',
    default_categorical='mode'
)

# Or specify strategies per column
cleaned_df = handler.handle(
    strategies={
        'age': 'median',
        'name': 'unknown',
        'date': 'interpolate'
    }
)

# Get report of what was done
report = handler.get_report()
print(report)
```

### 2. Duplicate Detector and Resolver
```python
from duplicate_detector import DuplicateDetector

df = pd.read_csv('data.csv')
detector = DuplicateDetector(df)

# Find exact duplicates
exact_dups = detector.find_exact_duplicates(subset=['email', 'phone'])
print(f"Found {len(exact_dups)} exact duplicates")

# Find fuzzy duplicates (similar names)
fuzzy_dups = detector.find_fuzzy_duplicates(
    match_columns=['name'],
    threshold=0.85,
    method='jaro_winkler'
)

# Get duplicate report
report = detector.get_duplicate_report()
print(report)

# Resolve duplicates (keep most complete record)
resolved_df = detector.resolve_duplicates(
    survivorship='most_complete'
)
```

### 3. Data Type Fixer and Standardizer
```python
from datatype_fixer import DataTypeFixer

df = pd.read_csv('data.csv')
fixer = DataTypeFixer(df)

# Analyze what types should be
analysis = fixer.infer_types()
print(analysis)

# Auto-fix all types
fixed_df = fixer.fix_types(
    auto_detect=True,
    coerce_errors=True
)

# Or specify types manually
fixed_df = fixer.fix_types(
    type_mapping={
        'price': 'numeric',
        'date': 'datetime',
        'is_active': 'boolean'
    }
)

# Get conversion report
report = fixer.get_report()
print(report)
```

### 4. Outlier Detector and Treatment
```python
from outlier_detector import OutlierDetector

df = pd.read_csv('data.csv')
detector = OutlierDetector(df)

# Detect outliers using IQR method
outliers = detector.detect(
    method='iqr',
    threshold=1.5
)
print(outliers)

# Treat outliers by capping
treated_df = detector.treat(
    strategy='cap'  # Options: 'remove', 'cap', 'winsorize', 'flag'
)

# Get outlier report
report = detector.get_report()
print(report)

# Get outliers for specific column
age_outliers = detector.get_outliers('age')
print(age_outliers)
```

### 5. Text Data Cleaner and Normalizer
```python
from text_cleaner import TextCleaner

df = pd.read_csv('data.csv')
cleaner = TextCleaner(df)

# Clean all columns by type
cleaned_df = cleaner.clean_all({
    'customer_name': 'name',
    'address': 'address',
    'description': 'description',
    'product_code': 'code'
})

# Or clean individual columns with custom pipeline
cleaned_df['email'] = cleaner.clean_custom(
    'email',
    operations=['strip_whitespace', 'lowercase', 'remove_special_chars']
)

# Get cleaning report
report = cleaner.get_report()
print(report)
```

---

## 🔄 Complete Cleaning Pipeline

```python
import pandas as pd
from missing_value_handler import MissingValueHandler
from duplicate_detector import DuplicateDetector
from datatype_fixer import DataTypeFixer
from outlier_detector import OutlierDetector
from text_cleaner import TextCleaner

# Load data
df = pd.read_csv('raw_data.csv')
print(f"Original shape: {df.shape}")

# Step 1: Fix data types
print("\n1. Fixing data types...")
fixer = DataTypeFixer(df)
df = fixer.fix_types(auto_detect=True)

# Step 2: Clean text fields
print("2. Cleaning text fields...")
cleaner = TextCleaner(df)
df = cleaner.clean_all({
    'name': 'name',
    'address': 'address',
    'notes': 'description'
})

# Step 3: Remove duplicates
print("3. Removing duplicates...")
detector = DuplicateDetector(df)
detector.find_exact_duplicates()
detector.find_fuzzy_duplicates(match_columns=['name'], threshold=0.9)
df = detector.resolve_duplicates(survivorship='most_complete')

# Step 4: Handle outliers
print("4. Handling outliers...")
outlier_detector = OutlierDetector(df)
outlier_detector.detect(method='iqr')
df = outlier_detector.treat(strategy='cap')

# Step 5: Handle missing values
print("5. Handling missing values...")
handler = MissingValueHandler(df)
df = handler.handle(default_numeric='median', default_categorical='mode')

print(f"\nFinal shape: {df.shape}")

# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved!")
```

---
