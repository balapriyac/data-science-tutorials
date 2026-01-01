# Data Quality Validation Scripts

A comprehensive collection of Python scripts for automated data quality validation. These tools help identify and report on common data quality issues before they impact your analysis or production systems.

## Overview

These scripts provide automated validation for:
- **Missing Data**: Identify patterns in incomplete records
- **Data Types**: Ensure fields contain expected formats
- **Duplicates**: Find exact and fuzzy duplicate records
- **Outliers**: Detect statistical and domain-specific anomalies
- **Consistency**: Validate cross-field logical relationships

Each script can be run standalone or integrated into data pipelines for continuous validation.

## 📦 Dependencies

### Core Dependencies
```
pandas          # Data manipulation and analysis
numpy          # Numerical operations
matplotlib      # Plotting and visualization
seaborn        # Statistical visualizations
scipy          # Scientific computing (for outlier detection)
```

### Optional Dependencies
```
openpyxl>=3.0.0        # Excel file support (.xlsx)
xlrd>=2.0.0            # Excel file support (.xls)
python-Levenshtein>=0.12.0  # Faster fuzzy string matching (optional)
```

### Installing All Dependencies
Create a `requirements.txt` file:
```txt
pandas
numpy
matplotlib
seaborn
scipy
openpyxl
xlrd
```

Then install:
```bash
pip install -r requirements.txt
```

## Scripts

### 1. Missing Data Analyzer
**File**: `missing_data_analyzer.py`

Comprehensively analyzes missing data patterns across your dataset.

**Features**:
- Detects various representations of missing values (None, NaN, "N/A", etc.)
- Calculates completeness scores by column and row
- Identifies correlated missingness patterns
- Generates visual reports with heatmaps and charts
- Provides actionable recommendations

**Basic Usage**:
```bash
python missing_data_analyzer.py data.csv
```

**Output**: Console report, JSON statistics, PNG visualization

---

### 2. Data Type Validator
**File**: `data_type_validator.py`

Validates that each column contains the expected data type and format.

**Supported Types**:
- Integer, Float, String, Boolean
- Date/DateTime (with format specification)
- Email, URL, Phone number
- Categorical (with allowed values)

**Basic Usage**:
```bash
python data_type_validator.py data.csv schema.json
```

**Schema Example**:
```json
{
  "user_id": {"type": "integer", "nullable": false, "range": [1, 1000000]},
  "email": {"type": "email", "nullable": false},
  "age": {"type": "integer", "range": [0, 120]},
  "status": {"type": "categorical", "values": ["active", "inactive"]}
}
```

**Output**: Console report, JSON violations file

---

### 3. Duplicate Record Detector
**File**: `duplicate_detector.py`

Identifies exact and fuzzy duplicate records using multiple detection strategies.

**Features**:
- Exact duplicate detection (hash-based)
- Fuzzy matching using string similarity (Levenshtein distance)
- Partial duplicate detection on specified columns
- Confidence scoring for fuzzy matches
- Safe deduplication with rollback capability

**Basic Usage**:
```bash
# Find exact duplicates
python duplicate_detector.py data.csv

# Include fuzzy matching
python duplicate_detector.py data.csv --fuzzy

# Check specific columns only
python duplicate_detector.py data.csv --columns=name,email
```

**Output**: Console report, JSON duplicate groups, optional deduplicated CSV

---

### 4. Outlier Detection Engine
**File**: `outlier_detector.py`

Detects statistical and domain-specific outliers in numeric data.

**Detection Methods**:
- **IQR (Interquartile Range)**: Default method, robust to skewed distributions
- **Z-Score**: Suitable for normally distributed data
- **Modified Z-Score**: Uses median absolute deviation (MAD)
- **Domain Rules**: Custom min/max constraints

**Basic Usage**:
```bash
# Default (IQR method)
python outlier_detector.py data.csv

# Use Z-score method with threshold 3
python outlier_detector.py data.csv --method zscore --threshold 3
```

**Domain Rules** (edit script or pass as config):
```python
domain_rules = {
    'age': {'min': 0, 'max': 120},
    'price': {'min': 0},
    'percentage': {'min': 0, 'max': 100}
}
```

**Output**: Console report, JSON outlier details, PNG box plot visualizations

---

### 5. Cross-Field Consistency Checker
**File**: `consistency_checker.py`

Validates logical relationships between fields based on business rules.

**Rule Types**:
- **Temporal**: Start dates before end dates
- **Mathematical**: Total = subtotal + tax
- **Referential**: Foreign keys exist in parent tables
- **Conditional**: If status="shipped", then tracking_number is not null
- **Custom**: Python functions for complex logic

**Basic Usage**:
```bash
python consistency_checker.py data.csv rules.json
```

**Rules Example**:
```json
{
  "temporal_rules": [
    {"start": "start_date", "end": "end_date", "name": "date_sequence"}
  ],
  "mathematical_rules": [
    {
      "columns": ["subtotal", "tax", "total"],
      "formula": "abs((subtotal + tax) - total) < 0.01",
      "name": "total_calculation"
    }
  ],
  "conditional_rules": [
    {
      "if": {"column": "status", "equals": "shipped"},
      "then": {"column": "tracking_number", "not_null": true},
      "name": "shipped_has_tracking"
    }
  ]
}
```

**Output**: Console report, JSON violations file

---


## 🎯 Quick Start

### Example 1: Complete Data Quality Check
```bash
# Check for missing data
python missing_data_analyzer.py sales_data.csv

# Validate data types
python data_type_validator.py sales_data.csv schema.json

# Find duplicates
python duplicate_detector.py sales_data.csv --fuzzy

# Detect outliers
python outlier_detector.py sales_data.csv

# Check consistency
python consistency_checker.py sales_data.csv rules.json
```

### Example 2: Python Integration
```python
from missing_data_analyzer import MissingDataAnalyzer
from data_type_validator import DataTypeValidator

# Analyze missing data
analyzer = MissingDataAnalyzer('data.csv')
report = analyzer.analyze()
analyzer.print_report()

# Validate types
schema = {
    'id': {'type': 'integer', 'nullable': False},
    'email': {'type': 'email', 'nullable': False}
}
validator = DataTypeValidator('data.csv', schema)
results = validator.validate()
validator.print_report()
```

## Detailed Usage

### Missing Data Analyzer

**Command Line Options**:
```bash
python missing_data_analyzer.py <filepath>
```

**Programmatic Usage**:
```python
analyzer = MissingDataAnalyzer(
    filepath='data.csv',
    missing_indicators=['N/A', 'Unknown', '-']  # Custom missing indicators
)

# Run analysis
report = analyzer.analyze()

# Generate outputs
analyzer.print_report()
analyzer.export_report('report.json')
analyzer.generate_visualization('missing_data.png')

# Access results
print(f"Overall completeness: {report['overall']['completeness_percentage']}%")
print(f"Columns with issues: {len([c for c in report['columns'] if c['status'] != 'Complete'])}")
```

---

### Data Type Validator

**Schema Configuration**:
```python
schema = {
    'column_name': {
        'type': 'integer',           # Required
        'nullable': False,            # Optional (default: True)
        'range': [0, 1000],          # Optional (for numeric types)
        'pattern': r'^\d{3}-\d{3}$', # Optional (for string types)
        'values': ['A', 'B', 'C'],   # Optional (for categorical)
        'format': '%Y-%m-%d'         # Optional (for date types)
    }
}
```

**Available Types**:
- `integer`, `float`, `string`, `boolean`
- `date` (requires `format` parameter)
- `email`, `url`, `phone`
- `categorical` (requires `values` parameter)

**Programmatic Usage**:
```python
validator = DataTypeValidator('data.csv', schema)
results = validator.validate()

if results['validation_passed']:
    print("All validations passed!")
else:
    print(f"Found {results['total_violations']} violations")
    validator.export_violations('violations.json')
```

---

### Duplicate Record Detector

**Command Line Options**:
```bash
# Basic duplicate detection
python duplicate_detector.py data.csv

# Include fuzzy matching (slower, more thorough)
python duplicate_detector.py data.csv --fuzzy

# Check specific columns only
python duplicate_detector.py data.csv --columns=name,email,phone
```

**Programmatic Usage**:
```python
detector = DuplicateDetector(
    filepath='data.csv',
    key_columns=['name', 'email'],  # Or None for all columns
    fuzzy_threshold=0.85             # Similarity threshold (0-1)
)

# Find different types of duplicates
exact_dupes = detector.find_exact_duplicates()
fuzzy_dupes = detector.find_fuzzy_duplicates()
partial_dupes = detector.find_partial_duplicates(['email'])

# Or run complete analysis
stats = detector.analyze_all(
    include_fuzzy=True,
    fuzzy_columns=['name', 'address'],
    partial_key_columns=[['email'], ['phone']]
)

# Get recommendations
recommendations = detector.get_deduplication_recommendations()

# Export deduplicated data
detector.export_deduplicated('clean_data.csv', strategy='keep_first')
```

---

### Outlier Detection Engine

**Command Line Options**:
```bash
# Use IQR method (default)
python outlier_detector.py data.csv

# Use Z-score method
python outlier_detector.py data.csv --method zscore --threshold 3

# Use modified Z-score
python outlier_detector.py data.csv --method modified_zscore --threshold 3.5
```

**Programmatic Usage**:
```python
# Define domain rules
domain_rules = {
    'age': {'min': 0, 'max': 120},
    'price': {'min': 0},
    'temperature': {'min': -50, 'max': 50},
    'percentage': {
        'min': 0, 
        'max': 100,
        'validator': lambda x: 0 <= x <= 100  # Custom function
    }
}

detector = OutlierDetector('data.csv', domain_rules=domain_rules)

# Run detection
stats = detector.analyze_all(
    statistical_method='iqr',  # or 'zscore', 'modified_zscore'
    threshold=1.5               # IQR multiplier or z-score threshold
)

# Generate outputs
detector.print_report()
detector.export_outliers('outliers.json')
detector.generate_visualization('outliers.png')

# Export cleaned data
detector.export_cleaned_data('cleaned.csv', strategy='remove')  # or 'flag'
```

---

### Cross-Field Consistency Checker

**Rules Configuration**:
```json
{
  "temporal_rules": [
    {
      "start": "hire_date",
      "end": "termination_date",
      "name": "employment_period"
    }
  ],
  "mathematical_rules": [
    {
      "columns": ["hours", "rate", "total_pay"],
      "formula": "abs((hours * rate) - total_pay) < 0.01",
      "tolerance": 0.01,
      "name": "pay_calculation"
    }
  ],
  "referential_rules": [
    {
      "child": "department_id",
      "parent_table": "departments.csv",
      "parent_key": "id",
      "name": "valid_department"
    }
  ],
  "conditional_rules": [
    {
      "if": {"column": "employment_status", "equals": "active"},
      "then": {"column": "termination_date", "not_null": false},
      "name": "active_not_terminated"
    }
  ],
  "custom_rules": [
    {
      "name": "custom_validation",
      "columns": ["value1", "value2"],
      "function": "lambda row: row['value1'] + row['value2'] > 100"
    }
  ]
}
```

**Programmatic Usage**:
```python
checker = ConsistencyChecker('data.csv', rules_config='rules.json')

# Or define rules in code
rules = {
    'temporal_rules': [
        {'start': 'start_date', 'end': 'end_date', 'name': 'date_order'}
    ],
    'mathematical_rules': [
        {
            'columns': ['quantity', 'price', 'total'],
            'formula': 'abs((quantity * price) - total) < 0.01',
            'name': 'total_check'
        }
    ]
}

checker = ConsistencyChecker('data.csv', rules_config=rules)
results = checker.validate_all()

checker.print_report()
checker.export_violations('violations.json')
```
