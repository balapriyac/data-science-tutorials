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

