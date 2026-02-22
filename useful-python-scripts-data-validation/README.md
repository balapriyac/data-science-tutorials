# Advanced Data Validation Scripts

Five Python scripts for validating data beyond simple quality checks. 

## Install Dependencies


Or install individually:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## Scripts

### 1. Time-Series Continuity Validator

Validates temporal integrity and pattern consistency in time-series data.

#### Features
- Detects missing timestamps in expected sequences
- Identifies gaps and overlaps in time-series data
- Validates sequence ordering (detects backward jumps)
- Checks velocity constraints (rate of change limits)
- Detects seasonal pattern violations
- Supports multiple frequencies (minute, hour, day, week, month)

#### Command Line Usage
```bash
# Basic validation
python timeseries_validator.py sensor_data.csv --timestamp-column timestamp

# With expected frequency
python timeseries_validator.py sales_data.csv -t order_date -f D

# Export results
python timeseries_validator.py data.csv -t timestamp -e report.json
```

#### Programmatic Usage
```python
from timeseries_validator import TimeSeriesValidator

# Define velocity rules (max change per hour)
velocity_rules = {
    'temperature': 10,  # Max 10 degree change per hour
    'pressure': 5,      # Max 5 unit change per hour
}

validator = TimeSeriesValidator(
    filepath='sensor_data.csv',
    timestamp_column='timestamp',
    expected_frequency='H',  # Hourly data
    velocity_rules=velocity_rules
)

results = validator.analyze_all()
validator.print_report()
validator.export_report('timeseries_report.json')
```

#### Output
```json
{
  "metadata": {
    "total_records": 8760,
    "date_range": {
      "start": "2024-01-01T00:00:00",
      "end": "2024-12-31T23:00:00"
    }
  },
  "gap_analysis": {
    "expected_frequency": "H",
    "completeness_percentage": 98.5,
    "missing_records": 132,
    "total_gaps": 15
  },
  "sequence_analysis": {
    "total_violations": 3,
    "backward_jumps": 2,
    "duplicate_timestamps": 1
  },
  "velocity_analysis": {
    "total_violations": 5
  }
}
```

---

### 2. Semantic Validity Checker

Validates data against complex business rules and domain knowledge.

#### Features
- Age/education consistency checks
- Date progression validation (start before end)
- State machine transition validation
- Mutually exclusive field validation
- Conditional requirement checks (if X then Y)
- Custom business constraint expressions

#### Command Line Usage
```bash
# Validate against rules file
python semantic_validator.py employee_data.csv --rules rules.json

# Export violations
python semantic_validator.py data.csv -r rules.json -e violations.json
```

#### Rules Configuration
Create a `rules.json` file:

```json
{
  "age_education_rules": [
    {
      "age_column": "age",
      "education_column": "education_level",
      "education_level": "PhD",
      "min_age": 24
    },
    {
      "age_column": "age",
      "education_column": "education_level",
      "education_level": "Masters",
      "min_age": 22
    }
  ],
  "date_progression_rules": [
    {
      "name": "employment_period",
      "earlier_date": "hire_date",
      "later_date": "termination_date",
      "allow_equal": false
    },
    {
      "name": "order_fulfillment",
      "earlier_date": "order_date",
      "later_date": "ship_date",
      "allow_equal": true
    }
  ],
  "state_transition_rules": [
    {
      "state_column": "order_status",
      "sequence_column": "status_timestamp",
      "valid_transitions": {
        "pending": ["confirmed", "cancelled"],
        "confirmed": ["processing", "cancelled"],
        "processing": ["shipped", "cancelled"],
        "shipped": ["delivered", "returned"],
        "delivered": ["returned"],
        "cancelled": [],
        "returned": []
      }
    }
  ],
  "mutually_exclusive_rules": [
    {
      "name": "contact_method",
      "field_groups": ["email", "phone", "mailing_address"]
    }
  ],
  "conditional_requirement_rules": [
    {
      "name": "active_employee_no_termination",
      "if_column": "employment_status",
      "if_value": "active",
      "then_column": "termination_date",
      "then_check": "is_null"
    },
    {
      "name": "completed_order_has_payment",
      "if_column": "order_status",
      "if_value": "completed",
      "then_column": "payment_date",
      "then_check": "not_null"
    }
  ],
  "business_constraint_rules": [
    {
      "name": "discount_percentage",
      "columns": ["original_price", "discount_percentage", "final_price"],
      "expression": "abs(original_price * (1 - discount_percentage/100) - final_price) < 0.01",
      "severity": "high"
    }
  ]
}
```

#### Programmatic Usage
```python
from semantic_validator import SemanticValidator

rules = {
    'age_education_rules': [
        {
            'age_column': 'age',
            'education_column': 'degree',
            'education_level': 'PhD',
            'min_age': 24
        }
    ],
    'date_progression_rules': [
        {
            'name': 'project_timeline',
            'earlier_date': 'start_date',
            'later_date': 'end_date',
            'allow_equal': False
        }
    ]
}

validator = SemanticValidator('employee_data.csv', rules_config=rules)
results = validator.analyze_all()
validator.print_report(results)

# Access specific violation types
high_severity = [v for v in results['violations'] if v['severity'] == 'high']
print(f"High severity violations: {len(high_severity)}")
```

---

### 3. Data Drift Detector

Monitors datasets for structural and statistical drift over time.

#### Features
- Schema drift detection (new/removed columns, type changes)
- Statistical distribution drift (mean, std, range changes)
- Categorical value drift (new/removed categories)
- Null pattern drift
- Drift scoring (0-100 scale)
- Baseline profile creation and comparison

#### Command Line Usage
```bash
# Create baseline profile
python drift_detector.py current_data.csv --export-profile baseline.json

# Compare against baseline
python drift_detector.py new_data.csv --baseline baseline.json

# Compare two datasets
python drift_detector.py current_data.csv --baseline old_data.csv

# Export drift report
python drift_detector.py new_data.csv -b baseline.json -r drift_report.json
```

#### Programmatic Usage
```python
from drift_detector import DataDriftDetector
import json

# First run: Create baseline
detector = DataDriftDetector(current_filepath='week1_data.csv')
baseline = detector.analyze_all()

# Save baseline for future comparisons
with open('baseline.json', 'w') as f:
    json.dump(baseline['profile'], f, default=str)

# Later: Compare new data against baseline
with open('baseline.json', 'r') as f:
    baseline_profile = json.load(f)

detector = DataDriftDetector(
    current_filepath='week2_data.csv',
    baseline_profile=baseline_profile
)

results = detector.analyze_all()
detector.print_report()

# Check drift score
if results['drift_score'] > 30:
    print("WARNING: Significant drift detected!")
    
    # Examine specific drift types
    schema_drift = results['schema_drift']
    if schema_drift['new_columns']:
        print(f"New columns: {schema_drift['new_columns']}")
```

#### Drift Interpretation
- **Drift Score < 10**: Minimal drift, data structure stable
- **Drift Score 10-30**: Low drift, monitor but acceptable
- **Drift Score 30-60**: Moderate drift, investigate changes
- **Drift Score > 60**: Significant drift, review pipeline

---

### 4. Hierarchical Relationship Validator

Validates graph and tree structures in relational data.

#### Features
- Circular reference detection
- Self-reference detection
- Orphaned node detection (missing parents)
- Multiple parent detection (tree violations)
- Depth constraint validation
- Disconnected subgraph identification
- Hierarchy statistics

#### Command Line Usage
```bash
# Basic validation
python hierarchy_validator.py org_chart.csv --node-id employee_id --parent-id manager_id

# With depth constraint
python hierarchy_validator.py categories.csv -n category_id -p parent_category_id -d 5

# Export results
python hierarchy_validator.py data.csv -n id -p parent_id -e hierarchy_report.json
```

#### Programmatic Usage
```python
from hierarchy_validator import HierarchyValidator

validator = HierarchyValidator(
    filepath='org_structure.csv',
    node_id_col='employee_id',
    parent_id_col='manager_id',
    max_depth=7  # Maximum 7 levels of management
)

results = validator.analyze_all()
validator.print_report()

# Check specific violation types
violations = results['violations']

if violations['circular_references']:
    print("CRITICAL: Circular references found!")
    for cycle in violations['circular_references']:
        print(f"Cycle: {' -> '.join(cycle['cycle'])}")

if violations['orphaned_nodes']:
    print(f"Found {len(violations['orphaned_nodes'])} orphaned nodes")

# Check connectivity
if not results['connectivity']['is_connected']:
    print(f"WARNING: {results['connectivity']['total_components']} disconnected components")
```

---

