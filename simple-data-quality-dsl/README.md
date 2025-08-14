# Data Quality DSL

A lightweight Python Domain-Specific Language (DSL) for data validation that reads like English but runs with pandas performance.


## Quick Start

```python
import pandas as pd
from data_quality_dsl import DataValidator, Rule, between, matches_pattern

# Sample data
df = pd.DataFrame({
    'user_id': [1, 2, 2, 3],
    'email': ['user@test.com', 'invalid', 'user@real.com', ''],
    'age': [25, -5, 30, 150]
})

# Build validator
validator = DataValidator()
validator.add_rule(Rule("Unique users", unique_values('user_id'), "User IDs must be unique"))
validator.add_rule(Rule("Valid emails", matches_pattern('email', r'^[^@]+@[^@]+\.[^@]+$'), "Invalid email format"))
validator.add_rule(Rule("Reasonable ages", between('age', 0, 120), "Age must be 0-120"))

# Run validation
issues = validator.validate(df)
for issue in issues:
    print(f"❌ {issue['rule']}: {issue['violations']} violations")
```

## Features

### Core Components
- **Rule**: Wraps validation logic with descriptive names and error messages
- **DataValidator**: Manages collections of rules and executes validation
- **Helper Functions**: Common validation patterns (`between`, `not_null`, `unique_values`, etc.)

### Built-in Validations
- Range checks (`between`)
- Null/missing value detection (`not_null`)
- Uniqueness validation (`unique_values`) 
- Regex pattern matching (`matches_pattern`)
- Date format validation (`valid_date_format`)
- Statistical outlier detection (`within_standard_deviations`)
- Cross-table referential integrity (`foreign_key_exists`)

### Advanced Features
- **Cross-column validation**: Rules that involve multiple columns
- **Custom business logic**: Write your own validation functions
- **Production decorators**: Validate data before processing
- **Sampling support**: Handle large datasets efficiently
- **Detailed error reporting**: Get violation counts and sample rows

## Installation

Just copy `data_quality_dsl.py` to your project. No external dependencies beyond pandas.

```bash
pip install pandas  # Only dependency
```

## Usage Examples

### Basic Validation
```python
validator = DataValidator()
validator.add_rule(Rule("No nulls", not_null('name'), "Name cannot be empty"))
validator.add_rule(Rule("Valid range", between('score', 0, 100), "Score must be 0-100"))

issues = validator.validate(df)
```

### Cross-Column Rules
```python
def price_quantity_consistent(df):
    return df['total'] == (df['price'] * df['quantity'])

validator.add_rule(Rule("Math checks out", price_quantity_consistent, "Total must equal price × quantity"))
```

### Production Integration
```python
@validate_dataframe(validator)
def process_data(df):
    return df.groupby('category').sum()

# Validation runs automatically before processing
result = process_data(df)  # Raises ValueError if validation fails
```

### Large Dataset Handling
```python
# Validate sample for performance
issues = validate_with_sampling(large_df, validator, sample_size=10000)
```

## API Reference

### Core Classes

#### `Rule(name, condition, error_msg)`
- `name`: Descriptive rule name
- `condition`: Function that takes DataFrame and returns boolean Series
- `error_msg`: Error message for violations

#### `DataValidator()`
- `add_rule(rule)`: Add validation rule (chainable)
- `validate(df)`: Run all rules and return violation list

### Helper Functions

#### Basic Checks
- `not_null(column)`: Check for non-null values
- `unique_values(column)`: Check for uniqueness
- `between(column, min_val, max_val)`: Range validation

#### Pattern Matching
- `matches_pattern(column, regex)`: Regex validation
- `valid_date_format(column, format)`: Date format validation

#### More Features
- `within_standard_deviations(column, std_devs)`: Outlier detection
- `foreign_key_exists(column, ref_df, ref_column)`: Referential integrity

### Violation Report Format
```python
{
    'rule': 'Rule Name',
    'message': 'Error description',
    'violations': 5,
    'sample_rows': [2, 7, 12]
}
```

