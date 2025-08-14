import pandas as pd
import re

# Core DSL Classes
class Rule:
    def __init__(self, name, condition, error_msg):
        self.name = name
        self.condition = condition
        self.error_msg = error_msg
    
    def check(self, df):
        violations = df[~self.condition(df)]
        if not violations.empty:
            return {
                'rule': self.name,
                'message': self.error_msg,
                'violations': len(violations),
                'sample_rows': violations.head(3).index.tolist()
            }
        return None

class DataValidator:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        self.rules.append(rule)
        return self
    
    def validate(self, df):
        results = []
        for rule in self.rules:
            violation = rule.check(df)
            if violation:
                results.append(violation)
        return results

# Helper Functions
def not_null(column):
    """Checks that column values are not null/NaN"""
    return lambda df: df[column].notna()

def unique_values(column):
    """Checks that all values in column are unique"""
    return lambda df: ~df.duplicated(subset=[column])

def between(column, min_val, max_val):
    """Checks that column values fall within specified range"""
    return lambda df: df[column].between(min_val, max_val)

def matches_pattern(column, pattern):
    """Checks that column values match regex pattern"""
    return lambda df: df[column].str.match(pattern, na=False)

def valid_date_format(column, date_format='%Y-%m-%d'):
    """Checks that column values are valid dates in specified format"""
    def check_dates(df):
        try:
            parsed_dates = pd.to_datetime(df[column], format=date_format, errors='coerce')
            return df[column].notna() & parsed_dates.notna()
        except:
            return pd.Series([False] * len(df), index=df.index)
    return check_dates

# Advanced Validation Functions
def high_spender_email_required(df):
    """High spending customers must have valid email addresses"""
    high_spenders = df['total_spent'] > 500
    has_valid_email = df['email'].str.contains('@', na=False)
    return ~high_spenders | has_valid_email

def within_standard_deviations(column, std_devs=3):
    """Checks for statistical outliers beyond specified standard deviations"""
    return lambda df: abs(df[column] - df[column].mean()) <= std_devs * df[column].std()

def foreign_key_exists(column, reference_df, reference_column):
    """Checks referential integrity against another DataFrame"""
    return lambda df: df[column].isin(reference_df[reference_column])

# Production Integration
def validate_dataframe(validator):
    """Decorator that validates DataFrame before processing"""
    def decorator(func):
        def wrapper(df, *args, **kwargs):
            issues = validator.validate(df)
            if issues:
                error_details = [f"{issue['rule']}: {issue['violations']} violations" for issue in issues]
                raise ValueError(f"Data validation failed: {'; '.join(error_details)}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

def validate_with_sampling(df, validator, sample_size=50000):
    """Validate large datasets using sampling"""
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
        return validator.validate(sample_df)
    return validator.validate(df)

# Sample Data
customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 103, 105],
    'email': ['john@gmail.com', 'invalid-email', '', 'sarah@yahoo.com', 'mike@domain.co'],
    'age': [25, -5, 35, 200, 28],
    'total_spent': [250.50, 1200.00, 0.00, -50.00, 899.99],
    'join_date': ['2023-01-15', '2023-13-45', '2023-02-20', '2023-02-20', '']
})

# Build Validator
validator = DataValidator()

validator.add_rule(Rule(
    "Unique customer IDs", 
    unique_values('customer_id'),
    "Customer IDs must be unique across all records"
))

validator.add_rule(Rule(
    "Valid email format",
    matches_pattern('email', r'^[^@\s]+@[^@\s]+\.[^@\s]+$'),
    "Email addresses must contain @ symbol and domain"
))

validator.add_rule(Rule(
    "Reasonable customer age",
    between('age', 13, 120),
    "Customer age must be between 13 and 120 years"
))

validator.add_rule(Rule(
    "Non-negative spending",
    lambda df: df['total_spent'] >= 0,
    "Total spending amount cannot be negative"
))

validator.add_rule(Rule(
    "High spenders need valid email",
    high_spender_email_required,
    "Customers spending over $500 must have valid email addresses"
))

validator.add_rule(Rule(
    "Valid join dates",
    valid_date_format('join_date'),
    "Join dates must follow YYYY-MM-DD format"
))

# Run Validation
if __name__ == "__main__":
    print("Customer Dataset:")
    print(customers)
    print("\n" + "="*50 + "\n")
    
    print("Validation Results:")
    issues = validator.validate(customers)
    
    if issues:
        for issue in issues:
            print(f"❌ Rule: {issue['rule']}")
            print(f"   Problem: {issue['message']}")
            print(f"   Violations: {issue['violations']}")
            print(f"   Affected rows: {issue['sample_rows']}")
            print()
    else:
        print("✅ All validation rules passed!")
    
    print("\n" + "="*50 + "\n")
    
    # Example with decorator
    @validate_dataframe(validator)
    def process_customer_data(df):
        return df.groupby('age').agg({'total_spent': 'sum'})
    
    print("Attempting to process data with validation decorator:")
    try:
        result = process_customer_data(customers)
        print("Processing successful:")
        print(result)
    except ValueError as e:
        print(f"Processing blocked: {e}")
