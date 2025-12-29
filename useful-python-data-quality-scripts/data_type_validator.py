"""
Data Type Validator
Validates data types and formats against expected schema
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import json

class DataTypeValidator:
    def __init__(self, filepath, schema_config):
        """
        Initialize validator with dataset and schema
        
        Args:
            filepath: Path to data file
            schema_config: Dict defining expected types for each column
                Example: {
                    'column_name': {
                        'type': 'integer|float|string|date|email|url|phone|boolean',
                        'nullable': True/False,
                        'range': [min, max],  # optional for numeric
                        'pattern': 'regex',    # optional for string
                        'values': ['allowed', 'values']  # optional for categorical
                    }
                }
        """
        self.filepath = Path(filepath)
        self.df = self._load_data()
        self.schema = schema_config
        self.violations = []
        self.summary = {}
        
    def _load_data(self):
        """Load data from file"""
        suffix = self.filepath.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(self.filepath)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(self.filepath)
        elif suffix == '.json':
            return pd.read_json(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def validate(self):
        """Run complete validation against schema"""
        print(f"Validating {len(self.df)} rows against schema...")
        
        for column, rules in self.schema.items():
            if column not in self.df.columns:
                self.violations.append({
                    'column': column,
                    'type': 'missing_column',
                    'message': f"Column '{column}' defined in schema but not found in data",
                    'severity': 'critical'
                })
                continue
            
            self._validate_column(column, rules)
        
        # Check for unexpected columns
        schema_cols = set(self.schema.keys())
        data_cols = set(self.df.columns)
        unexpected = data_cols - schema_cols
        
        if unexpected:
            for col in unexpected:
                self.violations.append({
                    'column': col,
                    'type': 'unexpected_column',
                    'message': f"Column '{col}' found in data but not defined in schema",
                    'severity': 'warning'
                })
        
        self._generate_summary()
        return self.summary
    
    def _validate_column(self, column, rules):
        """Validate a single column against its rules"""
        data_type = rules.get('type', 'string')
        nullable = rules.get('nullable', True)
        
        col_data = self.df[column]
        
        # Check for null values
        null_mask = col_data.isna()
        null_count = null_mask.sum()
        
        if null_count > 0 and not nullable:
            for idx in self.df[null_mask].index[:5]:  # Report first 5
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': None,
                    'type': 'null_violation',
                    'message': f"Null value in non-nullable column",
                    'severity': 'high'
                })
        
        # Validate non-null values based on type
        non_null_data = col_data[~null_mask]
        
        if data_type == 'integer':
            self._validate_integer(column, non_null_data, rules)
        elif data_type == 'float':
            self._validate_float(column, non_null_data, rules)
        elif data_type == 'string':
            self._validate_string(column, non_null_data, rules)
        elif data_type == 'date':
            self._validate_date(column, non_null_data, rules)
        elif data_type == 'email':
            self._validate_email(column, non_null_data)
        elif data_type == 'url':
            self._validate_url(column, non_null_data)
        elif data_type == 'phone':
            self._validate_phone(column, non_null_data)
        elif data_type == 'boolean':
            self._validate_boolean(column, non_null_data)
        elif data_type == 'categorical':
            self._validate_categorical(column, non_null_data, rules)
    
    def _validate_integer(self, column, data, rules):
        """Validate integer type"""
        for idx, value in data.items():
            try:
                int_val = int(value)
                # Check range if specified
                if 'range' in rules:
                    min_val, max_val = rules['range']
                    if int_val < min_val or int_val > max_val:
                        self.violations.append({
                            'column': column,
                            'row': int(idx),
                            'value': value,
                            'type': 'range_violation',
                            'message': f"Value {int_val} outside range [{min_val}, {max_val}]",
                            'severity': 'medium'
                        })
            except (ValueError, TypeError):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'type_violation',
                    'message': f"Expected integer, got '{value}' ({type(value).__name__})",
                    'severity': 'high'
                })
    
    def _validate_float(self, column, data, rules):
        """Validate float type"""
        for idx, value in data.items():
            try:
                float_val = float(value)
                if np.isnan(float_val) or np.isinf(float_val):
                    self.violations.append({
                        'column': column,
                        'row': int(idx),
                        'value': value,
                        'type': 'special_value',
                        'message': f"Invalid numeric value: {value}",
                        'severity': 'high'
                    })
                # Check range
                elif 'range' in rules:
                    min_val, max_val = rules['range']
                    if float_val < min_val or float_val > max_val:
                        self.violations.append({
                            'column': column,
                            'row': int(idx),
                            'value': value,
                            'type': 'range_violation',
                            'message': f"Value {float_val} outside range [{min_val}, {max_val}]",
                            'severity': 'medium'
                        })
            except (ValueError, TypeError):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'type_violation',
                    'message': f"Expected float, got '{value}' ({type(value).__name__})",
                    'severity': 'high'
                })
    
    def _validate_string(self, column, data, rules):
        """Validate string type and pattern"""
        pattern = rules.get('pattern')
        
        for idx, value in data.items():
            if not isinstance(value, str):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'type_violation',
                    'message': f"Expected string, got {type(value).__name__}",
                    'severity': 'high'
                })
            elif pattern and not re.match(pattern, str(value)):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'pattern_violation',
                    'message': f"Value doesn't match pattern: {pattern}",
                    'severity': 'medium'
                })
    
    def _validate_date(self, column, data, rules):
        """Validate date format"""
        date_format = rules.get('format', '%Y-%m-%d')
        
        for idx, value in data.items():
            try:
                if isinstance(value, (pd.Timestamp, datetime)):
                    continue
                datetime.strptime(str(value), date_format)
            except (ValueError, TypeError):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'format_violation',
                    'message': f"Invalid date format. Expected {date_format}",
                    'severity': 'high'
                })
    
    def _validate_email(self, column, data):
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for idx, value in data.items():
            if not re.match(email_pattern, str(value)):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'format_violation',
                    'message': f"Invalid email format",
                    'severity': 'high'
                })
    
    def _validate_url(self, column, data):
        """Validate URL format"""
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        for idx, value in data.items():
            if not re.match(url_pattern, str(value), re.IGNORECASE):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'format_violation',
                    'message': f"Invalid URL format",
                    'severity': 'high'
                })
    
    def _validate_phone(self, column, data):
        """Validate phone number format"""
        # Accepts formats: (123) 456-7890, 123-456-7890, 1234567890
        phone_pattern = r'^[\d\s\-\(\)\+]+$'
        
        for idx, value in data.items():
            cleaned = re.sub(r'[\s\-\(\)\+]', '', str(value))
            if not (10 <= len(cleaned) <= 15 and cleaned.isdigit()):
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'format_violation',
                    'message': f"Invalid phone number format",
                    'severity': 'medium'
                })
    
    def _validate_boolean(self, column, data):
        """Validate boolean type"""
        valid_bools = {True, False, 'true', 'false', 'True', 'False', 
                       'TRUE', 'FALSE', 1, 0, '1', '0', 'yes', 'no', 
                       'Yes', 'No', 'YES', 'NO'}
        
        for idx, value in data.items():
            if value not in valid_bools:
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'type_violation',
                    'message': f"Invalid boolean value: {value}",
                    'severity': 'high'
                })
    
    def _validate_categorical(self, column, data, rules):
        """Validate categorical values"""
        allowed_values = set(rules.get('values', []))
        
        for idx, value in data.items():
            if value not in allowed_values:
                self.violations.append({
                    'column': column,
                    'row': int(idx),
                    'value': value,
                    'type': 'invalid_category',
                    'message': f"Value '{value}' not in allowed set: {allowed_values}",
                    'severity': 'high'
                })
    
    def _generate_summary(self):
        """Generate validation summary"""
        total_violations = len(self.violations)
        
        # Count by severity
        severity_counts = {}
        for v in self.violations:
            sev = v['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Count by column
        column_counts = {}
        for v in self.violations:
            col = v['column']
            column_counts[col] = column_counts.get(col, 0) + 1
        
        # Count by type
        type_counts = {}
        for v in self.violations:
            vtype = v['type']
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        self.summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns_validated': len(self.schema),
            'total_violations': total_violations,
            'violations_by_severity': severity_counts,
            'violations_by_column': column_counts,
            'violations_by_type': type_counts,
            'validation_passed': total_violations == 0
        }
    
    def print_report(self):
        """Print validation report"""
        print("\n" + "="*70)
        print("DATA TYPE VALIDATION REPORT")
        print("="*70)
        print(f"Dataset: {self.filepath.name}")
        print(f"Rows: {self.summary['total_rows']:,}")
        print(f"Columns: {self.summary['total_columns']}")
        print("="*70)
        
        if self.summary['validation_passed']:
            print("\n✓ VALIDATION PASSED - No violations found!")
        else:
            print(f"\n✗ VALIDATION FAILED - {self.summary['total_violations']} violations found")
            
            print(f"\nViolations by Severity:")
            for severity, count in sorted(self.summary['violations_by_severity'].items()):
                print(f"  {severity.upper()}: {count}")
            
            print(f"\nViolations by Type:")
            for vtype, count in sorted(self.summary['violations_by_type'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {vtype}: {count}")
            
            print(f"\nTop Columns with Violations:")
            for col, count in sorted(self.summary['violations_by_column'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {col}: {count} violations")
            
            print(f"\nSample Violations (first 10):")
            for v in self.violations[:10]:
                print(f"  Row {v['row']}, Column '{v['column']}': {v['message']}")
                print(f"    Value: {v['value']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_violations(self, output_path='validation_violations.json'):
        """Export detailed violations to JSON"""
        output = {
            'summary': self.summary,
            'violations': self.violations
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Violations exported to {output_path}")


if __name__ == "__main__":
    # Example usage with sample schema
    schema = {
        'user_id': {'type': 'integer', 'nullable': False, 'range': [1, 1000000]},
        'email': {'type': 'email', 'nullable': False},
        'age': {'type': 'integer', 'range': [0, 120]},
        'registration_date': {'type': 'date', 'format': '%Y-%m-%d'},
        'status': {'type': 'categorical', 'values': ['active', 'inactive', 'suspended']},
        'website': {'type': 'url', 'nullable': True}
    }
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_type_validator.py <filepath> [schema.json]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load schema from file if provided
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            schema = json.load(f)
    
    validator = DataTypeValidator(filepath, schema)
    validator.validate()
    validator.print_report()
    validator.export_violations()


