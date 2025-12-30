"""
Cross-Field Consistency Checker
Validates logical relationships and consistency between fields
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class ConsistencyChecker:
    def __init__(self, filepath, rules_config=None):
        """
        Initialize consistency checker
        
        Args:
            filepath: Path to data file
            rules_config: Dict or path to JSON file defining validation rules
                Example: {
                    'temporal_rules': [
                        {'start': 'start_date', 'end': 'end_date', 'name': 'date_sequence'}
                    ],
                    'mathematical_rules': [
                        {'columns': ['subtotal', 'tax', 'total'], 
                         'formula': 'subtotal + tax == total', 'name': 'total_calculation'}
                    ],
                    'referential_rules': [
                        {'child': 'order_id', 'parent_table': 'orders.csv', 
                         'parent_key': 'id', 'name': 'order_exists'}
                    ],
                    'conditional_rules': [
                        {'if': {'column': 'status', 'equals': 'shipped'},
                         'then': {'column': 'tracking_number', 'not_null': True},
                         'name': 'shipped_has_tracking'}
                    ]
                }
        """
        self.filepath = Path(filepath)
        self.df = self._load_data()
        
        if isinstance(rules_config, (str, Path)):
            with open(rules_config, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = rules_config or {}
        
        self.violations = []
        self.stats = {}
    
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
    
    def validate_all(self):
        """Run all consistency validations"""
        print(f"Validating consistency across {len(self.df)} rows...")
        
        self.violations = []
        
        # Validate temporal consistency
        if 'temporal_rules' in self.rules:
            self._validate_temporal()
        
        # Validate mathematical consistency
        if 'mathematical_rules' in self.rules:
            self._validate_mathematical()
        
        # Validate referential integrity
        if 'referential_rules' in self.rules:
            self._validate_referential()
        
        # Validate conditional logic
        if 'conditional_rules' in self.rules:
            self._validate_conditional()
        
        # Validate custom rules
        if 'custom_rules' in self.rules:
            self._validate_custom()
        
        self._calculate_stats()
        return self.stats
    
    def _validate_temporal(self):
        """Validate temporal/date relationships"""
        print("Checking temporal consistency...")
        
        for rule in self.rules.get('temporal_rules', []):
            start_col = rule['start']
            end_col = rule['end']
            rule_name = rule.get('name', f'{start_col}_before_{end_col}')
            
            if start_col not in self.df.columns or end_col not in self.df.columns:
                continue
            
            # Convert to datetime if needed
            try:
                start_dates = pd.to_datetime(self.df[start_col], errors='coerce')
                end_dates = pd.to_datetime(self.df[end_col], errors='coerce')
            except Exception as e:
                print(f"Error converting dates for rule {rule_name}: {e}")
                continue
            
            # Check where start > end
            for idx in self.df.index:
                start = start_dates.loc[idx]
                end = end_dates.loc[idx]
                
                if pd.notna(start) and pd.notna(end) and start > end:
                    self.violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'type': 'temporal',
                        'columns': [start_col, end_col],
                        'values': {start_col: str(start), end_col: str(end)},
                        'message': f"{start_col} ({start}) is after {end_col} ({end})",
                        'severity': 'high'
                    })
    
    def _validate_mathematical(self):
        """Validate mathematical relationships"""
        print("Checking mathematical consistency...")
        
        for rule in self.rules.get('mathematical_rules', []):
            rule_name = rule.get('name', 'math_rule')
            columns = rule.get('columns', [])
            formula = rule.get('formula', '')
            tolerance = rule.get('tolerance', 0.01)  # For floating point comparisons
            
            # Check all columns exist
            if not all(col in self.df.columns for col in columns):
                continue
            
            # Evaluate formula for each row
            for idx in self.df.index:
                try:
                    # Create local namespace with column values
                    namespace = {col: self.df.loc[idx, col] for col in columns}
                    
                    # Skip if any values are null
                    if any(pd.isna(v) for v in namespace.values()):
                        continue
                    
                    # Evaluate the formula
                    result = eval(formula, {"__builtins__": {}}, namespace)
                    
                    # For comparisons, result should be boolean
                    if isinstance(result, bool):
                        if not result:
                            self.violations.append({
                                'row': int(idx),
                                'rule': rule_name,
                                'type': 'mathematical',
                                'columns': columns,
                                'values': namespace,
                                'message': f"Formula '{formula}' failed",
                                'severity': 'high'
                            })
                    # For calculations, check if close to zero (difference)
                    elif isinstance(result, (int, float)):
                        if abs(result) > tolerance:
                            self.violations.append({
                                'row': int(idx),
                                'rule': rule_name,
                                'type': 'mathematical',
                                'columns': columns,
                                'values': namespace,
                                'message': f"Calculation mismatch: {formula} = {result}",
                                'severity': 'medium'
                            })
                
                except Exception as e:
                    self.violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'type': 'mathematical',
                        'columns': columns,
                        'message': f"Error evaluating formula: {str(e)}",
                        'severity': 'low'
                    })
    
    def _validate_referential(self):
        """Validate referential integrity"""
        print("Checking referential integrity...")
        
        for rule in self.rules.get('referential_rules', []):
            child_col = rule['child']
            parent_table = rule['parent_table']
            parent_key = rule['parent_key']
            rule_name = rule.get('name', f'{child_col}_reference')
            
            if child_col not in self.df.columns:
                continue
            
            try:
                # Load parent table
                parent_path = Path(parent_table)
                if not parent_path.is_absolute():
                    parent_path = self.filepath.parent / parent_table
                
                if parent_path.suffix == '.csv':
                    parent_df = pd.read_csv(parent_path)
                elif parent_path.suffix in ['.xlsx', '.xls']:
                    parent_df = pd.read_excel(parent_path)
                else:
                    continue
                
                if parent_key not in parent_df.columns:
                    continue
                
                # Check for orphaned records
                valid_keys = set(parent_df[parent_key].dropna())
                
                for idx, value in self.df[child_col].items():
                    if pd.notna(value) and value not in valid_keys:
                        self.violations.append({
                            'row': int(idx),
                            'rule': rule_name,
                            'type': 'referential',
                            'columns': [child_col],
                            'values': {child_col: value},
                            'message': f"Foreign key '{value}' not found in {parent_table}",
                            'severity': 'high'
                        })
            
            except Exception as e:
                print(f"Error checking referential integrity for {rule_name}: {e}")
    
    def _validate_conditional(self):
        """Validate conditional logic rules"""
        print("Checking conditional logic...")
        
        for rule in self.rules.get('conditional_rules', []):
            rule_name = rule.get('name', 'conditional_rule')
            if_clause = rule.get('if', {})
            then_clause = rule.get('then', {})
            
            if_col = if_clause.get('column')
            if_value = if_clause.get('equals')
            if_not_null = if_clause.get('not_null', False)
            
            then_col = then_clause.get('column')
            then_not_null = then_clause.get('not_null', False)
            then_equals = then_clause.get('equals')
            then_in = then_clause.get('in')
            
            if if_col not in self.df.columns:
                continue
            
            # Find rows matching the IF condition
            if if_not_null:
                condition_mask = self.df[if_col].notna()
            elif if_value is not None:
                condition_mask = self.df[if_col] == if_value
            else:
                continue
            
            # Check THEN clause for matching rows
            for idx in self.df[condition_mask].index:
                violation = False
                message = ""
                
                if then_col and then_col in self.df.columns:
                    then_value = self.df.loc[idx, then_col]
                    
                    if then_not_null and pd.isna(then_value):
                        violation = True
                        message = f"When {if_col}={if_value}, {then_col} must not be null"
                    
                    elif then_equals is not None and then_value != then_equals:
                        violation = True
                        message = f"When {if_col}={if_value}, {then_col} should be {then_equals}, got {then_value}"
                    
                    elif then_in is not None and then_value not in then_in:
                        violation = True
                        message = f"When {if_col}={if_value}, {then_col} should be in {then_in}, got {then_value}"
                
                if violation:
                    self.violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'type': 'conditional',
                        'columns': [if_col, then_col] if then_col else [if_col],
                        'message': message,
                        'severity': 'medium'
                    })
    
    def _validate_custom(self):
        """Validate custom Python function rules"""
        print("Checking custom rules...")
        
        for rule in self.rules.get('custom_rules', []):
            rule_name = rule.get('name', 'custom_rule')
            func_str = rule.get('function', '')
            columns = rule.get('columns', [])
            
            if not all(col in self.df.columns for col in columns):
                continue
            
            try:
                # Create function from string
                func = eval(func_str)
                
                for idx in self.df.index:
                    row_data = {col: self.df.loc[idx, col] for col in columns}
                    
                    # Skip if any required values are null
                    if any(pd.isna(v) for v in row_data.values()):
                        continue
                    
                    # Call validation function
                    result = func(row_data)
                    
                    if not result:
                        self.violations.append({
                            'row': int(idx),
                            'rule': rule_name,
                            'type': 'custom',
                            'columns': columns,
                            'values': row_data,
                            'message': f"Custom validation failed: {rule_name}",
                            'severity': 'medium'
                        })
            
            except Exception as e:
                print(f"Error in custom rule {rule_name}: {e}")
    
    def _calculate_stats(self):
        """Calculate consistency statistics"""
        total_violations = len(self.violations)
        
        # Count by type
        type_counts = {}
        for v in self.violations:
            vtype = v['type']
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for v in self.violations:
            sev = v['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Count by rule
        rule_counts = {}
        for v in self.violations:
            rule = v['rule']
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        self.stats = {
            'total_rows': len(self.df),
            'total_violations': total_violations,
            'rows_with_violations': len(set(v['row'] for v in self.violations)),
            'violation_percentage': round((len(set(v['row'] for v in self.violations)) / len(self.df)) * 100, 2),
            'violations_by_type': type_counts,
            'violations_by_severity': severity_counts,
            'violations_by_rule': rule_counts
        }
    
    def print_report(self):
        """Print consistency validation report"""
        print("\n" + "="*70)
        print("CROSS-FIELD CONSISTENCY REPORT")
        print("="*70)
        print(f"Dataset: {self.filepath.name}")
        print(f"Total Rows: {self.stats['total_rows']:,}")
        print("="*70)
        
        if self.stats['total_violations'] == 0:
            print("\n✓ All consistency checks passed!")
        else:
            print(f"\nCONSISTENCY VIOLATIONS:")
            print(f"  Total Violations: {self.stats['total_violations']}")
            print(f"  Rows with Violations: {self.stats['rows_with_violations']} ({self.stats['violation_percentage']}%)")
            
            print(f"\nBY TYPE:")
            for vtype, count in sorted(self.stats['violations_by_type'].items()):
                print(f"  {vtype.capitalize()}: {count}")
            
            print(f"\nBY SEVERITY:")
            for severity in ['high', 'medium', 'low']:
                count = self.stats['violations_by_severity'].get(severity, 0)
                if count > 0:
                    print(f"  {severity.upper()}: {count}")
            
            print(f"\nBY RULE:")
            sorted_rules = sorted(self.stats['violations_by_rule'].items(), 
                                key=lambda x: x[1], reverse=True)
            for rule, count in sorted_rules[:10]:
                print(f"  {rule}: {count} violations")
            
            print(f"\nSAMPLE VIOLATIONS (first 10):")
            for v in self.violations[:10]:
                print(f"\n  Row {v['row']}, Rule '{v['rule']}':")
                print(f"    Type: {v['type']}")
                print(f"    Severity: {v['severity'].upper()}")
                print(f"    Message: {v['message']}")
                if 'values' in v:
                    print(f"    Values: {v['values']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_violations(self, output_path='consistency_violations.json'):
        """Export detailed violations"""
        output = {
            'statistics': self.stats,
            'violations': self.violations
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Violations exported to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python consistency_checker.py <filepath> [rules.json]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load rules from file or use defaults
    rules = None
    if len(sys.argv) > 2:
        rules = sys.argv[2]
    else:
        # Example default rules
        rules = {
            'temporal_rules': [
                {'start': 'start_date', 'end': 'end_date', 'name': 'date_sequence'}
            ],
            'mathematical_rules': [
                {'columns': ['quantity', 'price', 'total'], 
                 'formula': 'abs((quantity * price) - total) < 0.01',
                 'name': 'total_calculation'}
            ]
        }
    
    checker = ConsistencyChecker(filepath, rules_config=rules)
    checker.validate_all()
    checker.print_report()
    checker.export_violations()

