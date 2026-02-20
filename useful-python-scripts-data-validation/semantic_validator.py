"""
Semantic Validity Checker
Validates data against complex business rules and domain knowledge
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Callable
import warnings
warnings.filterwarnings('ignore')


class SemanticValidator:
    """Validates data against business rules and semantic constraints"""
    
    def __init__(self, filepath: str, rules_config: Dict = None):
        """
        Initialize the validator
        
        Args:
            filepath: Path to data file
            rules_config: Dictionary or path to JSON file with validation rules
        """
        self.filepath = filepath
        self.df = None
        self.violations = []
        
        # Load rules
        if isinstance(rules_config, str):
            with open(rules_config, 'r') as f:
                self.rules = json.load(f)
        elif isinstance(rules_config, dict):
            self.rules = rules_config
        else:
            self.rules = {}
    
    def load_data(self):
        """Load dataset"""
        if self.filepath.endswith('.csv'):
            self.df = pd.read_csv(self.filepath)
        elif self.filepath.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.filepath)
        elif self.filepath.endswith('.json'):
            self.df = pd.read_json(self.filepath)
        else:
            raise ValueError("Unsupported file format")
    
    def validate_age_education_rules(self) -> List[Dict]:
        """Validate logical consistency between age and education level"""
        violations = []
        
        age_education_rules = self.rules.get('age_education_rules', [])
        
        for rule in age_education_rules:
            education_col = rule.get('education_column')
            age_col = rule.get('age_column')
            min_age = rule.get('min_age')
            education_level = rule.get('education_level')
            
            if education_col not in self.df.columns or age_col not in self.df.columns:
                continue
            
            # Find violations
            mask = (self.df[education_col] == education_level) & (self.df[age_col] < min_age)
            violation_indices = self.df[mask].index
            
            for idx in violation_indices:
                violations.append({
                    'row': int(idx),
                    'rule': 'age_education_consistency',
                    'severity': 'high',
                    'education_level': str(self.df.loc[idx, education_col]),
                    'age': int(self.df.loc[idx, age_col]),
                    'min_required_age': min_age,
                    'message': f"Age {self.df.loc[idx, age_col]} too young for {education_level}"
                })
        
        return violations
    
    def validate_date_progression_rules(self) -> List[Dict]:
        """Validate logical date progressions"""
        violations = []
        
        date_rules = self.rules.get('date_progression_rules', [])
        
        for rule in date_rules:
            earlier_col = rule.get('earlier_date')
            later_col = rule.get('later_date')
            rule_name = rule.get('name', 'date_progression')
            allow_equal = rule.get('allow_equal', False)
            
            if earlier_col not in self.df.columns or later_col not in self.df.columns:
                continue
            
            # Convert to datetime
            earlier_dates = pd.to_datetime(self.df[earlier_col], errors='coerce')
            later_dates = pd.to_datetime(self.df[later_col], errors='coerce')
            
            # Check progression
            if allow_equal:
                mask = earlier_dates > later_dates
            else:
                mask = earlier_dates >= later_dates
            
            # Exclude null dates
            mask = mask & earlier_dates.notna() & later_dates.notna()
            violation_indices = self.df[mask].index
            
            for idx in violation_indices:
                violations.append({
                    'row': int(idx),
                    'rule': rule_name,
                    'severity': 'high',
                    'earlier_date': str(earlier_dates.loc[idx]),
                    'later_date': str(later_dates.loc[idx]),
                    'message': f"{earlier_col} should be before {later_col}"
                })
        
        return violations
    
    def validate_state_transitions(self) -> List[Dict]:
        """Validate state machine transitions"""
        violations = []
        
        state_rules = self.rules.get('state_transition_rules', [])
        
        for rule in state_rules:
            state_col = rule.get('state_column')
            valid_transitions = rule.get('valid_transitions', {})
            sequence_col = rule.get('sequence_column')  # e.g., timestamp or order_id
            
            if state_col not in self.df.columns:
                continue
            
            # Sort by sequence
            if sequence_col and sequence_col in self.df.columns:
                sorted_df = self.df.sort_values(sequence_col)
            else:
                sorted_df = self.df
            
            # Check transitions
            for i in range(1, len(sorted_df)):
                prev_state = sorted_df.iloc[i-1][state_col]
                current_state = sorted_df.iloc[i][state_col]
                
                if pd.isna(prev_state) or pd.isna(current_state):
                    continue
                
                # Check if transition is valid
                allowed_next_states = valid_transitions.get(str(prev_state), [])
                
                if str(current_state) not in allowed_next_states:
                    violations.append({
                        'row': int(sorted_df.index[i]),
                        'rule': 'state_transition',
                        'severity': 'high',
                        'previous_state': str(prev_state),
                        'current_state': str(current_state),
                        'allowed_transitions': allowed_next_states,
                        'message': f"Invalid transition from {prev_state} to {current_state}"
                    })
        
        return violations
    
    def validate_mutually_exclusive_fields(self) -> List[Dict]:
        """Validate that mutually exclusive fields don't both have values"""
        violations = []
        
        exclusive_rules = self.rules.get('mutually_exclusive_rules', [])
        
        for rule in exclusive_rules:
            field_groups = rule.get('field_groups', [])
            rule_name = rule.get('name', 'mutually_exclusive')
            
            # Check each row
            for idx in self.df.index:
                filled_fields = []
                
                for field in field_groups:
                    if field in self.df.columns:
                        value = self.df.loc[idx, field]
                        if pd.notna(value) and value != '' and value != 0:
                            filled_fields.append(field)
                
                # If more than one field is filled, it's a violation
                if len(filled_fields) > 1:
                    violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'severity': 'medium',
                        'fields_with_values': filled_fields,
                        'message': f"Only one of {field_groups} should have a value"
                    })
        
        return violations
    
    def validate_conditional_requirements(self) -> List[Dict]:
        """Validate conditional field requirements (if X then Y must exist)"""
        violations = []
        
        conditional_rules = self.rules.get('conditional_requirement_rules', [])
        
        for rule in conditional_rules:
            condition_col = rule.get('if_column')
            condition_value = rule.get('if_value')
            required_col = rule.get('then_column')
            required_check = rule.get('then_check', 'not_null')
            rule_name = rule.get('name', 'conditional_requirement')
            
            if condition_col not in self.df.columns or required_col not in self.df.columns:
                continue
            
            # Find rows that meet condition
            condition_mask = self.df[condition_col] == condition_value
            
            # Check required field
            if required_check == 'not_null':
                violation_mask = condition_mask & (self.df[required_col].isna() | (self.df[required_col] == ''))
            elif required_check == 'is_null':
                violation_mask = condition_mask & (self.df[required_col].notna() & (self.df[required_col] != ''))
            else:
                continue
            
            violation_indices = self.df[violation_mask].index
            
            for idx in violation_indices:
                violations.append({
                    'row': int(idx),
                    'rule': rule_name,
                    'severity': 'high',
                    'condition': f"{condition_col} = {condition_value}",
                    'requirement': f"{required_col} should {required_check}",
                    'actual_value': str(self.df.loc[idx, required_col]),
                    'message': f"When {condition_col}={condition_value}, {required_col} must {required_check}"
                })
        
        return violations
    
    def validate_business_constraints(self) -> List[Dict]:
        """Validate custom business constraint expressions"""
        violations = []
        
        constraint_rules = self.rules.get('business_constraint_rules', [])
        
        for rule in constraint_rules:
            rule_name = rule.get('name', 'business_constraint')
            columns = rule.get('columns', [])
            expression = rule.get('expression')
            severity = rule.get('severity', 'medium')
            
            # Verify all columns exist
            if not all(col in self.df.columns for col in columns):
                continue
            
            try:
                # Evaluate expression for each row
                for idx in self.df.index:
                    row_data = {col: self.df.loc[idx, col] for col in columns}
                    
                    # Skip if any values are null
                    if any(pd.isna(v) for v in row_data.values()):
                        continue
                    
                    # Evaluate expression
                    # Note: In production, use safer evaluation method
                    is_valid = eval(expression, {"__builtins__": {}}, row_data)
                    
                    if not is_valid:
                        violations.append({
                            'row': int(idx),
                            'rule': rule_name,
                            'severity': severity,
                            'values': {k: str(v) for k, v in row_data.items()},
                            'expression': expression,
                            'message': f"Business constraint violated: {rule_name}"
                        })
            except Exception as e:
                print(f"Warning: Could not evaluate rule {rule_name}: {e}")
        
        return violations
    
    def analyze_all(self) -> Dict:
        """Run all semantic validations"""
        self.load_data()
        self.violations = []
        
        # Run all validation checks
        self.violations.extend(self.validate_age_education_rules())
        self.violations.extend(self.validate_date_progression_rules())
        self.violations.extend(self.validate_state_transitions())
        self.violations.extend(self.validate_mutually_exclusive_fields())
        self.violations.extend(self.validate_conditional_requirements())
        self.violations.extend(self.validate_business_constraints())
        
        # Compile statistics
        results = {
            'metadata': {
                'filepath': self.filepath,
                'total_records': len(self.df),
                'total_violations': len(self.violations)
            },
            'summary': {
                'violations_by_rule': {},
                'violations_by_severity': {
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            },
            'violations': self.violations
        }
        
        # Count by rule type
        for violation in self.violations:
            rule = violation.get('rule', 'unknown')
            results['summary']['violations_by_rule'][rule] = \
                results['summary']['violations_by_rule'].get(rule, 0) + 1
            
            severity = violation.get('severity', 'medium')
            results['summary']['violations_by_severity'][severity] += 1
        
        return results
    
    def print_report(self, results: Dict = None):
        """Print formatted validation report"""
        if results is None:
            results = self.analyze_all()
        
        print("\n" + "="*80)
        print("SEMANTIC VALIDITY VALIDATION REPORT")
        print("="*80)
        
        meta = results['metadata']
        print(f"\nDataset: {meta['filepath']}")
        print(f"Total Records: {meta['total_records']:,}")
        print(f"Total Violations: {meta['total_violations']:,}")
        
        if meta['total_violations'] > 0:
            violation_rate = (meta['total_violations'] / meta['total_records']) * 100
            print(f"Violation Rate: {violation_rate:.2f}%")
        
        # Summary by rule
        print("\n" + "-"*80)
        print("VIOLATIONS BY RULE TYPE")
        print("-"*80)
        for rule, count in sorted(results['summary']['violations_by_rule'].items()):
            print(f"{rule}: {count}")
        
        # Summary by severity
        print("\n" + "-"*80)
        print("VIOLATIONS BY SEVERITY")
        print("-"*80)
        for severity, count in results['summary']['violations_by_severity'].items():
            if count > 0:
                print(f"{severity.upper()}: {count}")
        
        # Sample violations
        if results['violations']:
            print("\n" + "-"*80)
            print("SAMPLE VIOLATIONS (showing up to 10)")
            print("-"*80)
            for violation in results['violations'][:10]:
                print(f"\nRow {violation['row']} - {violation['rule'].upper()}")
                print(f"  Severity: {violation['severity']}")
                print(f"  Message: {violation['message']}")
                if 'values' in violation:
                    print(f"  Values: {violation['values']}")
        
        print("\n" + "="*80)
    
    def export_report(self, output_path: str, results: Dict = None):
        """Export results to JSON"""
        if results is None:
            results = self.analyze_all()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate data against semantic business rules'
    )
    parser.add_argument('filepath', help='Path to data file')
    parser.add_argument('--rules', '-r', required=True,
                       help='Path to rules JSON file')
    parser.add_argument('--export', '-e',
                       help='Export report to JSON file')
    
    args = parser.parse_args()
    
    validator = SemanticValidator(
        filepath=args.filepath,
        rules_config=args.rules
    )
    
    results = validator.analyze_all()
    validator.print_report(results)
    
    if args.export:
        validator.export_report(args.export, results)


if __name__ == '__main__':
    main()

