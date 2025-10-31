# ===================================================================
# 5. DATA QUALITY ASSERTION FRAMEWORK 
# ===================================================================
"""
Provides a declarative framework for defining and running data quality
assertions with detailed failure reporting.
"""

import pandas as pd
from datetime import datetime
from typing import Callable, List, Dict, Any
import json


class DataQualityAssertion:
    def __init__(self, name: str, description: str, check_func: Callable, 
                 severity: str = 'error'):
        """
        Define a data quality assertion.
        
        Args:
            name: unique name for the assertion
            description: human-readable description
            check_func: function that takes DataFrame and returns (passed: bool, details: dict)
            severity: 'error', 'warning', or 'info'
        """
        self.name = name
        self.description = description
        self.check_func = check_func
        self.severity = severity
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the assertion.
        
        Returns:
            dict with assertion results
        """
        try:
            passed, details = self.check_func(df)
            return {
                'name': self.name,
                'description': self.description,
                'severity': self.severity,
                'passed': passed,
                'details': details,
                'timestamp': datetime.now(),
                'error': None
            }
        except Exception as e:
            return {
                'name': self.name,
                'description': self.description,
                'severity': 'error',
                'passed': False,
                'details': {},
                'timestamp': datetime.now(),
                'error': str(e)
            }


class DataQualityFramework:
    def __init__(self, dataset_name: str):
        """
        Initialize data quality framework.
        
        Args:
            dataset_name: name of the dataset being validated
        """
        self.dataset_name = dataset_name
        self.assertions: List[DataQualityAssertion] = []
        self.results = []
    
    def add_assertion(self, assertion: DataQualityAssertion):
        """Add an assertion to the framework."""
        self.assertions.append(assertion)
    
    def assert_row_count_range(self, min_rows: int, max_rows: int = None):
        """Assert row count is within expected range."""
        def check(df):
            count = len(df)
            if max_rows:
                passed = min_rows <= count <= max_rows
                details = {
                    'actual_count': count,
                    'expected_range': f'{min_rows}-{max_rows}',
                    'status': 'pass' if passed else 'fail'
                }
            else:
                passed = count >= min_rows
                details = {
                    'actual_count': count,
                    'minimum_expected': min_rows,
                    'status': 'pass' if passed else 'fail'
                }
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'row_count_range_{min_rows}_{max_rows}',
            f'Row count should be between {min_rows} and {max_rows or "infinity"}',
            check
        ))
    
    def assert_no_nulls(self, columns: List[str]):
        """Assert specified columns have no null values."""
        def check(df):
            null_counts = {}
            all_passed = True
            
            for col in columns:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    null_counts[col] = int(null_count)
                    if null_count > 0:
                        all_passed = False
                else:
                    null_counts[col] = 'column_not_found'
                    all_passed = False
            
            return all_passed, {'null_counts': null_counts}
        
        self.add_assertion(DataQualityAssertion(
            f'no_nulls_{"-".join(columns[:3])}',
            f'Columns {columns} should have no null values',
            check,
            severity='error'
        ))
    
    def assert_unique(self, column: str):
        """Assert column values are unique."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            total = len(df)
            unique = df[column].nunique()
            duplicates = total - unique
            passed = duplicates == 0
            
            details = {
                'total_rows': total,
                'unique_values': unique,
                'duplicate_count': duplicates
            }
            
            if not passed:
                # Find duplicate values
                dup_values = df[df.duplicated(subset=[column], keep=False)][column].unique()
                details['sample_duplicates'] = [str(x) for x in dup_values[:5]]
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'unique_{column}',
            f'Column {column} should contain unique values',
            check,
            severity='error'
        ))
    
    def assert_value_range(self, column: str, min_value: float, max_value: float):
        """Assert numeric column values are within range."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            values = df[column].dropna()
            below_min = (values < min_value).sum()
            above_max = (values > max_value).sum()
            passed = below_min == 0 and above_max == 0
            
            details = {
                'expected_range': f'{min_value}-{max_value}',
                'actual_range': f'{values.min()}-{values.max()}',
                'values_below_min': int(below_min),
                'values_above_max': int(above_max)
            }
            
            if not passed:
                invalid_values = df[(df[column] < min_value) | (df[column] > max_value)][column]
                details['sample_invalid'] = [float(x) for x in invalid_values.head(5)]
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'range_{column}_{min_value}_{max_value}',
            f'Column {column} values should be between {min_value} and {max_value}',
            check
        ))
    
    def assert_foreign_key(self, column: str, reference_df: pd.DataFrame, 
                          reference_column: str):
        """Assert foreign key integrity."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            values = set(df[column].dropna().unique())
            reference_values = set(reference_df[reference_column].dropna().unique())
            
            orphaned = values - reference_values
            passed = len(orphaned) == 0
            
            details = {
                'total_distinct_values': len(values),
                'orphaned_count': len(orphaned),
                'orphaned_sample': [str(x) for x in list(orphaned)[:10]] if orphaned else []
            }
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'fk_{column}_references_{reference_column}',
            f'Foreign key {column} should reference {reference_column}',
            check,
            severity='error'
        ))
    
    def assert_pattern_match(self, column: str, pattern: str):
        """Assert values match a regex pattern."""
        import re
        
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            values = df[column].dropna().astype(str)
            matches = values.str.match(pattern)
            non_matches = (~matches).sum()
            passed = non_matches == 0
            
            details = {
                'pattern': pattern,
                'total_values': len(values),
                'matching': int(matches.sum()),
                'non_matching': int(non_matches)
            }
            
            if not passed:
                invalid = values[~matches].head(5).tolist()
                details['sample_invalid'] = invalid
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'pattern_{column}',
            f'Column {column} should match pattern: {pattern}',
            check
        ))
    
    def assert_no_duplicates(self, columns: List[str]):
        """Assert combination of columns is unique."""
        def check(df):
            total = len(df)
            duplicates = df.duplicated(subset=columns, keep=False)
            dup_count = duplicates.sum()
            passed = dup_count == 0
            
            details = {
                'total_rows': total,
                'duplicate_rows': int(dup_count),
                'columns_checked': columns
            }
            
            if not passed:
                dup_sample = df[duplicates][columns].head(5).to_dict('records')
                details['sample_duplicates'] = dup_sample
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'no_duplicates_{"-".join(columns[:3])}',
            f'Combination of {columns} should be unique',
            check,
            severity='error'
        ))
    
    def assert_column_exists(self, columns: List[str]):
        """Assert required columns exist."""
        def check(df):
            missing = [col for col in columns if col not in df.columns]
            passed = len(missing) == 0
            
            details = {
                'required_columns': columns,
                'missing_columns': missing,
                'existing_columns': [col for col in columns if col in df.columns]
            }
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'columns_exist',
            f'Required columns should exist: {columns}',
            check,
            severity='error'
        ))
    
    def assert_custom(self, name: str, description: str, check_func: Callable, 
                     severity: str = 'error'):
        """Add a custom assertion."""
        self.add_assertion(DataQualityAssertion(name, description, check_func, severity))
    
    def run_all_assertions(self, df: pd.DataFrame, stop_on_error: bool = False) -> bool:
        """
        Run all assertions and collect results.
        
        Args:
            df: DataFrame to validate
            stop_on_error: if True, stop on first error
        
        Returns:
            True if all critical assertions passed
        """
        print(f"\nRunning {len(self.assertions)} assertions on {self.dataset_name}...")
        
        self.results = []
        all_passed = True
        
        for idx, assertion in enumerate(self.assertions, 1):
            print(f"  [{idx}/{len(self.assertions)}] {assertion.name}...", end=" ")
            result = assertion.run(df)
            self.results.append(result)
            
            if result['passed']:
                print("✓")
            else:
                print("✗")
                if result['severity'] == 'error':
                    all_passed = False
                    if stop_on_error:
                        print(f"\n⚠ Stopping on error: {assertion.name}")
                        break
        
        return all_passed
    
    def generate_report(self, verbose: bool = True) -> str:
        """
        Generate detailed quality report.
        
        Args:
            verbose: if True, include all details
        
        Returns:
            formatted report string
        """
        if not self.results:
            return "No assertions have been run yet."
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        errors = sum(1 for r in self.results if not r['passed'] and r['severity'] == 'error')
        warnings = sum(1 for r in self.results if not r['passed'] and r['severity'] == 'warning')
        
        report = f"""
{'='*70}
DATA QUALITY REPORT
Dataset: {self.dataset_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

SUMMARY:
  Total Assertions: {total}
  Passed: {passed} ✓
  Failed: {failed} ✗
    - Errors: {errors}
    - Warnings: {warnings}
  Success Rate: {(passed/total*100):.1f}%
