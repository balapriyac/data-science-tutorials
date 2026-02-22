"""
Referential Integrity Validator
Validates foreign key relationships and cross-table data consistency
"""

import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, List, Optional, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ReferentialIntegrityValidator:
    """Validates referential integrity across multiple tables/datasets"""
    
    def __init__(self, primary_filepath: str, reference_tables: Dict[str, str] = None):
        """
        Initialize validator
        
        Args:
            primary_filepath: Path to main data file
            reference_tables: Dict of table_name -> filepath for reference tables
        """
        self.primary_filepath = primary_filepath
        self.reference_tables = reference_tables or {}
        self.primary_df = None
        self.reference_dfs = {}
        self.violations = []
    
    def load_data(self):
        """Load all datasets"""
        # Load primary table
        self.primary_df = self._load_file(self.primary_filepath)
        
        # Load reference tables
        for table_name, filepath in self.reference_tables.items():
            self.reference_dfs[table_name] = self._load_file(filepath)
    
    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Load a single file"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def validate_foreign_keys(self, foreign_key_rules: List[Dict]) -> List[Dict]:
        """Validate foreign key relationships"""
        violations = []
        
        for rule in foreign_key_rules:
            fk_column = rule.get('foreign_key_column')
            ref_table = rule.get('reference_table')
            ref_column = rule.get('reference_column')
            rule_name = rule.get('name', 'foreign_key')
            allow_null = rule.get('allow_null', True)
            
            if fk_column not in self.primary_df.columns:
                continue
            
            if ref_table not in self.reference_dfs:
                violations.append({
                    'rule': rule_name,
                    'type': 'missing_reference_table',
                    'severity': 'high',
                    'message': f"Reference table '{ref_table}' not loaded"
                })
                continue
            
            ref_df = self.reference_dfs[ref_table]
            
            if ref_column not in ref_df.columns:
                violations.append({
                    'rule': rule_name,
                    'type': 'missing_reference_column',
                    'severity': 'high',
                    'message': f"Column '{ref_column}' not found in table '{ref_table}'"
                })
                continue
            
            # Get valid reference values
            valid_values = set(ref_df[ref_column].dropna().unique())
            
            # Check each foreign key value
            for idx, fk_value in self.primary_df[fk_column].items():
                # Handle nulls
                if pd.isna(fk_value):
                    if not allow_null:
                        violations.append({
                            'row': int(idx),
                            'rule': rule_name,
                            'type': 'null_foreign_key',
                            'foreign_key_column': fk_column,
                            'severity': 'high',
                            'message': f"NULL value in foreign key column '{fk_column}' (not allowed)"
                        })
                    continue
                
                # Check if value exists in reference table
                if fk_value not in valid_values:
                    violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'type': 'invalid_foreign_key',
                        'foreign_key_column': fk_column,
                        'foreign_key_value': str(fk_value),
                        'reference_table': ref_table,
                        'reference_column': ref_column,
                        'severity': 'high',
                        'message': f"Value '{fk_value}' not found in {ref_table}.{ref_column}"
                    })
        
        return violations
    
    def detect_orphaned_records(self, orphan_rules: List[Dict]) -> List[Dict]:
        """Detect records that should have related records but don't"""
        violations = []
        
        for rule in orphan_rules:
            parent_table = rule.get('parent_table')
            parent_key = rule.get('parent_key_column')
            child_table = rule.get('child_table', 'primary')
            child_key = rule.get('child_key_column')
            rule_name = rule.get('name', 'orphan_detection')
            
            # Determine which dataframe is the parent
            if parent_table == 'primary':
                parent_df = self.primary_df
            elif parent_table in self.reference_dfs:
                parent_df = self.reference_dfs[parent_table]
            else:
                continue
            
            # Determine which dataframe is the child
            if child_table == 'primary':
                child_df = self.primary_df
            elif child_table in self.reference_dfs:
                child_df = self.reference_dfs[child_table]
            else:
                continue
            
            if parent_key not in parent_df.columns or child_key not in child_df.columns:
                continue
            
            # Find parent records without children
            parent_values = set(parent_df[parent_key].dropna().unique())
            child_values = set(child_df[child_key].dropna().unique())
            
            orphaned_parents = parent_values - child_values
            
            for orphan_value in orphaned_parents:
                # Find row(s) with this value
                matching_rows = parent_df[parent_df[parent_key] == orphan_value].index
                
                for idx in matching_rows:
                    violations.append({
                        'row': int(idx),
                        'rule': rule_name,
                        'type': 'orphaned_parent',
                        'parent_table': parent_table,
                        'parent_key': str(orphan_value),
                        'severity': 'medium',
                        'message': f"Parent record '{orphan_value}' has no related child records"
                    })
        
        return violations
    
    def validate_cardinality_constraints(self, cardinality_rules: List[Dict]) -> List[Dict]:
        """Validate one-to-one, one-to-many constraints"""
        violations = []
        
        for rule in cardinality_rules:
            constraint_type = rule.get('type')  # 'one-to-one', 'one-to-many'
            child_table = rule.get('child_table', 'primary')
            child_key = rule.get('child_key_column')
            rule_name = rule.get('name', 'cardinality')
            
            # Get child dataframe
            if child_table == 'primary':
                child_df = self.primary_df
            elif child_table in self.reference_dfs:
                child_df = self.reference_dfs[child_table]
            else:
                continue
            
            if child_key not in child_df.columns:
                continue
            
            # Count occurrences of each key value
            value_counts = child_df[child_key].value_counts()
            
            if constraint_type == 'one-to-one':
                # Check for duplicate keys
                duplicates = value_counts[value_counts > 1]
                
                for key_value, count in duplicates.items():
                    matching_rows = child_df[child_df[child_key] == key_value].index.tolist()
                    
                    violations.append({
                        'rows': matching_rows,
                        'rule': rule_name,
                        'type': 'cardinality_violation',
                        'constraint': 'one-to-one',
                        'key_value': str(key_value),
                        'occurrence_count': int(count),
                        'severity': 'high',
                        'message': f"Key '{key_value}' appears {count} times (expected 1)"
                    })
        
        return violations
    
    def validate_composite_keys(self, composite_key_rules: List[Dict]) -> List[Dict]:
        """Validate composite primary/foreign keys"""
        violations = []
        
        for rule in composite_key_rules:
            key_columns = rule.get('key_columns', [])
            uniqueness = rule.get('uniqueness', 'unique')  # 'unique' or 'non-unique'
            rule_name = rule.get('name', 'composite_key')
            
            # Check all columns exist
            if not all(col in self.primary_df.columns for col in key_columns):
                continue
            
            if uniqueness == 'unique':
                # Check for duplicates in composite key
                duplicates = self.primary_df[self.primary_df.duplicated(subset=key_columns, keep=False)]
                
                if len(duplicates) > 0:
                    # Group by composite key to find duplicates
                    for key_values, group in duplicates.groupby(key_columns):
                        if len(group) > 1:
                            violations.append({
                                'rows': group.index.tolist(),
                                'rule': rule_name,
                                'type': 'duplicate_composite_key',
                                'key_columns': key_columns,
                                'key_values': {col: str(val) for col, val in zip(key_columns, key_values)},
                                'occurrence_count': len(group),
                                'severity': 'high',
                                'message': f"Composite key {dict(zip(key_columns, key_values))} is duplicated"
                            })
        
        return violations
    
    def validate_cascading_deletes(self, cascade_rules: List[Dict]) -> List[Dict]:
        """Identify records that would be affected by cascading deletes"""
        violations = []
        
        for rule in cascade_rules:
            parent_table = rule.get('parent_table')
            parent_key = rule.get('parent_key_column')
            child_table = rule.get('child_table', 'primary')
            child_fk = rule.get('child_foreign_key_column')
            rule_name = rule.get('name', 'cascade_check')
            
            # Get parent dataframe
            if parent_table == 'primary':
                parent_df = self.primary_df
            elif parent_table in self.reference_dfs:
                parent_df = self.reference_dfs[parent_table]
            else:
                continue
            
            # Get child dataframe
            if child_table == 'primary':
                child_df = self.primary_df
            elif child_table in self.reference_dfs:
                child_df = self.reference_dfs[child_table]
            else:
                continue
            
            if parent_key not in parent_df.columns or child_fk not in child_df.columns:
                continue
            
            # For each parent record, count affected children
            for idx, parent_value in parent_df[parent_key].items():
                if pd.isna(parent_value):
                    continue
                
                affected_children = child_df[child_df[child_fk] == parent_value]
                
                if len(affected_children) > 0:
                    violations.append({
                        'parent_row': int(idx),
                        'parent_table': parent_table,
                        'parent_key': str(parent_value),
                        'affected_child_count': len(affected_children),
                        'affected_child_rows': affected_children.index.tolist()[:10],  # First 10
                        'rule': rule_name,
                        'type': 'cascade_impact',
                        'severity': 'medium',
                        'message': f"Deleting parent '{parent_value}' would affect {len(affected_children)} child records"
                    })
        
        return violations
    
    def detect_circular_references_across_tables(self, relationship_chain: List[Dict]) -> List[Dict]:
        """Detect circular references across multiple tables"""
        violations = []
        
        # Build graph of relationships
        # This is a simplified version - full implementation would use graph traversal
        
        for i, link in enumerate(relationship_chain):
            table_a = link.get('table_a')
            key_a = link.get('key_a')
            table_b = link.get('table_b')
            key_b = link.get('key_b')
            
            # Get dataframes
            df_a = self.primary_df if table_a == 'primary' else self.reference_dfs.get(table_a)
            df_b = self.primary_df if table_b == 'primary' else self.reference_dfs.get(table_b)
            
            if df_a is None or df_b is None:
                continue
            
            if key_a not in df_a.columns or key_b not in df_b.columns:
                continue
            
            # Check if the chain creates a cycle
            # For now, just check if values reference back
            a_values = set(df_a[key_a].dropna().unique())
            b_values = set(df_b[key_b].dropna().unique())
            
            # If there's overlap and bidirectional reference, potential cycle
            overlap = a_values & b_values
            if overlap and i == len(relationship_chain) - 1:  # Last link in chain
                violations.append({
                    'type': 'potential_circular_reference',
                    'chain': relationship_chain,
                    'overlapping_values': list(overlap)[:10],
                    'severity': 'medium',
                    'message': f"Potential circular reference detected across {len(relationship_chain)} tables"
                })
        
        return violations
    
    def analyze_all(self, validation_config: Dict) -> Dict:
        """Run all referential integrity validations"""
        self.load_data()
        
        results = {
            'metadata': {
                'primary_file': self.primary_filepath,
                'primary_records': len(self.primary_df),
                'reference_tables': {
                    name: len(df) for name, df in self.reference_dfs.items()
                }
            },
            'violations': {
                'foreign_keys': [],
                'orphaned_records': [],
                'cardinality': [],
                'composite_keys': [],
                'cascade_impact': [],
                'circular_references': []
            }
        }
        
        # Run validations based on config
        if 'foreign_key_rules' in validation_config:
            results['violations']['foreign_keys'] = self.validate_foreign_keys(
                validation_config['foreign_key_rules']
            )
        
        if 'orphan_rules' in validation_config:
            results['violations']['orphaned_records'] = self.detect_orphaned_records(
                validation_config['orphan_rules']
            )
        
        if 'cardinality_rules' in validation_config:
            results['violations']['cardinality'] = self.validate_cardinality_constraints(
                validation_config['cardinality_rules']
            )
        
        if 'composite_key_rules' in validation_config:
            results['violations']['composite_keys'] = self.validate_composite_keys(
                validation_config['composite_key_rules']
            )
        
        if 'cascade_rules' in validation_config:
            results['violations']['cascade_impact'] = self.validate_cascading_deletes(
                validation_config['cascade_rules']
            )
        
        if 'relationship_chain' in validation_config:
            results['violations']['circular_references'] = self.detect_circular_references_across_tables(
                validation_config['relationship_chain']
            )
        
        # Calculate total violations
        total_violations = sum(
            len(v) if isinstance(v, list) else 0 
            for v in results['violations'].values()
        )
        results['metadata']['total_violations'] = total_violations
        
        return results
    
    def print_report(self, results: Dict = None):
        """Print formatted validation report"""
        if results is None:
            raise ValueError("No results to print. Run analyze_all() first.")
        
        print("\n" + "="*80)
        print("REFERENTIAL INTEGRITY VALIDATION REPORT")
        print("="*80)
        
        meta = results['metadata']
        print(f"\nPrimary Table: {meta['primary_file']}")
        print(f"Primary Records: {meta['primary_records']:,}")
        print(f"\nReference Tables:")
        for table, count in meta['reference_tables'].items():
            print(f"  {table}: {count:,} records")
        print(f"\nTotal Violations: {meta['total_violations']:,}")
        
        violations = results['violations']
        
        # Foreign Key Violations
        if violations['foreign_keys']:
            print("\n" + "-"*80)
            print(f"FOREIGN KEY VIOLATIONS: {len(violations['foreign_keys'])}")
            print("-"*80)
            for v in violations['foreign_keys'][:10]:
                if v['type'] == 'invalid_foreign_key':
                    print(f"Row {v['row']}: {v['foreign_key_column']}='{v['foreign_key_value']}' "
                          f"not found in {v['reference_table']}")
        
        # Orphaned Records
        if violations['orphaned_records']:
            print("\n" + "-"*80)
            print(f"ORPHANED RECORDS: {len(violations['orphaned_records'])}")
            print("-"*80)
            for v in violations['orphaned_records'][:10]:
                print(f"Row {v['row']}: {v['message']}")
        
        # Cardinality Violations
        if violations['cardinality']:
            print("\n" + "-"*80)
            print(f"CARDINALITY VIOLATIONS: {len(violations['cardinality'])}")
            print("-"*80)
            for v in violations['cardinality'][:10]:
                print(f"Rows {v['rows']}: {v['message']}")
        
        # Composite Key Violations
        if violations['composite_keys']:
            print("\n" + "-"*80)
            print(f"COMPOSITE KEY VIOLATIONS: {len(violations['composite_keys'])}")
            print("-"*80)
            for v in violations['composite_keys'][:10]:
                print(f"Rows {v['rows']}: {v['message']}")
        
        # Cascade Impact
        if violations['cascade_impact']:
            print("\n" + "-"*80)
            print(f"CASCADE DELETE IMPACTS: {len(violations['cascade_impact'])}")
            print("-"*80)
            for v in violations['cascade_impact'][:10]:
                print(f"Parent row {v['parent_row']}: {v['message']}")
        
        # Circular References
        if violations['circular_references']:
            print("\n" + "-"*80)
            print(f"CIRCULAR REFERENCES: {len(violations['circular_references'])}")
            print("-"*80)
            for v in violations['circular_references']:
                print(f"  {v['message']}")
        
        print("\n" + "="*80)
    
    def export_report(self, output_path: str, results: Dict = None):
        """Export results to JSON"""
        if results is None:
            raise ValueError("No results to export. Run analyze_all() first.")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate referential integrity across tables'
    )
    parser.add_argument('primary_file', help='Path to primary data file')
    parser.add_argument('--config', '-c', required=True,
                       help='Path to validation config JSON')
    parser.add_argument('--export', '-e',
                       help='Export report to JSON file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract reference tables from config
    reference_tables = config.get('reference_tables', {})
    
    validator = ReferentialIntegrityValidator(
        primary_filepath=args.primary_file,
        reference_tables=reference_tables
    )
    
    results = validator.analyze_all(config)
    validator.print_report(results)
    
    if args.export:
        validator.export_report(args.export, results)


if __name__ == '__main__':
    main()


