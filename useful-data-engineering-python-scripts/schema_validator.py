
# ===================================================================
# 2. SCHEMA VALIDATOR AND CHANGE DETECTOR
# ===================================================================
"""
Validates schemas against baselines and detects schema drift
before it breaks downstream pipelines.
"""

import pandas as pd
import json
from datetime import datetime
from sqlalchemy import create_engine, inspect


class SchemaValidator:
    def __init__(self, baseline_dir='schema_baselines'):
        """
        Initialize schema validator.
        
        Args:
            baseline_dir: directory to store baseline schema definitions
        """
        self.baseline_dir = baseline_dir
        self.changes_detected = []
    
    def extract_schema_from_db(self, engine, table_name):
        """
        Extract schema from database table.
        
        Args:
            engine: SQLAlchemy engine
            table_name: name of table to inspect
        
        Returns:
            dict with schema definition
        """
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)
        
        schema = {
            'table_name': table_name,
            'columns': [],
            'primary_key': pk['constrained_columns'] if pk else []
        }
        
        for col in columns:
            schema['columns'].append({
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col['nullable'],
                'default': str(col['default']) if col['default'] else None
            })
        
        return schema
    
    def extract_schema_from_dataframe(self, df, table_name):
        """
        Extract schema from pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            table_name: logical name for the dataset
        
        Returns:
            dict with schema definition
        """
        schema = {
            'table_name': table_name,
            'columns': [],
            'row_count': len(df)
        }
        
        for col in df.columns:
            schema['columns'].append({
                'name': col,
                'type': str(df[col].dtype),
                'nullable': df[col].isna().any(),
                'unique_count': df[col].nunique()
            })
        
        return schema
    
    def save_baseline(self, schema, baseline_name):
        """Save schema as baseline."""
        import os
        os.makedirs(self.baseline_dir, exist_ok=True)
        
        filepath = f"{self.baseline_dir}/{baseline_name}.json"
        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2)
        
        print(f"✓ Baseline saved: {filepath}")
    
    def load_baseline(self, baseline_name):
        """Load baseline schema."""
        filepath = f"{self.baseline_dir}/{baseline_name}.json"
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def compare_schemas(self, baseline, current):
        """
        Compare current schema against baseline.
        
        Returns:
            dict with detected changes
        """
        changes = {
            'added_columns': [],
            'removed_columns': [],
            'modified_columns': [],
            'type_changes': []
        }
        
        baseline_cols = {col['name']: col for col in baseline['columns']}
        current_cols = {col['name']: col for col in current['columns']}
        
        # Find added columns
        for col_name in current_cols:
            if col_name not in baseline_cols:
                changes['added_columns'].append(col_name)
        
        # Find removed columns
        for col_name in baseline_cols:
            if col_name not in current_cols:
                changes['removed_columns'].append(col_name)
        
        # Find modified columns
        for col_name in baseline_cols:
            if col_name in current_cols:
                baseline_col = baseline_cols[col_name]
                current_col = current_cols[col_name]
                
                # Check type changes
                if baseline_col['type'] != current_col['type']:
                    changes['type_changes'].append({
                        'column': col_name,
                        'old_type': baseline_col['type'],
                        'new_type': current_col['type']
                    })
                
                # Check nullable changes
                if baseline_col['nullable'] != current_col['nullable']:
                    changes['modified_columns'].append({
                        'column': col_name,
                        'change': 'nullable',
                        'old_value': baseline_col['nullable'],
                        'new_value': current_col['nullable']
                    })
        
        return changes
    
    def validate_and_report(self, current_schema, baseline_name):
        """
        Validate current schema against baseline and generate report.
        
        Args:
            current_schema: current schema dict
            baseline_name: name of baseline to compare against
        
        Returns:
            validation report string
        """
        baseline = self.load_baseline(baseline_name)
        
        if baseline is None:
            return f"⚠ No baseline found for {baseline_name}. Use save_baseline() to create one."
        
        changes = self.compare_schemas(baseline, current_schema)
        
        # Check if any changes detected
        has_changes = (len(changes['added_columns']) > 0 or 
                      len(changes['removed_columns']) > 0 or 
                      len(changes['modified_columns']) > 0 or
                      len(changes['type_changes']) > 0)
        
        report = f"""
{'='*60}
SCHEMA VALIDATION REPORT
Table: {current_schema['table_name']}
Baseline: {baseline_name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

"""
        if not has_changes:
            report += "✓ No schema changes detected. Schema is valid.\n"
        else:
            report += "⚠ SCHEMA CHANGES DETECTED:\n\n"
            
            if changes['added_columns']:
                report += "ADDED COLUMNS:\n"
                for col in changes['added_columns']:
                    report += f"  + {col}\n"
                report += "\n"
            
            if changes['removed_columns']:
                report += "REMOVED COLUMNS:\n"
                for col in changes['removed_columns']:
                    report += f"  - {col}\n"
                report += "\n"
            
            if changes['type_changes']:
                report += "TYPE CHANGES:\n"
                for change in changes['type_changes']:
                    report += f"  ~ {change['column']}: {change['old_type']} → {change['new_type']}\n"
                report += "\n"
            
            if changes['modified_columns']:
                report += "OTHER MODIFICATIONS:\n"
                for mod in changes['modified_columns']:
                    report += f"  ~ {mod['column']} ({mod['change']}): {mod['old_value']} → {mod['new_value']}\n"
        
        report += f"{'='*60}\n"
        
        self.changes_detected.append({
            'timestamp': datetime.now(),
            'table': current_schema['table_name'],
            'changes': changes
        })
        
        return report

# Example usage
    
if __name__ == "__main__":
    # Sample DataFrame
    df_original = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
    })
    
    # Create baseline
    validator = SchemaValidator()
    schema = validator.extract_schema_from_dataframe(df_original, 'users')
    validator.save_baseline(schema, 'users_baseline')
    
    # Simulate schema change
    df_modified = df_original.copy()
    df_modified['phone'] = ['555-1234', '555-5678', '555-9012']  # New column
    df_modified = df_modified.drop('email', axis=1)  # Removed column
    
    # Validate
    current_schema = validator.extract_schema_from_dataframe(df_modified, 'users')
    report = validator.validate_and_report(current_schema, 'users_baseline')\
    print(report)
    
