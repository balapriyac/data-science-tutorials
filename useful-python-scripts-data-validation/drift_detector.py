"""
Data Drift Detector
Monitors datasets for structural and statistical drift over time
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataDriftDetector:
    """Detects schema changes and statistical drift in datasets"""
    
    def __init__(self, current_filepath: str, baseline_filepath: Optional[str] = None,
                 baseline_profile: Optional[Dict] = None):
        """
        Initialize drift detector
        
        Args:
            current_filepath: Path to current data file
            baseline_filepath: Path to baseline data file (optional)
            baseline_profile: Pre-computed baseline profile (optional)
        """
        self.current_filepath = current_filepath
        self.baseline_filepath = baseline_filepath
        self.baseline_profile = baseline_profile
        self.current_df = None
        self.baseline_df = None
        self.drift_results = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from file"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format")
    
    def create_profile(self, df: pd.DataFrame) -> Dict:
        """Create statistical profile of dataset"""
        profile = {
            'timestamp': datetime.now().isoformat(),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'schema_fingerprint': self._compute_schema_fingerprint(df)
        }
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(df[col].isna().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique())
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    'mean': float(df[col].mean()) if df[col].notna().any() else None,
                    'std': float(df[col].std()) if df[col].notna().any() else None,
                    'min': float(df[col].min()) if df[col].notna().any() else None,
                    'max': float(df[col].max()) if df[col].notna().any() else None,
                    'median': float(df[col].median()) if df[col].notna().any() else None,
                    'q25': float(df[col].quantile(0.25)) if df[col].notna().any() else None,
                    'q75': float(df[col].quantile(0.75)) if df[col].notna().any() else None,
                })
            
            # Categorical columns
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts()
                col_profile.update({
                    'unique_values': list(df[col].unique()[:100]),  # First 100 unique values
                    'top_values': value_counts.head(10).to_dict(),
                    'cardinality': int(df[col].nunique())
                })
            
            profile['columns'][col] = col_profile
        
        return profile
    
    def _compute_schema_fingerprint(self, df: pd.DataFrame) -> str:
        """Compute a fingerprint of the schema"""
        schema_str = ','.join([f"{col}:{df[col].dtype}" for col in sorted(df.columns)])
        return str(hash(schema_str))
    
    def detect_schema_drift(self, baseline: Dict, current: Dict) -> Dict:
        """Detect changes in schema structure"""
        drift = {
            'schema_changed': baseline['schema_fingerprint'] != current['schema_fingerprint'],
            'new_columns': [],
            'removed_columns': [],
            'type_changes': [],
            'row_count_change': {
                'baseline': baseline['row_count'],
                'current': current['row_count'],
                'difference': current['row_count'] - baseline['row_count'],
                'percent_change': ((current['row_count'] - baseline['row_count']) / 
                                 baseline['row_count'] * 100) if baseline['row_count'] > 0 else 0
            }
        }
        
        baseline_cols = set(baseline['columns'].keys())
        current_cols = set(current['columns'].keys())
        
        drift['new_columns'] = list(current_cols - baseline_cols)
        drift['removed_columns'] = list(baseline_cols - current_cols)
        
        # Check for type changes
        common_cols = baseline_cols & current_cols
        for col in common_cols:
            baseline_type = baseline['columns'][col]['dtype']
            current_type = current['columns'][col]['dtype']
            
            if baseline_type != current_type:
                drift['type_changes'].append({
                    'column': col,
                    'baseline_type': baseline_type,
                    'current_type': current_type
                })
        
        return drift
    
    def detect_statistical_drift(self, baseline: Dict, current: Dict, 
                                 threshold: float = 0.05) -> Dict:
        """Detect statistical drift in numeric columns"""
        drift = {
            'numeric_drift': [],
            'categorical_drift': []
        }
        
        common_cols = set(baseline['columns'].keys()) & set(current['columns'].keys())
        
        for col in common_cols:
            baseline_col = baseline['columns'][col]
            current_col = current['columns'][col]
            
            # Numeric drift detection
            if 'mean' in baseline_col and 'mean' in current_col:
                if baseline_col['mean'] is not None and current_col['mean'] is not None:
                    # Calculate percentage change in mean
                    mean_change = abs((current_col['mean'] - baseline_col['mean']) / 
                                    baseline_col['mean'] * 100) if baseline_col['mean'] != 0 else 0
                    
                    # Calculate change in std
                    std_change = abs((current_col['std'] - baseline_col['std']) / 
                                   baseline_col['std'] * 100) if baseline_col['std'] and baseline_col['std'] != 0 else 0
                    
                    # Check range changes
                    range_expanded = (current_col['min'] < baseline_col['min'] or 
                                    current_col['max'] > baseline_col['max'])
                    
                    if mean_change > 10 or std_change > 20 or range_expanded:
                        drift['numeric_drift'].append({
                            'column': col,
                            'baseline_mean': baseline_col['mean'],
                            'current_mean': current_col['mean'],
                            'mean_change_percent': round(mean_change, 2),
                            'baseline_std': baseline_col['std'],
                            'current_std': current_col['std'],
                            'std_change_percent': round(std_change, 2),
                            'baseline_range': [baseline_col['min'], baseline_col['max']],
                            'current_range': [current_col['min'], current_col['max']],
                            'range_expanded': range_expanded,
                            'severity': 'high' if mean_change > 50 else 'medium' if mean_change > 25 else 'low'
                        })
            
            # Categorical drift detection
            if 'unique_values' in baseline_col and 'unique_values' in current_col:
                baseline_values = set(baseline_col['unique_values'])
                current_values = set(current_col['unique_values'])
                
                new_values = current_values - baseline_values
                removed_values = baseline_values - current_values
                
                # Check cardinality change
                cardinality_change = abs((current_col['cardinality'] - baseline_col['cardinality']) /
                                       baseline_col['cardinality'] * 100) if baseline_col['cardinality'] > 0 else 0
                
                if new_values or removed_values or cardinality_change > 20:
                    drift['categorical_drift'].append({
                        'column': col,
                        'new_values': list(new_values)[:20],  # First 20
                        'removed_values': list(removed_values)[:20],
                        'baseline_cardinality': baseline_col['cardinality'],
                        'current_cardinality': current_col['cardinality'],
                        'cardinality_change_percent': round(cardinality_change, 2),
                        'severity': 'high' if new_values or removed_values else 'low'
                    })
        
        return drift
    
    def detect_null_drift(self, baseline: Dict, current: Dict) -> Dict:
        """Detect changes in null patterns"""
        drift = []
        
        common_cols = set(baseline['columns'].keys()) & set(current['columns'].keys())
        
        for col in common_cols:
            baseline_null_pct = baseline['columns'][col]['null_percentage']
            current_null_pct = current['columns'][col]['null_percentage']
            
            change = abs(current_null_pct - baseline_null_pct)
            
            # Flag significant changes in null percentage
            if change > 5:  # More than 5 percentage points change
                drift.append({
                    'column': col,
                    'baseline_null_percentage': round(baseline_null_pct, 2),
                    'current_null_percentage': round(current_null_pct, 2),
                    'change': round(change, 2),
                    'severity': 'high' if change > 20 else 'medium' if change > 10 else 'low'
                })
        
        return drift
    
    def calculate_drift_score(self, drift_results: Dict) -> float:
        """Calculate overall drift score (0-100, higher = more drift)"""
        score = 0
        
        # Schema changes (high weight)
        if drift_results['schema_drift']['schema_changed']:
            score += 30
        
        score += len(drift_results['schema_drift']['new_columns']) * 5
        score += len(drift_results['schema_drift']['removed_columns']) * 10
        score += len(drift_results['schema_drift']['type_changes']) * 8
        
        # Statistical drift (medium weight)
        for drift in drift_results['statistical_drift']['numeric_drift']:
            if drift['severity'] == 'high':
                score += 5
            elif drift['severity'] == 'medium':
                score += 3
            else:
                score += 1
        
        for drift in drift_results['statistical_drift']['categorical_drift']:
            if drift['severity'] == 'high':
                score += 4
            else:
                score += 1
        
        # Null drift (low weight)
        for drift in drift_results['null_drift']:
            if drift['severity'] == 'high':
                score += 3
            elif drift['severity'] == 'medium':
                score += 2
            else:
                score += 1
        
        return min(score, 100)  # Cap at 100
    
    def analyze_all(self) -> Dict:
        """Run complete drift analysis"""
        # Load current data
        self.current_df = self.load_data(self.current_filepath)
        current_profile = self.create_profile(self.current_df)
        
        # Get or create baseline profile
        if self.baseline_profile:
            baseline_profile = self.baseline_profile
        elif self.baseline_filepath:
            self.baseline_df = self.load_data(self.baseline_filepath)
            baseline_profile = self.create_profile(self.baseline_df)
        else:
            # No baseline - return current profile for future use
            return {
                'status': 'baseline_created',
                'message': 'No baseline provided. Current data profiled as baseline.',
                'profile': current_profile
            }
        
        # Perform drift detection
        self.drift_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'baseline_timestamp': baseline_profile.get('timestamp'),
            'schema_drift': self.detect_schema_drift(baseline_profile, current_profile),
            'statistical_drift': self.detect_statistical_drift(baseline_profile, current_profile),
            'null_drift': self.detect_null_drift(baseline_profile, current_profile),
            'current_profile': current_profile
        }
        
        # Calculate overall drift score
        self.drift_results['drift_score'] = self.calculate_drift_score(self.drift_results)
        
        return self.drift_results
    
    def print_report(self, results: Dict = None):
        """Print formatted drift report"""
        if results is None:
            results = self.drift_results
        
        if results.get('status') == 'baseline_created':
            print("\n" + "="*80)
            print("BASELINE PROFILE CREATED")
            print("="*80)
            print("\nNo baseline provided. Current data has been profiled.")
            print("Save this profile and use it as baseline for future comparisons.")
            return
        
        print("\n" + "="*80)
        print("DATA DRIFT DETECTION REPORT")
        print("="*80)
        
        print(f"\nDrift Score: {results['drift_score']:.1f}/100")
        
        if results['drift_score'] < 10:
            status = "MINIMAL DRIFT"
        elif results['drift_score'] < 30:
            status = "LOW DRIFT"
        elif results['drift_score'] < 60:
            status = "MODERATE DRIFT"
        else:
            status = "SIGNIFICANT DRIFT"
        
        print(f"Status: {status}")
        
        # Schema Drift
        print("\n" + "-"*80)
        print("SCHEMA DRIFT")
        print("-"*80)
        schema = results['schema_drift']
        print(f"Schema Changed: {schema['schema_changed']}")
        print(f"New Columns: {len(schema['new_columns'])}")
        if schema['new_columns']:
            print(f"  {', '.join(schema['new_columns'])}")
        print(f"Removed Columns: {len(schema['removed_columns'])}")
        if schema['removed_columns']:
            print(f"  {', '.join(schema['removed_columns'])}")
        print(f"Type Changes: {len(schema['type_changes'])}")
        for change in schema['type_changes']:
            print(f"  {change['column']}: {change['baseline_type']} → {change['current_type']}")
        
        # Row count change
        row_change = schema['row_count_change']
        print(f"\nRow Count Change: {row_change['baseline']:,} → {row_change['current']:,} "
              f"({row_change['percent_change']:+.1f}%)")
        
        # Statistical Drift
        print("\n" + "-"*80)
        print("STATISTICAL DRIFT")
        print("-"*80)
        
        numeric_drift = results['statistical_drift']['numeric_drift']
        print(f"Numeric Columns with Drift: {len(numeric_drift)}")
        for drift in numeric_drift[:5]:  # Show top 5
            print(f"\n  {drift['column']} ({drift['severity'].upper()}):")
            print(f"    Mean: {drift['baseline_mean']:.2f} → {drift['current_mean']:.2f} "
                  f"({drift['mean_change_percent']:+.1f}%)")
            print(f"    Std: {drift['baseline_std']:.2f} → {drift['current_std']:.2f} "
                  f"({drift['std_change_percent']:+.1f}%)")
        
        categorical_drift = results['statistical_drift']['categorical_drift']
        print(f"\nCategorical Columns with Drift: {len(categorical_drift)}")
        for drift in categorical_drift[:5]:
            print(f"\n  {drift['column']} ({drift['severity'].upper()}):")
            if drift['new_values']:
                print(f"    New values: {', '.join(str(v) for v in drift['new_values'][:5])}")
            if drift['removed_values']:
                print(f"    Removed values: {', '.join(str(v) for v in drift['removed_values'][:5])}")
        
        # Null Drift
        print("\n" + "-"*80)
        print("NULL PATTERN DRIFT")
        print("-"*80)
        null_drift = results['null_drift']
        print(f"Columns with Null Pattern Changes: {len(null_drift)}")
        for drift in null_drift[:5]:
            print(f"  {drift['column']}: {drift['baseline_null_percentage']:.1f}% → "
                  f"{drift['current_null_percentage']:.1f}% ({drift['severity'].upper()})")
        
        print("\n" + "="*80)
    
    def export_profile(self, output_path: str):
        """Export current profile for future baseline use"""
        if self.current_df is None:
            self.current_df = self.load_data(self.current_filepath)
        
        profile = self.create_profile(self.current_df)
        
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"Profile exported to {output_path}")
    
    def export_report(self, output_path: str):
        """Export drift results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.drift_results, f, indent=2, default=str)
        print(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect data drift and schema changes'
    )
    parser.add_argument('current', help='Path to current data file')
    parser.add_argument('--baseline', '-b',
                       help='Path to baseline data file or profile JSON')
    parser.add_argument('--export-profile', '-p',
                       help='Export current profile to JSON file')
    parser.add_argument('--export-report', '-r',
                       help='Export drift report to JSON file')
    
    args = parser.parse_args()
    
    # Load baseline if provided
    baseline_profile = None
    baseline_filepath = None
    
    if args.baseline:
        if args.baseline.endswith('.json'):
            with open(args.baseline, 'r') as f:
                baseline_profile = json.load(f)
        else:
            baseline_filepath = args.baseline
    
    detector = DataDriftDetector(
        current_filepath=args.current,
        baseline_filepath=baseline_filepath,
        baseline_profile=baseline_profile
    )
    
    results = detector.analyze_all()
    detector.print_report(results)
    
    if args.export_profile:
        detector.export_profile(args.export_profile)
    
    if args.export_report:
        detector.export_report(args.export_report)


if __name__ == '__main__':
    main()
