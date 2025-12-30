"""
Outlier Detection Engine
Identifies statistical and domain-specific outliers in datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

class OutlierDetector:
    def __init__(self, filepath, domain_rules=None):
        """
        Initialize outlier detector
        
        Args:
            filepath: Path to data file
            domain_rules: Dict of domain-specific validation rules
                Example: {
                    'age': {'min': 0, 'max': 120},
                    'price': {'min': 0},
                    'percentage': {'min': 0, 'max': 100}
                }
        """
        self.filepath = Path(filepath)
        self.df = self._load_data()
        self.domain_rules = domain_rules or {}
        self.outliers = []
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
    
    def detect_statistical_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers using statistical methods
        
        Args:
            method: 'iqr', 'zscore', or 'modified_zscore'
            threshold: Sensitivity threshold (1.5 for IQR, 3 for z-score)
        """
        print(f"Detecting statistical outliers using {method} method...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                continue
            
            if method == 'iqr':
                outlier_indices = self._iqr_outliers(col, data, threshold)
            elif method == 'zscore':
                outlier_indices = self._zscore_outliers(col, data, threshold)
            elif method == 'modified_zscore':
                outlier_indices = self._modified_zscore_outliers(col, data, threshold)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            for idx in outlier_indices:
                value = self.df.loc[idx, col]
                self.outliers.append({
                    'column': col,
                    'row': int(idx),
                    'value': float(value),
                    'method': method,
                    'type': 'statistical',
                    'severity': self._calculate_severity(col, value, data),
                    'message': f"Statistical outlier detected ({method})"
                })
    
    def _iqr_outliers(self, column, data, threshold):
        """Detect outliers using Interquartile Range method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return self.df[outlier_mask].index.tolist()
    
    def _zscore_outliers(self, column, data, threshold):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        outlier_mask = z_scores > threshold
        
        # Map back to original indices
        return data[outlier_mask].index.tolist()
    
    def _modified_zscore_outliers(self, column, data, threshold):
        """Detect outliers using Modified Z-score (based on MAD)"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return []
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return data[outlier_mask].index.tolist()
    
    def _calculate_severity(self, column, value, data):
        """Calculate outlier severity based on distance from normal range"""
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return 'low'
        
        z_score = abs((value - mean) / std)
        
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'
    
    def detect_domain_outliers(self):
        """Detect outliers based on domain-specific rules"""
        print("Detecting domain-specific outliers...")
        
        for column, rules in self.domain_rules.items():
            if column not in self.df.columns:
                continue
            
            data = self.df[column]
            
            # Check minimum value
            if 'min' in rules:
                min_val = rules['min']
                violations = self.df[data < min_val]
                
                for idx, row in violations.iterrows():
                    self.outliers.append({
                        'column': column,
                        'row': int(idx),
                        'value': float(row[column]),
                        'method': 'domain_rule',
                        'type': 'domain',
                        'severity': 'high',
                        'message': f"Value {row[column]} below minimum {min_val}"
                    })
            
            # Check maximum value
            if 'max' in rules:
                max_val = rules['max']
                violations = self.df[data > max_val]
                
                for idx, row in violations.iterrows():
                    self.outliers.append({
                        'column': column,
                        'row': int(idx),
                        'value': float(row[column]),
                        'method': 'domain_rule',
                        'type': 'domain',
                        'severity': 'high',
                        'message': f"Value {row[column]} exceeds maximum {max_val}"
                    })
            
            # Check custom validation function
            if 'validator' in rules:
                validator_func = rules['validator']
                for idx, value in data.items():
                    if pd.notna(value) and not validator_func(value):
                        self.outliers.append({
                            'column': column,
                            'row': int(idx),
                            'value': float(value),
                            'method': 'custom_validator',
                            'type': 'domain',
                            'severity': 'medium',
                            'message': f"Failed custom validation"
                        })
    
    def detect_impossible_values(self):
        """Detect mathematically or logically impossible values"""
        print("Detecting impossible values...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col]
            
            # Check for infinity
            inf_mask = np.isinf(data)
            if inf_mask.any():
                for idx in self.df[inf_mask].index:
                    self.outliers.append({
                        'column': col,
                        'row': int(idx),
                        'value': str(data[idx]),
                        'method': 'impossible_value',
                        'type': 'impossible',
                        'severity': 'critical',
                        'message': "Infinite value detected"
                    })
            
            # Check for extreme values (beyond float limits)
            extreme_mask = (data > 1e308) | (data < -1e308)
            if extreme_mask.any():
                for idx in self.df[extreme_mask].index:
                    self.outliers.append({
                        'column': col,
                        'row': int(idx),
                        'value': float(data[idx]),
                        'method': 'impossible_value',
                        'type': 'impossible',
                        'severity': 'critical',
                        'message': "Extreme value beyond normal range"
                    })
    
    def analyze_all(self, statistical_method='iqr', threshold=1.5):
        """
        Run complete outlier detection
        
        Args:
            statistical_method: Method for statistical outlier detection
            threshold: Threshold for statistical detection
        """
        print(f"Analyzing {len(self.df)} rows for outliers...")
        
        # Reset outliers
        self.outliers = []
        
        # Detect different types of outliers
        self.detect_impossible_values()
        self.detect_domain_outliers()
        self.detect_statistical_outliers(method=statistical_method, threshold=threshold)
        
        # Remove duplicates (same row/column combinations)
        seen = set()
        unique_outliers = []
        for outlier in self.outliers:
            key = (outlier['row'], outlier['column'])
            if key not in seen:
                seen.add(key)
                unique_outliers.append(outlier)
        
        self.outliers = unique_outliers
        
        # Calculate statistics
        self._calculate_stats()
        
        return self.stats
    
    def _calculate_stats(self):
        """Calculate outlier statistics"""
        total_outliers = len(self.outliers)
        
        # Count by column
        column_counts = {}
        for outlier in self.outliers:
            col = outlier['column']
            column_counts[col] = column_counts.get(col, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for outlier in self.outliers:
            sev = outlier['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Count by type
        type_counts = {}
        for outlier in self.outliers:
            otype = outlier['type']
            type_counts[otype] = type_counts.get(otype, 0) + 1
        
        self.stats = {
            'total_rows': len(self.df),
            'total_outliers': total_outliers,
            'outlier_percentage': round((total_outliers / len(self.df)) * 100, 2),
            'columns_affected': len(column_counts),
            'outliers_by_column': column_counts,
            'outliers_by_severity': severity_counts,
            'outliers_by_type': type_counts
        }
    
    def generate_visualization(self, output_path='outliers_report.png', max_cols=6):
        """Generate visualization of outliers"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:max_cols]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            # Create box plot
            data = self.df[col].dropna()
            ax.boxplot(data, vert=True)
            ax.set_title(col)
            ax.set_ylabel('Value')
            
            # Mark outliers from our detection
            col_outliers = [o for o in self.outliers if o['column'] == col]
            if col_outliers:
                outlier_values = [o['value'] for o in col_outliers]
                ax.scatter([1] * len(outlier_values), outlier_values, 
                          color='red', s=50, alpha=0.6, label='Detected Outliers')
                ax.legend()
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    def print_report(self):
        """Print outlier detection report"""
        print("\n" + "="*70)
        print("OUTLIER DETECTION REPORT")
        print("="*70)
        print(f"Dataset: {self.filepath.name}")
        print(f"Total Rows: {self.stats['total_rows']:,}")
        print("="*70)
        
        if self.stats['total_outliers'] == 0:
            print("\n✓ No outliers detected!")
        else:
            print(f"\nOUTLIER SUMMARY:")
            print(f"  Total Outliers: {self.stats['total_outliers']} ({self.stats['outlier_percentage']}% of rows)")
            print(f"  Columns Affected: {self.stats['columns_affected']}")
            
            print(f"\nBY SEVERITY:")
            for severity in ['critical', 'high', 'medium', 'low']:
                count = self.stats['outliers_by_severity'].get(severity, 0)
                if count > 0:
                    print(f"  {severity.upper()}: {count}")
            
            print(f"\nBY TYPE:")
            for otype, count in self.stats['outliers_by_type'].items():
                print(f"  {otype.capitalize()}: {count}")
            
            print(f"\nTOP COLUMNS WITH OUTLIERS:")
            sorted_cols = sorted(self.stats['outliers_by_column'].items(), 
                               key=lambda x: x[1], reverse=True)
            for col, count in sorted_cols[:10]:
                print(f"  {col}: {count} outliers")
            
            print(f"\nSAMPLE OUTLIERS (highest severity first):")
            sorted_outliers = sorted(self.outliers, 
                                    key=lambda x: {'critical': 0, 'high': 1, 
                                                  'medium': 2, 'low': 3}[x['severity']])
            
            for outlier in sorted_outliers[:10]:
                print(f"\n  Row {outlier['row']}, Column '{outlier['column']}':")
                print(f"    Value: {outlier['value']}")
                print(f"    Severity: {outlier['severity'].upper()}")
                print(f"    Reason: {outlier['message']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_outliers(self, output_path='outliers_detailed.json'):
        """Export detailed outlier analysis"""
        output = {
            'statistics': self.stats,
            'outliers': self.outliers
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Outlier analysis exported to {output_path}")
    
    def export_cleaned_data(self, output_path=None, strategy='remove'):
        """
        Export dataset with outliers handled
        
        Args:
            output_path: Output file path
            strategy: 'remove', 'cap', or 'flag'
        """
        if output_path is None:
            output_path = self.filepath.stem + '_cleaned' + self.filepath.suffix
        
        clean_df = self.df.copy()
        
        if strategy == 'remove':
            # Remove rows with critical/high severity outliers
            rows_to_remove = set()
            for outlier in self.outliers:
                if outlier['severity'] in ['critical', 'high']:
                    rows_to_remove.add(outlier['row'])
            
            clean_df = clean_df.drop(index=list(rows_to_remove))
            print(f"Removed {len(rows_to_remove)} rows with critical/high severity outliers")
        
        elif strategy == 'flag':
            # Add flag column
            clean_df['outlier_flag'] = False
            for outlier in self.outliers:
                clean_df.loc[outlier['row'], 'outlier_flag'] = True
        
        clean_df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python outlier_detector.py <filepath> [--method iqr|zscore] [--threshold N]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Parse arguments
    method = 'iqr'
    threshold = 1.5
    for i, arg in enumerate(sys.argv):
        if arg == '--method' and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
        elif arg == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
    
    # Example domain rules
    domain_rules = {
        'age': {'min': 0, 'max': 120},
        'price': {'min': 0},
        'percentage': {'min': 0, 'max': 100}
    }
    
    detector = OutlierDetector(filepath, domain_rules=domain_rules)
    detector.analyze_all(statistical_method=method, threshold=threshold)
    detector.print_report()
    detector.export_outliers()
    detector.generate_visualization()

