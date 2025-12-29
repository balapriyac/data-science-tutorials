"""
Missing Data Analyzer
Comprehensive scanner for missing data patterns in datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

class MissingDataAnalyzer:
    def __init__(self, filepath, missing_indicators=None):
        """
        Initialize the analyzer with a dataset
        
        Args:
            filepath: Path to CSV, Excel, or JSON file
            missing_indicators: List of additional values to treat as missing
        """
        self.filepath = Path(filepath)
        self.df = self._load_data()
        self.missing_indicators = missing_indicators or ['N/A', 'NA', 'n/a', 'Unknown', 
                                                          'unknown', 'UNKNOWN', '', ' ', 'null', 'NULL']
        self.report = {}
        
    def _load_data(self):
        """Load data from various file formats"""
        suffix = self.filepath.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(self.filepath)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(self.filepath)
        elif suffix == '.json':
            return pd.read_json(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _standardize_missing(self):
        """Replace various missing indicators with NaN"""
        for indicator in self.missing_indicators:
            self.df.replace(indicator, np.nan, inplace=True)
    
    def analyze(self):
        """Perform comprehensive missing data analysis"""
        self._standardize_missing()
        
        # Overall statistics
        total_cells = self.df.size
        missing_cells = self.df.isna().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        self.report['overall'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'completeness_percentage': round(completeness, 2)
        }
        
        # Column-level analysis
        column_stats = []
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            column_stats.append({
                'column': col,
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'present_count': int(len(self.df) - missing_count),
                'status': self._get_status(missing_pct)
            })
        
        column_stats.sort(key=lambda x: x['missing_percentage'], reverse=True)
        self.report['columns'] = column_stats
        
        # Row-level analysis
        rows_with_missing = self.df.isna().any(axis=1).sum()
        rows_complete = len(self.df) - rows_with_missing
        
        self.report['rows'] = {
            'complete_rows': int(rows_complete),
            'rows_with_missing': int(rows_with_missing),
            'rows_with_missing_percentage': round((rows_with_missing / len(self.df)) * 100, 2)
        }
        
        # Patterns in missingness
        self._analyze_patterns()
        
        return self.report
    
    def _get_status(self, missing_pct):
        """Categorize column status based on missing percentage"""
        if missing_pct == 0:
            return 'Complete'
        elif missing_pct < 5:
            return 'Excellent'
        elif missing_pct < 20:
            return 'Good'
        elif missing_pct < 50:
            return 'Fair'
        elif missing_pct < 80:
            return 'Poor'
        else:
            return 'Critical'
    
    def _analyze_patterns(self):
        """Identify patterns in missing data"""
        # Find columns with correlated missingness
        missing_matrix = self.df.isna().astype(int)
        correlations = missing_matrix.corr()
        
        high_correlations = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr_value = correlations.iloc[i, j]
                if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                    high_correlations.append({
                        'column1': correlations.columns[i],
                        'column2': correlations.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        self.report['patterns'] = {
            'correlated_missingness': high_correlations,
            'note': 'High correlation suggests missing values appear together systematically'
        }
    
    def generate_visualization(self, output_path='missing_data_report.png'):
        """Generate visualization of missing data patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Missing Data Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Missing percentage by column
        col_stats = pd.DataFrame(self.report['columns'])
        top_cols = col_stats.head(10)
        
        axes[0, 0].barh(top_cols['column'], top_cols['missing_percentage'], color='coral')
        axes[0, 0].set_xlabel('Missing Percentage (%)')
        axes[0, 0].set_title('Top 10 Columns by Missing Data')
        axes[0, 0].invert_yaxis()
        
        # 2. Overall completeness pie chart
        complete = self.report['overall']['total_cells'] - self.report['overall']['missing_cells']
        missing = self.report['overall']['missing_cells']
        
        axes[0, 1].pie([complete, missing], labels=['Complete', 'Missing'], 
                       autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
        axes[0, 1].set_title('Overall Data Completeness')
        
        # 3. Missing data heatmap (sample)
        sample_size = min(50, len(self.df))
        missing_matrix = self.df.head(sample_size).isna()
        
        sns.heatmap(missing_matrix.T, cbar=False, cmap='RdYlGn_r', ax=axes[1, 0])
        axes[1, 0].set_title(f'Missing Data Pattern (First {sample_size} Rows)')
        axes[1, 0].set_xlabel('Row Index')
        axes[1, 0].set_ylabel('Columns')
        
        # 4. Status distribution
        status_counts = col_stats['status'].value_counts()
        axes[1, 1].bar(status_counts.index, status_counts.values, color='skyblue')
        axes[1, 1].set_xlabel('Status Category')
        axes[1, 1].set_ylabel('Number of Columns')
        axes[1, 1].set_title('Column Status Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        
    def print_report(self):
        """Print formatted report to console"""
        print("\n" + "="*70)
        print("MISSING DATA ANALYSIS REPORT")
        print("="*70)
        print(f"Dataset: {self.filepath.name}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Overall statistics
        overall = self.report['overall']
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Rows: {overall['total_rows']:,}")
        print(f"  Total Columns: {overall['total_columns']}")
        print(f"  Total Cells: {overall['total_cells']:,}")
        print(f"  Missing Cells: {overall['missing_cells']:,}")
        print(f"  Completeness: {overall['completeness_percentage']}%")
        
        # Row statistics
        rows = self.report['rows']
        print(f"\nROW ANALYSIS:")
        print(f"  Complete Rows: {rows['complete_rows']:,}")
        print(f"  Rows with Missing Data: {rows['rows_with_missing']:,} ({rows['rows_with_missing_percentage']}%)")
        
        # Column details
        print(f"\nCOLUMN DETAILS (sorted by missing percentage):")
        print(f"{'Column':<30} {'Missing':<10} {'%':<8} {'Status':<12}")
        print("-"*70)
        for col in self.report['columns'][:15]:  # Show top 15
            print(f"{col['column']:<30} {col['missing_count']:<10} {col['missing_percentage']:<8.2f} {col['status']:<12}")
        
        if len(self.report['columns']) > 15:
            print(f"... and {len(self.report['columns']) - 15} more columns")
        
        # Patterns
        patterns = self.report['patterns']
        if patterns['correlated_missingness']:
            print(f"\nCORRELATED MISSINGNESS DETECTED:")
            for corr in patterns['correlated_missingness'][:5]:
                print(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f}")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        critical = [c for c in self.report['columns'] if c['status'] == 'Critical']
        poor = [c for c in self.report['columns'] if c['status'] == 'Poor']
        
        if critical:
            print(f"  - Consider removing {len(critical)} columns with >80% missing data")
        if poor:
            print(f"  - Investigate {len(poor)} columns with 50-80% missing data")
        if self.report['patterns']['correlated_missingness']:
            print(f"  - Review {len(self.report['patterns']['correlated_missingness'])} correlated missing patterns")
        
        print("="*70 + "\n")
    
    def export_report(self, output_path='missing_data_report.json'):
        """Export detailed report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"Detailed report exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python missing_data_analyzer.py <filepath>")
        print("Example: python missing_data_analyzer.py data.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Initialize analyzer
    analyzer = MissingDataAnalyzer(filepath)
    
    # Run analysis
    analyzer.analyze()
    
    # Generate outputs
    analyzer.print_report()
    analyzer.export_report()
    analyzer.generate_visualization()


  
