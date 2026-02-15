"""
Missing Data Pattern Analyzer
Analyzes patterns in missing data, classifies missingness mechanisms,
and provides visualization and imputation recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MissingDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the missing data analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        
    def get_missing_summary(self) -> pd.DataFrame:
        """Get summary of missing values across all columns."""
        missing_data = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_data.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': round(missing_count / len(self.df) * 100, 2),
                    'dtype': str(self.df[col].dtype),
                    'non_missing_count': len(self.df) - missing_count
                })
        
        if not missing_data:
            return pd.DataFrame()
        
        return pd.DataFrame(missing_data).sort_values('missing_percentage', ascending=False)
    
    def analyze_missingness_patterns(self) -> pd.DataFrame:
        """Analyze patterns of co-occurring missing values."""
        # Create binary missingness matrix
        missing_matrix = self.df.isna().astype(int)
        
        # Find columns with missing values
        cols_with_missing = [col for col in missing_matrix.columns if missing_matrix[col].sum() > 0]
        
        if len(cols_with_missing) < 2:
            return pd.DataFrame()
        
        # Calculate correlation between missingness patterns
        missing_corr = missing_matrix[cols_with_missing].corr()
        
        # Find significant correlations
        patterns = []
        for i in range(len(missing_corr.columns)):
            for j in range(i + 1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.3:  # Significant correlation
                    col1, col2 = missing_corr.columns[i], missing_corr.columns[j]
                    
                    # Count co-occurrence
                    both_missing = ((self.df[col1].isna()) & (self.df[col2].isna())).sum()
                    
                    patterns.append({
                        'column_1': col1,
                        'column_2': col2,
                        'correlation': round(corr_val, 4),
                        'both_missing_count': both_missing,
                        'both_missing_pct': round(both_missing / len(self.df) * 100, 2)
                    })
        
        if not patterns:
            return pd.DataFrame()
        
        return pd.DataFrame(patterns).sort_values('correlation', ascending=False, key=abs)
    
    def classify_missingness_type(self, column: str, test_columns: Optional[List[str]] = None) -> Dict:
        """
        Classify missingness mechanism for a column.
        
        Args:
            column: Column to analyze
            test_columns: Columns to test for MAR pattern
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        missing_mask = self.df[column].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            return {'column': column, 'type': 'No missing values', 'confidence': 'N/A'}
        
        # Test columns (use numeric columns by default)
        if test_columns is None:
            test_columns = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                          if c != column and self.df[c].notna().sum() > 0]
        
        # Test for MAR: is missingness related to other variables?
        mar_evidence = []
        
        for test_col in test_columns[:10]:  # Limit to 10 tests
            if self.df[test_col].dtype in ['object', 'category']:
                # Chi-square test for categorical
                try:
                    contingency = pd.crosstab(self.df[test_col].fillna('_missing_'), missing_mask)
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                    if p_value < 0.05:
                        mar_evidence.append((test_col, p_value))
                except:
                    pass
            else:
                # T-test for numeric
                try:
                    group1 = self.df.loc[missing_mask, test_col].dropna()
                    group2 = self.df.loc[~missing_mask, test_col].dropna()
                    
                    if len(group1) > 1 and len(group2) > 1:
                        _, p_value = stats.ttest_ind(group1, group2)
                        if p_value < 0.05:
                            mar_evidence.append((test_col, p_value))
                except:
                    pass
        
        # Classify based on evidence
        if len(mar_evidence) > 0:
            missingness_type = 'MAR (Missing At Random)'
            confidence = 'High' if len(mar_evidence) >= 3 else 'Medium'
            related_vars = [col for col, _ in mar_evidence[:5]]
        else:
            # Check if completely random (MCAR) vs not random (MNAR)
            # Simple heuristic: if missing values are scattered (no obvious pattern), likely MCAR
            if missing_count / len(self.df) < 0.05:
                missingness_type = 'MCAR (Missing Completely At Random)'
                confidence = 'Low'
            else:
                missingness_type = 'Possibly MNAR (Missing Not At Random)'
                confidence = 'Low'
            related_vars = []
        
        return {
            'column': column,
            'missingness_type': missingness_type,
            'confidence': confidence,
            'related_variables': related_vars,
            'evidence_count': len(mar_evidence)
        }
    
    def recommend_strategy(self, column: str) -> str:
        """Recommend imputation strategy based on missingness analysis."""
        classification = self.classify_missingness_type(column)
        missing_pct = self.df[column].isna().mean() * 100
        dtype = self.df[column].dtype
        
        # High missing rate
        if missing_pct > 50:
            return "Consider dropping column (>50% missing)"
        
        # Based on missingness type
        miss_type = classification['missingness_type']
        
        if 'MCAR' in miss_type:
            if pd.api.types.is_numeric_dtype(dtype):
                return "Mean/Median imputation (MCAR)"
            else:
                return "Mode imputation (MCAR)"
        
        elif 'MAR' in miss_type:
            return "Predictive imputation using related variables"
        
        else:  # MNAR
            return "Domain-specific imputation or create missing indicator"
    
    def plot_missing_heatmap(self, figsize: Tuple[int, int] = (12, 8), max_cols: int = 30):
        """
        Plot heatmap showing missing value patterns.
        
        Args:
            figsize: Figure size
            max_cols: Maximum columns to display
        """
        # Get columns with missing values
        cols_with_missing = [col for col in self.df.columns if self.df[col].isna().sum() > 0]
        cols_with_missing = cols_with_missing[:max_cols]
        
        if not cols_with_missing:
            print("No missing values to visualize")
            return
        
        plt.figure(figsize=figsize)
        
        # Create missingness matrix
        missing_matrix = self.df[cols_with_missing].isna()
        
        # Plot heatmap
        sns.heatmap(
            missing_matrix.T,
            cbar=False,
            yticklabels=True,
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1
        )
        
        plt.title('Missing Value Pattern (Yellow = Missing)')
        plt.xlabel('Row Index')
        plt.ylabel('Column')
        plt.tight_layout()
        plt.show()
    
    def plot_missing_bar(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot bar chart of missing value percentages."""
        summary = self.get_missing_summary()
        
        if summary.empty:
            print("No missing values to plot")
            return
        
        plt.figure(figsize=figsize)
        
        plt.barh(summary['column'], summary['missing_percentage'], color='coral', edgecolor='black')
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Column')
        plt.title('Missing Value Percentage by Column')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (col, pct) in enumerate(zip(summary['column'], summary['missing_percentage'])):
            plt.text(pct + 1, i, f'{pct:.1f}%', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_missing_correlation(self, figsize: Tuple[int, int] = (10, 8)):
        """Plot correlation matrix of missingness patterns."""
        # Create binary missingness matrix
        missing_matrix = self.df.isna().astype(int)
        
        # Get columns with missing values
        cols_with_missing = [col for col in missing_matrix.columns if missing_matrix[col].sum() > 0]
        
        if len(cols_with_missing) < 2:
            print("Need at least 2 columns with missing values")
            return
        
        # Calculate correlation
        missing_corr = missing_matrix[cols_with_missing].corr()
        
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            missing_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation of Missing Value Patterns')
        plt.tight_layout()
        plt.show()
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive missing data report."""
        report = {
            'summary': self.get_missing_summary(),
            'patterns': self.analyze_missingness_patterns(),
            'classifications': [],
            'recommendations': []
        }
        
        # Classify each column with missing values
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                classification = self.classify_missingness_type(col)
                recommendation = self.recommend_strategy(col)
                
                report['classifications'].append(classification)
                report['recommendations'].append({
                    'column': col,
                    'recommendation': recommendation
                })
        
        report['classifications'] = pd.DataFrame(report['classifications'])
        report['recommendations'] = pd.DataFrame(report['recommendations'])
        
        return report


# Example usage
if __name__ == "__main__":
    # Create dataset with missing values in different patterns
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'id': range(n),
        'age': np.random.normal(35, 10, n),
        'income': np.random.exponential(50000, n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n)
    })
    
    # MCAR: Random missing values
    df.loc[np.random.choice(df.index, 50), 'score'] = np.nan
    
    # MAR: Income missing when age > 60 (related to age)
    df.loc[df['age'] > 60, 'income'] = np.nan
    
    # Pattern: Age and score often missing together
    missing_both = np.random.choice(df.index, 80)
    df.loc[missing_both, 'age'] = np.nan
    df.loc[missing_both, 'score'] = np.nan
    
    # High missing rate column
    df.loc[np.random.choice(df.index, 600), 'city'] = np.nan
    
    print("Sample Data:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print("\n" + "="*70 + "\n")
    
    # Initialize analyzer
    analyzer = MissingDataAnalyzer(df)
    
    # Generate full report
    report = analyzer.generate_full_report()
    
    print("Missing Value Summary:")
    print(report['summary'].to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    print("Missingness Patterns (Co-occurrence):")
    if not report['patterns'].empty:
        print(report['patterns'].to_string(index=False))
    else:
        print("No significant patterns detected")
    print("\n" + "="*70 + "\n")
    
    print("Missingness Classifications:")
    print(report['classifications'].to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    print("Imputation Recommendations:")
    print(report['recommendations'].to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Visualizations
    print("Generating missing data visualizations...")
    analyzer.plot_missing_bar()
    analyzer.plot_missing_heatmap()
    analyzer.plot_missing_correlation()

