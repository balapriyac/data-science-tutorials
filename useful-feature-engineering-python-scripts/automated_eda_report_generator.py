"""
Automated EDA Report Generator
Generates comprehensive exploratory data analysis reports with
statistics, visualizations, and data quality insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    memory_usage: int
    
@dataclass
class NumericProfile(ColumnProfile):
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    outliers_count: int
    zeros_count: int

@dataclass
class CategoricalProfile(ColumnProfile):
    mode: Any
    mode_freq: int
    mode_pct: float
    top_categories: Dict[str, int]

class EDAGenerator:
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df = df.copy()
        self.target = target
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if target and target in self.numeric_cols:
            self.numeric_cols.remove(target)
        if target and target in self.categorical_cols:
            self.categorical_cols.remove(target)
    
    def generate_overview(self) -> pd.DataFrame:
        """Generate high-level dataset overview."""
        overview = {
            'Total Rows': len(self.df),
            'Total Columns': len(self.df.columns),
            'Numeric Columns': len(self.numeric_cols),
            'Categorical Columns': len(self.categorical_cols),
            'Datetime Columns': len(self.datetime_cols),
            'Duplicate Rows': self.df.duplicated().sum(),
            'Total Missing Values': self.df.isna().sum().sum(),
            'Memory Usage (MB)': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return pd.DataFrame([overview]).T.rename(columns={0: 'Value'})
    
    def profile_numeric_columns(self) -> List[NumericProfile]:
        """Profile all numeric columns."""
        profiles = []
        
        for col in self.numeric_cols:
            series = self.df[col]
            
            # Basic stats
            missing = series.isna().sum()
            unique = series.nunique()
            
            # Descriptive statistics
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            
            # Advanced stats
            skew = series.skew()
            kurt = series.kurtosis()
            
            # Outliers (IQR method)
            IQR = q75 - q25
            lower_bound = q25 - 1.5 * IQR
            upper_bound = q75 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            
            # Zeros
            zeros = (series == 0).sum()
            
            profiles.append(NumericProfile(
                name=col,
                dtype=str(series.dtype),
                missing_count=missing,
                missing_pct=round(missing / len(self.df) * 100, 2),
                unique_count=unique,
                unique_pct=round(unique / len(self.df) * 100, 2),
                memory_usage=series.memory_usage(deep=True),
                mean=round(mean_val, 4),
                median=round(median_val, 4),
                std=round(std_val, 4),
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
                q25=round(q25, 4),
                q75=round(q75, 4),
                skewness=round(skew, 4),
                kurtosis=round(kurt, 4),
                outliers_count=outliers,
                zeros_count=zeros
            ))
        
        return profiles
    
    def profile_categorical_columns(self) -> List[CategoricalProfile]:
        """Profile all categorical columns."""
        profiles = []
        
        for col in self.categorical_cols:
            series = self.df[col]
            
            missing = series.isna().sum()
            unique = series.nunique()
            
            value_counts = series.value_counts()
            mode_val = value_counts.index[0] if len(value_counts) > 0 else None
            mode_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            # Top 5 categories
            top_cats = value_counts.head(5).to_dict()
            
            profiles.append(CategoricalProfile(
                name=col,
                dtype=str(series.dtype),
                missing_count=missing,
                missing_pct=round(missing / len(self.df) * 100, 2),
                unique_count=unique,
                unique_pct=round(unique / len(self.df) * 100, 2),
                memory_usage=series.memory_usage(deep=True),
                mode=mode_val,
                mode_freq=mode_freq,
                mode_pct=round(mode_freq / len(self.df) * 100, 2),
                top_categories=top_cats
            ))
        
        return profiles
    
    def detect_data_quality_issues(self) -> pd.DataFrame:
        """Detect potential data quality issues."""
        issues = []
        
        # High missing value columns
        for col in self.df.columns:
            missing_pct = self.df[col].isna().mean() * 100
            if missing_pct > 50:
                issues.append({
                    'column': col,
                    'issue_type': 'High Missing Rate',
                    'severity': 'High',
                    'details': f'{missing_pct:.1f}% missing values'
                })
        
        # High cardinality categorical columns
        for col in self.categorical_cols:
            unique_pct = self.df[col].nunique() / len(self.df) * 100
            if unique_pct > 90:
                issues.append({
                    'column': col,
                    'issue_type': 'High Cardinality',
                    'severity': 'Medium',
                    'details': f'{self.df[col].nunique()} unique values ({unique_pct:.1f}%)'
                })
        
        # Constant or near-constant columns
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < 0.01 and self.df[col].nunique() > 1:
                issues.append({
                    'column': col,
                    'issue_type': 'Low Variance',
                    'severity': 'Medium',
                    'details': f'Only {self.df[col].nunique()} unique values'
                })
        
        # Highly skewed numeric columns
        for col in self.numeric_cols:
            skew = abs(self.df[col].skew())
            if skew > 3:
                issues.append({
                    'column': col,
                    'issue_type': 'High Skewness',
                    'severity': 'Low',
                    'details': f'Skewness = {skew:.2f}'
                })
        
        return pd.DataFrame(issues) if issues else pd.DataFrame(columns=['column', 'issue_type', 'severity', 'details'])
    
    def calculate_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        if len(self.numeric_cols) == 0:
            return pd.DataFrame()
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        return corr_matrix
    
    def find_high_correlations(self, threshold: float = 0.8) -> pd.DataFrame:
        """Find pairs of highly correlated features."""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        corr_matrix = self.calculate_correlations()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': round(corr_matrix.iloc[i, j], 4)
                    })
        
        return pd.DataFrame(high_corr).sort_values('correlation', ascending=False, key=abs) if high_corr else pd.DataFrame()
    
    def target_analysis(self) -> Dict[str, Any]:
        """Analyze relationship between features and target."""
        if not self.target or self.target not in self.df.columns:
            return {}
        
        results = {'target': self.target, 'correlations': {}, 'importance': {}}
        
        target_series = self.df[self.target]
        
        # For numeric target
        if pd.api.types.is_numeric_dtype(target_series):
            for col in self.numeric_cols:
                corr = self.df[col].corr(target_series)
                results['correlations'][col] = round(corr, 4)
            
            results['correlations'] = dict(sorted(
                results['correlations'].items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
        
        return results
    
    def plot_distributions(self, max_cols: int = 10, figsize: Tuple[int, int] = (15, 10)):
        """Plot distributions for numeric columns."""
        cols_to_plot = self.numeric_cols[:max_cols]
        if not cols_to_plot:
            return
        
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            axes[idx].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}\nSkew: {self.df[col].skew():.2f}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10)):
        """Plot correlation heatmap for numeric features."""
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for correlation heatmap")
            return
        
        corr_matrix = self.calculate_correlations()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate complete EDA report."""
        report = {
            'overview': self.generate_overview(),
            'numeric_profiles': pd.DataFrame([vars(p) for p in self.profile_numeric_columns()]),
            'categorical_profiles': pd.DataFrame([vars(p) for p in self.profile_categorical_columns()]),
            'data_quality_issues': self.detect_data_quality_issues(),
            'high_correlations': self.find_high_correlations(),
            'target_analysis': self.target_analysis()
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(35, 10, 100),
        'income': np.random.exponential(50000, 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 100),
        'target': np.random.binomial(1, 0.3, 100)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 10), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'category'] = np.nan
    
    print("Sample Dataset:")
    print(df.head())
    print("\n" + "="*60 + "\n")
    
    # Generate EDA report
    eda = EDAGenerator(df, target='target')
    report = eda.generate_full_report()
    
    print("Dataset Overview:")
    print(report['overview'])
    print("\n" + "="*60 + "\n")
    
    print("Numeric Column Profiles:")
    print(report['numeric_profiles'])
    print("\n" + "="*60 + "\n")
    
    print("Categorical Column Profiles:")
    print(report['categorical_profiles'])
    print("\n" + "="*60 + "\n")
    
    print("Data Quality Issues:")
    print(report['data_quality_issues'])
    print("\n" + "="*60 + "\n")
    
    print("High Correlations:")
    print(report['high_correlations'])
    print("\n" + "="*60 + "\n")
    
    print("Target Analysis:")
    print(report['target_analysis'])
    
    # Generate visualizations
    eda.plot_distributions()
  
    eda.plot_correlation_heatmap()
