"""
Comprehensive Data Profiler
Automatically generates complete profiles of datasets including
data types, statistics, cardinality, and data quality indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    memory_bytes: int

@dataclass
class NumericProfile(ColumnProfile):
    min_value: float
    max_value: float
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    zeros_count: int
    zeros_percentage: float
    negative_count: int

@dataclass
class CategoricalProfile(ColumnProfile):
    top_value: Any
    top_frequency: int
    top_percentage: float
    top_5_values: Dict[Any, int]
    is_high_cardinality: bool

class DataProfiler:
    def __init__(self, df: pd.DataFrame, high_cardinality_threshold: float = 0.5):
        """
        Initialize the profiler.
        
        Args:
            df: DataFrame to profile
            high_cardinality_threshold: Ratio of unique/total values to flag high cardinality
        """
        self.df = df
        self.high_cardinality_threshold = high_cardinality_threshold
        self.numeric_profiles: List[NumericProfile] = []
        self.categorical_profiles: List[CategoricalProfile] = []
        
    def generate_overview(self) -> Dict[str, Any]:
        """Generate high-level dataset overview."""
        overview = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_missing_cells': self.df.isna().sum().sum(),
            'total_memory_mb': round(self.df.memory_usage(deep=True).sum() / (1024**2), 2),
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_percentage': round(self.df.duplicated().sum() / len(self.df) * 100, 2)
        }
        
        # Count column types
        overview['numeric_columns'] = len(self.df.select_dtypes(include=[np.number]).columns)
        overview['categorical_columns'] = len(self.df.select_dtypes(include=['object', 'category']).columns)
        overview['datetime_columns'] = len(self.df.select_dtypes(include=['datetime64']).columns)
        overview['boolean_columns'] = len(self.df.select_dtypes(include=['bool']).columns)
        
        return overview
    
    def profile_numeric_column(self, col: str) -> NumericProfile:
        """Profile a single numeric column."""
        series = self.df[col]
        
        # Basic info
        null_count = series.isna().sum()
        non_null = series.dropna()
        
        # Statistics
        stats = {
            'min_value': non_null.min() if len(non_null) > 0 else np.nan,
            'max_value': non_null.max() if len(non_null) > 0 else np.nan,
            'mean': non_null.mean() if len(non_null) > 0 else np.nan,
            'median': non_null.median() if len(non_null) > 0 else np.nan,
            'std': non_null.std() if len(non_null) > 0 else np.nan,
            'q25': non_null.quantile(0.25) if len(non_null) > 0 else np.nan,
            'q75': non_null.quantile(0.75) if len(non_null) > 0 else np.nan,
            'skewness': non_null.skew() if len(non_null) > 2 else np.nan,
            'kurtosis': non_null.kurtosis() if len(non_null) > 3 else np.nan
        }
        
        # Special counts
        zeros = (series == 0).sum()
        negatives = (series < 0).sum()
        
        return NumericProfile(
            name=col,
            dtype=str(series.dtype),
            non_null_count=len(non_null),
            null_count=null_count,
            null_percentage=round(null_count / len(self.df) * 100, 2),
            unique_count=series.nunique(),
            unique_percentage=round(series.nunique() / len(self.df) * 100, 2),
            memory_bytes=series.memory_usage(deep=True),
            min_value=round(stats['min_value'], 4) if not pd.isna(stats['min_value']) else None,
            max_value=round(stats['max_value'], 4) if not pd.isna(stats['max_value']) else None,
            mean=round(stats['mean'], 4) if not pd.isna(stats['mean']) else None,
            median=round(stats['median'], 4) if not pd.isna(stats['median']) else None,
            std=round(stats['std'], 4) if not pd.isna(stats['std']) else None,
            q25=round(stats['q25'], 4) if not pd.isna(stats['q25']) else None,
            q75=round(stats['q75'], 4) if not pd.isna(stats['q75']) else None,
            skewness=round(stats['skewness'], 4) if not pd.isna(stats['skewness']) else None,
            kurtosis=round(stats['kurtosis'], 4) if not pd.isna(stats['kurtosis']) else None,
            zeros_count=zeros,
            zeros_percentage=round(zeros / len(self.df) * 100, 2),
            negative_count=negatives
        )
    
    def profile_categorical_column(self, col: str) -> CategoricalProfile:
        """Profile a single categorical column."""
        series = self.df[col]
        
        # Basic info
        null_count = series.isna().sum()
        unique_count = series.nunique()
        
        # Value counts
        value_counts = series.value_counts()
        top_value = value_counts.index[0] if len(value_counts) > 0 else None
        top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        # Top 5 values
        top_5 = value_counts.head(5).to_dict()
        
        # Check if high cardinality
        is_high_card = (unique_count / len(self.df)) > self.high_cardinality_threshold
        
        return CategoricalProfile(
            name=col,
            dtype=str(series.dtype),
            non_null_count=len(series) - null_count,
            null_count=null_count,
            null_percentage=round(null_count / len(self.df) * 100, 2),
            unique_count=unique_count,
            unique_percentage=round(unique_count / len(self.df) * 100, 2),
            memory_bytes=series.memory_usage(deep=True),
            top_value=top_value,
            top_frequency=top_freq,
            top_percentage=round(top_freq / len(self.df) * 100, 2),
            top_5_values=top_5,
            is_high_cardinality=is_high_card
        )
    
    def detect_issues(self) -> List[Dict[str, Any]]:
        """Detect potential data quality issues."""
        issues = []
        
        for col in self.df.columns:
            series = self.df[col]
            
            # High missing rate
            missing_pct = series.isna().mean() * 100
            if missing_pct > 50:
                issues.append({
                    'column': col,
                    'issue': 'High Missing Rate',
                    'severity': 'High',
                    'detail': f'{missing_pct:.1f}% missing'
                })
            
            # Zero variance (constant column)
            if series.nunique() <= 1:
                issues.append({
                    'column': col,
                    'issue': 'Zero Variance',
                    'severity': 'High',
                    'detail': 'Column has only one unique value'
                })
            
            # High cardinality for categoricals
            if series.dtype in ['object', 'category']:
                unique_ratio = series.nunique() / len(self.df)
                if unique_ratio > self.high_cardinality_threshold:
                    issues.append({
                        'column': col,
                        'issue': 'High Cardinality',
                        'severity': 'Medium',
                        'detail': f'{series.nunique()} unique values ({unique_ratio*100:.1f}%)'
                    })
            
            # Potential data type mismatch
            if series.dtype == 'object':
                # Check if numeric strings
                try:
                    pd.to_numeric(series.dropna().head(100), errors='raise')
                    issues.append({
                        'column': col,
                        'issue': 'Potential Type Mismatch',
                        'severity': 'Low',
                        'detail': 'Stored as object but appears numeric'
                    })
                except (ValueError, TypeError):
                    pass
        
        return issues
    
    def generate_full_profile(self) -> Dict[str, Any]:
        """Generate complete data profile."""
        print("Generating data profile...")
        
        # Overview
        overview = self.generate_overview()
        
        # Profile numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.numeric_profiles.append(self.profile_numeric_column(col))
        
        # Profile categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.categorical_profiles.append(self.profile_categorical_column(col))
        
        # Detect issues
        issues = self.detect_issues()
        
        # Compile report
        report = {
            'overview': overview,
            'numeric_profiles': pd.DataFrame([vars(p) for p in self.numeric_profiles]),
            'categorical_profiles': pd.DataFrame([vars(p) for p in self.categorical_profiles]),
            'data_quality_issues': pd.DataFrame(issues) if issues else pd.DataFrame()
        }
        
        print("Profile generation complete!")
        return report
    
    def print_summary(self):
        """Print a human-readable summary."""
        report = self.generate_full_profile()
        
        print("\n" + "="*70)
        print("DATASET OVERVIEW")
        print("="*70)
        for key, value in report['overview'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*70)
        print("NUMERIC COLUMNS SUMMARY")
        print("="*70)
        if not report['numeric_profiles'].empty:
            summary_cols = ['name', 'null_percentage', 'mean', 'std', 'min_value', 'max_value', 'skewness']
            print(report['numeric_profiles'][summary_cols].to_string(index=False))
        else:
            print("No numeric columns found")
        
        print("\n" + "="*70)
        print("CATEGORICAL COLUMNS SUMMARY")
        print("="*70)
        if not report['categorical_profiles'].empty:
            summary_cols = ['name', 'unique_count', 'null_percentage', 'top_value', 'top_percentage', 'is_high_cardinality']
            print(report['categorical_profiles'][summary_cols].to_string(index=False))
        else:
            print("No categorical columns found")
        
        print("\n" + "="*70)
        print("DATA QUALITY ISSUES")
        print("="*70)
        if not report['data_quality_issues'].empty:
            print(report['data_quality_issues'].to_string(index=False))
        else:
            print("No major issues detected")
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.exponential(50000, 1000),
        'score': np.random.uniform(0, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000),
        'name': [f'User_{i}' for i in range(1000)],  # High cardinality
        'constant': ['same_value'] * 1000,  # Zero variance
        'numeric_string': [str(i) for i in range(1000)]  # Type mismatch
    })
    
    # Add missing values
    df.loc[np.random.choice(df.index, 600), 'score'] = np.nan  # High missing rate
    df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
    
    print("Sample Data:")
    print(df.head())
    
    # Generate profile
    profiler = DataProfiler(df, high_cardinality_threshold=0.5)
    profiler.print_summary()
    
    # Get detailed report
    report = profiler.generate_full_profile()
    
    # Save to CSV if needed
    # report['numeric_profiles'].to_csv('numeric_profile.csv', index=False)
    # report['categorical_profiles'].to_csv('categorical_profile.csv', index=False)

