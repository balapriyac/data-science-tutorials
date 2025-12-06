"""
Outlier Detector and Treatment
Detects outliers using multiple statistical methods and applies
configurable treatment strategies with detailed reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

class DetectionMethod(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    PERCENTILE = "percentile"

class TreatmentStrategy(Enum):
    REMOVE = "remove"
    CAP = "cap"
    WINSORIZE = "winsorize"
    FLAG = "flag"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"

@dataclass
class OutlierReport:
    column: str
    method: str
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_pct: float
    treatment: str
    outlier_values: List[float]

class OutlierDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.outlier_masks: Dict[str, pd.Series] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.reports: List[OutlierReport] = []
    
    def detect(
        self,
        columns: Optional[List[str]] = None,
        method: Literal['iqr', 'zscore', 'modified_zscore', 'percentile'] = 'iqr',
        threshold: float = 1.5,
        percentile_range: Tuple[float, float] = (0.01, 0.99)
    ) -> pd.DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            columns: Columns to check (default: all numeric)
            method: Detection method to use
            threshold: Sensitivity threshold (IQR multiplier or Z-score cutoff)
            percentile_range: Lower and upper percentile bounds for percentile method
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            series = self.df[col].dropna()
            
            if method == 'iqr':
                mask, lower, upper = self._detect_iqr(col, threshold)
            elif method == 'zscore':
                mask, lower, upper = self._detect_zscore(col, threshold)
            elif method == 'modified_zscore':
                mask, lower, upper = self._detect_modified_zscore(col, threshold)
            elif method == 'percentile':
                mask, lower, upper = self._detect_percentile(col, percentile_range)
            
            self.outlier_masks[col] = mask
            self.bounds[col] = (lower, upper)
            
            outlier_count = mask.sum()
            outlier_values = self.df.loc[mask, col].tolist()
            
            outlier_summary.append({
                'column': col,
                'method': method,
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'outlier_count': outlier_count,
                'outlier_pct': round(outlier_count / len(self.df) * 100, 2),
                'min_outlier': min(outlier_values) if outlier_values else None,
                'max_outlier': max(outlier_values) if outlier_values else None
            })
        
        return pd.DataFrame(outlier_summary)
    
    def _detect_iqr(self, col: str, threshold: float) -> Tuple[pd.Series, float, float]:
        """Detect outliers using Interquartile Range method."""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        mask = (self.df[col] < lower) | (self.df[col] > upper)
        return mask, lower, upper
    
    def _detect_zscore(self, col: str, threshold: float) -> Tuple[pd.Series, float, float]:
        """Detect outliers using Z-score method."""
        mean = self.df[col].mean()
        std = self.df[col].std()
        
        z_scores = np.abs((self.df[col] - mean) / std)
        
        lower = mean - threshold * std
        upper = mean + threshold * std
        
        mask = z_scores > threshold
        return mask, lower, upper
    
    def _detect_modified_zscore(self, col: str, threshold: float) -> Tuple[pd.Series, float, float]:
        """Detect outliers using Modified Z-score (MAD-based) method."""
        median = self.df[col].median()
        mad = np.median(np.abs(self.df[col] - median))
        
        # Avoid division by zero
        mad = mad if mad != 0 else 1e-10
        
        modified_z = 0.6745 * (self.df[col] - median) / mad
        
        lower = median - threshold * mad / 0.6745
        upper = median + threshold * mad / 0.6745
        
        mask = np.abs(modified_z) > threshold
        return mask, lower, upper
    
    def _detect_percentile(self, col: str, pct_range: Tuple[float, float]) -> Tuple[pd.Series, float, float]:
        """Detect outliers using percentile bounds."""
        lower = self.df[col].quantile(pct_range[0])
        upper = self.df[col].quantile(pct_range[1])
        
        mask = (self.df[col] < lower) | (self.df[col] > upper)
        return mask, lower, upper
    
    def treat(
        self,
        strategy: Literal['remove', 'cap', 'winsorize', 'flag', 'impute_mean', 'impute_median'] = 'cap',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Treat detected outliers using specified strategy.
        
        Args:
            strategy: Treatment strategy to apply
            columns: Columns to treat (default: all detected)
        """
        if columns is None:
            columns = list(self.outlier_masks.keys())
        
        indices_to_remove = set()
        
        for col in columns:
            if col not in self.outlier_masks:
                continue
            
            mask = self.outlier_masks[col]
            lower, upper = self.bounds[col]
            outlier_values = self.df.loc[mask, col].tolist()
            
            if strategy == 'remove':
                indices_to_remove.update(self.df[mask].index.tolist())
                treatment_desc = 'rows_removed'
            
            elif strategy == 'cap':
                self.df.loc[self.df[col] < lower, col] = lower
                self.df.loc[self.df[col] > upper, col] = upper
                treatment_desc = f'capped_to_[{lower:.2f}, {upper:.2f}]'
            
            elif strategy == 'winsorize':
                p5, p95 = self.df[col].quantile(0.05), self.df[col].quantile(0.95)
                self.df.loc[self.df[col] < p5, col] = p5
                self.df.loc[self.df[col] > p95, col] = p95
                treatment_desc = f'winsorized_to_[{p5:.2f}, {p95:.2f}]'
            
            elif strategy == 'flag':
                self.df[f'{col}_is_outlier'] = mask.astype(int)
                treatment_desc = 'flagged'
            
            elif strategy == 'impute_mean':
                mean_val = self.df.loc[~mask, col].mean()
                self.df.loc[mask, col] = mean_val
                treatment_desc = f'imputed_with_mean_{mean_val:.2f}'
            
            elif strategy == 'impute_median':
                median_val = self.df.loc[~mask, col].median()
                self.df.loc[mask, col] = median_val
                treatment_desc = f'imputed_with_median_{median_val:.2f}'
            
            self.reports.append(OutlierReport(
                column=col,
                method=strategy,
                lower_bound=lower,
                upper_bound=upper,
                outlier_count=mask.sum(),
                outlier_pct=round(mask.sum() / len(self.df) * 100, 2),
                treatment=treatment_desc,
                outlier_values=outlier_values[:10]
            ))
        
        if indices_to_remove:
            self.df = self.df.drop(index=list(indices_to_remove))
        
        return self.df
    
    def get_outliers(self, col: str) -> pd.DataFrame:
        """Get all outlier records for a specific column."""
        if col not in self.outlier_masks:
            return pd.DataFrame()
        return self.df[self.outlier_masks[col]]
    
    def get_report(self) -> pd.DataFrame:
        """Get treatment report as DataFrame."""
        return pd.DataFrame([{
            'column': r.column,
            'lower_bound': r.lower_bound,
            'upper_bound': r.upper_bound,
            'outlier_count': r.outlier_count,
            'outlier_pct': r.outlier_pct,
            'treatment': r.treatment,
            'sample_outliers': str(r.outlier_values)
        } for r in self.reports])


# Example usage
if __name__ == "__main__":
    # Create sample data with outliers
    np.random.seed(42)
    df = pd.DataFrame({
        'age': [25, 30, 35, 28, 999, 32, 45, 38, -5, 29],  # 999 and -5 are outliers
        'salary': [50000, 55000, 60000, 52000, 58000, 1000000, 54000, 51000, 53000, 49000],  # 1M is outlier
        'score': np.concatenate([np.random.normal(75, 10, 8), [150, 5]]),  # 150 and 5 are outliers
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    })
    
    print("Original Data:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    detector = OutlierDetector(df)
    
    # Detect outliers using IQR method
    print("Outlier Detection (IQR method):")
    detection_results = detector.detect(method='iqr', threshold=1.5)
    print(detection_results)
    print("\n" + "="*60 + "\n")
    
    # Show outliers for specific column
    print("Outliers in 'salary' column:")
    print(detector.get_outliers('salary'))
    print("\n" + "="*60 + "\n")
    
    # Treat outliers by capping
    print("Treated Data (capping strategy):")
    treated_df = detector.treat(strategy='cap')
    print(treated_df)
    print("\n" + "="*60 + "\n")
    
    print("Treatment Report:")
    print(detector.get_report())


