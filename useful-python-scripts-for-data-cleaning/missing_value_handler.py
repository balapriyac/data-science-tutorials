"""
Missing Value Handler
Automatically analyzes and handles missing values in your dataset
with intelligent imputation strategies based on data type and patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal
from dataclasses import dataclass

@dataclass
class MissingReport:
    column: str
    missing_count: int
    missing_pct: float
    strategy_used: str
    fill_value: any

class MissingValueHandler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report: list[MissingReport] = []
        
    def analyze(self) -> pd.DataFrame:
        """Analyze missing value patterns across all columns."""
        analysis = []
        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            total = len(self.df)
            analysis.append({
                'column': col,
                'dtype': str(self.df[col].dtype),
                'missing_count': missing,
                'missing_pct': round(missing / total * 100, 2),
                'unique_values': self.df[col].nunique(),
                'recommended_strategy': self._recommend_strategy(col)
            })
        return pd.DataFrame(analysis)
    
    def _recommend_strategy(self, col: str) -> str:
        """Recommend handling strategy based on column characteristics."""
        dtype = self.df[col].dtype
        missing_pct = self.df[col].isna().mean() * 100
        
        if missing_pct > 70:
            return 'drop_column'
        elif missing_pct > 50:
            return 'flag_missing'
        elif pd.api.types.is_numeric_dtype(dtype):
            skew = self.df[col].skew() if self.df[col].notna().sum() > 2 else 0
            return 'median' if abs(skew) > 1 else 'mean'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'interpolate'
        else:
            return 'mode'
    
    def handle(
        self,
        strategies: Optional[Dict[str, str]] = None,
        default_numeric: Literal['mean', 'median', 'zero'] = 'median',
        default_categorical: Literal['mode', 'unknown'] = 'mode',
        drop_threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Handle missing values with specified or auto-detected strategies.
        
        Args:
            strategies: Dict mapping column names to strategies
            default_numeric: Default strategy for numeric columns
            default_categorical: Default strategy for categorical columns
            drop_threshold: Drop columns with missingness above this threshold
        """
        strategies = strategies or {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count == 0:
                continue
                
            missing_pct = missing_count / len(self.df)
            
            # Get strategy for this column
            if col in strategies:
                strategy = strategies[col]
            elif missing_pct > drop_threshold:
                strategy = 'drop_column'
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                strategy = default_numeric
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                strategy = 'interpolate'
            else:
                strategy = default_categorical
            
            # Apply strategy
            fill_value = self._apply_strategy(col, strategy)
            
            self.report.append(MissingReport(
                column=col,
                missing_count=missing_count,
                missing_pct=round(missing_pct * 100, 2),
                strategy_used=strategy,
                fill_value=fill_value
            ))
        
        return self.df
    
    def _apply_strategy(self, col: str, strategy: str) -> any:
        """Apply the specified imputation strategy to a column."""
        fill_value = None
        
        if strategy == 'drop_column':
            self.df.drop(columns=[col], inplace=True)
            fill_value = 'DROPPED'
        elif strategy == 'drop_rows':
            self.df.dropna(subset=[col], inplace=True)
            fill_value = 'ROWS_DROPPED'
        elif strategy == 'mean':
            fill_value = self.df[col].mean()
            self.df[col].fillna(fill_value, inplace=True)
        elif strategy == 'median':
            fill_value = self.df[col].median()
            self.df[col].fillna(fill_value, inplace=True)
        elif strategy == 'mode':
            fill_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'UNKNOWN'
            self.df[col].fillna(fill_value, inplace=True)
        elif strategy == 'zero':
            fill_value = 0
            self.df[col].fillna(0, inplace=True)
        elif strategy == 'unknown':
            fill_value = 'UNKNOWN'
            self.df[col].fillna('UNKNOWN', inplace=True)
        elif strategy == 'interpolate':
            self.df[col] = self.df[col].interpolate(method='linear')
            fill_value = 'INTERPOLATED'
        elif strategy == 'ffill':
            self.df[col].fillna(method='ffill', inplace=True)
            fill_value = 'FORWARD_FILL'
        elif strategy == 'bfill':
            self.df[col].fillna(method='bfill', inplace=True)
            fill_value = 'BACKWARD_FILL'
        elif strategy == 'flag_missing':
            self.df[f'{col}_is_missing'] = self.df[col].isna().astype(int)
            fill_value = 'FLAGGED'
        
        return fill_value
    
    def get_report(self) -> pd.DataFrame:
        """Get a DataFrame summarizing all imputation actions taken."""
        return pd.DataFrame([vars(r) for r in self.report])


# Example usage
if __name__ == "__main__":
    # Create sample data with missing values
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', None, 'Diana', 'Eve', None],
        'age': [25, None, 35, 28, None, 42],
        'salary': [50000, 60000, None, 55000, 70000, None],
        'department': ['Sales', 'IT', 'IT', None, 'Sales', 'HR'],
        'join_date': pd.to_datetime(['2020-01-15', None, '2019-06-01', '2021-03-20', None, '2018-11-10'])
    })
    
    print("Original Data:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    # Initialize handler
    handler = MissingValueHandler(df)
    
    # Analyze missing patterns
    print("Missing Value Analysis:")
    print(handler.analyze())
    print("\n" + "="*60 + "\n")
    
    # Handle missing values with custom strategies
    cleaned_df = handler.handle(
        strategies={'name': 'unknown'},
        default_numeric='median',
        default_categorical='mode'
    )
    
    print("Cleaned Data:")
    print(cleaned_df)
    print("\n" + "="*60 + "\n")
    
    print("Imputation Report:")
    print(handler.get_report())

