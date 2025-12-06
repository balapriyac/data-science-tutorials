"""
Data Type Fixer and Standardizer
Automatically detects intended data types, standardizes formats,
and converts columns to proper types with detailed reporting.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversionResult:
    column: str
    original_dtype: str
    new_dtype: str
    conversion_success: int
    conversion_failed: int
    failed_values: List[Any]

class DataTypeFixer:
    # Common date formats to try
    DATE_FORMATS = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
        '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y',
        '%Y%m%d', '%m%d%Y'
    ]
    
    # Boolean mappings
    BOOL_MAPPINGS = {
        'true': True, 'false': False,
        'yes': True, 'no': False,
        'y': True, 'n': False,
        '1': True, '0': False,
        't': True, 'f': False,
        'on': True, 'off': False
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: List[ConversionResult] = []
        
    def infer_types(self) -> pd.DataFrame:
        """Analyze each column and infer the intended data type."""
        analysis = []
        for col in self.df.columns:
            sample = self.df[col].dropna().head(100)
            inferred = self._infer_column_type(sample)
            analysis.append({
                'column': col,
                'current_dtype': str(self.df[col].dtype),
                'inferred_dtype': inferred,
                'sample_values': sample.head(3).tolist(),
                'null_count': self.df[col].isna().sum()
            })
        return pd.DataFrame(analysis)
    
    def _infer_column_type(self, sample: pd.Series) -> str:
        """Infer the intended type from sample values."""
        if sample.empty:
            return 'unknown'
        
        str_sample = sample.astype(str).str.lower().str.strip()
        
        # Check for boolean
        bool_matches = str_sample.isin(self.BOOL_MAPPINGS.keys()).sum()
        if bool_matches / len(sample) > 0.9:
            return 'boolean'
        
        # Check for numeric (including currency/formatted numbers)
        numeric_pattern = r'^[\$€£¥]?[\s]?[-+]?[\d,]+\.?\d*%?$'
        numeric_matches = str_sample.str.match(numeric_pattern, na=False).sum()
        if numeric_matches / len(sample) > 0.8:
            if any('%' in str(v) for v in sample):
                return 'percentage'
            return 'numeric'
        
        # Check for datetime
        date_matches = 0
        for val in sample.head(20):
            if self._try_parse_date(str(val)):
                date_matches += 1
        if date_matches / min(len(sample), 20) > 0.7:
            return 'datetime'
        
        # Check for integer-like strings
        int_matches = str_sample.str.match(r'^-?\d+$', na=False).sum()
        if int_matches / len(sample) > 0.9:
            return 'integer'
        
        return 'string'
    
    def _try_parse_date(self, value: str) -> Optional[datetime]:
        """Try to parse a string as a date using multiple formats."""
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        # Try pandas parser as fallback
        try:
            return pd.to_datetime(value, format='mixed')
        except:
            return None
    
    def fix_types(
        self,
        type_mapping: Optional[Dict[str, str]] = None,
        auto_detect: bool = True,
        coerce_errors: bool = True
    ) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            type_mapping: Dict mapping column names to target types
            auto_detect: If True, automatically detect types for unmapped columns
            coerce_errors: If True, convert failed values to NaN instead of raising
        """
        type_mapping = type_mapping or {}
        
        for col in self.df.columns:
            original_dtype = str(self.df[col].dtype)
            
            # Determine target type
            if col in type_mapping:
                target_type = type_mapping[col]
            elif auto_detect:
                sample = self.df[col].dropna().head(100)
                target_type = self._infer_column_type(sample)
            else:
                continue
            
            # Apply conversion
            success, failed, failed_vals = self._convert_column(col, target_type, coerce_errors)
            
            self.results.append(ConversionResult(
                column=col,
                original_dtype=original_dtype,
                new_dtype=str(self.df[col].dtype),
                conversion_success=success,
                conversion_failed=failed,
                failed_values=failed_vals[:10]  # Keep first 10 failed values
            ))
        
        return self.df
    
    def _convert_column(
        self, col: str, target_type: str, coerce: bool
    ) -> Tuple[int, int, List]:
        """Convert a single column to the target type."""
        original = self.df[col].copy()
        failed_values = []
        success, failed = 0, 0
        
        try:
            if target_type == 'numeric':
                self.df[col] = self._to_numeric(self.df[col], coerce)
            elif target_type == 'integer':
                self.df[col] = self._to_integer(self.df[col], coerce)
            elif target_type == 'percentage':
                self.df[col] = self._to_percentage(self.df[col], coerce)
            elif target_type == 'boolean':
                self.df[col] = self._to_boolean(self.df[col])
            elif target_type == 'datetime':
                self.df[col] = self._to_datetime(self.df[col], coerce)
            elif target_type == 'string':
                self.df[col] = self.df[col].astype(str).replace('nan', np.nan)
            
            # Count successes and failures
            for orig, new in zip(original, self.df[col]):
                if pd.notna(orig):
                    if pd.notna(new):
                        success += 1
                    else:
                        failed += 1
                        failed_values.append(orig)
                else:
                    success += 1  # NaN to NaN is success
                    
        except Exception as e:
            failed = len(original)
            failed_values = original.dropna().tolist()[:10]
        
        return success, failed, failed_values
    
    def _to_numeric(self, series: pd.Series, coerce: bool) -> pd.Series:
        """Convert to numeric, handling currency symbols and commas."""
        cleaned = series.astype(str).str.replace(r'[\$€£¥,\s]', '', regex=True)
        errors = 'coerce' if coerce else 'raise'
        return pd.to_numeric(cleaned, errors=errors)
    
    def _to_integer(self, series: pd.Series, coerce: bool) -> pd.Series:
        """Convert to integer type."""
        numeric = self._to_numeric(series, coerce)
        return numeric.astype('Int64')  # Nullable integer
    
    def _to_percentage(self, series: pd.Series, coerce: bool) -> pd.Series:
        """Convert percentage strings to decimal values."""
        def parse_pct(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip()
            if '%' in val:
                val = val.replace('%', '')
                return float(val.replace(',', '')) / 100
            return float(val.replace(',', ''))
        
        if coerce:
            return series.apply(lambda x: parse_pct(x) if pd.notna(x) else np.nan)
        return series.apply(parse_pct)
    
    def _to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert to boolean using common mappings."""
        def parse_bool(val):
            if pd.isna(val):
                return pd.NA
            return self.BOOL_MAPPINGS.get(str(val).lower().strip(), pd.NA)
        return series.apply(parse_bool).astype('boolean')
    
    def _to_datetime(self, series: pd.Series, coerce: bool) -> pd.Series:
        """Convert to datetime trying multiple formats."""
        errors = 'coerce' if coerce else 'raise'
        return pd.to_datetime(series, format='mixed', errors=errors)
    
    def get_report(self) -> pd.DataFrame:
        """Get conversion report as DataFrame."""
        report_data = []
        for r in self.results:
            report_data.append({
                'column': r.column,
                'original_dtype': r.original_dtype,
                'new_dtype': r.new_dtype,
                'success_count': r.conversion_success,
                'failed_count': r.conversion_failed,
                'sample_failures': str(r.failed_values) if r.failed_values else None
            })
        return pd.DataFrame(report_data)


# Example usage
if __name__ == "__main__":
    # Create sample data with messy types
    df = pd.DataFrame({
        'price': ['$1,234.56', '$999.00', '2500', '$45.99', 'N/A'],
        'quantity': ['100', '50', '75', '200', '30'],
        'is_active': ['Yes', 'No', 'Y', 'N', 'true'],
        'discount': ['10%', '25%', '5%', '15%', '0%'],
        'created_date': ['2024-01-15', '15/02/2024', 'Jan 20, 2024', '2024-03-01', '03-15-2024'],
        'rating': ['4.5', '3.8', '5.0', '4.2', 'four']
    })
    
    print("Original Data:")
    print(df)
    print("\nOriginal Data Types:")
    print(df.dtypes)
    print("\n" + "="*60 + "\n")
    
    fixer = DataTypeFixer(df)
    
    # Infer types
    print("Type Inference:")
    print(fixer.infer_types())
    print("\n" + "="*60 + "\n")
    
    # Fix types automatically
    fixed_df = fixer.fix_types(auto_detect=True, coerce_errors=True)
    
    print("Fixed Data:")
    print(fixed_df)
    print("\nFixed Data Types:")
    print(fixed_df.dtypes)
    print("\n" + "="*60 + "\n")
    
    print("Conversion Report:")
    print(fixer.get_report())


