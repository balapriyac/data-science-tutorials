"""
Datetime Feature Extractor
Extracts comprehensive datetime features including time components,
cyclical encodings, and derived temporal features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DatetimeFeatures:
    column: str
    features_created: List[str]
    feature_count: int

class DatetimeExtractor:
    def __init__(
        self,
        include_cyclical: bool = True,
        include_flags: bool = True,
        include_time_diff: bool = True,
        reference_date: Optional[datetime] = None
    ):
        """
        Initialize the datetime feature extractor.
        
        Args:
            include_cyclical: Include sin/cos encodings for cyclical features
            include_flags: Include boolean flags (weekend, month_start, etc.)
            include_time_diff: Include time differences from reference date
            reference_date: Reference date for time difference calculations
        """
        self.include_cyclical = include_cyclical
        self.include_flags = include_flags
        self.include_time_diff = include_time_diff
        self.reference_date = reference_date or datetime.now()
        self.extracted_features: List[DatetimeFeatures] = []
    
    def extract_basic_features(
        self,
        df: pd.DataFrame,
        datetime_col: str
    ) -> pd.DataFrame:
        """
        Extract basic datetime components.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
        """
        dt = pd.to_datetime(df[datetime_col])
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        # Time components
        features[f'{datetime_col}_year'] = dt.dt.year
        features[f'{datetime_col}_month'] = dt.dt.month
        features[f'{datetime_col}_day'] = dt.dt.day
        features[f'{datetime_col}_dayofweek'] = dt.dt.dayofweek  # Monday=0, Sunday=6
        features[f'{datetime_col}_dayofyear'] = dt.dt.dayofyear
        features[f'{datetime_col}_week'] = dt.dt.isocalendar().week
        features[f'{datetime_col}_quarter'] = dt.dt.quarter
        
        feature_names.extend([
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter'
        ])
        
        # Time of day (if timestamp includes time)
        if dt.dt.hour.sum() > 0:  # Check if time component exists
            features[f'{datetime_col}_hour'] = dt.dt.hour
            features[f'{datetime_col}_minute'] = dt.dt.minute
            features[f'{datetime_col}_second'] = dt.dt.second
            feature_names.extend(['hour', 'minute', 'second'])
        
        return features, feature_names
    
    def extract_cyclical_features(
        self,
        df: pd.DataFrame,
        datetime_col: str
    ) -> pd.DataFrame:
        """
        Extract cyclical encodings using sin/cos transformations.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
        """
        dt = pd.to_datetime(df[datetime_col])
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        # Month cyclical (12 months)
        features[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
        features[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        
        # Day of week cyclical (7 days)
        features[f'{datetime_col}_dayofweek_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        features[f'{datetime_col}_dayofweek_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        
        # Day of month cyclical (assuming 31 days)
        features[f'{datetime_col}_day_sin'] = np.sin(2 * np.pi * dt.dt.day / 31)
        features[f'{datetime_col}_day_cos'] = np.cos(2 * np.pi * dt.dt.day / 31)
        
        feature_names.extend([
            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
            'day_sin', 'day_cos'
        ])
        
        # Hour cyclical (if time component exists)
        if dt.dt.hour.sum() > 0:
            features[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
            features[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
            feature_names.extend(['hour_sin', 'hour_cos'])
        
        return features, feature_names
    
    def extract_flag_features(
        self,
        df: pd.DataFrame,
        datetime_col: str
    ) -> pd.DataFrame:
        """
        Extract boolean flag features.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
        """
        dt = pd.to_datetime(df[datetime_col])
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        # Weekend flag
        features[f'{datetime_col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        
        # Start/End of period flags
        features[f'{datetime_col}_is_month_start'] = dt.dt.is_month_start.astype(int)
        features[f'{datetime_col}_is_month_end'] = dt.dt.is_month_end.astype(int)
        features[f'{datetime_col}_is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
        features[f'{datetime_col}_is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
        features[f'{datetime_col}_is_year_start'] = dt.dt.is_year_start.astype(int)
        features[f'{datetime_col}_is_year_end'] = dt.dt.is_year_end.astype(int)
        
        feature_names.extend([
            'is_weekend', 'is_month_start', 'is_month_end',
            'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'
        ])
        
        # Business day flag (Monday-Friday)
        features[f'{datetime_col}_is_business_day'] = (dt.dt.dayofweek < 5).astype(int)
        feature_names.append('is_business_day')
        
        # Season (Northern Hemisphere)
        month = dt.dt.month
        features[f'{datetime_col}_is_spring'] = month.isin([3, 4, 5]).astype(int)
        features[f'{datetime_col}_is_summer'] = month.isin([6, 7, 8]).astype(int)
        features[f'{datetime_col}_is_fall'] = month.isin([9, 10, 11]).astype(int)
        features[f'{datetime_col}_is_winter'] = month.isin([12, 1, 2]).astype(int)
        
        feature_names.extend(['is_spring', 'is_summer', 'is_fall', 'is_winter'])
        
        return features, feature_names
    
    def extract_time_differences(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Extract time differences from reference date.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            reference_date: Reference date for calculations
        """
        dt = pd.to_datetime(df[datetime_col])
        ref_date = reference_date or self.reference_date
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        # Days since reference
        time_diff = dt - ref_date
        features[f'{datetime_col}_days_since_ref'] = time_diff.dt.days
        
        # Weeks since reference
        features[f'{datetime_col}_weeks_since_ref'] = (time_diff.dt.days / 7).astype(int)
        
        # Months since reference (approximate)
        features[f'{datetime_col}_months_since_ref'] = (
            (dt.dt.year - ref_date.year) * 12 + (dt.dt.month - ref_date.month)
        )
        
        # Years since reference
        features[f'{datetime_col}_years_since_ref'] = dt.dt.year - ref_date.year
        
        feature_names.extend([
            'days_since_ref', 'weeks_since_ref', 'months_since_ref', 'years_since_ref'
        ])
        
        # Days since epoch (useful for trend modeling)
        epoch = pd.Timestamp('1970-01-01')
        features[f'{datetime_col}_days_since_epoch'] = (dt - epoch).dt.days
        feature_names.append('days_since_epoch')
        
        return features, feature_names
    
    def extract_lag_features(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        sort_by_date: bool = True
    ) -> pd.DataFrame:
        """
        Extract lag-based features (time between consecutive records).
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            sort_by_date: Whether to sort by datetime before calculating lags
        """
        dt = pd.to_datetime(df[datetime_col])
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        if sort_by_date:
            sorted_idx = dt.argsort()
            dt_sorted = dt.iloc[sorted_idx]
        else:
            dt_sorted = dt
        
        # Time since previous record
        time_since_prev = dt_sorted.diff()
        features[f'{datetime_col}_days_since_prev'] = time_since_prev.dt.days
        features[f'{datetime_col}_hours_since_prev'] = time_since_prev.dt.total_seconds() / 3600
        
        feature_names.extend(['days_since_prev', 'hours_since_prev'])
        
        # Reorder back if sorted
        if sort_by_date:
            features = features.iloc[sorted_idx.argsort()]
        
        return features, feature_names
    
    def extract_all(
        self,
        df: pd.DataFrame,
        datetime_cols: Optional[List[str]] = None,
        include_lag_features: bool = False
    ) -> pd.DataFrame:
        """
        Extract all datetime features from specified columns.
        
        Args:
            df: Input DataFrame
            datetime_cols: List of datetime columns (auto-detect if None)
            include_lag_features: Whether to include lag-based features
        """
        if datetime_cols is None:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_cols:
            print("No datetime columns found")
            return df
        
        all_features = pd.DataFrame(index=df.index)
        
        for col in datetime_cols:
            print(f"Processing datetime column: {col}")
            all_feature_names = []
            
            # Basic features
            basic_features, basic_names = self.extract_basic_features(df, col)
            all_features = pd.concat([all_features, basic_features], axis=1)
            all_feature_names.extend(basic_names)
            
            # Cyclical features
            if self.include_cyclical:
                cyclical_features, cyclical_names = self.extract_cyclical_features(df, col)
                all_features = pd.concat([all_features, cyclical_features], axis=1)
                all_feature_names.extend(cyclical_names)
            
            # Flag features
            if self.include_flags:
                flag_features, flag_names = self.extract_flag_features(df, col)
                all_features = pd.concat([all_features, flag_features], axis=1)
                all_feature_names.extend(flag_names)
            
            # Time difference features
            if self.include_time_diff:
                diff_features, diff_names = self.extract_time_differences(df, col)
                all_features = pd.concat([all_features, diff_features], axis=1)
                all_feature_names.extend(diff_names)
            
            # Lag features
            if include_lag_features:
                lag_features, lag_names = self.extract_lag_features(df, col)
                all_features = pd.concat([all_features, lag_features], axis=1)
                all_feature_names.extend(lag_names)
            
            self.extracted_features.append(DatetimeFeatures(
                column=col,
                features_created=all_feature_names,
                feature_count=len(all_feature_names)
            ))
        
        return all_features
    
    def get_extraction_report(self) -> pd.DataFrame:
        """Get report of extracted features."""
        return pd.DataFrame([{
            'original_column': f.column,
            'features_created': f.feature_count,
            'feature_list': ', '.join(f.features_created[:10]) + ('...' if len(f.features_created) > 10 else '')
        } for f in self.extracted_features])


# Example usage
if __name__ == "__main__":
    # Create sample dataset with datetime
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'timestamp': pd.date_range('2023-01-01 08:00:00', periods=100, freq='6H'),
        'value': np.random.randn(100)
    })
    
    print("Original Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\n" + "="*60 + "\n")
    
    # Initialize extractor
    extractor = DatetimeExtractor(
        include_cyclical=True,
        include_flags=True,
        include_time_diff=True,
        reference_date=datetime(2023, 1, 1)
    )
    
    # Extract all datetime features
    datetime_features = extractor.extract_all(
        df,
        datetime_cols=['date', 'timestamp'],
        include_lag_features=True
    )
    
    print("\nExtracted Datetime Features:")
    print(datetime_features.head(10))
    print(f"\nShape: {datetime_features.shape}")
    print("\n" + "="*60 + "\n")
    
    print("Extraction Report:")
    print(extractor.get_extraction_report())
    print("\n" + "="*60 + "\n")
    
    # Combine with original data
    df_enhanced = pd.concat([df, datetime_features], axis=1)
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    print(f"Original columns: {df.shape[1]}, New columns: {datetime_features.shape[1]}")
  
