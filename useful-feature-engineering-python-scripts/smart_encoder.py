"""
Smart Feature Encoder
Automatically selects and applies appropriate encoding strategies
for categorical variables based on their characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EncodingStrategy:
    column: str
    strategy: str
    cardinality: int
    rare_categories: int
    mapping: Dict[Any, Any]

class SmartEncoder:
    def __init__(
        self, 
        cardinality_threshold: int = 10,
        rare_threshold: float = 0.01,
        handle_unknown: Literal['error', 'ignore', 'encode_rare'] = 'encode_rare'
    ):
        """
        Initialize the encoder.
        
        Args:
            cardinality_threshold: Max unique values for one-hot encoding
            rare_threshold: Min frequency to not be considered rare
            handle_unknown: How to handle unseen categories in test data
        """
        self.cardinality_threshold = cardinality_threshold
        self.rare_threshold = rare_threshold
        self.handle_unknown = handle_unknown
        self.encodings: Dict[str, EncodingStrategy] = {}
        self.fitted = False
    
    def _determine_strategy(
        self, 
        series: pd.Series, 
        target: Optional[pd.Series] = None,
        force_strategy: Optional[str] = None
    ) -> str:
        """Determine the best encoding strategy for a column."""
        if force_strategy:
            return force_strategy
        
        cardinality = series.nunique()
        
        # One-hot for low cardinality
        if cardinality <= self.cardinality_threshold:
            return 'onehot'
        
        # Target encoding if target is provided and cardinality is moderate
        if target is not None and cardinality <= 50:
            return 'target'
        
        # Frequency encoding for high cardinality
        if cardinality > 50:
            return 'frequency'
        
        # Default to label encoding
        return 'label'
    
    def _identify_rare_categories(self, series: pd.Series) -> List[Any]:
        """Identify rare categories below the threshold."""
        value_counts = series.value_counts(normalize=True)
        rare = value_counts[value_counts < self.rare_threshold].index.tolist()
        return rare
    
    def _label_encode(self, series: pd.Series, mapping: Optional[Dict] = None) -> Tuple[pd.Series, Dict]:
        """Apply label encoding."""
        if mapping is None:
            unique_vals = series.dropna().unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        
        encoded = series.map(mapping)
        
        # Handle unknown values
        if self.handle_unknown == 'encode_rare':
            encoded = encoded.fillna(-1)
        
        return encoded, mapping
    
    def _onehot_encode(self, df: pd.DataFrame, col: str, categories: Optional[List] = None) -> pd.DataFrame:
        """Apply one-hot encoding."""
        if categories is None:
            categories = df[col].dropna().unique().tolist()
        
        for cat in categories:
            df[f'{col}_{cat}'] = (df[col] == cat).astype(int)
        
        return df
    
    def _target_encode(
        self, 
        series: pd.Series, 
        target: pd.Series, 
        mapping: Optional[Dict] = None,
        smoothing: float = 1.0
    ) -> Tuple[pd.Series, Dict]:
        """Apply target encoding with smoothing."""
        if mapping is None:
            # Calculate target mean for each category
            global_mean = target.mean()
            stats = pd.DataFrame({'target': target, 'category': series})
            
            agg = stats.groupby('category').agg(
                count=('target', 'count'),
                mean=('target', 'mean')
            )
            
            # Apply smoothing
            agg['smoothed_mean'] = (
                agg['count'] * agg['mean'] + smoothing * global_mean
            ) / (agg['count'] + smoothing)
            
            mapping = agg['smoothed_mean'].to_dict()
            mapping['__global_mean__'] = global_mean
        
        global_mean = mapping.get('__global_mean__', target.mean() if target is not None else 0)
        encoded = series.map(lambda x: mapping.get(x, global_mean))
        
        return encoded, mapping
    
    def _frequency_encode(self, series: pd.Series, mapping: Optional[Dict] = None) -> Tuple[pd.Series, Dict]:
        """Apply frequency encoding."""
        if mapping is None:
            freq = series.value_counts(normalize=True)
            mapping = freq.to_dict()
        
        encoded = series.map(mapping)
        
        # Handle unknown values
        if self.handle_unknown == 'encode_rare':
            encoded = encoded.fillna(0)
        
        return encoded, mapping
    
    def _binary_encode(self, series: pd.Series, mapping: Optional[Dict] = None) -> pd.DataFrame:
        """Apply binary encoding."""
        if mapping is None:
            unique_vals = series.dropna().unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        
        # Convert to integers first
        int_encoded = series.map(mapping).fillna(-1).astype(int)
        
        # Determine number of binary columns needed
        max_val = max(mapping.values()) if mapping else 0
        n_bits = max_val.bit_length()
        
        # Create binary columns
        binary_df = pd.DataFrame()
        for i in range(n_bits):
            binary_df[f'{series.name}_bin_{i}'] = (int_encoded >> i) & 1
        
        return binary_df, mapping
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        column_strategies: Optional[Dict[str, str]] = None
    ) -> 'SmartEncoder':
        """
        Fit encoder on training data.
        
        Args:
            X: DataFrame with categorical columns to encode
            y: Optional target variable for target encoding
            column_strategies: Optional dict to force specific strategies per column
        """
        column_strategies = column_strategies or {}
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            series = X[col]
            
            # Group rare categories
            rare_cats = self._identify_rare_categories(series)
            if rare_cats:
                series = series.copy()
                series = series.replace(rare_cats, '__RARE__')
            
            # Determine encoding strategy
            strategy = self._determine_strategy(
                series, 
                target=y,
                force_strategy=column_strategies.get(col)
            )
            
            # Create mapping based on strategy
            if strategy == 'label':
                _, mapping = self._label_encode(series)
            elif strategy == 'target':
                _, mapping = self._target_encode(series, y)
            elif strategy == 'frequency':
                _, mapping = self._frequency_encode(series)
            elif strategy == 'onehot':
                mapping = series.dropna().unique().tolist()
            elif strategy == 'binary':
                _, mapping = self._binary_encode(series)
            
            self.encodings[col] = EncodingStrategy(
                column=col,
                strategy=strategy,
                cardinality=series.nunique(),
                rare_categories=len(rare_cats),
                mapping=mapping
            )
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encodings."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        X_encoded = X.copy()
        cols_to_drop = []
        
        for col, encoding in self.encodings.items():
            if col not in X_encoded.columns:
                continue
            
            series = X_encoded[col].copy()
            
            # Handle rare categories
            if encoding.rare_categories > 0:
                rare_mask = ~series.isin(encoding.mapping.keys())
                series.loc[rare_mask] = '__RARE__'
            
            # Apply encoding
            if encoding.strategy == 'label':
                X_encoded[f'{col}_encoded'], _ = self._label_encode(series, encoding.mapping)
                cols_to_drop.append(col)
            
            elif encoding.strategy == 'target':
                X_encoded[f'{col}_target_enc'], _ = self._target_encode(
                    series, None, encoding.mapping
                )
                cols_to_drop.append(col)
            
            elif encoding.strategy == 'frequency':
                X_encoded[f'{col}_freq'], _ = self._frequency_encode(series, encoding.mapping)
                cols_to_drop.append(col)
            
            elif encoding.strategy == 'onehot':
                X_encoded = self._onehot_encode(X_encoded, col, encoding.mapping)
                cols_to_drop.append(col)
            
            elif encoding.strategy == 'binary':
                binary_df, _ = self._binary_encode(series, encoding.mapping)
                X_encoded = pd.concat([X_encoded, binary_df], axis=1)
                cols_to_drop.append(col)
        
        # Drop original categorical columns
        X_encoded = X_encoded.drop(columns=cols_to_drop)
        
        return X_encoded
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        column_strategies: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, column_strategies)
        return self.transform(X)
    
    def get_encoding_report(self) -> pd.DataFrame:
        """Get report of encoding strategies used."""
        report = []
        for col, encoding in self.encodings.items():
            report.append({
                'column': encoding.column,
                'strategy': encoding.strategy,
                'cardinality': encoding.cardinality,
                'rare_categories_grouped': encoding.rare_categories,
                'encoded_features': self._count_encoded_features(encoding)
            })
        return pd.DataFrame(report)
    
    def _count_encoded_features(self, encoding: EncodingStrategy) -> int:
        """Count number of features created by encoding."""
        if encoding.strategy == 'onehot':
            return len(encoding.mapping)
        elif encoding.strategy == 'binary':
            return max(encoding.mapping.values()).bit_length() if encoding.mapping else 0
        else:
            return 1


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'color': np.random.choice(['red', 'blue', 'green'], 100),
        'size': np.random.choice(['S', 'M', 'L', 'XL'], 100),
        'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Spain', 
                                     'Italy', 'Canada', 'Japan', 'China', 'India',
                                     'Brazil', 'Mexico', 'Australia'], 100),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'] + 
                                    [f'Cat_{i}' for i in range(20)], 100),
        'price': np.random.uniform(10, 100, 100)
    })
    
    # Create a target variable
    y = (df['price'] > 50).astype(int)
    
    print("Original Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\n" + "="*60 + "\n")
    
    # Initialize and fit encoder
    encoder = SmartEncoder(cardinality_threshold=5, rare_threshold=0.02)
    
    # Fit and transform
    df_encoded = encoder.fit_transform(df[['color', 'size', 'country', 'category']], y)
    
    print("Encoded Data:")
    print(df_encoded.head())
    print(f"\nShape: {df_encoded.shape}")
    print("\n" + "="*60 + "\n")
    
    print("Encoding Report:")
    print(encoder.get_encoding_report())
    print("\n" + "="*60 + "\n")
    
    # Test on new data
    new_data = pd.DataFrame({
        'color': ['red', 'purple'],  # purple is unseen
        'size': ['M', 'XXL'],  # XXL is unseen
        'country': ['USA', 'Unknown'],
        'category': ['A', 'New']
    })
    
    print("New Data (with unseen categories):")
    print(new_data)
    print("\n" + "="*60 + "\n")
    
    new_encoded = encoder.transform(new_data)
    print("New Data Encoded:")
    print(new_encoded)
  
