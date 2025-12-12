"""
Numerical Feature Transformer
Automatically tests and applies optimal transformations for numeric features
including scaling, normalization, and distribution transformations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TransformationResult:
    column: str
    original_skewness: float
    original_kurtosis: float
    best_transformation: str
    transformed_skewness: float
    transformed_kurtosis: float
    normality_improved: bool
    transformation_params: Dict

class NumericalTransformer:
    def __init__(self):
        """Initialize the numerical transformer."""
        self.transformations = {}
        self.scalers = {}
        self.results: List[TransformationResult] = []
        
    def evaluate_normality(self, data: pd.Series) -> Tuple[float, float, float]:
        """
        Evaluate how normal a distribution is.
        
        Returns:
            skewness, kurtosis, normality_score (lower is more normal)
        """
        skew = data.skew()
        kurt = data.kurtosis()
        
        # Normality score: combination of absolute skewness and excess kurtosis
        normality_score = abs(skew) + abs(kurt) / 3
        
        return skew, kurt, normality_score
    
    def log_transform(self, data: pd.Series, shift: float = 1) -> pd.Series:
        """
        Apply log transformation (handles zeros and negatives with shift).
        
        Args:
            data: Input series
            shift: Value to add before log (default 1)
        """
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
        
        return np.log(data + shift), {'shift': shift}
    
    def sqrt_transform(self, data: pd.Series) -> Tuple[pd.Series, Dict]:
        """Apply square root transformation."""
        min_val = data.min()
        if min_val < 0:
            # Shift to make all values non-negative
            shift = abs(min_val)
            return np.sqrt(data + shift), {'shift': shift}
        return np.sqrt(data), {'shift': 0}
    
    def cbrt_transform(self, data: pd.Series) -> Tuple[pd.Series, Dict]:
        """Apply cube root transformation (works with negative values)."""
        return np.cbrt(data), {}
    
    def boxcox_transform(self, data: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Apply Box-Cox transformation (requires positive values).
        
        Returns transformed data and lambda parameter
        """
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            data_shifted = data + shift
        else:
            shift = 0
            data_shifted = data
        
        try:
            transformed, lambda_param = stats.boxcox(data_shifted)
            return pd.Series(transformed, index=data.index), {'lambda': lambda_param, 'shift': shift}
        except:
            # If Box-Cox fails, return original
            return data, {'failed': True}
    
    def yeojohnson_transform(self, data: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Apply Yeo-Johnson transformation (works with any real values).
        
        Returns transformed data and lambda parameter
        """
        try:
            transformed, lambda_param = stats.yeojohnson(data)
            return pd.Series(transformed, index=data.index), {'lambda': lambda_param}
        except:
            return data, {'failed': True}
    
    def quantile_transform(self, data: pd.Series, n_quantiles: int = 1000) -> Tuple[pd.Series, Dict]:
        """Apply quantile transformation (makes distribution uniform)."""
        from sklearn.preprocessing import QuantileTransformer
        
        transformer = QuantileTransformer(n_quantiles=min(n_quantiles, len(data)))
        transformed = transformer.fit_transform(data.values.reshape(-1, 1)).flatten()
        
        return pd.Series(transformed, index=data.index), {'transformer': transformer}
    
    def find_best_transformation(
        self,
        data: pd.Series,
        transformations: List[str] = ['log', 'sqrt', 'boxcox', 'yeojohnson']
    ) -> Tuple[str, pd.Series, Dict, float]:
        """
        Test multiple transformations and return the best one.
        
        Args:
            data: Input series
            transformations: List of transformation names to try
            
        Returns:
            best_method, transformed_data, params, improvement_score
        """
        original_skew, original_kurt, original_score = self.evaluate_normality(data)
        
        results = {}
        
        for method in transformations:
            try:
                if method == 'log':
                    transformed, params = self.log_transform(data)
                elif method == 'sqrt':
                    transformed, params = self.sqrt_transform(data)
                elif method == 'cbrt':
                    transformed, params = self.cbrt_transform(data)
                elif method == 'boxcox':
                    transformed, params = self.boxcox_transform(data)
                elif method == 'yeojohnson':
                    transformed, params = self.yeojohnson_transform(data)
                elif method == 'quantile':
                    transformed, params = self.quantile_transform(data)
                else:
                    continue
                
                if 'failed' in params:
                    continue
                
                # Evaluate transformed distribution
                _, _, new_score = self.evaluate_normality(transformed)
                improvement = original_score - new_score
                
                results[method] = {
                    'transformed': transformed,
                    'params': params,
                    'score': new_score,
                    'improvement': improvement
                }
            except Exception as e:
                continue
        
        if not results:
            # No transformation improved the distribution
            return 'none', data, {}, 0
        
        # Select transformation with best (lowest) normality score
        best_method = min(results.keys(), key=lambda k: results[k]['score'])
        best_result = results[best_method]
        
        return best_method, best_result['transformed'], best_result['params'], best_result['improvement']
    
    def fit_scaler(
        self,
        data: pd.Series,
        method: Literal['standard', 'minmax', 'robust'] = 'standard'
    ) -> Tuple[pd.Series, object]:
        """
        Fit and apply a scaler.
        
        Args:
            data: Input series
            method: Scaling method
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        
        scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=data.index), scaler
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        auto_transform: bool = True,
        auto_scale: bool = True,
        scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard'
    ) -> pd.DataFrame:
        """
        Fit transformations and scalers on training data.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform (all numeric if None)
            auto_transform: Whether to automatically find best transformation
            auto_scale: Whether to apply scaling after transformation
            scaling_method: Type of scaling to apply
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_transformed = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if len(data) < 3:
                continue
            
            original_skew, original_kurt, _ = self.evaluate_normality(data)
            
            # Find best transformation
            if auto_transform:
                best_method, transformed, params, improvement = self.find_best_transformation(data)
                
                # Only apply if it actually improves
                if improvement > 0.1:
                    df_transformed.loc[data.index, col] = transformed
                    self.transformations[col] = {
                        'method': best_method,
                        'params': params
                    }
                else:
                    best_method = 'none'
                    self.transformations[col] = {'method': 'none', 'params': {}}
            else:
                best_method = 'none'
                self.transformations[col] = {'method': 'none', 'params': {}}
            
            # Apply scaling
            if auto_scale:
                scaled, scaler = self.fit_scaler(df_transformed[col].dropna(), scaling_method)
                df_transformed.loc[scaled.index, col] = scaled
                self.scalers[col] = scaler
            
            # Record results
            final_skew, final_kurt, _ = self.evaluate_normality(df_transformed[col].dropna())
            
            self.results.append(TransformationResult(
                column=col,
                original_skewness=round(original_skew, 4),
                original_kurtosis=round(original_kurt, 4),
                best_transformation=best_method,
                transformed_skewness=round(final_skew, 4),
                transformed_kurtosis=round(final_kurt, 4),
                normality_improved=abs(final_skew) < abs(original_skew),
                transformation_params=self.transformations[col]['params']
            ))
        
        return df_transformed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data.
        
        Args:
            df: New DataFrame to transform
        """
        df_transformed = df.copy()
        
        for col, transform_info in self.transformations.items():
            if col not in df.columns:
                continue
            
            data = df_transformed[col].copy()
            method = transform_info['method']
            params = transform_info['params']
            
            # Apply transformation
            if method == 'log':
                shift = params['shift']
                df_transformed[col] = np.log(data + shift)
            elif method == 'sqrt':
                shift = params.get('shift', 0)
                df_transformed[col] = np.sqrt(data + shift)
            elif method == 'cbrt':
                df_transformed[col] = np.cbrt(data)
            elif method == 'boxcox':
                if 'failed' not in params:
                    lambda_param = params['lambda']
                    shift = params.get('shift', 0)
                    df_transformed[col] = stats.boxcox(data + shift, lmbda=lambda_param)
            elif method == 'yeojohnson':
                if 'failed' not in params:
                    lambda_param = params['lambda']
                    df_transformed[col] = stats.yeojohnson(data, lmbda=lambda_param)
            elif method == 'quantile':
                if 'transformer' in params:
                    transformer = params['transformer']
                    df_transformed[col] = transformer.transform(data.values.reshape(-1, 1)).flatten()
            
            # Apply scaling
            if col in self.scalers:
                scaler = self.scalers[col]
                scaled = scaler.transform(df_transformed[col].values.reshape(-1, 1)).flatten()
                df_transformed[col] = scaled
        
        return df_transformed
    
    def get_transformation_report(self) -> pd.DataFrame:
        """Get report of all transformations applied."""
        return pd.DataFrame([{
            'column': r.column,
            'transformation': r.best_transformation,
            'original_skew': r.original_skewness,
            'transformed_skew': r.transformed_skewness,
            'original_kurtosis': r.original_kurtosis,
            'transformed_kurtosis': r.transformed_kurtosis,
            'improved': 'Yes' if r.normality_improved else 'No'
        } for r in self.results])


# Example usage
if __name__ == "__main__":
    # Create sample dataset with different distributions
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'normal': np.random.normal(50, 10, n),
        'right_skewed': np.random.exponential(5, n),
        'left_skewed': 100 - np.random.exponential(10, n),
        'heavy_tailed': np.random.standard_t(3, n) * 10 + 50,
        'uniform': np.random.uniform(0, 100, n),
        'with_negatives': np.random.normal(0, 20, n)
    })
    
    print("Original Data Statistics:")
    print(df.describe())
    print("\n" + "="*70 + "\n")
    
    print("Original Skewness:")
    print(df.skew())
    print("\n" + "="*70 + "\n")
    
    # Initialize transformer
    transformer = NumericalTransformer()
    
    # Fit and transform
    df_transformed = transformer.fit_transform(
        df,
        auto_transform=True,
        auto_scale=True,
        scaling_method='standard'
    )
    
    print("Transformed Data Statistics:")
    print(df_transformed.describe())
    print("\n" + "="*70 + "\n")
    
    print("Transformed Skewness:")
    print(df_transformed.skew())
    print("\n" + "="*70 + "\n")
    
    print("Transformation Report:")
    report = transformer.get_transformation_report()
    print(report.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Test on new data
    new_data = pd.DataFrame({
        'normal': np.random.normal(50, 10, 100),
        'right_skewed': np.random.exponential(5, 100),
        'left_skewed': 100 - np.random.exponential(10, 100),
        'heavy_tailed': np.random.standard_t(3, 100) * 10 + 50,
        'uniform': np.random.uniform(0, 100, 100),
        'with_negatives': np.random.normal(0, 20, 100)
    })
    
    print("Transforming new data with fitted transformations...")
    new_transformed = transformer.transform(new_data)
    print("New data transformed successfully!")
    print(new_transformed.head())

