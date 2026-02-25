"""
Variance Threshold Feature Selector

Automatically removes features with low or zero variance that provide
little information for prediction.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class VarianceThresholdSelector:
    """
    Removes low-variance features using configurable thresholds.
    Handles continuous and binary features appropriately.
    """
    
    def __init__(self, threshold: float = 0.01, normalize: bool = True):
        """
        Initialize the variance threshold selector.
        
        Parameters:
        -----------
        threshold : float
            Variance threshold below which features are removed (default: 0.01)
        normalize : bool
            Whether to normalize variance by feature range (default: True)
        """
        self.threshold = threshold
        self.normalize = normalize
        self.selected_features_ = None
        self.removed_features_ = None
        self.feature_variances_ = None
        
    def fit(self, X: pd.DataFrame, feature_names: List[str] = None) -> 'VarianceThresholdSelector':
        """
        Fit the selector on the training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        feature_names : list, optional
            Feature names (uses X.columns if DataFrame)
            
        Returns:
        --------
        self : VarianceThresholdSelector
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate variance for each feature
        variances = np.var(X_array, axis=0)
        
        # Normalize variance by range if requested
        if self.normalize:
            ranges = np.ptp(X_array, axis=0)  # peak-to-peak (max - min)
            # Avoid division by zero
            ranges[ranges == 0] = 1
            normalized_variances = variances / (ranges ** 2)
        else:
            normalized_variances = variances
            
        # Identify features to keep
        mask = normalized_variances > self.threshold
        
        # Store results
        self.selected_features_ = [f for f, m in zip(feature_names, mask) if m]
        self.removed_features_ = [f for f, m in zip(feature_names, mask) if not m]
        self.feature_variances_ = dict(zip(feature_names, normalized_variances))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by removing low-variance features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Feature matrix with low-variance features removed
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # If numpy array, return selected columns
            indices = [list(X.columns).index(f) for f in self.selected_features_]
            return X[:, indices]
    
    def fit_transform(self, X: pd.DataFrame, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        feature_names : list, optional
            Feature names
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, feature_names).transform(X)
    
    def get_report(self) -> pd.DataFrame:
        """
        Get detailed report of feature selection.
        
        Returns:
        --------
        report : pd.DataFrame
            Report with variance scores and selection status
        """
        if self.feature_variances_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        report_data = []
        for feature, variance in self.feature_variances_.items():
            selected = feature in self.selected_features_
            report_data.append({
                'Feature': feature,
                'Variance': variance,
                'Selected': selected,
                'Status': 'Kept' if selected else 'Removed'
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('Variance', ascending=False).reset_index(drop=True)


def demo_variance_selector():
    """Demonstrate variance threshold selector usage."""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'constant_feature': [1] * n_samples,  # Zero variance
        'low_variance': np.random.normal(10, 0.1, n_samples),  # Very low variance
        'binary_feature': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),  # Imbalanced binary
        'normal_feature': np.random.normal(0, 1, n_samples),  # Normal variance
        'high_variance': np.random.normal(0, 10, n_samples),  # High variance
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("Variance Threshold Feature Selection Demo")
    print("=" * 60)
    
    # Initialize and fit selector
    selector = VarianceThresholdSelector(threshold=0.01, normalize=True)
    selector.fit(df)
    
    # Display report
    print("\nFeature Variance Report:")
    print("-" * 60)
    report = selector.get_report()
    print(report.to_string(index=False))
    
    print(f"\nSelected Features: {len(selector.selected_features_)}")
    print(f"Removed Features: {len(selector.removed_features_)}")
    
    # Transform data
    df_selected = selector.transform(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Selected shape: {df_selected.shape}")
    print(f"\nRemaining features: {df_selected.columns.tolist()}")


if __name__ == "__main__":
    demo_variance_selector()
  
