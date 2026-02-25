"""
Correlation-Based Feature Selector

Identifies and removes highly correlated features to reduce redundancy
while preserving predictive power.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Tuple, List, Dict, Set
import warnings
warnings.filterwarnings('ignore')


class CorrelationSelector:
    """
    Removes highly correlated features based on correlation with target.
    Handles both numerical and categorical features.
    """
    
    def __init__(self, threshold: float = 0.95, method: str = 'pearson'):
        """
        Initialize correlation-based selector.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold above which features are considered redundant (default: 0.95)
        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'pearson')
        """
        self.threshold = threshold
        self.method = method
        self.selected_features_ = None
        self.removed_features_ = None
        self.correlation_pairs_ = None
        
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramér's V statistic for categorical features.
        
        Parameters:
        -----------
        x, y : pd.Series
            Categorical features
            
        Returns:
        --------
        cramers_v : float
            Cramér's V statistic (0 to 1)
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0:
            return 0.0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return cramers_v
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CorrelationSelector':
        """
        Fit the selector on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target variable (used to decide which correlated feature to keep)
            
        Returns:
        --------
        self : CorrelationSelector
        """
        # Calculate correlation matrix
        corr_matrix = X.corr(method=self.method).abs()
        
        # Find highly correlated pairs
        correlated_pairs = []
        features_to_remove = set()
        
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    feature_i = corr_matrix.columns[i]
                    feature_j = corr_matrix.columns[j]
                    
                    # Decide which feature to remove
                    if y is not None:
                        # Keep feature with higher correlation to target
                        corr_i = abs(X[feature_i].corr(y))
                        corr_j = abs(X[feature_j].corr(y))
                        
                        if corr_i >= corr_j:
                            to_remove = feature_j
                            to_keep = feature_i
                        else:
                            to_remove = feature_i
                            to_keep = feature_j
                    else:
                        # Keep first feature arbitrarily
                        to_remove = feature_j
                        to_keep = feature_i
                    
                    correlated_pairs.append({
                        'Feature_1': feature_i,
                        'Feature_2': feature_j,
                        'Correlation': corr_matrix.iloc[i, j],
                        'Removed': to_remove,
                        'Kept': to_keep
                    })
                    
                    features_to_remove.add(to_remove)
        
        # Store results
        self.correlation_pairs_ = pd.DataFrame(correlated_pairs)
        self.removed_features_ = list(features_to_remove)
        self.selected_features_ = [f for f in X.columns if f not in features_to_remove]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by removing correlated features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Feature matrix with correlated features removed
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_report(self) -> pd.DataFrame:
        """
        Get detailed report of correlated feature pairs.
        
        Returns:
        --------
        report : pd.DataFrame
            Report showing all correlated pairs and removal decisions
        """
        if self.correlation_pairs_ is None or len(self.correlation_pairs_) == 0:
            return pd.DataFrame({'Message': ['No highly correlated features found']})
        
        return self.correlation_pairs_.sort_values('Correlation', ascending=False).reset_index(drop=True)


def demo_correlation_selector():
    """Demonstrate correlation-based selector usage."""
    
    # Create sample dataset with correlated features
    np.random.seed(42)
    n_samples = 1000
    
    # Base features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Create correlated features
    data = {
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'feature_1_copy': x1 + np.random.normal(0, 0.01, n_samples),  # Highly correlated with feature_1
        'feature_2_copy': x2 + np.random.normal(0, 0.01, n_samples),  # Highly correlated with feature_2
        'feature_combined': 0.7 * x1 + 0.3 * x2,  # Moderately correlated with both
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable
    y = 2 * x1 + x2 + np.random.normal(0, 0.5, n_samples)
    
    print("=" * 60)
    print("Correlation-Based Feature Selection Demo")
    print("=" * 60)
    
    # Initialize and fit selector
    selector = CorrelationSelector(threshold=0.95)
    selector.fit(df, y)
    
    # Display report
    print("\nCorrelated Feature Pairs:")
    print("-" * 60)
    report = selector.get_report()
    if 'Feature_1' in report.columns:
        print(report.to_string(index=False))
    else:
        print(report.to_string(index=False))
    
    print(f"\nSelected Features: {len(selector.selected_features_)}")
    print(f"Removed Features: {len(selector.removed_features_)}")
    
    # Transform data
    df_selected = selector.transform(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Selected shape: {df_selected.shape}")
    print(f"\nRemaining features: {df_selected.columns.tolist()}")
    
    if selector.removed_features_:
        print(f"Removed features: {selector.removed_features_}")


if __name__ == "__main__":
    demo_correlation_selector()

