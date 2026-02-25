"""
Statistical Test Feature Selector

Selects features based on statistical significance tests appropriate
for feature and target types.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    f_classif, chi2, mutual_info_classif, 
    f_regression, mutual_info_regression
)
from scipy.stats import f_oneway
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class StatisticalTestSelector:
    """
    Selects features using statistical tests with multiple testing correction.
    Automatically chooses appropriate tests based on feature and target types.
    """
    
    def __init__(self, alpha: float = 0.05, correction: str = 'fdr', test_type: str = 'auto'):
        """
        Initialize statistical test selector.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        correction : str
            Multiple testing correction: 'bonferroni', 'fdr', or 'none' (default: 'fdr')
        test_type : str
            Test type: 'auto', 'anova', 'chi2', 'mutual_info', 'f_regression' (default: 'auto')
        """
        self.alpha = alpha
        self.correction = correction
        self.test_type = test_type
        self.selected_features_ = None
        self.feature_scores_ = None
        self.feature_pvalues_ = None
        
    def _bonferroni_correction(self, pvalues: np.ndarray) -> np.ndarray:
        """Apply Bonferroni correction to p-values."""
        n = len(pvalues)
        return np.minimum(pvalues * n, 1.0)
    
    def _fdr_correction(self, pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Returns:
        --------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        pvalues_corrected : np.ndarray
            FDR-corrected p-values
        """
        n = len(pvalues)
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = pvalues[sorted_indices]
        
        # Calculate critical values
        critical_values = (np.arange(1, n + 1) / n) * self.alpha
        
        # Find largest i where p[i] <= (i/n) * alpha
        rejected_indices = sorted_pvalues <= critical_values
        
        if np.any(rejected_indices):
            max_index = np.where(rejected_indices)[0][-1]
            rejected = np.zeros(n, dtype=bool)
            rejected[sorted_indices[:max_index + 1]] = True
        else:
            rejected = np.zeros(n, dtype=bool)
        
        # Calculate corrected p-values
        pvalues_corrected = np.minimum.accumulate(
            sorted_pvalues[::-1] * n / np.arange(n, 0, -1)
        )[::-1]
        pvalues_corrected = np.minimum(pvalues_corrected, 1.0)
        
        # Restore original order
        original_order = np.argsort(sorted_indices)
        pvalues_corrected = pvalues_corrected[original_order]
        
        return rejected, pvalues_corrected
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task: str = 'classification') -> 'StatisticalTestSelector':
        """
        Fit the selector on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        task : str
            Task type: 'classification' or 'regression' (default: 'classification')
            
        Returns:
        --------
        self : StatisticalTestSelector
        """
        feature_names = X.columns.tolist()
        
        # Select appropriate test
        if self.test_type == 'auto':
            if task == 'classification':
                test_func = f_classif
            else:
                test_func = f_regression
        elif self.test_type == 'anova':
            test_func = f_classif
        elif self.test_type == 'chi2':
            test_func = chi2
            # Chi2 requires non-negative values
            X = X - X.min() + 1e-10
        elif self.test_type == 'mutual_info':
            if task == 'classification':
                test_func = mutual_info_classif
            else:
                test_func = mutual_info_regression
        elif self.test_type == 'f_regression':
            test_func = f_regression
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
        
        # Compute test statistics and p-values
        if self.test_type == 'mutual_info':
            # Mutual information doesn't return p-values
            scores = test_func(X, y, random_state=42)
            # Use scores directly, treat as pseudo p-values (1 - normalized score)
            pvalues = 1 - (scores / scores.max() if scores.max() > 0 else scores)
        else:
            scores, pvalues = test_func(X, y)
        
        # Apply multiple testing correction
        if self.correction == 'bonferroni':
            pvalues_corrected = self._bonferroni_correction(pvalues)
            selected_mask = pvalues_corrected < self.alpha
        elif self.correction == 'fdr':
            selected_mask, pvalues_corrected = self._fdr_correction(pvalues)
        else:  # no correction
            pvalues_corrected = pvalues
            selected_mask = pvalues < self.alpha
        
        # Store results
        self.feature_scores_ = dict(zip(feature_names, scores))
        self.feature_pvalues_ = dict(zip(feature_names, pvalues_corrected))
        self.selected_features_ = [f for f, m in zip(feature_names, selected_mask) if m]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting significant features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Feature matrix with only significant features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task: str = 'classification') -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        task : str
            Task type
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, y, task).transform(X)
    
    def get_report(self) -> pd.DataFrame:
        """
        Get detailed report of feature selection.
        
        Returns:
        --------
        report : pd.DataFrame
            Report with test scores, p-values, and selection status
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        report_data = []
        for feature in self.feature_scores_.keys():
            report_data.append({
                'Feature': feature,
                'Score': self.feature_scores_[feature],
                'P_Value': self.feature_pvalues_[feature],
                'Selected': feature in self.selected_features_,
                'Status': 'Kept' if feature in self.selected_features_ else 'Removed'
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('P_Value').reset_index(drop=True)


def demo_statistical_selector():
    """Demonstrate statistical test selector usage."""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 500
    
    # Features with different relationships to target
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    noise1 = np.random.normal(0, 1, n_samples)
    noise2 = np.random.normal(0, 1, n_samples)
    
    # Target with strong relationship to x1, weak to x2, none to x3 and noise
    y = (2 * x1 + 0.5 * x2 + np.random.normal(0, 0.5, n_samples)) > 0
    y = y.astype(int)
    
    data = {
        'strong_feature': x1,
        'weak_feature': x2,
        'irrelevant_feature': x3,
        'noise_1': noise1,
        'noise_2': noise2,
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("Statistical Test Feature Selection Demo")
    print("=" * 60)
    
    # Initialize and fit selector
    selector = StatisticalTestSelector(alpha=0.05, correction='fdr', test_type='auto')
    selector.fit(df, pd.Series(y), task='classification')
    
    # Display report
    print("\nStatistical Test Results:")
    print("-" * 60)
    report = selector.get_report()
    print(report.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    
    print(f"\nSelected Features: {len(selector.selected_features_)}")
    print(f"Rejected Features: {len(df.columns) - len(selector.selected_features_)}")
    
    # Transform data
    df_selected = selector.transform(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Selected shape: {df_selected.shape}")
    print(f"\nRemaining features: {df_selected.columns.tolist()}")


if __name__ == "__main__":
    demo_statistical_selector()
