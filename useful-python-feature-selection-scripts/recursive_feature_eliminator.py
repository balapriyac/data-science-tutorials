"""
Recursive Feature Elimination Selector

Systematically removes features to find optimal subset through
iterative model training and performance evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RecursiveFeatureEliminator:
    """
    Performs recursive feature elimination to find optimal feature subset.
    Tracks performance at each step and identifies best configuration.
    """
    
    def __init__(self, estimator=None, step: int = 1, cv: int = 5, 
                 scoring: str = 'accuracy', random_state: int = 42):
        """
        Initialize recursive feature eliminator.
        
        Parameters:
        -----------
        estimator : sklearn estimator, optional
            Model to use for feature importance (default: RandomForestClassifier)
        step : int
            Number of features to remove at each iteration (default: 1)
        cv : int
            Number of cross-validation folds (default: 5)
        scoring : str
            Scoring metric: 'accuracy', 'f1', 'roc_auc' (default: 'accuracy')
        random_state : int
            Random state for reproducibility (default: 42)
        """
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.selected_features_ = None
        self.performance_history_ = None
        self.optimal_n_features_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RecursiveFeatureEliminator':
        """
        Fit the eliminator on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        self : RecursiveFeatureEliminator
        """
        # Initialize estimator if not provided
        if self.estimator is None:
            self.estimator = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state
            )
        
        # Start with all features
        current_features = X.columns.tolist()
        performance_history = []
        
        print(f"Starting RFE with {len(current_features)} features...")
        
        # Iteratively remove features
        while len(current_features) > 1:
            # Get current feature subset
            X_current = X[current_features]
            
            # Evaluate performance with cross-validation
            cv_scores = cross_val_score(
                self.estimator, X_current, y, 
                cv=self.cv, scoring=self.scoring
            )
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            # Store results
            performance_history.append({
                'n_features': len(current_features),
                'features': current_features.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            print(f"  {len(current_features)} features: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Fit model to get feature importances
            self.estimator.fit(X_current, y)
            
            # Get feature importances
            if hasattr(self.estimator, 'feature_importances_'):
                importances = self.estimator.feature_importances_
            elif hasattr(self.estimator, 'coef_'):
                importances = np.abs(self.estimator.coef_).flatten()
            else:
                raise ValueError("Estimator must have feature_importances_ or coef_ attribute")
            
            # Remove least important feature(s)
            n_to_remove = min(self.step, len(current_features) - 1)
            least_important_indices = np.argsort(importances)[:n_to_remove]
            features_to_remove = [current_features[i] for i in least_important_indices]
            
            current_features = [f for f in current_features if f not in features_to_remove]
        
        # Find optimal number of features
        self.performance_history_ = pd.DataFrame(performance_history)
        
        # Find configuration with best performance
        best_idx = self.performance_history_['mean_score'].idxmax()
        self.optimal_n_features_ = self.performance_history_.loc[best_idx, 'n_features']
        self.selected_features_ = self.performance_history_.loc[best_idx, 'features']
        
        print(f"\nOptimal number of features: {self.optimal_n_features_}")
        print(f"Best CV score: {self.performance_history_.loc[best_idx, 'mean_score']:.4f}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using optimal feature subset.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Feature matrix with optimal features
        """
        if self.selected_features_ is None:
            raise ValueError("Eliminator has not been fitted yet. Call fit() first.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_performance_curve(self) -> pd.DataFrame:
        """
        Get performance curve showing score vs number of features.
        
        Returns:
        --------
        curve : pd.DataFrame
            Performance metrics for each feature subset size
        """
        if self.performance_history_ is None:
            raise ValueError("Eliminator has not been fitted yet. Call fit() first.")
        
        return self.performance_history_[['n_features', 'mean_score', 'std_score']].copy()
    
    def plot_performance_curve(self):
        """Plot performance curve (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            if self.performance_history_ is None:
                raise ValueError("Eliminator has not been fitted yet. Call fit() first.")
            
            curve = self.get_performance_curve()
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                curve['n_features'], 
                curve['mean_score'], 
                yerr=curve['std_score'],
                marker='o', capsize=5, capthick=2
            )
            plt.axvline(
                x=self.optimal_n_features_, 
                color='r', 
                linestyle='--', 
                label=f'Optimal: {self.optimal_n_features_} features'
            )
            plt.xlabel('Number of Features')
            plt.ylabel(f'CV Score ({self.scoring})')
            plt.title('Recursive Feature Elimination Performance Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it to plot curves.")


def demo_rfe_selector():
    """Demonstrate recursive feature elimination usage."""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 500
    
    # Create features with varying importance
    x1 = np.random.normal(0, 1, n_samples)  # Very important
    x2 = np.random.normal(0, 1, n_samples)  # Important
    x3 = np.random.normal(0, 1, n_samples)  # Moderately important
    x4 = np.random.normal(0, 1, n_samples)  # Slightly important
    noise1 = np.random.normal(0, 1, n_samples)  # Noise
    noise2 = np.random.normal(0, 1, n_samples)  # Noise
    noise3 = np.random.normal(0, 1, n_samples)  # Noise
    
    # Create target
    y = (4 * x1 + 2 * x2 + 1 * x3 + 0.3 * x4 + np.random.normal(0, 0.5, n_samples)) > 0
    y = y.astype(int)
    
    data = {
        'very_important': x1,
        'important': x2,
        'moderately_important': x3,
        'slightly_important': x4,
        'noise_1': noise1,
        'noise_2': noise2,
        'noise_3': noise3,
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("Recursive Feature Elimination Demo")
    print("=" * 60)
    print()
    
    # Initialize and fit eliminator
    eliminator = RecursiveFeatureEliminator(
        step=1, cv=5, scoring='accuracy', random_state=42
    )
    eliminator.fit(df, pd.Series(y))
    
    # Display performance curve
    print("\nPerformance Curve:")
    print("-" * 60)
    curve = eliminator.get_performance_curve()
    print(curve.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Transform data
    df_selected = eliminator.transform(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Selected shape: {df_selected.shape}")
    print(f"\nSelected features: {df_selected.columns.tolist()}")


if __name__ == "__main__":
    demo_rfe_selector()

