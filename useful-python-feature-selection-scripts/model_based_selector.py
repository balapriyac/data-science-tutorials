"""
Model-Based Feature Selector

Selects features using importance scores from multiple models,
combining results into ensemble rankings.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelBasedSelector:
    """
    Selects features using ensemble importance from multiple model types.
    Supports tree-based, linear, and permutation importance methods.
    """
    
    def __init__(self, n_features: int = None, threshold: float = None, 
                 use_permutation: bool = False, random_state: int = 42):
        """
        Initialize model-based selector.
        
        Parameters:
        -----------
        n_features : int, optional
            Number of top features to select (default: None, uses threshold)
        threshold : float, optional
            Importance threshold (0-1) for feature selection (default: None, uses n_features)
        use_permutation : bool
            Whether to use permutation importance (default: False)
        random_state : int
            Random state for reproducibility (default: 42)
        """
        self.n_features = n_features
        self.threshold = threshold
        self.use_permutation = use_permutation
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None
        self.model_scores_ = None
        
    def _get_tree_importance(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance from tree-based model."""
        model.fit(X, y)
        importances = model.feature_importances_
        return dict(zip(X.columns, importances))
    
    def _get_linear_importance(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance from linear model coefficients."""
        # Standardize features for fair comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_).flatten()
        
        # Normalize to [0, 1]
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return dict(zip(X.columns, importances))
    
    def _get_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get permutation importance."""
        model.fit(X, y)
        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=self.random_state
        )
        importances = perm_importance.importances_mean
        
        # Normalize to [0, 1]
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return dict(zip(X.columns, importances))
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task: str = 'classification') -> 'ModelBasedSelector':
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
        self : ModelBasedSelector
        """
        feature_names = X.columns.tolist()
        all_importances = {}
        
        # Define models based on task
        if task == 'classification':
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=self.random_state
                ),
                'LogisticRegression': LogisticRegression(
                    penalty='l1', solver='liblinear', C=1.0, random_state=self.random_state
                )
            }
        else:  # regression
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=self.random_state
                ),
                'Lasso': Lasso(alpha=0.1, random_state=self.random_state)
            }
        
        # Get importance from each model
        for model_name, model in models.items():
            if self.use_permutation:
                importances = self._get_permutation_importance(model, X, y)
            elif model_name in ['RandomForest', 'GradientBoosting']:
                importances = self._get_tree_importance(model, X, y)
            else:  # Linear models
                importances = self._get_linear_importance(model, X, y)
            
            all_importances[model_name] = importances
        
        # Convert to DataFrame for easier manipulation
        importance_df = pd.DataFrame(all_importances)
        
        # Calculate ensemble importance (mean across models)
        importance_df['Ensemble_Mean'] = importance_df.mean(axis=1)
        importance_df['Ensemble_Rank'] = importance_df['Ensemble_Mean'].rank(ascending=False)
        
        # Normalize ensemble importance
        max_importance = importance_df['Ensemble_Mean'].max()
        if max_importance > 0:
            importance_df['Ensemble_Normalized'] = importance_df['Ensemble_Mean'] / max_importance
        else:
            importance_df['Ensemble_Normalized'] = 0
        
        # Select features
        if self.n_features is not None:
            # Select top N features
            selected_features = importance_df.nlargest(self.n_features, 'Ensemble_Mean').index.tolist()
        elif self.threshold is not None:
            # Select features above threshold
            selected_features = importance_df[
                importance_df['Ensemble_Normalized'] >= self.threshold
            ].index.tolist()
        else:
            # Default: select features with above-average importance
            mean_importance = importance_df['Ensemble_Mean'].mean()
            selected_features = importance_df[
                importance_df['Ensemble_Mean'] >= mean_importance
            ].index.tolist()
        
        # Store results
        self.selected_features_ = selected_features
        self.feature_importances_ = importance_df
        self.model_scores_ = all_importances
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting important features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pd.DataFrame
            Feature matrix with selected features
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
    
    def get_report(self, top_n: int = None) -> pd.DataFrame:
        """
        Get detailed report of feature importances.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to include in report (default: all)
            
        Returns:
        --------
        report : pd.DataFrame
            Report with importance scores from all models
        """
        if self.feature_importances_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        
        report = self.feature_importances_.copy()
        report['Selected'] = report.index.isin(self.selected_features_)
        report = report.sort_values('Ensemble_Mean', ascending=False)
        
        if top_n is not None:
            report = report.head(top_n)
        
        return report.reset_index().rename(columns={'index': 'Feature'})


def demo_model_based_selector():
    """Demonstrate model-based selector usage."""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with varying importance
    x1 = np.random.normal(0, 1, n_samples)  # Strong predictor
    x2 = np.random.normal(0, 1, n_samples)  # Moderate predictor
    x3 = np.random.normal(0, 1, n_samples)  # Weak predictor
    noise1 = np.random.normal(0, 1, n_samples)  # Noise
    noise2 = np.random.normal(0, 1, n_samples)  # Noise
    
    # Create target with different feature contributions
    y = (3 * x1 + 1.5 * x2 + 0.3 * x3 + np.random.normal(0, 0.5, n_samples)) > 0
    y = y.astype(int)
    
    data = {
        'strong_predictor': x1,
        'moderate_predictor': x2,
        'weak_predictor': x3,
        'noise_feature_1': noise1,
        'noise_feature_2': noise2,
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 70)
    print("Model-Based Feature Selection Demo")
    print("=" * 70)
    
    # Initialize and fit selector
    selector = ModelBasedSelector(n_features=3, use_permutation=False)
    selector.fit(df, pd.Series(y), task='classification')
    
    # Display report
    print("\nFeature Importance Report:")
    print("-" * 70)
    report = selector.get_report()
    print(report.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print(f"\nSelected Features: {len(selector.selected_features_)}")
    print(f"Removed Features: {len(df.columns) - len(selector.selected_features_)}")
    
    # Transform data
    df_selected = selector.transform(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Selected shape: {df_selected.shape}")
    print(f"\nSelected features: {df_selected.columns.tolist()}")


if __name__ == "__main__":
    demo_model_based_selector()

