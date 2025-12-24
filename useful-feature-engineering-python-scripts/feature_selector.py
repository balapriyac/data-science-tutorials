"""
Automated Feature Selector
Automatically selects the most valuable features using multiple
selection strategies and provides comprehensive importance rankings.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    chi2, mutual_info_classif, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureScore:
    feature: str
    importance_score: float
    selection_method: str
    rank: int

class FeatureSelector:
    def __init__(
        self,
        task: Literal['classification', 'regression'] = 'classification',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize the feature selector.
        
        Args:
            task: Type of prediction task
            n_features: Target number of features to select
            threshold: Minimum importance threshold
        """
        self.task = task
        self.n_features = n_features
        self.threshold = threshold
        self.feature_scores: Dict[str, List[FeatureScore]] = {}
        self.selected_features: List[str] = []
    
    def remove_low_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            X: Input features
            threshold: Variance threshold
        """
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_cols = X.columns[selector.get_support()].tolist()
        removed_cols = X.columns[~selector.get_support()].tolist()
        
        print(f"Removed {len(removed_cols)} low-variance features")
        
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index), removed_cols
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features, keeping the one most correlated with target.
        
        Args:
            X: Input features
            y: Target variable
            threshold: Correlation threshold
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Calculate correlation with target
        target_corr = X.corrwith(y).abs()
        
        # Find features to remove
        to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Keep the feature more correlated with target
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    
                    if target_corr[feat1] < target_corr[feat2]:
                        to_remove.add(feat1)
                    else:
                        to_remove.add(feat2)
        
        to_remove = list(to_remove)
        X_selected = X.drop(columns=to_remove)
        
        print(f"Removed {len(to_remove)} highly correlated features")
        
        return X_selected, to_remove
    
    def statistical_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Select features using statistical tests.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
        """
        k = k or min(50, X.shape[1])
        
        # Choose appropriate statistical test
        if self.task == 'classification':
            # Use ANOVA F-statistic for classification
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        else:
            # Use F-statistic for regression
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        
        selector.fit(X, y)
        
        # Get feature scores
        scores = dict(zip(X.columns, selector.scores_))
        
        return scores
    
    def mutual_information_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Select features using mutual information.
        
        Args:
            X: Input features
            y: Target variable
        """
        if self.task == 'classification':
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        
        return dict(zip(X.columns, scores))
    
    def tree_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100
    ) -> Dict[str, float]:
        """
        Select features using tree-based importance.
        
        Args:
            X: Input features
            y: Target variable
            n_estimators: Number of trees
        """
        if self.task == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X, y)
        
        return dict(zip(X.columns, model.feature_importances_))
    
    def l1_regularization_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Select features using L1 regularization.
        
        Args:
            X: Input features
            y: Target variable
        """
        if self.task == 'classification':
            model = LogisticRegressionCV(
                cv=5,
                penalty='l1',
                solver='saga',
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
        else:
            model = LassoCV(cv=5, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        
        # Get absolute coefficients as importance
        if self.task == 'classification':
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_)
        
        return dict(zip(X.columns, importances))
    
    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: Optional[int] = None
    ) -> List[str]:
        """
        Select features using recursive feature elimination.
        
        Args:
            X: Input features
            y: Target variable
            n_features_to_select: Number of features to select
        """
        n_features_to_select = n_features_to_select or max(1, X.shape[1] // 2)
        
        if self.task == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
        
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        
        return selected
    
    def ensemble_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = ['statistical', 'mutual_info', 'tree_based', 'l1']
    ) -> pd.DataFrame:
        """
        Combine multiple selection methods.
        
        Args:
            X: Input features
            y: Target variable
            methods: List of methods to use
        """
        all_scores = pd.DataFrame(index=X.columns)
        
        if 'statistical' in methods:
            print("Computing statistical scores...")
            scores = self.statistical_selection(X, y)
            all_scores['statistical'] = pd.Series(scores)
        
        if 'mutual_info' in methods:
            print("Computing mutual information scores...")
            scores = self.mutual_information_selection(X, y)
            all_scores['mutual_info'] = pd.Series(scores)
        
        if 'tree_based' in methods:
            print("Computing tree-based importance...")
            scores = self.tree_based_selection(X, y)
            all_scores['tree_based'] = pd.Series(scores)
        
        if 'l1' in methods:
            print("Computing L1 regularization importance...")
            scores = self.l1_regularization_selection(X, y)
            all_scores['l1'] = pd.Series(scores)
        
        # Normalize each method's scores to [0, 1]
        for col in all_scores.columns:
            min_val = all_scores[col].min()
            max_val = all_scores[col].max()
            if max_val > min_val:
                all_scores[col] = (all_scores[col] - min_val) / (max_val - min_val)
        
        # Calculate average rank
        all_scores['mean_score'] = all_scores.mean(axis=1)
        all_scores['rank'] = all_scores['mean_score'].rank(ascending=False)
        
        return all_scores.sort_values('mean_score', ascending=False)
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        remove_low_variance: bool = True,
        remove_correlated: bool = True,
        correlation_threshold: float = 0.95,
        selection_methods: List[str] = ['statistical', 'mutual_info', 'tree_based'],
        n_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete feature selection pipeline.
        
        Args:
            X: Input features
            y: Target variable
            remove_low_variance: Whether to remove low-variance features
            remove_correlated: Whether to remove correlated features
            correlation_threshold: Correlation threshold for removal
            selection_methods: List of selection methods to use
            n_features: Number of features to select (None = use threshold)
        """
        print(f"Starting with {X.shape[1]} features")
        print("="*60)
        
        X_filtered = X.copy()
        
        # Step 1: Remove low variance features
        if remove_low_variance:
            X_filtered, _ = self.remove_low_variance(X_filtered)
            print(f"After variance filter: {X_filtered.shape[1]} features")
        
        # Step 2: Remove correlated features
        if remove_correlated and self.task == 'regression':
            X_filtered, _ = self.remove_correlated_features(
                X_filtered, y, correlation_threshold
            )
            print(f"After correlation filter: {X_filtered.shape[1]} features")
        
        # Step 3: Ensemble selection
        print("\nApplying ensemble feature selection...")
        feature_scores = self.ensemble_selection(X_filtered, y, selection_methods)
        
        # Step 4: Select top features
        n_features = n_features or self.n_features
        if n_features:
            selected_features = feature_scores.head(n_features).index.tolist()
        elif self.threshold:
            selected_features = feature_scores[
                feature_scores['mean_score'] >= self.threshold
            ].index.tolist()
        else:
            # Default: select top 50% of features
            n_select = max(1, len(feature_scores) // 2)
            selected_features = feature_scores.head(n_select).index.tolist()
        
        self.selected_features = selected_features
        
        print(f"\nSelected {len(selected_features)} features")
        print("="*60)
        
        return X_filtered[selected_features], feature_scores
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features
    
    def get_feature_importance_report(
        self,
        feature_scores: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get formatted feature importance report.
        
        Args:
            feature_scores: DataFrame with feature scores
            top_n: Number of top features to show
        """
        report = feature_scores.head(top_n).copy()
        report['feature'] = report.index
        report = report.reset_index(drop=True)
        
        # Reorder columns
        cols = ['feature', 'mean_score', 'rank'] + [
            c for c in report.columns if c not in ['feature', 'mean_score', 'rank']
        ]
        
        return report[cols].round(4)


# Example usage
if __name__ == "__main__":
    # Create sample dataset with relevant and irrelevant features
    np.random.seed(42)
    n_samples = 500
    
    # Relevant features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    
    # Target depends on X1, X2, X3
    y = (2 * X1 + 3 * X2 - 1.5 * X3 + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
    
    # Create DataFrame with relevant and irrelevant features
    X = pd.DataFrame({
        'relevant_1': X1,
        'relevant_2': X2,
        'relevant_3': X3,
        'irrelevant_1': np.random.normal(0, 1, n_samples),
        'irrelevant_2': np.random.normal(0, 1, n_samples),
        'irrelevant_3': np.random.uniform(0, 1, n_samples),
        'low_variance': np.ones(n_samples),
        'correlated_1': X1 + np.random.normal(0, 0.1, n_samples),  # Highly correlated with X1
        'correlated_2': X2 + np.random.normal(0, 0.1, n_samples),  # Highly correlated with X2
    })
    
    # Add interaction feature
    X['interaction'] = X['relevant_1'] * X['relevant_2']
    
    print("Original Features:")
    print(X.head())
    print(f"\nShape: {X.shape}")
    print("\n" + "="*60 + "\n")
    
    # Initialize selector
    selector = FeatureSelector(
        task='classification',
        n_features=5
    )
    
    # Select features
    X_selected, scores = selector.select_features(
        X, y,
        remove_low_variance=True,
        remove_correlated=True,
        correlation_threshold=0.9,
        selection_methods=['statistical', 'mutual_info', 'tree_based', 'l1']
    )
    
    print("\nSelected Features:")
    print(X_selected.head())
    print(f"\nShape: {X_selected.shape}")
    print("\n" + "="*60 + "\n")
    
    print("Top 10 Features by Importance:")
    print(selector.get_feature_importance_report(scores, top_n=10))
    print("\n" + "="*60 + "\n")
    
    print("Selected Feature Names:")
    print(selector.get_selected_features())

