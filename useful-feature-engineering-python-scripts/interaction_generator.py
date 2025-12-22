"""
Feature Interaction Generator
Automatically generates and evaluates feature interactions,
keeping only the most predictive combinations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from itertools import combinations
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InteractionFeature:
    name: str
    feature1: str
    feature2: str
    operation: str
    importance_score: float

class InteractionGenerator:
    def __init__(
        self,
        max_interactions: int = 50,
        min_importance: float = 0.01,
        task: Literal['classification', 'regression'] = 'classification'
    ):
        """
        Initialize the interaction generator.
        
        Args:
            max_interactions: Maximum number of interactions to keep
            min_importance: Minimum importance score to consider
            task: Type of prediction task
        """
        self.max_interactions = max_interactions
        self.min_importance = min_importance
        self.task = task
        self.interactions: List[InteractionFeature] = []
    
    def generate_numeric_interactions(
        self,
        X: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        operations: List[str] = ['multiply', 'divide', 'add', 'subtract']
    ) -> pd.DataFrame:
        """
        Generate arithmetic interactions between numeric features.
        
        Args:
            X: Input DataFrame
            numeric_cols: Columns to create interactions from
            operations: List of operations to apply
        """
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        interactions_df = pd.DataFrame(index=X.index)
        
        for col1, col2 in combinations(numeric_cols, 2):
            if 'multiply' in operations:
                interactions_df[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            
            if 'divide' in operations:
                # Avoid division by zero
                interactions_df[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
                interactions_df[f'{col2}_div_{col1}'] = X[col2] / (X[col1] + 1e-8)
            
            if 'add' in operations:
                interactions_df[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
            
            if 'subtract' in operations:
                interactions_df[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
                interactions_df[f'{col2}_minus_{col1}'] = X[col2] - X[col1]
        
        return interactions_df
    
    def generate_polynomial_features(
        self,
        X: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Generate polynomial features.
        
        Args:
            X: Input DataFrame
            numeric_cols: Columns to create polynomials from
            degree: Polynomial degree
        """
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        poly_df = pd.DataFrame(index=X.index)
        
        for col in numeric_cols:
            for d in range(2, degree + 1):
                poly_df[f'{col}_pow_{d}'] = X[col] ** d
        
        return poly_df
    
    def generate_ratio_features(
        self,
        X: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate ratio features normalized by totals.
        
        Args:
            X: Input DataFrame
            numeric_cols: Columns to create ratios from
        """
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        ratio_df = pd.DataFrame(index=X.index)
        
        # Sum of all specified columns
        total = X[numeric_cols].sum(axis=1) + 1e-8
        
        for col in numeric_cols:
            ratio_df[f'{col}_ratio'] = X[col] / total
        
        return ratio_df
    
    def generate_categorical_interactions(
        self,
        X: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate interactions between categorical features.
        
        Args:
            X: Input DataFrame
            categorical_cols: Categorical columns to combine
        """
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) < 2:
            return pd.DataFrame(index=X.index)
        
        interaction_df = pd.DataFrame(index=X.index)
        
        for col1, col2 in combinations(categorical_cols, 2):
            # Combine categories
            interaction_df[f'{col1}_and_{col2}'] = (
                X[col1].astype(str) + '_' + X[col2].astype(str)
            )
        
        return interaction_df
    
    def evaluate_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Literal['mutual_info', 'random_forest'] = 'mutual_info'
    ) -> Dict[str, float]:
        """
        Evaluate feature importance scores.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Importance calculation method
        """
        importance_scores = {}
        
        if method == 'mutual_info':
            # Use mutual information
            if self.task == 'classification':
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            
            for col, score in zip(X.columns, scores):
                importance_scores[col] = score
        
        elif method == 'random_forest':
            # Use random forest feature importance
            if self.task == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            
            for col, importance in zip(X.columns, model.feature_importances_):
                importance_scores[col] = importance
        
        return importance_scores
    
    def select_top_interactions(
        self,
        interaction_df: pd.DataFrame,
        importance_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Select top interactions based on importance scores.
        
        Args:
            interaction_df: DataFrame with candidate interactions
            importance_scores: Dict of feature importance scores
        """
        # Filter by minimum importance
        valid_features = [
            col for col, score in importance_scores.items()
            if score >= self.min_importance
        ]
        
        # Sort by importance and take top N
        sorted_features = sorted(
            valid_features,
            key=lambda x: importance_scores[x],
            reverse=True
        )[:self.max_interactions]
        
        return interaction_df[sorted_features]
    
    def generate_and_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_operations: List[str] = ['multiply', 'divide'],
        include_polynomials: bool = True,
        polynomial_degree: int = 2,
        include_ratios: bool = True,
        include_categorical: bool = True,
        importance_method: Literal['mutual_info', 'random_forest'] = 'mutual_info'
    ) -> pd.DataFrame:
        """
        Generate all interactions and select the most important ones.
        
        Args:
            X: Input features
            y: Target variable
            numeric_operations: Operations for numeric interactions
            include_polynomials: Whether to include polynomial features
            polynomial_degree: Degree of polynomial features
            include_ratios: Whether to include ratio features
            include_categorical: Whether to include categorical interactions
            importance_method: Method to calculate feature importance
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        all_interactions = pd.DataFrame(index=X.index)
        
        # Generate numeric interactions
        if numeric_cols:
            print(f"Generating numeric interactions for {len(numeric_cols)} columns...")
            numeric_interactions = self.generate_numeric_interactions(
                X, numeric_cols, numeric_operations
            )
            all_interactions = pd.concat([all_interactions, numeric_interactions], axis=1)
        
        # Generate polynomial features
        if include_polynomials and numeric_cols:
            print(f"Generating polynomial features up to degree {polynomial_degree}...")
            poly_features = self.generate_polynomial_features(
                X, numeric_cols, polynomial_degree
            )
            all_interactions = pd.concat([all_interactions, poly_features], axis=1)
        
        # Generate ratio features
        if include_ratios and numeric_cols:
            print(f"Generating ratio features...")
            ratio_features = self.generate_ratio_features(X, numeric_cols)
            all_interactions = pd.concat([all_interactions, ratio_features], axis=1)
        
        # Generate categorical interactions
        if include_categorical and len(categorical_cols) >= 2:
            print(f"Generating categorical interactions...")
            cat_interactions = self.generate_categorical_interactions(X, categorical_cols)
            
            # Encode categorical interactions
            for col in cat_interactions.columns:
                cat_interactions[col] = pd.Categorical(cat_interactions[col]).codes
            
            all_interactions = pd.concat([all_interactions, cat_interactions], axis=1)
        
        print(f"\nGenerated {len(all_interactions.columns)} candidate interactions")
        
        # Handle infinite and NaN values
        all_interactions = all_interactions.replace([np.inf, -np.inf], np.nan)
        all_interactions = all_interactions.fillna(0)
        
        # Evaluate importance
        print(f"Evaluating feature importance using {importance_method}...")
        importance_scores = self.evaluate_importance(
            all_interactions, y, method=importance_method
        )
        
        # Select top interactions
        print(f"Selecting top {self.max_interactions} interactions...")
        selected_interactions = self.select_top_interactions(
            all_interactions, importance_scores
        )
        
        # Store interaction details
        for col in selected_interactions.columns:
            parts = col.split('_')
            self.interactions.append(InteractionFeature(
                name=col,
                feature1=parts[0] if len(parts) > 0 else '',
                feature2=parts[-1] if len(parts) > 1 else '',
                operation='_'.join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else 'unknown',
                importance_score=importance_scores[col]
            ))
        
        print(f"\nSelected {len(selected_interactions.columns)} interactions")
        
        return selected_interactions
    
    def get_interaction_report(self) -> pd.DataFrame:
        """Get report of selected interactions."""
        return pd.DataFrame([{
            'interaction_name': i.name,
            'feature_1': i.feature1,
            'feature_2': i.feature2,
            'operation': i.operation,
            'importance_score': round(i.importance_score, 6)
        } for i in self.interactions]).sort_values('importance_score', ascending=False)


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(20, 5, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'feature4': np.random.exponential(5, n_samples),
        'category1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category2': np.random.choice(['X', 'Y'], n_samples)
    })
    
    # Create target with some interaction effects
    y = (
        X['feature1'] * X['feature2'] * 0.1 +
        X['feature3'] / (X['feature4'] + 1) +
        (X['category1'] == 'A').astype(int) * 50 +
        np.random.normal(0, 10, n_samples)
    )
    
    # For classification
    y_class = (y > y.median()).astype(int)
    
    print("Original Features:")
    print(X.head())
    print(f"\nShape: {X.shape}")
    print("\n" + "="*60 + "\n")
    
    # Initialize generator
    generator = InteractionGenerator(
        max_interactions=20,
        min_importance=0.001,
        task='regression'
    )
    
    # Generate and select interactions
    interactions = generator.generate_and_select(
        X, y,
        numeric_operations=['multiply', 'divide', 'add'],
        include_polynomials=True,
        polynomial_degree=2,
        include_ratios=True,
        include_categorical=True,
        importance_method='mutual_info'
    )
    
    print("\n" + "="*60 + "\n")
    print("Selected Interactions:")
    print(interactions.head())
    print(f"\nShape: {interactions.shape}")
    print("\n" + "="*60 + "\n")
    
    print("Interaction Importance Report:")
    print(generator.get_interaction_report().head(15))
    
    # Combine with original features
    X_enhanced = pd.concat([X, interactions], axis=1)
    print(f"\nEnhanced feature set shape: {X_enhanced.shape}")

