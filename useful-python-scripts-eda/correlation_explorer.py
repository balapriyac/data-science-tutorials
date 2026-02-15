"""
Correlation and Relationship Explorer
Analyzes relationships between variables using multiple correlation
methods and detects multicollinearity issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class CorrelationExplorer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the explorer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def calculate_correlations(
        self,
        method: str = 'pearson',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Args:
            method: 'pearson', 'spearman', or 'kendall'
            columns: Specific columns to include
        """
        cols = columns or self.numeric_cols
        
        if len(cols) < 2:
            print("Need at least 2 numeric columns for correlation")
            return pd.DataFrame()
        
        return self.df[cols].corr(method=method)
    
    def find_high_correlations(
        self,
        threshold: float = 0.7,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Find pairs of highly correlated features.
        
        Args:
            threshold: Minimum absolute correlation to report
            method: Correlation method
        """
        corr_matrix = self.calculate_correlations(method=method)
        
        if corr_matrix.empty:
            return pd.DataFrame()
        
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 4),
                        'abs_correlation': round(abs(corr_val), 4),
                        'strength': self._classify_correlation(abs(corr_val))
                    })
        
        if not high_corr:
            return pd.DataFrame()
        
        df_corr = pd.DataFrame(high_corr)
        return df_corr.sort_values('abs_correlation', ascending=False)
    
    def _classify_correlation(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.9:
            return 'Very Strong'
        elif abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.5:
            return 'Moderate'
        elif abs_corr >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def calculate_vif(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for multicollinearity detection.
        
        Args:
            columns: Columns to analyze (all numeric if None)
        """
        from sklearn.linear_model import LinearRegression
        
        cols = columns or self.numeric_cols
        
        if len(cols) < 2:
            print("Need at least 2 columns for VIF")
            return pd.DataFrame()
        
        vif_data = []
        
        for i, col in enumerate(cols):
            # Use other columns to predict this column
            X = self.df[cols].drop(columns=[col])
            y = self.df[col]
            
            # Remove rows with missing values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 2:
                continue
            
            # Fit model
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            
            # Calculate R²
            r_squared = model.score(X_clean, y_clean)
            
            # Calculate VIF
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            
            vif_data.append({
                'feature': col,
                'vif': round(vif, 4),
                'r_squared': round(r_squared, 4),
                'multicollinearity': self._classify_vif(vif)
            })
        
        return pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    
    def _classify_vif(self, vif: float) -> str:
        """Classify VIF severity."""
        if vif > 10:
            return 'High (Remove)'
        elif vif > 5:
            return 'Moderate (Consider removing)'
        else:
            return 'Low (Acceptable)'
    
    def mutual_information_analysis(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.
        
        Args:
            target_col: Target variable
            feature_cols: Features to analyze
        """
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        cols = feature_cols or [c for c in self.numeric_cols if c != target_col]
        
        X = self.df[cols].fillna(0)
        y = self.df[target_col].fillna(0)
        
        # Determine if classification or regression
        if self.df[target_col].nunique() < 10:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': cols,
            'mutual_information': mi_scores,
            'mi_normalized': mi_scores / mi_scores.max() if mi_scores.max() > 0 else 0
        })
        
        return mi_df.sort_values('mutual_information', ascending=False)
    
    def plot_correlation_heatmap(
        self,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (12, 10),
        annot: bool = True,
        cmap: str = 'coolwarm'
    ):
        """
        Plot correlation heatmap.
        
        Args:
            method: Correlation method
            figsize: Figure size
            annot: Whether to annotate cells
            cmap: Color map
        """
        corr_matrix = self.calculate_correlations(method=method)
        
        if corr_matrix.empty:
            print("No correlation matrix to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_comparison(
        self,
        figsize: Tuple[int, int] = (18, 5)
    ):
        """Plot correlation matrices using different methods side by side."""
        methods = ['pearson', 'spearman', 'kendall']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, method in enumerate(methods):
            corr = self.calculate_correlations(method=method)
            
            if corr.empty:
                continue
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(
                corr,
                mask=mask,
                ax=axes[idx],
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot=True if len(corr) < 10 else False,
                fmt='.2f'
            )
            
            axes[idx].set_title(f'{method.capitalize()} Correlation')
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_matrix(
        self,
        columns: Optional[List[str]] = None,
        max_cols: int = 5,
        figsize: Tuple[int, int] = (12, 12)
    ):
        """
        Plot scatter matrix for selected columns.
        
        Args:
            columns: Specific columns to plot
            max_cols: Maximum columns to include
            figsize: Figure size
        """
        cols = (columns or self.numeric_cols)[:max_cols]
        
        if len(cols) < 2:
            print("Need at least 2 columns for scatter matrix")
            return
        
        pd.plotting.scatter_matrix(
            self.df[cols],
            figsize=figsize,
            diagonal='hist',
            alpha=0.6,
            hist_kwds={'bins': 20, 'edgecolor': 'black'}
        )
        
        plt.suptitle('Scatter Matrix', y=1.0)
        plt.tight_layout()
        plt.show()
    
    def plot_top_correlations(
        self,
        n_pairs: int = 10,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot scatter plots for top correlated pairs.
        
        Args:
            n_pairs: Number of pairs to plot
            method: Correlation method
            figsize: Figure size
        """
        high_corr = self.find_high_correlations(threshold=0.0, method=method)
        
        if high_corr.empty:
            print("No correlations to plot")
            return
        
        top_pairs = high_corr.head(n_pairs)
        
        n_cols = 3
        n_rows = (len(top_pairs) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, row in top_pairs.iterrows():
            ax = axes[idx] if idx < len(axes) else None
            
            if ax is None:
                break
            
            feat1, feat2 = row['feature_1'], row['feature_2']
            corr = row['correlation']
            
            ax.scatter(self.df[feat1], self.df[feat2], alpha=0.5, s=20)
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.set_title(f'r = {corr:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(top_pairs), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample dataset with correlations
    np.random.seed(42)
    n = 500
    
    x1 = np.random.normal(50, 10, n)
    x2 = x1 + np.random.normal(0, 5, n)  # Highly correlated with x1
    x3 = 100 - x1 + np.random.normal(0, 8, n)  # Negatively correlated with x1
    x4 = np.random.normal(30, 15, n)  # Independent
    x5 = x1 * 0.5 + x4 * 0.5 + np.random.normal(0, 3, n)  # Related to both
    
    df = pd.DataFrame({
        'feature_A': x1,
        'feature_B': x2,
        'feature_C': x3,
        'feature_D': x4,
        'feature_E': x5,
        'target': x1 * 2 + x4 - x3 * 0.5 + np.random.normal(0, 10, n)
    })
    
    print("Sample Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\n" + "="*70 + "\n")
    
    # Initialize explorer
    explorer = CorrelationExplorer(df)
    
    # Find high correlations
    print("High Correlations (threshold = 0.5):")
    high_corr = explorer.find_high_correlations(threshold=0.5)
    print(high_corr.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Calculate VIF
    print("Variance Inflation Factors:")
    vif = explorer.calculate_vif()
    print(vif.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Mutual information with target
    print("Mutual Information with Target:")
    mi = explorer.mutual_information_analysis('target')
    print(mi.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Visualizations
    print("Generating correlation heatmap...")
    explorer.plot_correlation_heatmap()
    
    print("Generating correlation comparison...")
    explorer.plot_correlation_comparison()
    
    print("Generating scatter matrix...")
    explorer.plot_scatter_matrix(max_cols=4)
    
    print("Generating top correlation pairs...")
    explorer.plot_top_correlations(n_pairs=6)


