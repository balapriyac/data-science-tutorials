"""
Outlier Detection and Analysis Suite
Detects outliers using multiple methods and provides comprehensive
analysis of their impact and patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OutlierSuite:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the outlier detection suite.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.outlier_masks = {}
        
    def detect_iqr_outliers(
        self,
        column: str,
        multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using Interquartile Range method.
        
        Args:
            column: Column name
            multiplier: IQR multiplier (1.5 = standard, 3 = extreme)
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
    
    def detect_zscore_outliers(
        self,
        column: str,
        threshold: float = 3
    ) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            column: Column name
            threshold: Z-score threshold
        """
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        
        # Create boolean mask with same index as original
        mask = pd.Series(False, index=self.df.index)
        mask.loc[self.df[column].dropna().index] = z_scores > threshold
        
        return mask
    
    def detect_modified_zscore_outliers(
        self,
        column: str,
        threshold: float = 3.5
    ) -> pd.Series:
        """
        Detect outliers using Modified Z-score (MAD-based).
        
        Args:
            column: Column name
            threshold: Modified Z-score threshold
        """
        median = self.df[column].median()
        mad = np.median(np.abs(self.df[column] - median))
        
        if mad == 0:
            return pd.Series(False, index=self.df.index)
        
        modified_z_scores = 0.6745 * (self.df[column] - median) / mad
        
        return np.abs(modified_z_scores) > threshold
    
    def detect_isolation_forest_outliers(
        self,
        columns: Optional[List[str]] = None,
        contamination: float = 0.1
    ) -> pd.Series:
        """
        Detect multivariate outliers using Isolation Forest.
        
        Args:
            columns: Columns to use (all numeric if None)
            contamination: Expected proportion of outliers
        """
        cols = columns or self.numeric_cols
        
        if len(cols) == 0:
            return pd.Series(False, index=self.df.index)
        
        X = self.df[cols].fillna(self.df[cols].mean())
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X)
        
        return pd.Series(predictions == -1, index=self.df.index)
    
    def detect_mahalanobis_outliers(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 0.95
    ) -> pd.Series:
        """
        Detect multivariate outliers using Mahalanobis distance.
        
        Args:
            columns: Columns to use
            threshold: Chi-square probability threshold
        """
        cols = columns or self.numeric_cols
        
        if len(cols) < 2:
            return pd.Series(False, index=self.df.index)
        
        X = self.df[cols].fillna(self.df[cols].mean())
        
        # Use Elliptic Envelope for robust covariance estimation
        try:
            detector = EllipticEnvelope(contamination=0.1, random_state=42)
            predictions = detector.fit_predict(X)
            return pd.Series(predictions == -1, index=self.df.index)
        except:
            return pd.Series(False, index=self.df.index)
    
    def analyze_all_methods(
        self,
        column: str,
        iqr_mult: float = 1.5,
        z_threshold: float = 3,
        modified_z_threshold: float = 3.5
    ) -> pd.DataFrame:
        """
        Apply all univariate outlier detection methods to a column.
        
        Args:
            column: Column to analyze
            iqr_mult: IQR multiplier
            z_threshold: Z-score threshold
            modified_z_threshold: Modified Z-score threshold
        """
        results = pd.DataFrame(index=self.df.index)
        
        results['iqr'] = self.detect_iqr_outliers(column, iqr_mult)
        results['zscore'] = self.detect_zscore_outliers(column, z_threshold)
        results['modified_zscore'] = self.detect_modified_zscore_outliers(column, modified_z_threshold)
        
        # Consensus: flagged by at least 2 methods
        results['consensus'] = results.sum(axis=1) >= 2
        
        return results
    
    def get_outlier_summary(self, column: str) -> Dict:
        """Get summary statistics for outliers in a column."""
        analysis = self.analyze_all_methods(column)
        
        summary = {
            'column': column,
            'total_outliers_iqr': analysis['iqr'].sum(),
            'total_outliers_zscore': analysis['zscore'].sum(),
            'total_outliers_modified_zscore': analysis['modified_zscore'].sum(),
            'consensus_outliers': analysis['consensus'].sum(),
            'percentage_consensus': round(analysis['consensus'].mean() * 100, 2)
        }
        
        return summary
    
    def compare_methods_all_columns(self) -> pd.DataFrame:
        """Compare outlier detection methods across all numeric columns."""
        summaries = []
        
        for col in self.numeric_cols:
            summary = self.get_outlier_summary(col)
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def plot_outlier_comparison(
        self,
        column: str,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize outlier detection across methods for a column.
        
        Args:
            column: Column to analyze
            figsize: Figure size
        """
        analysis = self.analyze_all_methods(column)
        data = self.df[column].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        methods = ['iqr', 'zscore', 'modified_zscore', 'consensus']
        titles = ['IQR Method', 'Z-Score Method', 'Modified Z-Score', 'Consensus (≥2 methods)']
        
        for ax, method, title in zip(axes.flatten(), methods, titles):
            outliers = analysis[method]
            
            # Scatter plot
            ax.scatter(
                data.index[~outliers],
                data[~outliers],
                c='blue',
                alpha=0.5,
                s=20,
                label='Normal'
            )
            ax.scatter(
                data.index[outliers],
                data[outliers],
                c='red',
                alpha=0.7,
                s=50,
                label='Outlier'
            )
            
            ax.set_title(f'{title}\n{outliers.sum()} outliers ({outliers.mean()*100:.1f}%)')
            ax.set_xlabel('Index')
            ax.set_ylabel(column)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multivariate_outliers(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Visualize multivariate outlier detection.
        
        Args:
            columns: Columns to use (first 2-3 numeric if None)
            figsize: Figure size
        """
        cols = columns or self.numeric_cols[:3]
        
        if len(cols) < 2:
            print("Need at least 2 columns for multivariate outlier detection")
            return
        
        # Detect outliers
        iso_outliers = self.detect_isolation_forest_outliers(cols)
        maha_outliers = self.detect_mahalanobis_outliers(cols)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Isolation Forest
        ax = axes[0]
        ax.scatter(
            self.df.loc[~iso_outliers, cols[0]],
            self.df.loc[~iso_outliers, cols[1]],
            c='blue',
            alpha=0.5,
            s=20,
            label='Normal'
        )
        ax.scatter(
            self.df.loc[iso_outliers, cols[0]],
            self.df.loc[iso_outliers, cols[1]],
            c='red',
            alpha=0.7,
            s=50,
            label='Outlier'
        )
        ax.set_title(f'Isolation Forest\n{iso_outliers.sum()} outliers')
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mahalanobis
        ax = axes[1]
        ax.scatter(
            self.df.loc[~maha_outliers, cols[0]],
            self.df.loc[~maha_outliers, cols[1]],
            c='blue',
            alpha=0.5,
            s=20,
            label='Normal'
        )
        ax.scatter(
            self.df.loc[maha_outliers, cols[0]],
            self.df.loc[maha_outliers, cols[1]],
            c='red',
            alpha=0.7,
            s=50,
            label='Outlier'
        )
        ax.set_title(f'Mahalanobis Distance\n{maha_outliers.sum()} outliers')
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_outlier_impact(self, column: str) -> Dict:
        """Analyze the impact of outliers on statistics."""
        analysis = self.analyze_all_methods(column)
        consensus_outliers = analysis['consensus']
        
        data_with = self.df[column]
        data_without = self.df.loc[~consensus_outliers, column]
        
        impact = {
            'column': column,
            'mean_with_outliers': round(data_with.mean(), 4),
            'mean_without_outliers': round(data_without.mean(), 4),
            'mean_difference': round(data_with.mean() - data_without.mean(), 4),
            'median_with_outliers': round(data_with.median(), 4),
            'median_without_outliers': round(data_without.median(), 4),
            'std_with_outliers': round(data_with.std(), 4),
            'std_without_outliers': round(data_without.std(), 4)
        }
        
        return impact


# Example usage
if __name__ == "__main__":
    # Create dataset with outliers
    np.random.seed(42)
    n = 500
    
    # Normal data with injected outliers
    normal_data = np.random.normal(50, 10, n-30)
    outlier_data = np.random.uniform(120, 150, 30)
    
    df = pd.DataFrame({
        'feature1': np.concatenate([normal_data, outlier_data]),
        'feature2': np.random.normal(100, 15, n),
        'feature3': np.random.exponential(20, n),
        'feature4': np.random.uniform(0, 100, n)
    })
    
    # Add some extreme outliers
    df.loc[np.random.choice(df.index, 10), 'feature2'] = np.random.uniform(200, 250, 10)
    
    print("Sample Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\n" + "="*70 + "\n")
    
    # Initialize suite
    suite = OutlierSuite(df)
    
    # Compare methods across all columns
    print("Outlier Detection Summary:")
    summary = suite.compare_methods_all_columns()
    print(summary.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Analyze impact for specific column
    print("Outlier Impact Analysis for 'feature1':")
    impact = suite.analyze_outlier_impact('feature1')
    for key, value in impact.items():
        print(f"{key}: {value}")
    print("\n" + "="*70 + "\n")
    
    # Visualizations
    print("Generating outlier comparison plot...")
    suite.plot_outlier_comparison('feature1')
    
    print("Generating multivariate outlier plot...")
    suite.plot_multivariate_outliers(['feature1', 'feature2'])

