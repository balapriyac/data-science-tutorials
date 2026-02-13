"""
Distribution Analyzer and Visualizer
Automatically generates comprehensive distribution visualizations
and statistical tests for all features in a dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DistributionAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def analyze_numeric_distribution(self, col: str) -> Dict[str, any]:
        """
        Analyze distribution characteristics for a numeric column.
        
        Args:
            col: Column name to analyze
        """
        series = self.df[col].dropna()
        
        if len(series) == 0:
            return {'error': 'No non-null values'}
        
        # Basic statistics
        mean = series.mean()
        median = series.median()
        mode = series.mode()[0] if len(series.mode()) > 0 else np.nan
        std = series.std()
        
        # Shape statistics
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # Normality test (Shapiro-Wilk for sample size < 5000)
        if len(series) < 5000:
            _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
            is_normal = p_value > 0.05
        else:
            # Use Anderson-Darling for larger samples
            result = stats.anderson(series.sample(5000))
            is_normal = result.statistic < result.critical_values[2]  # 5% significance
            p_value = None
        
        # Detect distribution type
        dist_type = self._classify_distribution(skewness, kurtosis, is_normal)
        
        # Outlier detection (IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        
        return {
            'column': col,
            'count': len(series),
            'mean': round(mean, 4),
            'median': round(median, 4),
            'mode': round(mode, 4) if not pd.isna(mode) else None,
            'std': round(std, 4),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'is_normal': is_normal,
            'normality_p_value': round(p_value, 4) if p_value else None,
            'distribution_type': dist_type,
            'outlier_count': outliers,
            'outlier_percentage': round(outliers / len(series) * 100, 2)
        }
    
    def _classify_distribution(self, skew: float, kurt: float, is_normal: bool) -> str:
        """Classify distribution type based on statistics."""
        if is_normal and abs(skew) < 0.5:
            return 'Normal'
        elif skew > 1:
            return 'Right-skewed (Positive)'
        elif skew < -1:
            return 'Left-skewed (Negative)'
        elif abs(skew) < 0.5 and kurt > 3:
            return 'Leptokurtic (Heavy-tailed)'
        elif abs(skew) < 0.5 and kurt < 0:
            return 'Platykurtic (Light-tailed)'
        else:
            return 'Approximately symmetric'
    
    def plot_numeric_distributions(
        self, 
        columns: Optional[List[str]] = None,
        max_cols: int = 10,
        figsize: Tuple[int, int] = (15, 12)
    ):
        """
        Plot distributions for numeric columns.
        
        Args:
            columns: Specific columns to plot (None = all numeric)
            max_cols: Maximum number of columns to plot
            figsize: Figure size
        """
        cols_to_plot = columns or self.numeric_cols
        cols_to_plot = cols_to_plot[:max_cols]
        
        if not cols_to_plot:
            print("No numeric columns to plot")
            return
        
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(col)
                continue
            
            # Histogram with KDE
            ax.hist(data, bins=30, alpha=0.6, color='skyblue', edgecolor='black', density=True)
            
            # KDE overlay
            try:
                data_range = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                ax.plot(data_range, kde(data_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
            
            # Statistics
            skew = data.skew()
            kurt = data.kurtosis()
            
            # Title with statistics
            ax.set_title(f'{col}\nSkew: {skew:.2f}, Kurt: {kurt:.2f}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(
        self,
        columns: Optional[List[str]] = None,
        max_cols: int = 10,
        figsize: Tuple[int, int] = (15, 8)
    ):
        """
        Plot box plots for numeric columns to show outliers.
        
        Args:
            columns: Specific columns to plot
            max_cols: Maximum number of columns to plot
            figsize: Figure size
        """
        cols_to_plot = columns or self.numeric_cols
        cols_to_plot = cols_to_plot[:max_cols]
        
        if not cols_to_plot:
            print("No numeric columns to plot")
            return
        
        n_cols = min(4, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(col)
                continue
            
            # Box plot
            bp = ax.boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Count outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            
            ax.set_title(f'{col}\n{outliers} outliers ({outliers/len(data)*100:.1f}%)')
            ax.set_ylabel('Value')
        
        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_distributions(
        self,
        columns: Optional[List[str]] = None,
        max_categories: int = 20,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot bar charts for categorical columns.
        
        Args:
            columns: Specific columns to plot
            max_categories: Maximum categories to show per column
            figsize: Figure size
        """
        cols_to_plot = columns or self.categorical_cols
        
        if not cols_to_plot:
            print("No categorical columns to plot")
            return
        
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            
            # Get value counts
            value_counts = self.df[col].value_counts().head(max_categories)
            
            if len(value_counts) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(col)
                continue
            
            # Bar plot
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            
            ax.set_title(f'{col}\n{self.df[col].nunique()} unique values')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_distribution_report(self) -> pd.DataFrame:
        """Generate comprehensive distribution report for numeric columns."""
        reports = []
        
        for col in self.numeric_cols:
            report = self.analyze_numeric_distribution(col)
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def plot_qq_plots(
        self,
        columns: Optional[List[str]] = None,
        max_cols: int = 9,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot Q-Q plots to assess normality.
        
        Args:
            columns: Specific columns to plot
            max_cols: Maximum number of columns to plot
            figsize: Figure size
        """
        cols_to_plot = columns or self.numeric_cols
        cols_to_plot = cols_to_plot[:max_cols]
        
        if not cols_to_plot:
            print("No numeric columns to plot")
            return
        
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            
            if len(data) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_title(col)
                continue
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'{col}\nQ-Q Plot')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample dataset with different distributions
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'normal': np.random.normal(50, 10, n),
        'right_skewed': np.random.exponential(5, n),
        'left_skewed': 100 - np.random.exponential(5, n),
        'uniform': np.random.uniform(0, 100, n),
        'bimodal': np.concatenate([np.random.normal(30, 5, n//2), np.random.normal(70, 5, n//2)]),
        'with_outliers': np.concatenate([np.random.normal(50, 10, n-50), np.random.uniform(150, 200, 50)]),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'segment': np.random.choice(['High', 'Medium', 'Low'], n, p=[0.2, 0.5, 0.3])
    })
    
    print("Sample Data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\n" + "="*70 + "\n")
    
    # Initialize analyzer
    analyzer = DistributionAnalyzer(df)
    
    # Generate distribution report
    print("Distribution Analysis Report:")
    report = analyzer.generate_distribution_report()
    print(report.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    # Plot distributions
    print("Generating distribution plots...")
    analyzer.plot_numeric_distributions()
    
    print("Generating box plots...")
    analyzer.plot_boxplots()
    
    print("Generating Q-Q plots...")
    analyzer.plot_qq_plots()
    
    print("Generating categorical distribution plots...")
    analyzer.plot_categorical_distributions()

