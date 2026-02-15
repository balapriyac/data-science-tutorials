# 5 Useful Python Scripts for Data Exploration

Automate exploratory data analysis with these 5 comprehensive Python scripts that save hours of manual work.

## Scripts Included

1. **Comprehensive Data Profiler** - Complete dataset profiling with statistics and quality checks
2. **Distribution Analyzer and Visualizer** - Distribution analysis with plots and normality tests
3. **Correlation and Relationship Explorer** - Multi-method correlation analysis and VIF detection
4. **Outlier Detection and Analysis Suite** - Multiple outlier detection methods with consensus
5. **Missing Data Pattern Analyzer** - Missingness pattern detection and mechanism classification

---

## 📚 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

---

## Quick Start Examples

### 1. Comprehensive Data Profiler
```python
from data_profiler import DataProfiler
import pandas as pd

df = pd.read_csv('data.csv')

# Create profiler
profiler = DataProfiler(df, high_cardinality_threshold=0.5)

# Print summary to console
profiler.print_summary()

# Or get detailed report
report = profiler.generate_full_profile()
print(report['overview'])
print(report['numeric_profiles'])
print(report['categorical_profiles'])
print(report['data_quality_issues'])

# Save reports
report['numeric_profiles'].to_csv('numeric_profile.csv', index=False)
report['data_quality_issues'].to_csv('issues.csv', index=False)
```

### 2. Distribution Analyzer and Visualizer
```python
from distribution_analyzer import DistributionAnalyzer

df = pd.read_csv('data.csv')
analyzer = DistributionAnalyzer(df)

# Generate distribution report
report = analyzer.generate_distribution_report()
print(report)

# Identify highly skewed columns
skewed = report[abs(report['skewness']) > 2]
print(f"Highly skewed columns: {skewed['column'].tolist()}")

# Visualize distributions
analyzer.plot_numeric_distributions(max_cols=10)
analyzer.plot_boxplots()
analyzer.plot_qq_plots()
analyzer.plot_categorical_distributions()
```

### 3. Correlation and Relationship Explorer
```python
from correlation_explorer import CorrelationExplorer

df = pd.read_csv('data.csv')
explorer = CorrelationExplorer(df)

# Find high correlations
high_corr = explorer.find_high_correlations(threshold=0.7, method='pearson')
print(high_corr)

# Check for multicollinearity
vif = explorer.calculate_vif()
problematic = vif[vif['vif'] > 10]
print(f"Features with high multicollinearity:\n{problematic}")

# Mutual information with target
if 'target' in df.columns:
    mi_scores = explorer.mutual_information_analysis('target')
    print(f"Top features:\n{mi_scores.head(10)}")

# Visualize
explorer.plot_correlation_heatmap(method='pearson')
explorer.plot_correlation_comparison()
explorer.plot_scatter_matrix(max_cols=5)
explorer.plot_top_correlations(n_pairs=10)
```

### 4. Outlier Detection and Analysis Suite
```python
from outlier_suite import OutlierSuite

df = pd.read_csv('data.csv')
suite = OutlierSuite(df)

# Compare methods across all columns
summary = suite.compare_methods_all_columns()
print(summary)

# Analyze specific column
suite.plot_outlier_comparison('column_name')

# Detect multivariate outliers
iso_outliers = suite.detect_isolation_forest_outliers(contamination=0.1)
print(f"Found {iso_outliers.sum()} multivariate outliers")

suite.plot_multivariate_outliers(['feature1', 'feature2'])

# Analyze outlier impact
impact = suite.analyze_outlier_impact('column_name')
print(impact)
```

### 5. Missing Data Pattern Analyzer
```python
from missing_data_analyzer import MissingDataAnalyzer

df = pd.read_csv('data.csv')
analyzer = MissingDataAnalyzer(df)

# Generate full report
report = analyzer.generate_full_report()

print("Missing Value Summary:")
print(report['summary'])

print("\nMissingness Patterns (co-occurrence):")
print(report['patterns'])

print("\nMissingness Classifications:")
print(report['classifications'])

print("\nImputation Recommendations:")
print(report['recommendations'])

# Visualize patterns
analyzer.plot_missing_bar()
analyzer.plot_missing_heatmap(max_cols=30)
analyzer.plot_missing_correlation()

# Classify specific column
classification = analyzer.classify_missingness_type('column_name')
recommendation = analyzer.recommend_strategy('column_name')
print(f"Missingness type: {classification['missingness_type']}")
print(f"Recommendation: {recommendation}")
```

---

