#  Useful Python Scripts for Feature Selection

A collection of five automated feature selection scripts for machine learning projects. These scripts help identify the most relevant/valuable features, remove redundancy, and optimize your feature set for better model performance.

## Overview

Feature selection is super important for building efficient, accurate machine learning models. This repository provides five complementary approaches:

1. **Variance Threshold Selector** - Removes features with low or zero variance
2. **Correlation-Based Selector** - Eliminates redundant highly correlated features
3. **Statistical Test Selector** - Selects features based on statistical significance
4. **Model-Based Selector** - Ranks features using ensemble importance from multiple models
5. **Recursive Feature Eliminator** - Finds optimal feature subset through iterative elimination

Each script can be used independently or combined into a complete feature selection pipeline.

## Dependencies

Install required packages:

```bash
pip install numpy pandas scikit-learn scipy
```

Optional for visualization:
```bash
pip install matplotlib seaborn
```

## Quick Start

### 1. Variance Threshold Selector

Removes constant or near-constant features that provide no information.

```python
from variance_threshold_selector import VarianceThresholdSelector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)

# Initialize and fit selector
selector = VarianceThresholdSelector(threshold=0.01, normalize=True)
selector.fit(X)

# Get report
print(selector.get_report())

# Transform data
X_selected = selector.transform(X)
print(f"Reduced from {X.shape[1]} to {X_selected.shape[1]} features")
```

**Parameters:**
- `threshold`: Variance threshold (default: 0.01)
- `normalize`: Whether to normalize variance by feature range (default: True)

### 2. Correlation-Based Selector

Removes redundant features by identifying highly correlated pairs.

```python
from correlation_selector import CorrelationSelector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize and fit selector
selector = CorrelationSelector(threshold=0.95, method='pearson')
selector.fit(X, y)

# Get correlated pairs report
print(selector.get_report())

# Transform data
X_selected = selector.transform(X)
print(f"Removed {len(selector.removed_features_)} correlated features")
```

**Parameters:**
- `threshold`: Correlation threshold above which features are redundant (default: 0.95)
- `method`: Correlation method - 'pearson', 'spearman', or 'kendall' (default: 'pearson')

### 3. Statistical Test Selector

Selects features with statistically significant relationships to the target.

```python
from statistical_test_selector import StatisticalTestSelector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize and fit selector
selector = StatisticalTestSelector(alpha=0.05, correction='fdr', test_type='auto')
selector.fit(X, y, task='classification')

# Get statistical test results
print(selector.get_report())

# Transform data
X_selected = selector.transform(X)
print(f"Selected {len(selector.selected_features_)} significant features")
```

**Parameters:**
- `alpha`: Significance level (default: 0.05)
- `correction`: Multiple testing correction - 'bonferroni', 'fdr', or 'none' (default: 'fdr')
- `test_type`: Test type - 'auto', 'anova', 'chi2', 'mutual_info', or 'f_regression' (default: 'auto')

### 4. Model-Based Selector

Ranks features using importance scores from multiple models.

```python
from model_based_selector import ModelBasedSelector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize and fit selector
selector = ModelBasedSelector(n_features=10, use_permutation=False)
selector.fit(X, y, task='classification')

# Get importance report
print(selector.get_report())

# Transform data
X_selected = selector.transform(X)
print(f"Selected top {X_selected.shape[1]} features")
```

**Parameters:**
- `n_features`: Number of top features to select (default: None)
- `threshold`: Importance threshold for selection (default: None)
- `use_permutation`: Whether to use permutation importance (default: False)
- `random_state`: Random state for reproducibility (default: 42)

### 5. Recursive Feature Eliminator

Finds optimal feature subset through systematic elimination and cross-validation.

```python
from recursive_feature_eliminator import RecursiveFeatureEliminator
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize and fit eliminator
eliminator = RecursiveFeatureEliminator(step=1, cv=5, scoring='accuracy')
eliminator.fit(X, y)

# Get performance curve
print(eliminator.get_performance_curve())

# Transform data
X_selected = eliminator.transform(X)
print(f"Optimal subset: {X_selected.shape[1]} features")
print(f"Selected: {X_selected.columns.tolist()}")
```

**Parameters:**
- `estimator`: Model for importance scores (default: RandomForestClassifier)
- `step`: Number of features to remove per iteration (default: 1)
- `cv`: Number of cross-validation folds (default: 5)
- `scoring`: Scoring metric - 'accuracy', 'f1', 'roc_auc' (default: 'accuracy')


---



## Complete Pipeline Example

Combine multiple selectors for comprehensive feature selection:

```python
import pandas as pd
from variance_threshold_selector import VarianceThresholdSelector
from correlation_selector import CorrelationSelector
from statistical_test_selector import StatisticalTestSelector
from model_based_selector import ModelBasedSelector
from recursive_feature_eliminator import RecursiveFeatureEliminator

# Load data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

print(f"Starting with {X.shape[1]} features")

# Step 1: Remove low-variance features
variance_selector = VarianceThresholdSelector(threshold=0.01)
X = variance_selector.fit_transform(X)
print(f"After variance filter: {X.shape[1]} features")

# Step 2: Remove correlated features
corr_selector = CorrelationSelector(threshold=0.95)
X = corr_selector.fit_transform(X, y)
print(f"After correlation filter: {X.shape[1]} features")

# Step 3: Select statistically significant features
stat_selector = StatisticalTestSelector(alpha=0.05, correction='fdr')
X = stat_selector.fit_transform(X, y, task='classification')
print(f"After statistical tests: {X.shape[1]} features")

# Step 4: Select top features by model importance
model_selector = ModelBasedSelector(n_features=20)
X = model_selector.fit_transform(X, y, task='classification')
print(f"After model selection: {X.shape[1]} features")

# Step 5: Find optimal subset with RFE
rfe = RecursiveFeatureEliminator(step=1, cv=5)
X_final = rfe.fit_transform(X, y)
print(f"Final optimal subset: {X_final.shape[1]} features")
print(f"Selected features: {X_final.columns.tolist()}")
```

## When to Use Each Selector

| Selector | Best For | Typical Use Case |
|----------|----------|------------------|
| Variance Threshold | Removing uninformative features | First step to eliminate constants |
| Correlation-Based | Handling multicollinearity | Linear models, reducing redundancy |
| Statistical Test | Ensuring significance | Validating feature-target relationships |
| Model-Based | General-purpose ranking | Identifying most predictive features |
| RFE | Finding optimal subset | Final optimization, feature interactions |


## Tips and Best Practices

1. **Start with variance filtering** - Quick way to remove obvious noise
2. **Apply correlation filtering early** - Reduces dimensionality before expensive operations
3. **Use statistical tests for validation** - Ensure features have meaningful relationships
4. **Combine multiple methods as needed** - Different selectors capture different aspects
5. **Always use cross-validation** - Especially with RFE to avoid overfitting
6. **Monitor performance** - Track model metrics throughout the selection process
7. **Consider domain knowledge** - Please don't remove features that make business sense
