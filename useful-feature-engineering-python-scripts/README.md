# 5 Useful Python Scripts for Feature Engineering

Automate feature engineering with these 5 powerful Python scripts designed to create high-quality features systematically.

## Scripts Included

1. **Categorical Feature Encoder** - Intelligent encoding based on cardinality
2. **Numerical Feature Transformer** - Auto-finds optimal transformations
3. **Feature Interaction Generator** - Discovers valuable feature combinations
4. **Datetime Feature Extractor** - Extracts temporal patterns
5. **Automated Feature Selector** - Selects most predictive features

---

## 📚 Dependencies

```bash
pip install pandas numpy scikit-learn scipy dateutil
```

---

## Quick Start Examples

### 1. Categorical Feature Encoder
```python
from smart_encoder import SmartEncoder
import pandas as pd

df = pd.read_csv('data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Initialize encoder
encoder = SmartEncoder(
    cardinality_threshold=10,
    rare_threshold=0.01
)

# Fit and transform
X_encoded = encoder.fit_transform(X, y)

# Or fit and transform separately
encoder.fit(X, y)
X_encoded = encoder.transform(X)

# Get encoding report
report = encoder.get_encoding_report()
print(report)

# Force specific strategies
X_encoded = encoder.fit_transform(
    X, y,
    column_strategies={
        'category': 'target',
        'country': 'frequency',
        'status': 'onehot'
    }
)
```

### 2. Numerical Feature Transformer
```python
from numerical_transformer import NumericalTransformer

df = pd.read_csv('data.csv')

# Initialize transformer
transformer = NumericalTransformer()

# Fit and transform (finds best transformation for each column)
df_transformed = transformer.fit_transform(
    df,
    auto_transform=True,  # Find best transformation
    auto_scale=True,      # Apply scaling
    scaling_method='standard'  # 'standard', 'minmax', or 'robust'
)

# Get transformation report
report = transformer.get_transformation_report()
print(report)

# Apply to new data
new_data = pd.read_csv('test_data.csv')
new_transformed = transformer.transform(new_data)
```

### 3. Feature Interaction Generator
```python
from interaction_generator import InteractionGenerator

df = pd.read_csv('data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Initialize generator
generator = InteractionGenerator(
    max_interactions=50,
    min_importance=0.01,
    task='classification'  # or 'regression'
)

# Generate and select interactions
interactions = generator.generate_and_select(
    X, y,
    numeric_operations=['multiply', 'divide'],
    include_polynomials=True,
    polynomial_degree=2,
    include_ratios=True,
    include_categorical=True,
    importance_method='mutual_info'  # or 'random_forest'
)

# Get interaction report
report = generator.get_interaction_report()
print(report)

# Combine with original features
X_enhanced = pd.concat([X, interactions], axis=1)
```

### 4. Datetime Feature Extractor
```python
from datetime_extractor import DatetimeExtractor

df = pd.read_csv('data.csv')

# Initialize extractor
extractor = DatetimeExtractor(
    include_cyclical=True,
    include_flags=True,
    include_time_diff=True
)

# Extract all datetime features
datetime_features = extractor.extract_all(
    df,
    datetime_cols=['order_date', 'ship_date'],
    include_lag_features=True
)

# Get extraction report
report = extractor.get_extraction_report()
print(report)

# Combine with original data
df_enhanced = pd.concat([df, datetime_features], axis=1)
```

### 5. Automated Feature Selector
```python
from feature_selector import FeatureSelector

df = pd.read_csv('data.csv')
y = df['target']
X = df.drop('target', axis=1)

# Initialize selector
selector = FeatureSelector(
    task='classification',  # or 'regression'
    n_features=50  # Target number of features
)

# Select features
X_selected, feature_scores = selector.select_features(
    X, y,
    remove_low_variance=True,
    remove_correlated=True,
    correlation_threshold=0.95,
    selection_methods=['statistical', 'mutual_info', 'tree_based'],
    n_features=50
)

# Get feature importance report
report = selector.get_feature_importance_report(feature_scores, top_n=20)
print(report)

# Get selected feature names
selected_features = selector.get_selected_features()
print(f"Selected {len(selected_features)} features")
```

---


## Performance Tips

- Use `n_features` parameter to limit feature count
- Set `max_interactions` to avoid explosion of features
- Use `importance_method='mutual_info'` for faster selection
- For large datasets, use `contamination` parameter in outlier detection
- Reduce `n_quantiles` in quantile transform for large data
