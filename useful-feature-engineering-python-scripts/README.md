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

## Complete Feature Engineering Pipeline Example

```python
import pandas as pd
from smart_encoder import SmartEncoder
from numerical_transformer import NumericalTransformer
from interaction_generator import InteractionGenerator
from datetime_extractor import DatetimeExtractor
from feature_selector import FeatureSelector

# Load data
df = pd.read_csv('data.csv')
y = df['target']
X = df.drop('target', axis=1)

print(f"Starting with {X.shape[1]} features")

# Step 1: Extract datetime features
print("\n1. Extracting datetime features...")
dt_extractor = DatetimeExtractor()
datetime_features = dt_extractor.extract_all(X, include_lag_features=False)
X = pd.concat([X, datetime_features], axis=1)
print(f"After datetime extraction: {X.shape[1]} features")

# Step 2: Encode categorical features
print("\n2. Encoding categorical features...")
encoder = SmartEncoder(cardinality_threshold=10)
X_encoded = encoder.fit_transform(X, y)
print(f"After encoding: {X_encoded.shape[1]} features")

# Step 3: Transform numerical features
print("\n3. Transforming numerical features...")
transformer = NumericalTransformer()
X_transformed = transformer.fit_transform(X_encoded, auto_transform=True)
print(f"After transformation: {X_transformed.shape[1]} features")

# Step 4: Generate interactions
print("\n4. Generating feature interactions...")
generator = InteractionGenerator(max_interactions=30, task='classification')
interactions = generator.generate_and_select(
    X_transformed, y,
    numeric_operations=['multiply', 'divide'],
    include_polynomials=True
)
X_enhanced = pd.concat([X_transformed, interactions], axis=1)
print(f"After interactions: {X_enhanced.shape[1]} features")

# Step 5: Select best features
print("\n5. Selecting best features...")
selector = FeatureSelector(task='classification', n_features=50)
X_final, scores = selector.select_features(X_enhanced, y)
print(f"Final feature set: {X_final.shape[1]} features")

# Save for modeling
X_final.to_csv('engineered_features.csv', index=False)
print("\nEngineered features saved!")

# Get comprehensive report
print("\nTop 20 Features by Importance:")
print(selector.get_feature_importance_report(scores, top_n=20))
```

---

## 🎯 Pipeline for Train/Test Data

```python
# Train data
train_df = pd.read_csv('train.csv')
y_train = train_df['target']
X_train = train_df.drop('target', axis=1)

# Test data
test_df = pd.read_csv('test.csv')
y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

# Fit on training data only
encoder = SmartEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)

transformer = NumericalTransformer()
X_train_transformed = transformer.fit_transform(X_train_encoded)

generator = InteractionGenerator(max_interactions=30)
interactions_train = generator.generate_and_select(X_train_transformed, y_train)
X_train_final = pd.concat([X_train_transformed, interactions_train], axis=1)

selector = FeatureSelector(n_features=50)
X_train_selected, _ = selector.select_features(X_train_final, y_train)

# Apply to test data (using fitted transformations)
X_test_encoded = encoder.transform(X_test)
X_test_transformed = transformer.transform(X_test_encoded)

# For interactions on test: would need to regenerate using same logic
# For selection: use the selected feature names
selected_cols = selector.get_selected_features()
X_test_selected = X_test_transformed[selected_cols]

print(f"Train shape: {X_train_selected.shape}")
print(f"Test shape: {X_test_selected.shape}")
```


---

## Performance Tips

- Use `n_features` parameter to limit feature count
- Set `max_interactions` to avoid explosion of features
- Use `importance_method='mutual_info'` for faster selection
- For large datasets, use `contamination` parameter in outlier detection
- Reduce `n_quantiles` in quantile transform for large data
