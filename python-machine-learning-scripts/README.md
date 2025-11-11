# 5 Essential Python Scripts for Intermediate Machine Learning Practitioners

A collection of ready-to-use Python scripts designed to automate and streamline common machine learning pipeline tasks. 

## 🎯 What's Included

1. **Automated Feature Engineering Pipeline** - Intelligent preprocessing and feature generation
2. **Hyperparameter Optimization Manager** - Unified interface for multiple optimization strategies
3. **Model Performance Debugger** - Comprehensive model diagnostics and debugging
4. **Cross-Validation Strategy Manager** - Smart CV splitting for various data types
5. **Experiment Tracker** - Lightweight experiment tracking and comparison

## 📋 Requirements

### Core Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
scipy>=1.7.0
```

### Optional Dependencies
```bash
# For Bayesian optimization
pip install optuna>=3.0.0

# For advanced imputation
pip install scikit-learn[experimental]
```

## 🚀 Quick Start

### 1. Automated Feature Engineering Pipeline

Automatically handles feature preprocessing, encoding, scaling, and engineering.

```python
from feature_engineering_pipeline import FeatureEngineer
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')

# Initialize the pipeline
engineer = FeatureEngineer(
    target_column='target',
    numeric_strategy='robust',  # 'standard', 'robust', 'minmax'
    categorical_strategy='target_encoding',  # 'onehot', 'label', 'target_encoding'
    missing_strategy='iterative',  # 'simple', 'knn', 'iterative'
    generate_interactions=True,
    outlier_detection=True
)

# Fit and transform
X_train, y_train = engineer.fit_transform(df)

# Transform new data
X_test = engineer.transform(df_test)

# Get engineering report
report = engineer.generate_report()
print(f"Original features: {report['original_count']}")
print(f"Engineered features: {report['final_count']}")

# Save for production
engineer.save_pipeline('preprocessor.pkl')

# Load later
engineer = FeatureEngineer.load_pipeline('preprocessor.pkl')
```

---

### 2. Hyperparameter Optimization Manager

Unified interface for multiple optimization strategies with automatic tracking.

```python
from hyperparameter_optimizer import HyperparameterOptimizer
from sklearn.ensemble import RandomForestClassifier

# Define search space
param_space = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    model=RandomForestClassifier(random_state=42),
    param_space=param_space,
    optimization_strategy='bayesian',  # 'grid', 'random', 'bayesian', 'halving'
    cv=5,
    scoring='f1',
    n_iter=50,
    early_stopping_rounds=10
)

# Run optimization
results = optimizer.optimize(X_train, y_train)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")

# Analyze parameter importance
importance = optimizer.get_param_importance()
print(f"Most important parameters: {importance[:3]}")

# Visualize optimization
optimizer.plot_optimization_history(save_path='optimization.png')
optimizer.plot_param_distributions(save_path='param_dist.png')

# Save best model
optimizer.save_best_model('best_model.pkl')
```

---

### 3. Model Performance Debugger

Automatically diagnose model performance issues and detect problematic data segments.

```python
from model_debugger import ModelPerformanceDebugger
from sklearn.ensemble import GradientBoostingClassifier

# Train your model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Initialize debugger
debugger = ModelPerformanceDebugger(
    model=model,
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    task_type='classification',  # or 'regression'
    X_train=X_train  # Optional: for drift detection
)

# Run full diagnosis
report = debugger.run_full_diagnosis()

# Check problematic segments
print("Worst performing segments:")
for segment in report['problematic_segments'][:5]:
    print(f"  {segment['feature']}: {segment['value']}")
    print(f"    Accuracy: {segment['accuracy']:.3f}")
    print(f"    Sample size: {segment['n_samples']}")

# Check for data drift
if report['drift_detected']:
    print(f"\nDrift detected in: {report['drifted_features']}")

# Visualize performance breakdown
debugger.plot_feature_performance_breakdown(top_n=10)

# Export detailed report
debugger.export_html_report('debug_report.html')
```

---

### 4. Cross-Validation Strategy Manager

Intelligent CV splitting with automatic strategy recommendation.

```python
from cv_strategy_manager import CrossValidationManager
import pandas as pd

# Load dataset
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Initialize CV manager
cv_manager = CrossValidationManager(
    data=df,
    target_column='sales',
    datetime_column='date',  # For time-series splitting
    group_column='store_id',  # For grouped splitting
    task_type='regression'
)

# Get automatic recommendation
strategy = cv_manager.recommend_strategy()
print(f"Recommended: {strategy['name']}")
print(f"Reason: {strategy['rationale']}")

# Generate splits
splits = cv_manager.generate_splits(
    n_splits=5,
    strategy='time_series_group',  # or 'stratified', 'grouped', 'time_series', 'standard'
    test_size=0.2
)

# Validate split quality
validation = cv_manager.validate_splits(splits)
print(f"No temporal leakage: {validation['no_leakage']}")
print(f"Groups separated: {validation['groups_separated']}")

# Cross-validate with any model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

scores = cv_manager.cross_validate(
    model=model,
    splits=splits,
    scoring=['neg_mean_squared_error', 'r2']
)

print(f"CV RMSE: {np.sqrt(-scores['neg_mean_squared_error'].mean()):.3f}")

# Visualize splits
cv_manager.plot_split_distributions(splits, save_path='cv_splits.png')
```

---

### 5. Experiment Tracker

Lightweight experiment tracking with metadata, artifacts, and comparison.

```python
from experiment_tracker import ExperimentTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Initialize tracker
tracker = ExperimentTracker(
    project_name='customer_churn',
    experiment_name='rf_baseline_v1',
    tracking_dir='./ml_experiments'
)

# Log dataset info
tracker.log_dataset(
    name='customer_data_v2.csv',
    n_samples=X_train.shape[0],
    n_features=X_train.shape[1],
    feature_names=X_train.columns.tolist()
)

# Train model and track time
model = RandomForestClassifier(n_estimators=100, max_depth=10)
tracker.log_params(model.get_params())

tracker.start_timer()
model.fit(X_train, y_train)
training_time = tracker.stop_timer()

# Make predictions and log metrics
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'training_time': training_time
})

# Log feature importance
tracker.log_feature_importance(
    feature_names=X_train.columns,
    importance_values=model.feature_importances_
)

# Save artifacts
tracker.save_model(model, 'rf_model.pkl')
tracker.log_confusion_matrix(y_test, y_pred, labels=['No Churn', 'Churn'])

# Add metadata
tracker.log_notes("Baseline model with default features.")
tracker.add_tags(['baseline', 'random_forest', 'production_candidate'])

# Finish experiment
tracker.finish()

# Compare experiments
comparison = ExperimentTracker.compare_experiments(
    project_name='customer_churn',
    metric='roc_auc',
    top_n=5
)
print(comparison[['experiment_name', 'roc_auc', 'f1_score', 'training_time']])

# Visualize comparison
ExperimentTracker.plot_experiment_comparison(
    experiments=['rf_baseline_v1', 'xgb_tuned_v1', 'lgbm_v2'],
    metrics=['accuracy', 'f1_score', 'roc_auc'],
    project_name='customer_churn'
)
```

---

