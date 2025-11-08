import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
from typing import Dict, List, Optional, Union


class FeatureEngineer:
    """
    Automated feature engineering pipeline that handles common preprocessing tasks.
    """
    
    def __init__(
        self,
        target_column: str,
        numeric_strategy: str = 'standard',
        categorical_strategy: str = 'onehot',
        missing_strategy: str = 'simple',
        generate_interactions: bool = False,
        interaction_degree: int = 2,
        outlier_detection: bool = True,
        outlier_contamination: float = 0.1
    ):
        """
        Initialize the feature engineering pipeline.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column
        numeric_strategy : str
            Strategy for numeric scaling: 'standard', 'robust', 'minmax'
        categorical_strategy : str
            Strategy for categorical encoding: 'onehot', 'label', 'target_encoding'
        missing_strategy : str
            Strategy for missing values: 'simple', 'knn', 'iterative'
        generate_interactions : bool
            Whether to generate polynomial features and interactions
        interaction_degree : int
            Degree of polynomial features to generate
        outlier_detection : bool
            Whether to detect and cap outliers
        outlier_contamination : float
            Expected proportion of outliers in the dataset
        """
        self.target_column = target_column
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_strategy = missing_strategy
        self.generate_interactions = generate_interactions
        self.interaction_degree = interaction_degree
        self.outlier_detection = outlier_detection
        self.outlier_contamination = outlier_contamination
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.original_columns = []
        self.engineered_columns = []
        
        self.pipeline = None
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.target_encodings = {}
        self.feature_importance = {}
        
    def _detect_column_types(self, df: pd.DataFrame):
        """Automatically detect column types."""
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                self.categorical_cols.append(col)
    
    def _handle_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        df = df.copy()
        
        for col in self.datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            
            # Cyclical encoding for month and day
            df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
            df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
            df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[col].dt.day / 31)
            df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[col].dt.day / 31)
            
            df = df.drop(columns=[col])
            
            # Add these new columns to numeric columns
            new_cols = [f'{col}_year', f'{col}_month', f'{col}_day', 
                       f'{col}_dayofweek', f'{col}_quarter',
                       f'{col}_month_sin', f'{col}_month_cos',
                       f'{col}_day_sin', f'{col}_day_cos']
            self.numeric_cols.extend(new_cols)
        
        return df
    
    def _detect_and_cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Isolation Forest and cap them."""
        if not self.outlier_detection or len(self.numeric_cols) == 0:
            return df
        
        df = df.copy()
        numeric_data = df[self.numeric_cols].copy()
        
        # Remove any NaN values temporarily for outlier detection
        numeric_data_clean = numeric_data.fillna(numeric_data.median())
        
        iso_forest = IsolationForest(
            contamination=self.outlier_contamination,
            random_state=42
        )
        outliers = iso_forest.fit_predict(numeric_data_clean)
        
        # Cap outliers at 1st and 99th percentiles
        for col in self.numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _build_numeric_pipeline(self):
        """Build pipeline for numeric features."""
        steps = []
        
        # Missing value imputation
        if self.missing_strategy == 'simple':
            steps.append(('imputer', SimpleImputer(strategy='median')))
        elif self.missing_strategy == 'knn':
            steps.append(('imputer', KNNImputer(n_neighbors=5)))
        elif self.missing_strategy == 'iterative':
            steps.append(('imputer', IterativeImputer(random_state=42)))
        
        # Scaling
        if self.numeric_strategy == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.numeric_strategy == 'robust':
            steps.append(('scaler', RobustScaler()))
        
        # Polynomial features
        if self.generate_interactions:
            steps.append(('poly', PolynomialFeatures(
                degree=self.interaction_degree,
                include_bias=False
            )))
        
        self.numeric_pipeline = Pipeline(steps)
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        if len(self.categorical_cols) == 0:
            return df
        
        df = df.copy()
        
        if self.categorical_strategy == 'onehot':
            if fit:
                self.encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                )
                encoded = self.encoder.fit_transform(df[self.categorical_cols])
            else:
                encoded = self.encoder.transform(df[self.categorical_cols])
            
            # Get feature names
            feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df.index
            )
            
            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
            
        elif self.categorical_strategy == 'label':
            if fit:
                self.label_encoders = {}
            
            for col in self.categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].map(lambda x: x if x in le.classes_ else 'unknown')
                    if 'unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    df[col] = le.transform(df[col].astype(str))
        
        elif self.categorical_strategy == 'target_encoding':
            # Simple target encoding (mean of target per category)
            if fit:
                for col in self.categorical_cols:
                    encoding_map = df.groupby(col)[self.target_column].mean().to_dict()
                    self.target_encodings[col] = encoding_map
                    global_mean = df[self.target_column].mean()
                    df[col] = df[col].map(encoding_map).fillna(global_mean)
            else:
                for col in self.categorical_cols:
                    encoding_map = self.target_encodings[col]
                    global_mean = np.mean(list(encoding_map.values()))
                    df[col] = df[col].map(encoding_map).fillna(global_mean)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit the pipeline and transform the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        X_transformed : pd.DataFrame or np.ndarray
            Transformed features
        y : pd.Series
            Target variable
        """
        self.original_columns = df.columns.tolist()
        
        # Separate features and target
        y = df[self.target_column].copy()
        X = df.drop(columns=[self.target_column])
        
        # Detect column types
        self._detect_column_types(X)
        
        # Handle datetime features
        X = self._handle_datetime_features(X)
        
        # Detect and cap outliers
        X = self._detect_and_cap_outliers(X)
        
        # Encode categorical features
        X_with_target = X.copy()
        X_with_target[self.target_column] = y
        X = self._encode_categorical(X_with_target, fit=True)
        X = X.drop(columns=[self.target_column])
        
        # Build and fit numeric pipeline
        if len(self.numeric_cols) > 0:
            self._build_numeric_pipeline()
            
            # Get numeric columns after encoding
            numeric_cols_after_encoding = [col for col in X.columns 
                                          if col in self.numeric_cols or 
                                          any(nc in col for nc in self.numeric_cols)]
            
            X_numeric = self.numeric_pipeline.fit_transform(X[numeric_cols_after_encoding])
            
            # Handle polynomial feature names
            if self.generate_interactions:
                poly_features = self.numeric_pipeline.named_steps['poly']
                feature_names = poly_features.get_feature_names_out(numeric_cols_after_encoding)
            else:
                feature_names = numeric_cols_after_encoding
            
            X_numeric_df = pd.DataFrame(
                X_numeric,
                columns=feature_names,
                index=X.index
            )
            
            # Drop original numeric columns and add transformed ones
            X = X.drop(columns=numeric_cols_after_encoding)
            X = pd.concat([X, X_numeric_df], axis=1)
        
        self.engineered_columns = X.columns.tolist()
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed features
        """
        X = df.copy()
        
        # Handle datetime features
        X = self._handle_datetime_features(X)
        
        # Detect and cap outliers (using same thresholds)
        X = self._detect_and_cap_outliers(X)
        
        # Encode categorical features
        X = self._encode_categorical(X, fit=False)
        
        # Transform numeric features
        if len(self.numeric_cols) > 0 and self.numeric_pipeline is not None:
            numeric_cols_after_encoding = [col for col in X.columns 
                                          if col in self.numeric_cols or 
                                          any(nc in col for nc in self.numeric_cols)]
            
            X_numeric = self.numeric_pipeline.transform(X[numeric_cols_after_encoding])
            
            if self.generate_interactions:
                poly_features = self.numeric_pipeline.named_steps['poly']
                feature_names = poly_features.get_feature_names_out(numeric_cols_after_encoding)
            else:
                feature_names = numeric_cols_after_encoding
            
            X_numeric_df = pd.DataFrame(
                X_numeric,
                columns=feature_names,
                index=X.index
            )
            
            X = X.drop(columns=numeric_cols_after_encoding)
            X = pd.concat([X, X_numeric_df], axis=1)
        
        return X
    
    def generate_report(self) -> Dict:
        """Generate a report of feature engineering operations."""
        report = {
            'original_count': len(self.original_columns) - 1,  # Exclude target
            'final_count': len(self.engineered_columns),
            'numeric_features': len(self.numeric_cols),
            'categorical_features': len(self.categorical_cols),
            'datetime_features': len(self.datetime_cols),
            'strategies': {
                'numeric_scaling': self.numeric_strategy,
                'categorical_encoding': self.categorical_strategy,
                'missing_values': self.missing_strategy,
                'interactions_generated': self.generate_interactions
            },
            'top_features': self.engineered_columns[:20] if len(self.engineered_columns) > 20 else self.engineered_columns
        }
        
        return report
    
    def save_pipeline(self, filepath: str):
        """Save the pipeline to disk."""
        pipeline_data = {
            'numeric_pipeline': self.numeric_pipeline,
            'encoder': getattr(self, 'encoder', None),
            'label_encoders': getattr(self, 'label_encoders', None),
            'target_encodings': self.target_encodings,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'datetime_cols': self.datetime_cols,
            'engineered_columns': self.engineered_columns,
            'config': {
                'target_column': self.target_column,
                'numeric_strategy': self.numeric_strategy,
                'categorical_strategy': self.categorical_strategy,
                'missing_strategy': self.missing_strategy,
                'generate_interactions': self.generate_interactions
            }
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load a saved pipeline from disk."""
        pipeline_data = joblib.load(filepath)
        
        config = pipeline_data['config']
        engineer = cls(
            target_column=config['target_column'],
            numeric_strategy=config['numeric_strategy'],
            categorical_strategy=config['categorical_strategy'],
            missing_strategy=config['missing_strategy'],
            generate_interactions=config['generate_interactions']
        )
        
        engineer.numeric_pipeline = pipeline_data['numeric_pipeline']
        engineer.encoder = pipeline_data.get('encoder')
        engineer.label_encoders = pipeline_data.get('label_encoders')
        engineer.target_encodings = pipeline_data['target_encodings']
        engineer.numeric_cols = pipeline_data['numeric_cols']
        engineer.categorical_cols = pipeline_data['categorical_cols']
        engineer.datetime_cols = pipeline_data['datetime_cols']
        engineer.engineered_columns = pipeline_data['engineered_columns']
        
        print(f"Pipeline loaded from {filepath}")
        return engineer


