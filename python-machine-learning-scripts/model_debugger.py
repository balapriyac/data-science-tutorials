import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ModelPerformanceDebugger:
    """
    Automated model debugging and performance analysis tool.
    """
    
    def __init__(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_pred_proba: Optional[Union[pd.Series, np.ndarray]] = None,
        task_type: str = 'classification',
        X_train: Optional[pd.DataFrame] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize the model debugger.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : array-like
            True labels/values
        y_pred : array-like
            Model predictions
        y_pred_proba : array-like, optional
            Prediction probabilities (for classification)
        task_type : str
            'classification' or 'regression'
        X_train : pd.DataFrame, optional
            Training features for drift detection
        significance_level : float
            P-value threshold for statistical tests
        """
        self.model = model
        self.X_test = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
        self.task_type = task_type
        self.X_train = X_train if X_train is None or isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.significance_level = significance_level
        
        self.overall_metrics = {}
        self.segment_analysis = []
        self.drift_results = {}
        self.feature_importance = None
        
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall model metrics."""
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
            metrics['precision'] = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            
            if self.y_pred_proba is not None and len(np.unique(self.y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba)
            
            metrics['confusion_matrix'] = confusion_matrix(self.y_test, self.y_pred)
            
        else:  # regression
            metrics['mse'] = mean_squared_error(self.y_test, self.y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(self.y_test, self.y_pred)
            metrics['r2'] = r2_score(self.y_test, self.y_pred)
            
            # Calculate residuals
            residuals = self.y_test - self.y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
        
        return metrics
    
    def _analyze_segments(self) -> List[Dict]:
        """Analyze model performance across different data segments."""
        segments = []
        
        for col in self.X_test.columns:
            try:
                # For numeric features, create quartile-based segments
                if pd.api.types.is_numeric_dtype(self.X_test[col]):
                    quartiles = pd.qcut(self.X_test[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                    
                    for quartile in quartiles.unique():
                        mask = (quartiles == quartile)
                        if mask.sum() < 10:  # Skip segments with too few samples
                            continue
                        
                        segment_metrics = self._calculate_segment_metrics(mask)
                        segment_metrics['feature'] = col
                        segment_metrics['value'] = str(quartile)
                        segment_metrics['n_samples'] = mask.sum()
                        segments.append(segment_metrics)
                
                # For categorical features, analyze each category
                elif pd.api.types.is_object_dtype(self.X_test[col]) or pd.api.types.is_categorical_dtype(self.X_test[col]):
                    unique_vals = self.X_test[col].unique()
                    
                    # Limit to top 10 categories by frequency
                    if len(unique_vals) > 10:
                        top_vals = self.X_test[col].value_counts().head(10).index
                        unique_vals = [v for v in unique_vals if v in top_vals]
                    
                    for val in unique_vals:
                        mask = (self.X_test[col] == val)
                        if mask.sum() < 10:
                            continue
                        
                        segment_metrics = self._calculate_segment_metrics(mask)
                        segment_metrics['feature'] = col
                        segment_metrics['value'] = str(val)
                        segment_metrics['n_samples'] = mask.sum()
                        segments.append(segment_metrics)
            
            except Exception as e:
                # Skip features that cause errors
                continue
        
        return segments
    
    def _calculate_segment_metrics(self, mask: np.ndarray) -> Dict:
        """Calculate metrics for a specific segment."""
        metrics = {}
        
        y_true_seg = self.y_test[mask]
        y_pred_seg = self.y_pred[mask]
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true_seg, y_pred_seg)
            metrics['f1_score'] = f1_score(y_true_seg, y_pred_seg, average='weighted', zero_division=0)
            
            if self.y_pred_proba is not None and len(np.unique(y_true_seg)) == 2:
                try:
                    y_proba_seg = self.y_pred_proba[mask]
                    metrics['roc_auc'] = roc_auc_score(y_true_seg, y_proba_seg)
                except:
                    metrics['roc_auc'] = None
        else:
            metrics['mae'] = mean_absolute_error(y_true_seg, y_pred_seg)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))
            metrics['r2'] = r2_score(y_true_seg, y_pred_seg)
        
        return metrics
    
    def _detect_drift(self) -> Dict:
        """Detect data drift between train and test sets."""
        if self.X_train is None:
            return {'drift_detected': False, 'message': 'Training data not provided'}
        
        drift_results = {'drifted_features': [], 'drift_scores': {}}
        
        for col in self.X_test.columns:
            if col not in self.X_train.columns:
                continue
            
            try:
                if pd.api.types.is_numeric_dtype(self.X_test[col]):
                    # Use Kolmogorov-Smirnov test for numeric features
                    train_vals = self.X_train[col].dropna()
                    test_vals = self.X_test[col].dropna()
                    
                    statistic, p_value = stats.ks_2samp(train_vals, test_vals)
                    
                    drift_results['drift_scores'][col] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'drifted': p_value < self.significance_level
                    }
                    
                    if p_value < self.significance_level:
                        drift_results['drifted_features'].append(col)
                
                else:
                    # Use chi-square test for categorical features
                    train_dist = self.X_train[col].value_counts(normalize=True)
                    test_dist = self.X_test[col].value_counts(normalize=True)
                    
                    # Align distributions
                    all_categories = set(train_dist.index) | set(test_dist.index)
                    train_dist = train_dist.reindex(all_categories, fill_value=0)
                    test_dist = test_dist.reindex(all_categories, fill_value=0)
                    
                    # Population Stability Index (PSI)
                    psi = np.sum((test_dist - train_dist) * np.log((test_dist + 1e-10) / (train_dist + 1e-10)))
                    
                    drift_results['drift_scores'][col] = {
                        'psi': psi,
                        'drifted': psi > 0.2  # PSI > 0.2 indicates significant drift
                    }
                    
                    if psi > 0.2:
                        drift_results['drifted_features'].append(col)
            
            except Exception as e:
                continue
        
        drift_results['drift_detected'] = len(drift_results['drifted_features']) > 0
        
        return drift_results
    
    def _get_feature_importance(self) -> Optional[Dict]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.X_test.columns
                
                importance_dict = dict(zip(feature_names, importances))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            elif hasattr(self.model, 'coef_'):
                coefficients = np.abs(self.model.coef_)
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]
                feature_names = self.X_test.columns
                
                importance_dict = dict(zip(feature_names, coefficients))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except:
            pass
        
        return None
    
    def run_full_diagnosis(self) -> Dict:
        """
        Run comprehensive model diagnosis.
        
        Returns:
        --------
        report : dict
            Comprehensive diagnostic report
        """
        print("Running comprehensive model diagnosis...")
        
        # Calculate overall metrics
        print("  - Calculating overall metrics...")
        self.overall_metrics = self._calculate_overall_metrics()
        
        # Analyze segments
        print("  - Analyzing performance by segments...")
        self.segment_analysis = self._analyze_segments()
        
        # Detect drift
        print("  - Detecting data drift...")
        self.drift_results = self._detect_drift()
        
        # Get feature importance
        print("  - Extracting feature importance...")
        self.feature_importance = self._get_feature_importance()
        
        # Identify problematic segments
        problematic_segments = []
        if self.task_type == 'classification':
            overall_metric = self.overall_metrics['accuracy']
            metric_key = 'accuracy'
        else:
            overall_metric = self.overall_metrics['mae']
            metric_key = 'mae'
        
        for segment in self.segment_analysis:
            if self.task_type == 'classification':
                if segment[metric_key] < overall_metric - 0.1:  # 10% worse
                    problematic_segments.append(segment)
            else:
                if segment[metric_key] > overall_metric * 1.2:  # 20% worse
                    problematic_segments.append(segment)
        
        # Sort by performance difference
        if self.task_type == 'classification':
            problematic_segments.sort(key=lambda x: x[metric_key])
        else:
            problematic_segments.sort(key=lambda x: -x[metric_key])
        
        # Compile report
        report = {
            'overall_metrics': self.overall_metrics,
            'overall_accuracy': self.overall_metrics.get('accuracy'),
            'overall_mae': self.overall_metrics.get('mae'),
            'problematic_segments': problematic_segments,
            'drift_detected': self.drift_results.get('drift_detected', False),
            'drifted_features': self.drift_results.get('drifted_features', []),
            'feature_importance': self.feature_importance,
            'total_segments_analyzed': len(self.segment_analysis)
        }
        
        print("\nDiagnosis complete!")
        print(f"  - {len(problematic_segments)} problematic segments identified")
        print(f"  - {len(report['drifted_features'])} features with drift detected")
        
        return report
    
    def plot_feature_performance_breakdown(self, top_n: int = 10, save_path: Optional[str] = None):
        """Plot performance breakdown by top features."""
        if not self.segment_analysis:
            print("Run run_full_diagnosis() first.")
            return
        
        # Get top features by variance in performance
        feature_variance = {}
        for feature in self.X_test.columns:
            feature_segments = [s for s in self.segment_analysis if s['feature'] == feature]
            if not feature_segments:
                continue
            
            if self.task_type == 'classification':
                scores = [s['accuracy'] for s in feature_segments]
            else:
                scores = [s['mae'] for s in feature_segments]
            
            feature_variance[feature] = np.var(scores)
        
        top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Plot
        fig, axes = plt.subplots(min(5, len(top_features)), 1, figsize=(12, 4*min(5, len(top_features))))
        if len(top_features) == 1:
            axes = [axes]
        
        for idx, (feature, _) in enumerate(top_features[:5]):
            feature_segments = [s for s in self.segment_analysis if s['feature'] == feature]
            
            values = [s['value'] for s in feature_segments]
            if self.task_type == 'classification':
                scores = [s['accuracy'] for s in feature_segments]
                metric_name = 'Accuracy'
            else:
                scores = [s['mae'] for s in feature_segments]
                metric_name = 'MAE'
            
            axes[idx].bar(values, scores)
            axes[idx].axhline(y=self.overall_metrics.get('accuracy' if self.task_type == 'classification' else 'mae'),
                            color='r', linestyle='--', label='Overall')
            axes[idx].set_title(f'{feature} - {metric_name} by Segment')
            axes[idx].set_xlabel('Segment')
            axes[idx].set_ylabel(metric_name)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def export_html_report(self, filepath: str):
        """Export comprehensive HTML report."""
        html_content = f"""
        <html>
        <head>
            <title>Model Performance Debug Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Model Performance Debug Report</h1>
            
            <h2>Overall Metrics</h2>
            <div>
        """
        
        for key, value in self.overall_metrics.items():
            if key != 'confusion_matrix':
                html_content += f'<div class="metric"><strong>{key}:</strong> {value:.4f if isinstance(value, float) else value}</div>'
        
        html_content += "</div>"
        
        if self.drift_results.get('drift_detected'):
            html_content += f"""
            <div class="warning">
                <strong>⚠️ Data Drift Detected!</strong><br>
                Drifted features: {', '.join(self.drift_results['drifted_features'])}
            </div>
            """
        
        html_content += """
            <h2>Problematic Segments</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Samples</th>
        """
        
        if self.task_type == 'classification':
            html_content += "<th>Accuracy</th><th>F1 Score</th>"
        else:
            html_content += "<th>MAE</th><th>RMSE</th>"
        
        html_content += "</tr>"
        
        # Get problematic segments
        if self.task_type == 'classification':
            overall_metric = self.overall_metrics['accuracy']
            problematic = [s for s in self.segment_analysis if s['accuracy'] < overall_metric - 0.1]
        else:
            overall_metric = self.overall_metrics['mae']
            problematic = [s for s in self.segment_analysis if s['mae'] > overall_metric * 1.2]
        
        for segment in sorted(problematic, key=lambda x: x.get('accuracy', x.get('mae')))[:20]:
            html_content += f"""
            <tr>
                <td>{segment['feature']}</td>
                <td>{segment['value']}</td>
                <td>{segment['n_samples']}</td>
            """
            
            if self.task_type == 'classification':
                html_content += f"<td>{segment['accuracy']:.4f}</td><td>{segment['f1_score']:.4f}</td>"
            else:
                html_content += f"<td>{segment['mae']:.4f}</td><td>{segment['rmse']:.4f}</td>"
            
            html_content += "</tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report exported to {filepath}")


