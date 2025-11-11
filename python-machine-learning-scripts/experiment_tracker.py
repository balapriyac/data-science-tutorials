import os
import json
import time
import joblib
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys


class ExperimentTracker:
    """
    Lightweight experiment tracking for ML projects.
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        tracking_dir: str = './ml_experiments',
        auto_log_system_info: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Parameters:
        -----------
        project_name : str
            Name of the project
        experiment_name : str
            Name of this specific experiment
        tracking_dir : str
            Base directory for storing experiments
        auto_log_system_info : bool
            Whether to automatically log system information
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.experiment_dir = self.tracking_dir / project_name / experiment_name
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'models').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)
        
        # Initialize experiment data
        self.experiment_data = {
            'project_name': project_name,
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'params': {},
            'metrics': {},
            'dataset_info': {},
            'feature_importance': {},
            'tags': [],
            'notes': '',
            'artifacts': {},
            'system_info': {}
        }
        
        # Timer
        self.start_time = None
        self.elapsed_time = 0
        
        # Auto-log system info
        if auto_log_system_info:
            self._log_system_info()
        
        print(f"Initialized experiment: {project_name}/{experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
    
    def _log_system_info(self):
        """Log system and environment information."""
        try:
            import platform
            import sklearn
            
            self.experiment_data['system_info'] = {
                'python_version': sys.version,
                'platform': platform.platform(),
                'sklearn_version': sklearn.__version__,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            }
            
            # Try to get git info
            try:
                import subprocess
                git_hash = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL
                ).decode('ascii').strip()
                self.experiment_data['system_info']['git_commit'] = git_hash
            except:
                pass
        
        except Exception as e:
            print(f"Warning: Could not log all system info: {e}")
    
    def log_params(self, params: Dict):
        """
        Log model parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary of model parameters
        """
        self.experiment_data['params'].update(params)
        print(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict):
        """
        Log evaluation metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics (name -> value)
        """
        self.experiment_data['metrics'].update(metrics)
        print(f"Logged {len(metrics)} metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    
    def log_dataset(
        self,
        name: str,
        n_samples: int,
        n_features: int,
        feature_names: Optional[List[str]] = None,
        data_hash: Optional[str] = None
    ):
        """
        Log dataset information.
        
        Parameters:
        -----------
        name : str
            Dataset name
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        feature_names : list, optional
            List of feature names
        data_hash : str, optional
            Hash of the data for versioning
        """
        self.experiment_data['dataset_info'] = {
            'name': name,
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_names': feature_names,
            'data_hash': data_hash
        }
        print(f"Logged dataset: {name} ({n_samples} samples, {n_features} features)")
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20
    ):
        """
        Log feature importance.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        importance_values : array
            Array of importance values
        top_n : int
            Number of top features to store
        """
        # Sort by importance
        importance_dict = dict(zip(feature_names, importance_values))
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        self.experiment_data['feature_importance'] = {
            name: float(value) for name, value in sorted_importance
        }
        
        print(f"Logged top {min(top_n, len(sorted_importance))} features")
    
    def log_notes(self, notes: str):
        """
        Add notes to the experiment.
        
        Parameters:
        -----------
        notes : str
            Text notes about the experiment
        """
        self.experiment_data['notes'] = notes
        print("Notes logged")
    
    def add_tags(self, tags: Union[str, List[str]]):
        """
        Add tags to the experiment.
        
        Parameters:
        -----------
        tags : str or list of str
            Tags to add
        """
        if isinstance(tags, str):
            tags = [tags]
        
        self.experiment_data['tags'].extend(tags)
        self.experiment_data['tags'] = list(set(self.experiment_data['tags']))  # Remove duplicates
        print(f"Added tags: {tags}")
    
    def start_timer(self):
        """Start timing the experiment."""
        self.start_time = time.time()
        print("Timer started")
    
    def stop_timer(self) -> float:
        """
        Stop timing and return elapsed time.
        
        Returns:
        --------
        elapsed : float
            Elapsed time in seconds
        """
        if self.start_time is None:
            print("Warning: Timer was not started")
            return 0
        
        self.elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.2f} seconds")
        return self.elapsed_time
    
    def save_model(self, model, filename: str = 'model.pkl'):
        """
        Save model artifact.
        
        Parameters:
        -----------
        model : object
            Model to save
        filename : str
            Filename for the model
        """
        model_path = self.experiment_dir / 'models' / filename
        joblib.dump(model, model_path)
        
        self.experiment_data['artifacts'][filename] = str(model_path)
        print(f"Model saved: {model_path}")
    
    def save_plot(self, fig, filename: str):
        """
        Save a matplotlib figure.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Filename for the plot
        """
        plot_path = self.experiment_dir / 'plots' / filename
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        self.experiment_data['artifacts'][filename] = str(plot_path)
        print(f"Plot saved: {plot_path}")
    
    def finish(self):
        """
        Finish the experiment and save all metadata.
        """
        self.experiment_data['finished_at'] = datetime.now().isoformat()
        self.experiment_data['duration_seconds'] = self.elapsed_time
        
        # Save experiment metadata
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)
        
        print(f"\nExperiment finished and saved to: {self.experiment_dir}")
        print(f"Metadata: {metadata_path}")
    
    @staticmethod
    def load_experiment(project_name: str, experiment_name: str, tracking_dir: str = './ml_experiments') -> Dict:
        """
        Load a saved experiment.
        
        Parameters:
        -----------
        project_name : str
            Project name
        experiment_name : str
            Experiment name
        tracking_dir : str
            Tracking directory
            
        Returns:
        --------
        experiment_data : dict
            Experiment metadata
        """
        metadata_path = Path(tracking_dir) / project_name / experiment_name / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Experiment not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            experiment_data = json.load(f)
        
        return experiment_data
    
    @staticmethod
    def list_experiments(project_name: str, tracking_dir: str = './ml_experiments') -> List[str]:
        """
        List all experiments in a project.
        
        Parameters:
        -----------
        project_name : str
            Project name
        tracking_dir : str
            Tracking directory
            
        Returns:
        --------
        experiments : list
            List of experiment names
        """
        project_dir = Path(tracking_dir) / project_name
        
        if not project_dir.exists():
            return []
        
        experiments = [
            d.name for d in project_dir.iterdir()
            if d.is_dir() and (d / 'metadata.json').exists()
        ]
        
        return experiments
    
    @staticmethod
    def compare_experiments(
        project_name: str,
        metric: Optional[str] = None,
        top_n: Optional[int] = None,
        tracking_dir: str = './ml_experiments'
    ) -> pd.DataFrame:
        """
        Compare experiments in a project.
        
        Parameters:
        -----------
        project_name : str
            Project name
        metric : str, optional
            Metric to sort by
        top_n : int, optional
            Number of top experiments to return
        tracking_dir : str
            Tracking directory
            
        Returns:
        --------
        comparison : pd.DataFrame
            DataFrame comparing experiments
        """
        experiments = ExperimentTracker.list_experiments(project_name, tracking_dir)
        
        if not experiments:
            print(f"No experiments found for project: {project_name}")
            return pd.DataFrame()
        
        experiment_records = []
        
        for exp_name in experiments:
            try:
                exp_data = ExperimentTracker.load_experiment(project_name, exp_name, tracking_dir)
                
                record = {
                    'experiment_name': exp_name,
                    'created_at': exp_data.get('created_at', ''),
                    'duration_seconds': exp_data.get('duration_seconds', 0),
                    'tags': ', '.join(exp_data.get('tags', []))
                }
                
                # Add all metrics
                record.update(exp_data.get('metrics', {}))
                
                # Add key parameters
                params = exp_data.get('params', {})
                for key, value in params.items():
                    record[f'param_{key}'] = value
                
                experiment_records.append(record)
            
            except Exception as e:
                print(f"Warning: Could not load experiment {exp_name}: {e}")
                continue
        
        df = pd.DataFrame(experiment_records)
        
        # Sort by metric if specified
        if metric and metric in df.columns:
            # Determine if higher is better
            if 'accuracy' in metric or 'f1' in metric or 'r2' in metric or 'auc' in metric:
                df = df.sort_values(metric, ascending=False)
            else:
                df = df.sort_values(metric, ascending=True)
        
        # Limit to top_n
        if top_n:
            df = df.head(top_n)
        
        return df
    
    @staticmethod
    def plot_experiment_comparison(
        experiments: List[str],
        metrics: List[str],
        project_name: str,
        tracking_dir: str = './ml_experiments',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of experiments across metrics.
        
        Parameters:
        -----------
        experiments : list
            List of experiment names to compare
        metrics : list
            List of metrics to compare
        project_name : str
            Project name
        tracking_dir : str
            Tracking directory
        save_path : str, optional
            Path to save the plot
        """
        # Load experiment data
        exp_data_list = []
        for exp_name in experiments:
            try:
                exp_data = ExperimentTracker.load_experiment(project_name, exp_name, tracking_dir)
                exp_data_list.append(exp_data)
            except Exception as e:
                print(f"Warning: Could not load experiment {exp_name}: {e}")
        
        if not exp_data_list:
            print("No experiments to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            metric_values = []
            exp_names = []
            
            for exp_data in exp_data_list:
                if metric in exp_data.get('metrics', {}):
                    metric_values.append(exp_data['metrics'][metric])
                    exp_names.append(exp_data['experiment_name'])
            
            if metric_values:
                axes[idx].bar(range(len(metric_values)), metric_values)
                axes[idx].set_xticks(range(len(metric_values)))
                axes[idx].set_xticklabels(exp_names, rotation=45, ha='right')
                axes[idx].set_ylabel(metric)
                axes[idx].set_title(f'{metric} Comparison')
                axes[idx].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(metric_values):
                    axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def delete_experiment(
        project_name: str,
        experiment_name: str,
        tracking_dir: str = './ml_experiments'
    ):
        """
        Delete an experiment.
        
        Parameters:
        -----------
        project_name : str
            Project name
        experiment_name : str
            Experiment name
        tracking_dir : str
            Tracking directory
        """
        import shutil
        
        experiment_dir = Path(tracking_dir) / project_name / experiment_name
        
        if not experiment_dir.exists():
            print(f"Experiment not found: {experiment_dir}")
            return
        
        shutil.rmtree(experiment_dir)
        print(f"Deleted experiment: {experiment_dir}")
    
    def log_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Log and plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Label names
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        self.save_plot(fig, 'confusion_matrix.png')
        plt.close(fig)
        
        print("Confusion matrix logged")


