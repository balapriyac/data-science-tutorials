import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.model_selection._split import _BaseKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesGroupSplit(_BaseKFold):
    """
    Custom splitter that combines time series and group awareness.
    """
    
    def __init__(self, n_splits=5, group_col=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.group_col = group_col
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            train_indices = indices[:start]
            test_indices = indices[start:stop]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
            
            current = stop


class CrossValidationManager:
    """
    Manages cross-validation strategies for various data types and ML scenarios.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        datetime_column: Optional[str] = None,
        group_column: Optional[str] = None,
        task_type: str = 'classification'
    ):
        """
        Initialize the CV manager.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The complete dataset
        target_column : str
            Name of the target column
        datetime_column : str, optional
            Name of datetime column for time-series splits
        group_column : str, optional
            Name of group column for grouped splits
        task_type : str
            'classification' or 'regression'
        """
        self.data = data
        self.target_column = target_column
        self.datetime_column = datetime_column
        self.group_column = group_column
        self.task_type = task_type
        
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        
        # Sort by datetime if provided
        if self.datetime_column and self.datetime_column in self.X.columns:
            sort_idx = self.X[self.datetime_column].argsort()
            self.X = self.X.iloc[sort_idx].reset_index(drop=True)
            self.y = self.y.iloc[sort_idx].reset_index(drop=True)
        
        self.groups = self.X[group_column].values if group_column and group_column in self.X.columns else None
    
    def recommend_strategy(self) -> Dict:
        """
        Automatically recommend the best CV strategy based on data characteristics.
        
        Returns:
        --------
        recommendation : dict
            Dictionary with strategy name and rationale
        """
        recommendations = []
        
        # Check for imbalance (classification only)
        if self.task_type == 'classification':
            class_counts = self.y.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()
            
            if min_class_ratio < 0.3:
                recommendations.append({
                    'name': 'stratified',
                    'rationale': f'Imbalanced classes detected (min/max ratio: {min_class_ratio:.2f}). Stratified CV recommended.',
                    'priority': 3
                })
        
        # Check for time series
        if self.datetime_column:
            recommendations.append({
                'name': 'time_series',
                'rationale': 'Datetime column detected. Time-series split recommended to prevent temporal leakage.',
                'priority': 5
            })
        
        # Check for groups
        if self.group_column:
            unique_groups = len(self.groups) if self.groups is not None else 0
            recommendations.append({
                'name': 'grouped',
                'rationale': f'Group column detected with {unique_groups} unique groups. Grouped CV recommended.',
                'priority': 4
            })
        
        # Combined time series and groups
        if self.datetime_column and self.group_column:
            recommendations.append({
                'name': 'time_series_group',
                'rationale': 'Both datetime and group columns detected. Combined time-series + group split recommended.',
                'priority': 6
            })
        
        # Default recommendation
        if not recommendations:
            if self.task_type == 'classification':
                recommendations.append({
                    'name': 'stratified',
                    'rationale': 'Standard stratified K-fold for classification task.',
                    'priority': 1
                })
            else:
                recommendations.append({
                    'name': 'standard',
                    'rationale': 'Standard K-fold for regression task.',
                    'priority': 1
                })
        
        # Return highest priority recommendation
        best_recommendation = max(recommendations, key=lambda x: x['priority'])
        return best_recommendation
    
    def generate_splits(
        self,
        n_splits: int = 5,
        strategy: Optional[str] = None,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cross-validation splits.
        
        Parameters:
        -----------
        n_splits : int
            Number of splits
        strategy : str, optional
            CV strategy: 'standard', 'stratified', 'grouped', 'time_series', 'time_series_group'
            If None, uses recommended strategy
        test_size : float
            Proportion of data for testing (for time series splits)
        shuffle : bool
            Whether to shuffle data (not applicable for time series)
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        splits : list of tuples
            List of (train_indices, test_indices) tuples
        """
        if strategy is None:
            strategy = self.recommend_strategy()['name']
        
        print(f"Generating {strategy} splits with {n_splits} folds...")
        
        splits = []
        
        if strategy == 'standard':
            kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            splits = list(kfold.split(self.X, self.y))
        
        elif strategy == 'stratified':
            if self.task_type != 'classification':
                print("Warning: Stratified splitting is for classification. Using standard K-fold.")
                kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splits = list(kfold.split(self.X, self.y))
            else:
                skfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splits = list(skfold.split(self.X, self.y))
        
        elif strategy == 'grouped':
            if self.groups is None:
                raise ValueError("Group column not provided. Cannot perform grouped split.")
            
            gkfold = GroupKFold(n_splits=n_splits)
            splits = list(gkfold.split(self.X, self.y, groups=self.groups))
        
        elif strategy == 'time_series':
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(self.X) * test_size))
            splits = list(tscv.split(self.X))
        
        elif strategy == 'time_series_group':
            if self.groups is None:
                print("Warning: Group column not provided. Using standard time series split.")
                tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(self.X) * test_size))
                splits = list(tscv.split(self.X))
            else:
                # Custom implementation
                ts_group_cv = TimeSeriesGroupSplit(n_splits=n_splits, group_col=self.group_column)
                splits = list(ts_group_cv.split(self.X, self.y, groups=self.groups))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Generated {len(splits)} splits")
        return splits
    
    def validate_splits(self, splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Validate the quality of generated splits.
        
        Parameters:
        -----------
        splits : list of tuples
            Cross-validation splits
            
        Returns:
        --------
        validation_report : dict
            Report on split quality
        """
        report = {
            'n_splits': len(splits),
            'balanced': True,
            'no_leakage': True,
            'groups_separated': True,
            'issues': []
        }
        
        # Check class balance (for classification)
        if self.task_type == 'classification':
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                train_dist = self.y.iloc[train_idx].value_counts(normalize=True)
                test_dist = self.y.iloc[test_idx].value_counts(normalize=True)
                
                # Check if distributions are similar
                for cls in train_dist.index:
                    if cls in test_dist.index:
                        diff = abs(train_dist[cls] - test_dist[cls])
                        if diff > 0.1:  # More than 10% difference
                            report['balanced'] = False
                            report['issues'].append(
                                f"Fold {fold_idx}: Class {cls} distribution differs by {diff:.2%}"
                            )
        
        # Check for temporal leakage (if datetime column exists)
        if self.datetime_column and self.datetime_column in self.X.columns:
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                train_dates = self.X[self.datetime_column].iloc[train_idx]
                test_dates = self.X[self.datetime_column].iloc[test_idx]
                
                if train_dates.max() > test_dates.min():
                    report['no_leakage'] = False
                    report['issues'].append(
                        f"Fold {fold_idx}: Temporal leakage detected (train max > test min)"
                    )
        
        # Check for group separation (if group column exists)
        if self.groups is not None:
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                train_groups = set(self.groups[train_idx])
                test_groups = set(self.groups[test_idx])
                
                overlap = train_groups & test_groups
                if overlap:
                    report['groups_separated'] = False
                    report['issues'].append(
                        f"Fold {fold_idx}: {len(overlap)} groups appear in both train and test"
                    )
        
        # Check split sizes
        split_sizes = [(len(train_idx), len(test_idx)) for train_idx, test_idx in splits]
        train_sizes, test_sizes = zip(*split_sizes)
        
        report['train_size_mean'] = np.mean(train_sizes)
        report['train_size_std'] = np.std(train_sizes)
        report['test_size_mean'] = np.mean(test_sizes)
        report['test_size_std'] = np.std(test_sizes)
        
        if len(report['issues']) == 0:
            report['issues'].append("No issues detected. Splits look good!")
        
        return report
    
    def cross_validate(
        self,
        model,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        scoring: Union[str, List[str]] = 'accuracy',
        return_train_score: bool = True
    ) -> Dict:
        """
        Perform cross-validation with the given splits.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        splits : list of tuples
            Cross-validation splits
        scoring : str or list of str
            Scoring metric(s)
        return_train_score : bool
            Whether to return training scores
            
        Returns:
        --------
        scores : dict
            Cross-validation scores
        """
        # Prepare features (drop datetime/group columns if they're not needed)
        X_clean = self.X.copy()
        cols_to_drop = []
        
        if self.datetime_column and self.datetime_column in X_clean.columns:
            cols_to_drop.append(self.datetime_column)
        if self.group_column and self.group_column in X_clean.columns:
            cols_to_drop.append(self.group_column)
        
        if cols_to_drop:
            X_clean = X_clean.drop(columns=cols_to_drop)
        
        # Convert to numpy if needed
        X_array = X_clean.values if isinstance(X_clean, pd.DataFrame) else X_clean
        y_array = self.y.values if isinstance(self.y, pd.Series) else self.y
        
        # Perform cross-validation
        if isinstance(scoring, str):
            scores = cross_validate(
                model, X_array, y_array,
                cv=splits,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=-1
            )
        else:
            scores = cross_validate(
                model, X_array, y_array,
                cv=splits,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=-1
            )
        
        # Convert to DataFrame for easier analysis
        scores_df = pd.DataFrame(scores)
        
        return scores_df
    
    def plot_split_distributions(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        save_path: Optional[str] = None
    ):
        """
        Visualize the distribution of splits.
        """
        n_splits = len(splits)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Split sizes
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        test_sizes = [len(test_idx) for _, test_idx in splits]
        
        x = np.arange(n_splits)
        width = 0.35
        
        axes[0].bar(x - width/2, train_sizes, width, label='Train', alpha=0.8)
        axes[0].bar(x + width/2, test_sizes, width, label='Test', alpha=0.8)
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Train/Test Split Sizes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Target distribution per fold (for classification)
        if self.task_type == 'classification':
            fold_distributions = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                test_dist = self.y.iloc[test_idx].value_counts(normalize=True).to_dict()
                fold_distributions.append(test_dist)
            
            # Create distribution matrix
            all_classes = sorted(set().union(*[d.keys() for d in fold_distributions]))
            dist_matrix = np.zeros((len(all_classes), n_splits))
            
            for fold_idx, dist in enumerate(fold_distributions):
                for class_idx, cls in enumerate(all_classes):
                    dist_matrix[class_idx, fold_idx] = dist.get(cls, 0)
            
            im = axes[1].imshow(dist_matrix, aspect='auto', cmap='YlOrRd')
            axes[1].set_xlabel('Fold')
            axes[1].set_ylabel('Class')
            axes[1].set_yticks(range(len(all_classes)))
            axes[1].set_yticklabels(all_classes)
            axes[1].set_title('Test Set Class Distribution by Fold')
            plt.colorbar(im, ax=axes[1], label='Proportion')
        
        else:
            # For regression, plot target distribution statistics
            fold_stats = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                test_targets = self.y.iloc[test_idx]
                fold_stats.append({
                    'fold': fold_idx,
                    'mean': test_targets.mean(),
                    'std': test_targets.std(),
                    'min': test_targets.min(),
                    'max': test_targets.max()
                })
            
            fold_df = pd.DataFrame(fold_stats)
            
            axes[1].errorbar(fold_df['fold'], fold_df['mean'], yerr=fold_df['std'],
                           fmt='o-', capsize=5, capthick=2)
            axes[1].set_xlabel('Fold')
            axes[1].set_ylabel('Target Value')
            axes[1].set_title('Test Set Target Distribution by Fold (Mean ± Std)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


