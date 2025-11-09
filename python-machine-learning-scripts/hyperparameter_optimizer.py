
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterOptimizer:
    """
    Unified hyperparameter optimization manager supporting multiple strategies.
    """
    
    def __init__(
        self,
        model,
        param_space: Dict,
        optimization_strategy: str = 'random',
        cv: int = 5,
        scoring: str = 'accuracy',
        n_iter: int = 50,
        early_stopping_rounds: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Parameters:
        -----------
        model : sklearn estimator
            The model to optimize
        param_space : dict
            Dictionary defining the parameter search space
        optimization_strategy : str
            Strategy to use: 'grid', 'random', 'bayesian', 'halving'
        cv : int
            Number of cross-validation folds
        scoring : str or callable
            Scoring metric to optimize
        n_iter : int
            Number of iterations for random/bayesian search
        early_stopping_rounds : int, optional
            Stop if no improvement for n rounds (bayesian only)
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
        """
        self.model = model
        self.param_space = param_space
        self.optimization_strategy = optimization_strategy
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.cv_results_ = None
        self.optimization_history_ = []
        self.param_importance_ = None
        
    def _grid_search(self, X, y):
        """Perform grid search optimization."""
        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_model_ = search.best_estimator_
        self.cv_results_ = pd.DataFrame(search.cv_results_)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(self.cv_results_),
            'strategy': 'grid_search'
        }
    
    def _random_search(self, X, y):
        """Perform random search optimization."""
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_train_score=True
        )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_model_ = search.best_estimator_
        self.cv_results_ = pd.DataFrame(search.cv_results_)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(self.cv_results_),
            'strategy': 'random_search'
        }
    
    def _halving_search(self, X, y):
        """Perform successive halving search."""
        search = HalvingRandomSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            factor=3
        )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_model_ = search.best_estimator_
        self.cv_results_ = pd.DataFrame(search.cv_results_)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(self.cv_results_),
            'strategy': 'halving_search'
        }
    
    def _bayesian_search(self, X, y):
        """Perform Bayesian optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
        
        def objective(trial):
            # Build params from trial suggestions
            params = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            min(param_values),
                            max(param_values)
                        )
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(param_values),
                            max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_values
                        )
            
            # Create model with suggested params
            model = self.model.__class__(**params)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs
            )
            
            score = scores.mean()
            
            # Store history
            self.optimization_history_.append({
                'trial': trial.number,
                'params': params,
                'score': score,
                'std': scores.std()
            })
            
            return score
        
        # Create study
        direction = 'maximize' if 'accuracy' in self.scoring or 'f1' in self.scoring or 'roc' in self.scoring else 'minimize'
        
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        if self.early_stopping_rounds:
            study.optimize(
                objective,
                n_trials=self.n_iter,
                callbacks=[
                    lambda study, trial: study.stop() 
                    if trial.number - study.best_trial.number >= self.early_stopping_rounds 
                    else None
                ]
            )
        else:
            study.optimize(objective, n_trials=self.n_iter)
        
        # Store best results
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # Train final model with best params
        self.best_model_ = self.model.__class__(**self.best_params_)
        self.best_model_.fit(X, y)
        
        # Calculate parameter importance
        try:
            self.param_importance_ = optuna.importance.get_param_importances(study)
        except:
            self.param_importance_ = None
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(study.trials),
            'strategy': 'bayesian_optimization'
        }
    
    def optimize(self, X, y) -> Dict:
        """
        Run hyperparameter optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training target
            
        Returns:
        --------
        results : dict
            Dictionary containing optimization results
        """
        print(f"Starting {self.optimization_strategy} optimization...")
        print(f"Parameter space: {self.param_space}")
        print(f"CV folds: {self.cv}, Scoring: {self.scoring}")
        
        start_time = datetime.now()
        
        if self.optimization_strategy == 'grid':
            results = self._grid_search(X, y)
        elif self.optimization_strategy == 'random':
            results = self._random_search(X, y)
        elif self.optimization_strategy == 'halving':
            results = self._halving_search(X, y)
        elif self.optimization_strategy == 'bayesian':
            results = self._bayesian_search(X, y)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.optimization_strategy}")
        
        end_time = datetime.now()
        results['optimization_time'] = (end_time - start_time).total_seconds()
        
        print(f"\nOptimization complete!")
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Best parameters: {results['best_params']}")
        print(f"Total trials: {results['n_trials']}")
        print(f"Time taken: {results['optimization_time']:.2f} seconds")
        
        return results
    
    def get_param_importance(self) -> List[tuple]:
        """
        Get parameter importance rankings.
        
        Returns:
        --------
        importance : list of tuples
            List of (param_name, importance_score) sorted by importance
        """
        if self.optimization_strategy == 'bayesian' and self.param_importance_:
            return sorted(
                self.param_importance_.items(),
                key=lambda x: x[1],
                reverse=True
            )
        elif self.cv_results_ is not None:
            # Calculate importance based on variance in scores
            param_cols = [col for col in self.cv_results_.columns if col.startswith('param_')]
            importance = {}
            
            for col in param_cols:
                param_name = col.replace('param_', '')
                # Group by parameter value and calculate score variance
                grouped = self.cv_results_.groupby(col)['mean_test_score']
                variance = grouped.mean().std()
                importance[param_name] = variance
            
            return sorted(importance.items(), key=lambda x: x[1], reverse=True)
        else:
            return []
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history over trials."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        if self.optimization_strategy == 'bayesian' and self.optimization_history_:
            # Plot score progression
            trials = [h['trial'] for h in self.optimization_history_]
            scores = [h['score'] for h in self.optimization_history_]
            best_scores = np.maximum.accumulate(scores)
            
            axes[0].plot(trials, scores, 'o-', alpha=0.6, label='Trial scores')
            axes[0].plot(trials, best_scores, 'r-', linewidth=2, label='Best score')
            axes[0].set_xlabel('Trial')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Optimization History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
        elif self.cv_results_ is not None:
            # Plot for sklearn-based searches
            scores = self.cv_results_['mean_test_score'].values
            trials = np.arange(len(scores))
            best_scores = np.maximum.accumulate(scores)
            
            axes[0].plot(trials, scores, 'o-', alpha=0.6, label='Trial scores')
            axes[0].plot(trials, best_scores, 'r-', linewidth=2, label='Best score')
            axes[0].set_xlabel('Trial')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Optimization History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot parameter importance
        importance = self.get_param_importance()
        if importance:
            params, scores = zip(*importance[:10])  # Top 10
            axes[1].barh(params, scores)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Parameter Importance (Top 10)')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Parameter importance not available',
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_param_distributions(self, save_path: Optional[str] = None):
        """Plot distributions of tried parameter values."""
        if self.cv_results_ is None:
            print("No results available to plot.")
            return
        
        param_cols = [col for col in self.cv_results_.columns if col.startswith('param_')]
        n_params = len(param_cols)
        
        if n_params == 0:
            print("No parameters to plot.")
            return
        
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for idx, col in enumerate(param_cols):
            param_name = col.replace('param_', '')
            param_values = self.cv_results_[col]
            scores = self.cv_results_['mean_test_score']
            
            # Try to convert to numeric
            try:
                param_values = pd.to_numeric(param_values)
                axes[idx].scatter(param_values, scores, alpha=0.6)
                axes[idx].set_xlabel(param_name)
            except:
                # Categorical parameter
                unique_vals = param_values.unique()
                positions = range(len(unique_vals))
                score_by_val = [scores[param_values == val].values for val in unique_vals]
                axes[idx].boxplot(score_by_val, labels=unique_vals)
                axes[idx].set_xlabel(param_name)
                plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{param_name} vs Score')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_best_model(self, filepath: str):
        """Save the best model to disk."""
        if self.best_model_ is None:
            raise ValueError("No model has been trained yet. Run optimize() first.")
        
        model_data = {
            'model': self.best_model_,
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'optimization_strategy': self.optimization_strategy,
            'cv': self.cv,
            'scoring': self.scoring
        }
        
        joblib.dump(model_data, filepath)
        print(f"Best model saved to {filepath}")
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get a summary of all trials."""
        if self.cv_results_ is not None:
            summary = self.cv_results_[[
                col for col in self.cv_results_.columns 
                if col.startswith('param_') or 'score' in col or 'time' in col
            ]].copy()
            return summary.sort_values('mean_test_score', ascending=False)
        elif self.optimization_history_:
            return pd.DataFrame(self.optimization_history_).sort_values('score', ascending=False)
        else:
            return pd.DataFrame()


