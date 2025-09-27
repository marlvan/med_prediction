"""
CatBoost hyperparameter tuning using Optuna.

This script performs hyperparameter optimization for CatBoost models using
cross-validation and Optuna's TPE sampler.
"""

import numpy as np
import pandas as pd
import argparse
import pickle
import gc
from typing import List

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier


def objective_function(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    continuous_features: List[str],
    cv_splitter: StratifiedKFold
) -> float:
    """
    Optuna objective function for CatBoost hyperparameter tuning.

    Args:
        trial: Optuna trial object for hyperparameter suggestions
        X: Feature matrix
        y: Binary target vector
        continuous_features: List of continuous feature column names
        cv_splitter: StratifiedKFold cross-validator

    Returns:
        Mean ROC AUC across cross-validation folds
    """
    # Suggest hyperparameters
    n_iterations = trial.suggest_int('iterations', 50, 400)
    
    # Adjust learning rate based on number of iterations
    if n_iterations <= 200:
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.3, log=True)
    else:
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
    
    # Model parameters
    params = {
        'iterations': n_iterations,
        'learning_rate': learning_rate,
        'depth': trial.suggest_int('depth', 3, 6),
        'task_type': 'CPU',
        'thread_count': 1,
        'allow_writing_files': False,
        'eval_metric': 'AUC',
        'verbose': False,
        'random_seed': 42
    }

    # Cross-validation evaluation
    cv_scores = []
    for train_indices, valid_indices in cv_splitter.split(X, y):
        # Split data
        X_train, X_valid = X.iloc[train_indices].copy(), X.iloc[valid_indices].copy()
        y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
        
        # Scale continuous features
        scaler = StandardScaler()
        X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_valid[continuous_features] = scaler.transform(X_valid[continuous_features])
        
        # Train model
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=200
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        cv_scores.append(roc_auc_score(y_valid, y_pred_proba))
        
        # Clean up memory
        del model
        gc.collect()

    return np.mean(cv_scores)


def tune_catboost_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    continuous_features: List[str],
    n_trials: int = 100,
    n_cv_folds: int = 3,
    n_jobs: int = 1,
    storage_url: str = None,
    study_name: str = None
) -> tuple[CatBoostClassifier, dict]:
    """
    Tune CatBoost hyperparameters using Optuna with cross-validation.

    Args:
        X: Feature matrix
        y: Binary target vector  
        continuous_features: List of continuous feature column names
        n_trials: Number of Optuna trials to run
        n_cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel Optuna workers
        storage_url: Optuna storage URL for persistence
        study_name: Name for the Optuna study

    Returns:
        Tuple of (best_model, best_parameters)
    """
    # Configure Optuna study
    sampler = TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=20,
        warn_independent_sampling=False
    )
    pruner = MedianPruner(n_startup_trials=10)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True
    )

    # Enqueue a sensible default trial
    study.enqueue_trial({
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 4
    })

    # Count existing completed trials
    completed_trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE,)
    )
    remaining_trials = max(0, n_trials - len(completed_trials))

    # Set up cross-validation
    cv_splitter = StratifiedKFold(
        n_splits=n_cv_folds,
        shuffle=True,
        random_state=42
    )

    # Run optimization
    if remaining_trials > 0:
        print(f'Running {remaining_trials} optimization trials...')
        study.optimize(
            lambda trial: objective_function(
                trial, X, y, continuous_features, cv_splitter
            ),
            n_trials=remaining_trials,
            n_jobs=n_jobs
        )

    # Extract best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'verbose': False,
        'random_seed': 42,
        'task_type': 'CPU',
        'thread_count': 1,
        'allow_writing_files': False
    })

    # Build final model with best parameters
    best_model = CatBoostClassifier(**best_params)

    return best_model, best_params


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for CatBoost using Optuna'
    )

    # Required arguments
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to pickle file containing (X, y, continuous_features) tuple'
    )
    parser.add_argument(
        '--prefix', type=str, required=True,
        help='File prefix for saving results and Optuna database'
    )

    # Optional tuning configuration
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='Number of parallel Optuna jobs'
    )
    parser.add_argument(
        '--n_trials', type=int, default=100,
        help='Number of Optuna trials to run'
    )
    parser.add_argument(
        '--n_folds', type=int, default=3,
        help='Number of cross-validation folds'
    )

    # Optional Optuna study configuration
    parser.add_argument(
        '--storage', type=str, default=None,
        help='Optuna storage URL (e.g., sqlite:///path/to/study.db)'
    )
    parser.add_argument(
        '--study_name', type=str, default=None,
        help='Name for the Optuna study (used with persistent storage)'
    )

    args = parser.parse_args()

    # Load training data
    with open(args.data_path, "rb") as f:
        X, y, continuous_features = pickle.load(f)

    # Run hyperparameter tuning
    best_model, best_params = tune_catboost_hyperparameters(
        X=X,
        y=y,
        continuous_features=continuous_features,
        n_trials=args.n_trials,
        n_cv_folds=args.n_folds,
        n_jobs=args.n_jobs,
        storage_url=args.storage,
        study_name=args.study_name
    )

    # Save best parameters
    output_path = f"{args.prefix}_best_params.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)

    print(f'Hyperparameter tuning completed.')
    print(f'Best parameters saved to: {output_path}')
    print(f'Best ROC AUC: {optuna.load_study(storage=args.storage, study_name=args.study_name).best_value:.4f}')


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()