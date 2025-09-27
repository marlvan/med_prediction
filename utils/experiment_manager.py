"""
Experiment management utilities for machine learning workflows.

This module handles the generation of training data, cross-validation splits,
and command-line interfaces for distributed hyperparameter tuning.
"""

import os
import pickle
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .data_sampling import create_balanced_sample, generate_stratified_cv_splits
from .evaluation import create_results_dataframe


def create_experiment_directories(base_dir: str, outcome_vars: List[str]) -> Dict[str, str]:
    """
    Create standardized directory structure for machine learning experiments.
    
    Creates organized directory structure with separate folders for each outcome
    and model type to store indices, data, and tuning results.
    
    Args:
        base_dir: Root directory for the experiment
        outcome_vars: List of target outcome variable names
        
    Returns:
        Dictionary mapping directory purposes to their paths
        
    Example:
        >>> dirs = create_experiment_directories('/experiments', ['outcome1'])
        >>> print(dirs['outcome1_indices'])  # '/experiments/outcome1/indices'
    """
    directory_map = {}
    
    for outcome in outcome_vars:
        # Create main outcome directory
        outcome_dir = os.path.join(base_dir, outcome)
        os.makedirs(outcome_dir, exist_ok=True)
        
        # Create subdirectories for each component
        subdirs = ['indices', 'data', 'catboost_tuning', 'keras_tuning']
        for subdir in subdirs:
            subdir_path = os.path.join(outcome_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            directory_map[f'{outcome}_{subdir}'] = subdir_path
            
        directory_map[f'{outcome}_base'] = outcome_dir
    
    return directory_map


def generate_experiment_data(
    df: pd.DataFrame,
    subject_col: str,
    feature_cols: List[str],
    continuous_features: List[str],
    outcome_vars: List[str],
    n_subsamples: int,
    n_folds: int,
    base_dir: str
) -> Dict[str, List[str]]:
    """
    Generate subsampled datasets and cross-validation splits for ML experiments.
    
    Creates balanced subsamples for each outcome, generates stratified CV splits,
    saves all indices and data files, and returns command lists for distributed
    hyperparameter tuning.
    
    Args:
        df: Full input dataset
        subject_col: Column name for subject identifiers
        feature_cols: List of feature column names
        continuous_features: List of continuous feature column names
        outcome_vars: List of target variable names
        n_subsamples: Number of balanced subsamples per outcome
        n_folds: Number of cross-validation folds
        base_dir: Root directory for saving all outputs
        
    Returns:
        Dictionary with 'catboost_commands' and 'keras_commands' lists
        
    Example:
        >>> commands = generate_experiment_data(
        ...     df, 'subject_id', features, continuous_vars, 
        ...     outcomes, n_subsamples=10, n_folds=5, base_dir='/exp'
        ... )
        >>> print(f"Generated {len(commands['catboost_commands'])} CatBoost commands")
    """
    # Create directory structure
    dir_map = create_experiment_directories(base_dir, outcome_vars)
    
    # Initialize command lists for distributed tuning
    catboost_commands = []
    keras_commands = []
    
    for outcome_idx, outcome_var in enumerate(outcome_vars):
        _process_single_outcome(
            df=df,
            outcome_var=outcome_var,
            outcome_idx=outcome_idx,
            subject_col=subject_col,
            feature_cols=feature_cols,
            continuous_features=continuous_features,
            n_subsamples=n_subsamples,
            n_folds=n_folds,
            dir_map=dir_map,
            catboost_commands=catboost_commands,
            keras_commands=keras_commands
        )
    
    return {
        'catboost_commands': catboost_commands,
        'keras_commands': keras_commands
    }


def _process_single_outcome(
    df: pd.DataFrame,
    outcome_var: str,
    outcome_idx: int,
    subject_col: str,
    feature_cols: List[str],
    continuous_features: List[str],
    n_subsamples: int,
    n_folds: int,
    dir_map: Dict[str, str],
    catboost_commands: List[str],
    keras_commands: List[str]
) -> None:
    """
    Process a single outcome variable for experiment generation.
    
    Internal helper function that handles data generation and command creation
    for one specific outcome variable.
    """
    indices_dir = dir_map[f'{outcome_var}_indices']
    data_dir = dir_map[f'{outcome_var}_data']
    catboost_dir = dir_map[f'{outcome_var}_catboost_tuning']
    keras_dir = dir_map[f'{outcome_var}_keras_tuning']
    
    for subsample_idx in range(n_subsamples):
        # Generate or load balanced subsample
        subsample_file = os.path.join(
            indices_dir, 
            f'subsample_{str(subsample_idx).zfill(3)}.pkl'
        )
        
        if not os.path.exists(subsample_file):
            # Create balanced subsample with unique random state
            random_state = _calculate_random_state(outcome_idx, subsample_idx)
            original_indices, balanced_data = create_balanced_sample(
                df, outcome_var, random_state, subject_col
            )
            
            # Save indices and data
            with open(subsample_file, 'wb') as f:
                pickle.dump(original_indices, f)
            
            data_file = os.path.join(
                data_dir, 
                f'subsample_{str(subsample_idx).zfill(3)}_full.pkl'
            )
            with open(data_file, 'wb') as f:
                pickle.dump(balanced_data, f)
        else:
            # Load existing data
            with open(subsample_file, 'rb') as f:
                original_indices = pickle.load(f)
            data_file = os.path.join(
                data_dir, 
                f'subsample_{str(subsample_idx).zfill(3)}_full.pkl'
            )
            with open(data_file, 'rb') as f:
                balanced_data = pickle.load(f)
        
        # Generate cross-validation splits
        _generate_cv_splits_for_subsample(
            balanced_data=balanced_data,
            feature_cols=feature_cols,
            continuous_features=continuous_features,
            outcome_var=outcome_var,
            subsample_idx=subsample_idx,
            n_folds=n_folds,
            indices_dir=indices_dir,
            data_dir=data_dir,
            catboost_dir=catboost_dir,
            keras_dir=keras_dir,
            catboost_commands=catboost_commands,
            keras_commands=keras_commands
        )


def _generate_cv_splits_for_subsample(
    balanced_data: pd.DataFrame,
    feature_cols: List[str],
    continuous_features: List[str],
    outcome_var: str,
    subsample_idx: int,
    n_folds: int,
    indices_dir: str,
    data_dir: str,
    catboost_dir: str,
    keras_dir: str,
    catboost_commands: List[str],
    keras_commands: List[str]
) -> None:
    """Generate CV splits and training commands for a single subsample."""
    X = balanced_data[feature_cols].astype(float)
    y = balanced_data[outcome_var].astype(int)
    
    # Check if CV splits already exist
    splits_exist = all(
        os.path.exists(os.path.join(
            indices_dir, 
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold).zfill(2)}_train.pkl'
        )) and
        os.path.exists(os.path.join(
            indices_dir, 
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold).zfill(2)}_valid.pkl'
        ))
        for fold in range(n_folds)
    )
    
    # Generate CV splits if they don't exist
    if not splits_exist:
        # Use consistent random state
        cv_splits = generate_stratified_cv_splits(X, y, n_folds, random_state=42)
        
        for fold_idx, (train_indices, valid_indices) in enumerate(cv_splits):
            train_file = os.path.join(
                indices_dir,
                f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_train.pkl'
            )
            valid_file = os.path.join(
                indices_dir,
                f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
            )
            
            # Create directory if it doesn't exist
            os.makedirs(indices_dir, exist_ok=True)
            
            with open(train_file, 'wb') as f:
                pickle.dump(train_indices, f)
            with open(valid_file, 'wb') as f:
                pickle.dump(valid_indices, f)
    
    # Generate training data and commands for each fold
    for fold_idx in range(n_folds):
        _create_fold_training_data_and_commands(
            balanced_data=balanced_data,
            feature_cols=feature_cols,
            continuous_features=continuous_features,
            outcome_var=outcome_var,
            subsample_idx=subsample_idx,
            fold_idx=fold_idx,
            indices_dir=indices_dir,
            data_dir=data_dir,
            catboost_dir=catboost_dir,
            keras_dir=keras_dir,
            catboost_commands=catboost_commands,
            keras_commands=keras_commands
        )


def _create_fold_training_data_and_commands(
    balanced_data: pd.DataFrame,
    feature_cols: List[str],
    continuous_features: List[str],
    outcome_var: str,
    subsample_idx: int,
    fold_idx: int,
    indices_dir: str,
    data_dir: str,
    catboost_dir: str,
    keras_dir: str,
    catboost_commands: List[str],
    keras_commands: List[str]
) -> None:
    """Create training data and CLI commands for a specific fold."""
    # Define file paths
    train_data_file = os.path.join(
        data_dir,
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_train.pkl'
    )
    
    # Create training data if it doesn't exist
    if not os.path.exists(train_data_file):
        train_indices_file = os.path.join(
            indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_train.pkl'
        )
        
        with open(train_indices_file, 'rb') as f:
            train_indices = pickle.load(f)
        
        # Extract training subset
        X_train = balanced_data[feature_cols].iloc[train_indices]
        y_train = balanced_data[outcome_var].iloc[train_indices]
        
        # Save training data
        with open(train_data_file, 'wb') as f:
            pickle.dump([X_train, y_train, continuous_features], f)
    
    # Generate CLI commands for distributed tuning
    study_name = f'{outcome_var}_subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    
    # CatBoost command
    catboost_prefix = os.path.join(
        catboost_dir,
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    catboost_script = os.path.join(os.getcwd(), 'utils', 'catboost_tuning.py')
    catboost_commands.append(_build_catboost_tuning_command(
        script_path=catboost_script,
        data_path=train_data_file,
        prefix=catboost_prefix,
        study_name=study_name
    ))
    
    # Keras command
    keras_prefix = os.path.join(
        keras_dir,
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    keras_script = os.path.join(os.getcwd(), 'utils', 'keras_tuning.py')
    keras_commands.append(_build_keras_tuning_command(
        script_path=keras_script,
        data_path=train_data_file,
        prefix=keras_prefix,
        study_name=study_name,
    ))


def _build_catboost_tuning_command(
    script_path: str,
    data_path: str,
    prefix: str,
    study_name: str,
    n_trials: int = 100,
    n_folds: int = 5
) -> str:
    """Build a CLI command for hyperparameter tuning."""
    storage_path = f'{prefix}_optuna.db'
    log_path = f'{prefix}_log.txt'
    
    return (
        f'python {script_path} '
        f'--data_path {data_path} '
        f'--n_jobs 1 '
        f'--prefix {prefix} '
        f'--n_trials {n_trials} '
        f'--n_folds {n_folds} '
        f'--storage sqlite:///{storage_path} '
        f'--study_name {study_name} '
        f'>> {log_path} 2>&1'
    )

def _build_keras_tuning_command(
    script_path: str,
    data_path: str,
    prefix: str,
    study_name: str,
    n_trials: int = 100,
    n_folds: int = 5,
) -> str:
    """Build a CLI command for hyperparameter tuning."""
    storage_path = f'{prefix}_optuna.db'
    log_path = f'{prefix}_log.txt'
    
    return (
        f'python {script_path} '
        f'--data_path {data_path} '
        f'--n_jobs 1 '
        f'--prefix {prefix} '
        f'--n_trials {n_trials} '
        f'--n_folds {n_folds} '
        f'--storage sqlite:///{storage_path} '
        f'--study_name {study_name} '
        f'>> {log_path} 2>&1'
    )


def save_command_scripts(
    commands: Dict[str, List[str]],
    base_dir: str
) -> None:
    """
    Save command lists to executable shell scripts.
    
    Args:
        commands: Dictionary with 'catboost_commands' and 'keras_commands'
        base_dir: Directory where shell scripts should be saved
    """
    for model_type, command_list in commands.items():
        if command_list:  # Only create script if commands exist
            script_name = f'run_{model_type.replace("_commands", "")}_tuning.sh'
            script_path = os.path.join(base_dir, script_name)
            
            with open(script_path, 'w') as f:
                for cmd in command_list:
                    f.write(cmd + '\n')
            
            print(f'Run: parallel --progress :::: {script_path}')


def _calculate_random_state(outcome_idx: int, subsample_idx: int) -> int:
    """
    Calculate a unique random state for reproducible subsampling.
    
    Uses a deterministic formula to ensure different random states for
    different outcome/subsample combinations while maintaining reproducibility.
    """
    # Handle edge case for first outcome (outcome_idx=0)
    if outcome_idx == 0:
        return (outcome_idx + 3) * subsample_idx
    else:
        return outcome_idx * subsample_idx