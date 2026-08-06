"""
Keras neural network hyperparameter tuning using Optuna.

This script performs hyperparameter optimization for Keras neural networks using
cross-validation and Optuna's TPE sampler.
"""

import numpy as np
import pandas as pd
import argparse
import pickle
import gc
import warnings
import random
from typing import List

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC as KerasAUC

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_neural_network(
    input_dim: int,
    n_hidden_layers: int = 2,
    hidden_units: int = 64,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    l2_regularization: float = 0.0,
    activation: str = 'relu',
    optimizer_name: str = 'adam',
    **kwargs
) -> tf.keras.Model:
    """
    Build and compile a feedforward neural network for binary classification.
    
    Args:
        input_dim: Number of input features
        n_hidden_layers: Number of hidden layers
        hidden_units: Number of neurons in each hidden layer
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate applied after each hidden layer
        l2_regularization: L2 regularization strength
        activation: Activation function for hidden layers
        optimizer_name: Optimizer to use ('adam', 'rmsprop', 'sgd')
        
    Returns:
        Compiled Keras model ready for training
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Add hidden layers
    for layer_idx in range(n_hidden_layers):
        layer_kwargs = {
            'units': hidden_units,
            'activation': activation
        }

        # Add L2 regularization if specified
        if l2_regularization > 0:
            layer_kwargs['kernel_regularizer'] = regularizers.l2(l2_regularization)

        model.add(Dense(**layer_kwargs))
        model.add(Dropout(dropout_rate))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Configure optimizer
    optimizer_map = {
        'adam': Adam(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate)
    }
    optimizer = optimizer_map[optimizer_name]

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[KerasAUC(name='auc')]
    )

    return model


def objective_function(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    continuous_features: List[str],
    cv_splitter: StratifiedKFold
) -> float:
    """
    Optuna objective function for Keras hyperparameter tuning.

    Args:
        trial: Optuna trial object for hyperparameter suggestions
        X: Feature matrix
        y: Binary target vector
        continuous_features: List of continuous feature column names
        cv_splitter: StratifiedKFold cross-validator

    Returns:
        Mean ROC AUC across cross-validation folds
    """
    # Suggest hyperparameters with reasonable ranges
    params = {
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),  
        'hidden_units': trial.suggest_int('hidden_units', 48, 96),
        'learning_rate': trial.suggest_float('learning_rate', 3e-4, 3e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.20, 0.40),
        'batch_size': 32,  # Fixed for simplicity
        'activation': 'relu',  # Fixed for simplicity
        'optimizer_name': 'adam',  # Fixed for simplicity
        'l2_regularization': 0.0  # Fixed for simplicity
    }

    # Cross-validation evaluation
    cv_scores = []
    for train_indices, valid_indices in cv_splitter.split(X, y):
        # Split data
        X_train = X.iloc[train_indices].copy()
        X_valid = X.iloc[valid_indices].copy()
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        
        # Scale continuous features
        scaler = StandardScaler()
        X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_valid[continuous_features] = scaler.transform(X_valid[continuous_features])
        
        # Set seeds for reproducibility
        set_random_seeds(42)
        
        # Build and train model
        model = build_neural_network(input_dim=X.shape[1], **params)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=12,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(dtype=np.float32),
            validation_data=(X_valid.to_numpy(dtype=np.float32), y_valid.to_numpy(dtype=np.float32)),
            epochs=200,
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=callbacks
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_valid.to_numpy(dtype=np.float32), verbose=0).ravel()
        cv_scores.append(roc_auc_score(y_valid, y_pred_proba))
        
        # Clean up memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()

    return np.mean(cv_scores)


def tune_keras_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    continuous_features: List[str],
    n_trials: int = 100,
    n_cv_folds: int = 3,
    n_jobs: int = 1,
    storage_url: str = None,
    study_name: str = None
) -> tuple[tf.keras.Model, dict]:
    """
    Tune Keras hyperparameters using Optuna with cross-validation.

    Args:
        X: Feature matrix
        y: Binary target vector
        continuous_features: List of continuous feature column names
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
        'hidden_units': 64,
        'learning_rate': 1e-3,
        'dropout_rate': 0.3
    })

    # Count existing completed trials
    completed_trials = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,)
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
        'batch_size': 32,
        'activation': 'relu',
        'optimizer_name': 'adam',
        'l2_regularization': 0.0
    })

    # Build final model with best parameters
    best_model = build_neural_network(input_dim=X.shape[1], **best_params)

    return best_model, best_params


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for Keras neural networks using Optuna'
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
    best_model, best_params = tune_keras_hyperparameters(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
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