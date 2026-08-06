"""
Model utilities for building and managing machine learning models.

This module provides functions for building neural networks, loading/saving models,
and managing scalers for feature preprocessing.
"""

import pickle
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC as KerasAUC
from catboost import CatBoostClassifier


def build_neural_network(
    input_dim: int,
    n_hidden_layers: int = 2,
    hidden_units: int = 64,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    l2_regularization: float = 0.0,
    activation: str = 'relu',
    optimizer_name: str = 'adam',
    batch_size: int = 32,
    epochs: int = 100
) -> tf.keras.Model:
    """
    Build and compile a feedforward neural network for binary classification.
    
    Creates a fully connected neural network with configurable architecture,
    regularization, and optimization parameters.
    
    Args:
        input_dim: Number of input features
        n_hidden_layers: Number of hidden layers
        hidden_units: Number of neurons in each hidden layer
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate applied after each hidden layer (0.0-1.0)
        l2_regularization: L2 regularization strength (0.0 = no regularization)
        activation: Activation function for hidden layers ('relu', 'tanh', etc.)
        optimizer_name: Optimizer to use ('adam', 'rmsprop', 'sgd')
        batch_size: Training batch size (for reference, not used in model build)
        epochs: Number of training epochs (for reference, not used in model build)
        
    Returns:
        Compiled Keras model ready for training
        
    Example:
        >>> model = build_neural_network(
        ...     input_dim=50, 
        ...     n_hidden_layers=3, 
        ...     hidden_units=128,
        ...     learning_rate=0.001,
        ...     dropout_rate=0.2
        ... )
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Add hidden layers with optional L2 regularization
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
    
    # Add output layer for binary classification
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


def get_default_catboost_params() -> dict:
    """
    Get default CatBoost parameters when hyperparameter tuning is skipped.
    
    Returns:
        Dictionary with default CatBoost parameters
    """
    return {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 4,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU',
        'thread_count': 1,
        'allow_writing_files': False,
        'eval_metric': 'AUC'
    }


def get_default_keras_params() -> dict:
    """
    Get default Keras parameters when hyperparameter tuning is skipped.
    
    Args:
        
    Returns:
        Dictionary with default Keras parameters
    """
    return {
        'n_hidden_layers': 2,
        'hidden_units': 64,
        'learning_rate': 1e-3,
        'dropout_rate': 0.3,
        'batch_size': 32,
        'activation': 'relu',
        'optimizer_name': 'adam',
        'l2_regularization': 0.0
    }


def save_model_artifacts(
    model: Any,
    scaler: StandardScaler,
    best_params: Dict[str, Any],
    file_prefix: str,
    model_type: str = 'keras'
) -> None:
    """
    Save model, scaler, and parameters to disk.
    
    Args:
        model: Trained model (Keras or CatBoost)
        scaler: Fitted StandardScaler for features
        best_params: Dictionary of best hyperparameters
        file_prefix: Base filename prefix for all saved files
        model_type: Type of model ('keras' or 'catboost')
    """
    # Save scaler
    scaler_path = f"{file_prefix}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save parameters
    params_path = f"{file_prefix}_best_params.pkl"
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    
    # Save model (different methods for different model types)
    if model_type == 'keras':
        model_path = f"{file_prefix}_model.keras"
        model.save(model_path)
    elif model_type == 'catboost':
        model_path = f"{file_prefix}_model.pkl"
        model.save_model(model_path)


def load_model_artifacts(
    file_prefix: str,
    model_type: str = 'keras'
) -> tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    Load model, scaler, and parameters from disk.
    
    Args:
        file_prefix: Base filename prefix for all saved files
        model_type: Type of model ('keras' or 'catboost')
        
    Returns:
        Tuple of (model, scaler, best_params)
    """
    # Load scaler
    scaler_path = f"{file_prefix}_scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load parameters
    params_path = f"{file_prefix}_best_params.pkl"
    with open(params_path, 'rb') as f:
        best_params = pickle.load(f)
    
    # Load model
    if model_type == 'keras':
        model_path = f"{file_prefix}_model.keras"
        model = load_model(model_path)
    elif model_type == 'catboost':
        model_path = f"{file_prefix}_model.pkl"
        model = CatBoostClassifier()
        model.load_model(model_path)
    
    return model, scaler, best_params


def apply_feature_scaling(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    continuous_features: List[str],
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply standard scaling to continuous features.
    
    Fits a StandardScaler on training data and applies the same transformation
    to both training and validation data. Only specified continuous features
    are scaled.
    
    Args:
        X_train: Training features
        X_valid: Validation features
        continuous_features: List of column names to scale
        scaler: Pre-fitted scaler (optional, will create new if None)
        fit_scaler: Whether to fit the scaler on training data
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_valid, fitted_scaler)
        
    Example:
        >>> X_train_scaled, X_valid_scaled, scaler = apply_feature_scaling(
        ...     X_train, X_valid, ['age', 'income']
        ... )
    """
    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    
    # Create or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
    
    # Fit and transform training data, transform validation data
    if fit_scaler:
        X_train_scaled[continuous_features] = scaler.fit_transform(
            X_train_scaled[continuous_features]
        )
    else:
        X_train_scaled[continuous_features] = scaler.transform(
            X_train_scaled[continuous_features]
        )
        
    X_valid_scaled[continuous_features] = scaler.transform(
        X_valid_scaled[continuous_features]
    )
    
    return X_train_scaled, X_valid_scaled, scaler