"""
Keras neural network training and evaluation utilities.

This module provides functions for training Keras neural networks with cross-validation,
internal testing on training data, and external testing on new datasets.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from utils.model_utils import build_neural_network, apply_feature_scaling
from utils.evaluation import compute_classification_metrics, create_results_dataframe, validate_input_data, compute_roc_curve_interpolated
from utils.data_sampling import load_indices_from_file

# Add these imports and settings at the top of keras_trainer.py
import os
import warnings
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Disable eager execution warnings specifically
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasTrainer:
    """
    Trainer class for Keras neural networks with cross-validation support.
    
    Handles model training, evaluation, and persistence across multiple
    subsamples and cross-validation folds.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the Keras trainer.
        
        Args:
            base_dir: Root directory containing model artifacts and data
        """
        self.base_dir = base_dir
        self.model_type = 'keras'
    
    def train_and_evaluate_internal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_vars: List[str],
        n_subsamples: int,
        n_folds: int,
        use_tuned_params: bool = True
    ) -> pd.DataFrame:
        """
        Train and evaluate Keras models using internal cross-validation.
        
        Trains neural networks on each subsample and fold, evaluating on the 
        held-out validation set within the same dataset.
        
        Args:
            df: Full dataset containing features and targets
            feature_cols: List of feature column names
            continuous_features: List of continuous feature column names
            outcome_vars: List of target variable names
            n_subsamples: Number of subsamples to evaluate
            n_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with evaluation metrics for each fold
        """
        # Validate input data
        validate_input_data(df, feature_cols, outcome_vars, continuous_features)
        
        results_df = create_results_dataframe(outcome_vars, n_subsamples, n_folds)

        fpr_points = np.linspace(0, 1, 101)
        tprs_columns = ['Outcome', 'Bootstrap', 'Fold'] + [f'{fpr:.2f}' for fpr in fpr_points]
        tprs_df = pd.DataFrame(columns=tprs_columns)
        
        for outcome_var in outcome_vars:
            outcome_dir = os.path.join(self.base_dir, outcome_var)
            indices_dir = os.path.join(outcome_dir, 'indices')
            tuning_dir = os.path.join(outcome_dir, 'keras_tuning')
            
            for subsample_idx in range(n_subsamples):
                # Load subsample indices
                subsample_file = os.path.join(
                    indices_dir, f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                )
                
                # Check if file exists
                if not os.path.exists(subsample_file):
                    raise FileNotFoundError(f"Subsample file not found: {subsample_file}")
                
                original_indices = load_indices_from_file(subsample_file)
                subsample_data = df.iloc[original_indices].reset_index(drop=True)
                
                for fold_idx in range(n_folds):
                    try:
                        metrics = self._train_and_evaluate_fold(
                            data=subsample_data,
                            feature_cols=feature_cols,
                            continuous_features=continuous_features,
                            outcome_var=outcome_var,
                            subsample_idx=subsample_idx,
                            fold_idx=fold_idx,
                            indices_dir=indices_dir,
                            tuning_dir=tuning_dir,
                            use_tuned_params=use_tuned_params
                        )

                        # Extract TPR values
                        tpr_interpolated = metrics.pop('TPR_interpolated')
                        
                        results_df.loc[len(results_df)] = {
                            'Outcome': outcome_var,
                            'Subsample': subsample_idx,
                            'Fold': fold_idx,
                            **metrics
                        }

                        # Add to TPRs dataframe
                        tpr_row = {
                            'Outcome': outcome_var,
                            'Bootstrap': subsample_idx,
                            'Fold': fold_idx
                        }
                        
                        # Add TPR values for each FPR point
                        for i, fpr_val in enumerate(fpr_points):
                            tpr_row[f'{fpr_val:.2f}'] = tpr_interpolated[i]
                        
                        tprs_df.loc[len(tprs_df)] = tpr_row
                        
                    except Exception as e:
                        print(f"Error training Keras fold {fold_idx} for {outcome_var}, subsample {subsample_idx}: {e}")
                        # Clean up TensorFlow session on error
                        tf.keras.backend.clear_session()
                        continue
        
        return results_df, tprs_df
    
    def evaluate_external(
        self,
        external_df: pd.DataFrame,
        subject_col: str,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_vars: List[str],
        n_subsamples: int,
        n_folds: int,
        trained_model_dir: str,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Evaluate pre-trained Keras models on an external dataset.
        
        Uses neural networks trained on one dataset to make predictions on a 
        completely different dataset for external validation.
        
        Args:
            external_df: External dataset for evaluation
            subject_col: Column name for subject identifiers
            feature_cols: List of feature column names
            continuous_features: List of continuous feature column names
            outcome_vars: List of target variable names
            n_subsamples: Number of subsamples to evaluate
            n_folds: Number of cross-validation folds
            trained_model_dir: Directory containing pre-trained models
            output_dir: Directory to save evaluation results
            
        Returns:
            DataFrame with evaluation metrics on external data
        """
        # Validate input data
        validate_input_data(external_df, feature_cols, outcome_vars, continuous_features)
        
        from utils.experiment_manager import create_experiment_directories
        
        # Create output directory structure
        create_experiment_directories(output_dir, outcome_vars)
        
        results_df = create_results_dataframe(outcome_vars, n_subsamples, n_folds)

        fpr_points = np.linspace(0, 1, 101)
        tprs_columns = ['Outcome', 'Bootstrap', 'Fold'] + [f'{fpr:.2f}' for fpr in fpr_points]
        tprs_df = pd.DataFrame(columns=tprs_columns)
        
        for outcome_idx, outcome_var in enumerate(outcome_vars):
            output_outcome_dir = os.path.join(output_dir, outcome_var)
            output_indices_dir = os.path.join(output_outcome_dir, 'indices')
            trained_tuning_dir = os.path.join(trained_model_dir, outcome_var, 'keras_tuning')
            
            # Check if trained model directory exists
            if not os.path.exists(trained_tuning_dir):
                print(f"Warning: Trained model directory not found: {trained_tuning_dir}")
                continue
            
            for subsample_idx in range(n_subsamples):
                # Create or load balanced external sample
                try:
                    external_sample = self._get_external_sample(
                        external_df, outcome_var, outcome_idx, subsample_idx,
                        subject_col, output_indices_dir
                    )
                except Exception as e:
                    print(f"Error creating external sample for {outcome_var}, subsample {subsample_idx}: {e}")
                    continue
                
                for fold_idx in range(n_folds):
                    try:
                        metrics = self._evaluate_external_fold(
                            data=external_sample,
                            feature_cols=feature_cols,
                            continuous_features=continuous_features,
                            outcome_var=outcome_var,
                            subsample_idx=subsample_idx,
                            fold_idx=fold_idx,
                            trained_tuning_dir=trained_tuning_dir
                        )

                        # Extract TPR values
                        tpr_interpolated = metrics.pop('TPR_interpolated')
                        
                        results_df.loc[len(results_df)] = {
                            'Outcome': outcome_var,
                            'Subsample': subsample_idx,
                            'Fold': fold_idx,
                            **metrics
                        }

                        # Add to TPRs dataframe
                        tpr_row = {
                            'Outcome': outcome_var,
                            'Bootstrap': subsample_idx,
                            'Fold': fold_idx
                        }

                        # Add TPR values for each FPR point
                        for i, fpr_val in enumerate(fpr_points):
                            tpr_row[f'{fpr_val:.2f}'] = tpr_interpolated[i]
                        
                        tprs_df.loc[len(tprs_df)] = tpr_row
                        
                    except Exception as e:
                        print(f"Error evaluating external Keras fold {fold_idx} for {outcome_var}, subsample {subsample_idx}: {e}")
                        # Clean up TensorFlow session on error
                        tf.keras.backend.clear_session()
                        continue
        
        return results_df, tprs_df
    
    def _train_and_evaluate_fold(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_var: str,
        subsample_idx: int,
        fold_idx: int,
        indices_dir: str,
        tuning_dir: str,
        use_tuned_params: bool = True
    ) -> dict:
        """Train and evaluate a single fold."""
        from utils.model_utils import get_default_keras_params
        # Load train/validation indices
        train_file = os.path.join(
            indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_train.pkl'
        )
        valid_file = os.path.join(
            indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
        )
        
        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(valid_file):
            raise FileNotFoundError(f"CV index files not found: {train_file} or {valid_file}")
        
        train_indices = load_indices_from_file(train_file)
        valid_indices = load_indices_from_file(valid_file)
        
        # Split data
        X = data[feature_cols]
        y = data[outcome_var]
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_valid, y_valid = X.iloc[valid_indices], y.iloc[valid_indices]
        
        # Model file prefix
        model_prefix = os.path.join(
            tuning_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
        )
        
        # Train or load model
        model_path = f'{model_prefix}_model.keras'
        if not os.path.exists(model_path):
            # Scale features
            X_train_scaled, X_valid_scaled, scaler = apply_feature_scaling(
                X_train, X_valid, continuous_features, fit_scaler=True
            )
            
            if use_tuned_params:
                # Load best parameters
                params_file = f'{model_prefix}_best_params.pkl'
                if not os.path.exists(params_file):
                    raise FileNotFoundError(f"Best parameters file not found: {params_file}")
                    
                with open(params_file, 'rb') as f:
                    best_params = pickle.load(f)
            else:
                # Use default parameters
                best_params = get_default_keras_params()
            
            # Build and train model
            model = build_neural_network(input_dim=X.shape[1], **best_params)
            
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
                X_train_scaled.to_numpy(dtype=np.float32),
                y_train.values,
                validation_data=(
                    X_valid_scaled.to_numpy(dtype=np.float32),
                    y_valid.values
                ),
                epochs=best_params.get('epochs', 200),
                batch_size=best_params['batch_size'],
                verbose=0,
                callbacks=callbacks
            )
            
            # Save model and scaler
            model.save(model_path)
            scaler_path = f'{model_prefix}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        else:
            # Load existing model and scaler
            model = load_model(model_path)
            scaler_path = f'{model_prefix}_scaler.pkl'
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            X_train_scaled, X_valid_scaled, _ = apply_feature_scaling(
                X_train, X_valid, continuous_features, scaler, fit_scaler=False
            )
        
        # Make predictions and evaluate
        y_scores = model.predict(
            X_valid_scaled.to_numpy(dtype=np.float32), verbose=0
        ).flatten()
        
        auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(
            y_valid.values, y_scores
        )
        tpr_interpolated = compute_roc_curve_interpolated(y_valid.values, y_scores)

        
        # Clean up TensorFlow session
        tf.keras.backend.clear_session()
        
        return {
            'ROC AUC': auc,
            'PR AUC': pr_auc,
            'Accuracy': acc,
            'Confusion Matrix': cm, 
            'Sensitivity': sens,
            'Specificity': spec,
            'TPR_interpolated': tpr_interpolated
        }
    
    def _evaluate_external_fold(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_var: str,
        subsample_idx: int,
        fold_idx: int,
        trained_tuning_dir: str
    ) -> dict:
        """Evaluate a single fold on external data."""
        # Model file prefix
        model_prefix = os.path.join(
            trained_tuning_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
        )
        
        # Check if model files exist
        model_path = f'{model_prefix}_model.keras'
        scaler_path = f'{model_prefix}_scaler.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load model and scaler
        model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Apply same scaling as training
        data_scaled = data.copy()
        data_scaled[continuous_features] = scaler.transform(data[continuous_features])
        
        # Make predictions and evaluate
        y_scores = model.predict(
            data_scaled[feature_cols].to_numpy(dtype=np.float32), verbose=0
        ).flatten()
        
        auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(
            data[outcome_var], y_scores
        )
        tpr_interpolated = compute_roc_curve_interpolated(data[outcome_var], y_scores)
        
        # Clean up TensorFlow session
        tf.keras.backend.clear_session()
        
        return {
            'ROC AUC': auc,
            'PR AUC': pr_auc,
            'Accuracy': acc,
            'Confusion Matrix': cm,
            'Sensitivity': sens,
            'Specificity': spec,
            'TPR_interpolated': tpr_interpolated
        }
    
    def _get_external_sample(
        self,
        external_df: pd.DataFrame,
        outcome_var: str,
        outcome_idx: int,
        subsample_idx: int,
        subject_col: str,
        output_indices_dir: str
    ) -> pd.DataFrame:
        """Create or load a balanced sample from external dataset."""
        from utils.data_sampling import create_balanced_sample
        from utils.experiment_manager import _calculate_random_state
        
        sample_file = os.path.join(
            output_indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}.pkl'
        )
        
        # Create directory if it doesn't exist
        os.makedirs(output_indices_dir, exist_ok=True)
        
        if not os.path.exists(sample_file):
            # Create new balanced sample
            random_state = _calculate_random_state(outcome_idx, subsample_idx)
            original_indices, balanced_data = create_balanced_sample(
                external_df, outcome_var, random_state, subject_col
            )
            
            # Save indices for reproducibility
            with open(sample_file, 'wb') as f:
                pickle.dump(original_indices, f)
        else:
            # Load existing sample
            with open(sample_file, 'rb') as f:
                original_indices = pickle.load(f)
            balanced_data = external_df.iloc[original_indices].reset_index(drop=True)
        
        return balanced_data