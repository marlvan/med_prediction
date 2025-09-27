"""
Super Learner ensemble model using logistic regression meta-learner.

This module combines predictions from CatBoost and Keras models using
a logistic regression meta-learner with cross-validation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier
import tensorflow as tf

from utils.evaluation import compute_classification_metrics, create_results_dataframe, validate_input_data, compute_roc_curve_interpolated
from utils.data_sampling import load_indices_from_file


class SuperLearner:
    """
    Super Learner ensemble that combines CatBoost and Keras predictions.
    
    Uses a logistic regression meta-learner trained on out-of-fold predictions
    from base learners to create final ensemble predictions.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the Super Learner.
        
        Args:
            base_dir: Root directory containing base model artifacts
        """
        self.base_dir = base_dir
    
    def train_and_evaluate_internal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_vars: List[str],
        n_subsamples: int,
        n_folds: int
    ) -> pd.DataFrame:
        """
        Train and evaluate Super Learner using internal cross-validation.
        
        Creates meta-features from base learner predictions and trains
        a logistic regression meta-learner with fold-aware cross-validation.
        
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
            catboost_dir = os.path.join(outcome_dir, 'catboost_tuning')
            keras_dir = os.path.join(outcome_dir, 'keras_tuning')
            
            # Check if base model directories exist
            if not os.path.exists(catboost_dir) or not os.path.exists(keras_dir):
                raise FileNotFoundError(f"Base model directories not found for {outcome_var}")
            
            for subsample_idx in range(n_subsamples):
                # Load subsample data
                subsample_file = os.path.join(
                    indices_dir, f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                )
                
                if not os.path.exists(subsample_file):
                    raise FileNotFoundError(f"Subsample file not found: {subsample_file}")
                
                original_indices = load_indices_from_file(subsample_file)
                subsample_data = df.iloc[original_indices].reset_index(drop=True)
                
                # Generate meta-features for all folds
                try:
                    fold_predictions, fold_labels = self._generate_meta_features(
                        subsample_data, feature_cols, continuous_features, outcome_var,
                        subsample_idx, n_folds, indices_dir, catboost_dir, keras_dir
                    )
                except Exception as e:
                    print(f"Error generating meta-features for {outcome_var}, subsample {subsample_idx}: {e}")
                    continue
                
                # Evaluate meta-learner for each fold
                for fold_idx in range(n_folds):
                    try:
                        metrics = self._train_and_evaluate_meta_learner(
                            fold_predictions, fold_labels, fold_idx, n_folds,
                            outcome_var, subsample_idx, outcome_dir
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
                        print(f"Error training meta-learner for {outcome_var}, subsample {subsample_idx}, fold {fold_idx}: {e}")
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
        Evaluate Super Learner on external dataset.
        
        Uses pre-trained base learners and meta-learners to make predictions
        on a completely different dataset.
        
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
            trained_outcome_dir = os.path.join(trained_model_dir, outcome_var)
            
            # Check if trained model directories exist
            catboost_dir = os.path.join(trained_outcome_dir, 'catboost_tuning')
            keras_dir = os.path.join(trained_outcome_dir, 'keras_tuning')
            
            if not os.path.exists(catboost_dir) or not os.path.exists(keras_dir):
                print(f"Warning: Base model directories not found for {outcome_var}")
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
                            external_sample, feature_cols, continuous_features,
                            outcome_var, subsample_idx, fold_idx, trained_outcome_dir
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
                        print(f"Error evaluating external Super Learner fold {fold_idx} for {outcome_var}, subsample {subsample_idx}: {e}")
                        continue
        
        return results_df, tprs_df
    
    def _generate_meta_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_var: str,
        subsample_idx: int,
        n_folds: int,
        indices_dir: str,
        catboost_dir: str,
        keras_dir: str
    ) -> tuple[List[np.ndarray], List[pd.Series]]:
        """
        Generate meta-features from base learner predictions.
        
        Returns:
            Tuple of (fold_predictions, fold_labels) where fold_predictions
            contains stacked base learner predictions for each fold
        """
        fold_predictions = []
        fold_labels = []
        
        for fold_idx in range(n_folds):
            # Load validation indices and data
            valid_file = os.path.join(
                indices_dir,
                f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
            )
            
            if not os.path.exists(valid_file):
                raise FileNotFoundError(f"Validation indices file not found: {valid_file}")
            
            valid_indices = load_indices_from_file(valid_file)
            X_valid = data[feature_cols].iloc[valid_indices]
            y_valid = data[outcome_var].iloc[valid_indices]
            
            # Get predictions from base learners
            try:
                catboost_preds = self._get_catboost_predictions(
                    X_valid, continuous_features, subsample_idx, fold_idx, catboost_dir
                )
                keras_preds = self._get_keras_predictions(
                    X_valid, continuous_features, subsample_idx, fold_idx, keras_dir
                )
            except Exception as e:
                print(f"Error getting base learner predictions for fold {fold_idx}: {e}")
                raise
            
            # Stack predictions as meta-features
            meta_features = np.vstack([catboost_preds, keras_preds]).T
            fold_predictions.append(meta_features)
            fold_labels.append(y_valid)
        
        return fold_predictions, fold_labels
    
    def _get_catboost_predictions(
        self,
        X: pd.DataFrame,
        continuous_features: List[str],
        subsample_idx: int,
        fold_idx: int,
        catboost_dir: str
    ) -> np.ndarray:
        """Get predictions from CatBoost model."""
        model_prefix = os.path.join(
            catboost_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
        )
        
        # Load model and scaler
        model_path = f'{model_prefix}_model.pkl'
        scaler_path = f'{model_prefix}_scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"CatBoost model or scaler not found: {model_prefix}")
        
        model = CatBoostClassifier()
        model.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale features and predict
        X_scaled = X.copy()
        X_scaled[continuous_features] = scaler.transform(X[continuous_features])
        
        return model.predict_proba(X_scaled)[:, 1]
    
    def _get_keras_predictions(
        self,
        X: pd.DataFrame,
        continuous_features: List[str],
        subsample_idx: int,
        fold_idx: int,
        keras_dir: str
    ) -> np.ndarray:
        """Get predictions from Keras model."""
        model_prefix = os.path.join(
            keras_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
        )
        
        # Load model and scaler
        model_path = f'{model_prefix}_model.keras'
        scaler_path = f'{model_prefix}_scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Keras model or scaler not found: {model_prefix}")
        
        model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale features and predict
        X_scaled = X.copy()
        X_scaled[continuous_features] = scaler.transform(X[continuous_features])
        
        predictions = model.predict(
            X_scaled.to_numpy(dtype=np.float32), verbose=0
        ).flatten()
        
        # Clean up TensorFlow session
        tf.keras.backend.clear_session()
        
        return predictions
    
    def _train_and_evaluate_meta_learner(
        self,
        fold_predictions: List[np.ndarray],
        fold_labels: List[pd.Series],
        target_fold: int,
        n_folds: int,
        outcome_var: str,
        subsample_idx: int,
        outcome_dir: str
    ) -> dict:
        """
        Train and evaluate meta-learner for a specific fold.
        
        Uses all folds except target_fold for training the meta-learner,
        then evaluates on the target_fold.
        """
        # Prepare validation set (target fold)
        X_meta_valid = fold_predictions[target_fold]
        y_meta_valid = fold_labels[target_fold]
        
        # Prepare training set (all other folds)
        train_folds = [i for i in range(n_folds) if i != target_fold]
        X_meta_train = np.vstack([fold_predictions[i] for i in train_folds])
        y_meta_train = np.concatenate([fold_labels[i] for i in train_folds])
        
        # Train or load meta-learner
        model_path = os.path.join(
            outcome_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(target_fold).zfill(2)}_super_learner.pkl'
        )
        
        # Create directory if it doesn't exist
        os.makedirs(outcome_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            # Train new meta-learner
            meta_learner = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            meta_learner.fit(X_meta_train, y_meta_train)
            
            # Save meta-learner
            with open(model_path, 'wb') as f:
                pickle.dump(meta_learner, f)
        else:
            # Load existing meta-learner
            with open(model_path, 'rb') as f:
                meta_learner = pickle.load(f)
        
        # Make predictions and evaluate
        y_scores = meta_learner.predict_proba(X_meta_valid)[:, 1]
        auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(
            y_meta_valid, y_scores
        )
        tpr_interpolated = compute_roc_curve_interpolated(y_meta_valid.values, y_scores)

        
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
        trained_outcome_dir: str
    ) -> dict:
        """Evaluate a single fold on external data."""
        catboost_dir = os.path.join(trained_outcome_dir, 'catboost_tuning')
        keras_dir = os.path.join(trained_outcome_dir, 'keras_tuning')
        
        # Get base learner predictions
        try:
            catboost_preds = self._get_catboost_predictions(
                data[feature_cols], continuous_features, subsample_idx, fold_idx, catboost_dir
            )
            keras_preds = self._get_keras_predictions(
                data[feature_cols], continuous_features, subsample_idx, fold_idx, keras_dir
            )
        except Exception as e:
            print(f"Error getting base learner predictions for external evaluation: {e}")
            raise
        
        # Stack predictions as meta-features
        meta_features = np.vstack([catboost_preds, keras_preds]).T
        
        # Load and apply meta-learner
        meta_learner_path = os.path.join(
            trained_outcome_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_super_learner.pkl'
        )
        
        if not os.path.exists(meta_learner_path):
            raise FileNotFoundError(f"Meta-learner not found: {meta_learner_path}")
        
        with open(meta_learner_path, 'rb') as f:
            meta_learner = pickle.load(f)
        
        # Make final predictions and evaluate
        y_scores = meta_learner.predict_proba(meta_features)[:, 1]
        auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(
            data[outcome_var], y_scores
        )
        tpr_interpolated = compute_roc_curve_interpolated(data[outcome_var], y_scores)
        
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