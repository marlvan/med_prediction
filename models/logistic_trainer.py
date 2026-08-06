"""
Logistic regression training and evaluation utilities.

This module provides a conventional comparator for the Super Learner, trained and
evaluated on the same subsamples, cross-validation folds and feature scaling as the
other models.
"""

import os
import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression

from utils.model_utils import apply_feature_scaling
from utils.evaluation import compute_classification_metrics, create_results_dataframe, validate_input_data, compute_roc_curve_interpolated
from utils.data_sampling import load_indices_from_file


class LogisticTrainer:
    """
    Trainer class for logistic regression models with cross-validation support.

    Provides a conventional machine learning comparator for the Super Learner,
    using the same balanced subsamples and cross-validation folds as the other
    models so that performance differences are attributable to the algorithm.
    """

    def __init__(self, base_dir: str):
        """
        Initialize the logistic regression trainer.

        Args:
            base_dir: Root directory containing model artifacts and data
        """
        self.base_dir = base_dir
        self.model_type = 'logistic'

    def train_and_evaluate_internal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_vars: List[str],
        n_subsamples: int,
        n_folds: int,
        C: float = 1.0
    ) -> pd.DataFrame:
        """
        Train and evaluate logistic regression models using internal cross-validation.

        Trains models on each subsample and fold, evaluating on the held-out
        validation set within the same dataset.

        Args:
            df: Full dataset containing features and targets
            feature_cols: List of feature column names
            continuous_features: List of continuous feature column names
            outcome_vars: List of target variable names
            n_subsamples: Number of subsamples to evaluate
            n_folds: Number of cross-validation folds
            C: Inverse regularization strength

        Returns:
            DataFrame with evaluation metrics for each fold
        """
        validate_input_data(df, feature_cols, outcome_vars, continuous_features)

        results_df = create_results_dataframe(outcome_vars, n_subsamples, n_folds)

        fpr_points = np.linspace(0, 1, 101)
        tprs_columns = ['Outcome', 'Bootstrap', 'Fold'] + [f'{fpr:.2f}' for fpr in fpr_points]
        tprs_df = pd.DataFrame(columns=tprs_columns)

        for outcome_var in outcome_vars:
            indices_dir = os.path.join(self.base_dir, outcome_var, 'indices')

            for subsample_idx in range(n_subsamples):
                subsample_file = os.path.join(
                    indices_dir, f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                )

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
                            C=C
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
                        print(f"Error training fold {fold_idx} for {outcome_var}, subsample {subsample_idx}: {e}")
                        continue

        return results_df, tprs_df

    def evaluate_external(
        self,
        external_df: pd.DataFrame,
        feature_cols: List[str],
        continuous_features: List[str],
        outcome_vars: List[str],
        n_subsamples: int,
        n_folds: int,
        train_df: pd.DataFrame,
        trained_model_dir: str,
        output_dir: str,
        C: float = 1.0
    ) -> pd.DataFrame:
        """
        Evaluate logistic regression models on an external dataset.

        Models are fit on the training cohort's folds and evaluated on the whole
        balanced external subsample, mirroring SuperLearner.evaluate_external. The
        external subsamples created by the Super Learner evaluation are reused so
        that all models are compared on identical participants.

        Args:
            external_df: External dataset for evaluation
            feature_cols: List of feature column names
            continuous_features: List of continuous feature column names
            outcome_vars: List of target variable names
            n_subsamples: Number of subsamples to evaluate
            n_folds: Number of cross-validation folds
            train_df: Dataset the models are trained on
            trained_model_dir: Directory containing the training cohort's indices
            output_dir: Directory containing the external cohort's indices
            C: Inverse regularization strength

        Returns:
            DataFrame with evaluation metrics on external data
        """
        validate_input_data(external_df, feature_cols, outcome_vars, continuous_features)

        results_df = create_results_dataframe(outcome_vars, n_subsamples, n_folds)

        fpr_points = np.linspace(0, 1, 101)
        tprs_columns = ['Outcome', 'Bootstrap', 'Fold'] + [f'{fpr:.2f}' for fpr in fpr_points]
        tprs_df = pd.DataFrame(columns=tprs_columns)

        for outcome_var in outcome_vars:
            train_indices_dir = os.path.join(trained_model_dir, outcome_var, 'indices')
            external_indices_dir = os.path.join(output_dir, outcome_var, 'indices')

            for subsample_idx in range(n_subsamples):
                # Load the training and external balanced subsamples
                train_pool = train_df.iloc[
                    load_indices_from_file(os.path.join(
                        train_indices_dir, f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                    ))
                ].reset_index(drop=True)

                external_sample = external_df.iloc[
                    load_indices_from_file(os.path.join(
                        external_indices_dir, f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                    ))
                ].reset_index(drop=True)

                for fold_idx in range(n_folds):
                    try:
                        metrics = self._train_and_evaluate_fold(
                            data=train_pool,
                            feature_cols=feature_cols,
                            continuous_features=continuous_features,
                            outcome_var=outcome_var,
                            subsample_idx=subsample_idx,
                            fold_idx=fold_idx,
                            indices_dir=train_indices_dir,
                            C=C,
                            external_data=external_sample
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
                        print(f"Error evaluating external fold {fold_idx} for {outcome_var}, subsample {subsample_idx}: {e}")
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
        C: float = 1.0,
        external_data: pd.DataFrame = None
    ) -> dict:
        """Train and evaluate a single fold."""
        train_file = os.path.join(
            indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_train.pkl'
        )
        valid_file = os.path.join(
            indices_dir,
            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
        )

        if not os.path.exists(train_file) or not os.path.exists(valid_file):
            raise FileNotFoundError(f"CV index files not found: {train_file} or {valid_file}")

        train_indices = load_indices_from_file(train_file)
        valid_indices = load_indices_from_file(valid_file)

        X_train = data[feature_cols].iloc[train_indices]
        y_train = data[outcome_var].iloc[train_indices]

        # External evaluation uses the whole balanced external sample
        if external_data is not None:
            X_valid = external_data[feature_cols]
            y_valid = external_data[outcome_var]
        else:
            X_valid = data[feature_cols].iloc[valid_indices]
            y_valid = data[outcome_var].iloc[valid_indices]

        X_train_scaled, X_valid_scaled, _ = apply_feature_scaling(
            X_train, X_valid, continuous_features, fit_scaler=True
        )

        model = LogisticRegression(
            C=C, solver='liblinear', max_iter=1000, random_state=42
        )
        model.fit(X_train_scaled.to_numpy(dtype=float), y_train)

        y_scores = model.predict_proba(X_valid_scaled.to_numpy(dtype=float))[:, 1]
        auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(y_valid, y_scores)
        tpr_interpolated = compute_roc_curve_interpolated(np.asarray(y_valid), y_scores)

        return {
            'ROC AUC': auc,
            'PR AUC': pr_auc,
            'Accuracy': acc,
            'Confusion Matrix': cm,
            'Sensitivity': sens,
            'Specificity': spec,
            'TPR_interpolated': tpr_interpolated
        }
