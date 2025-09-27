"""
Model evaluation utilities for binary classification tasks.

This module provides functions for computing classification metrics,
ROC curve statistics, and summary statistics across multiple experiments.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import pickle
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
import shap
from catboost import CatBoostClassifier
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score, 
    confusion_matrix,
    roc_curve
)

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[float, float, float, np.ndarray, float, float]:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Calculates ROC-AUC, precision-recall AUC, accuracy, confusion matrix,
    sensitivity (recall), and specificity at a given threshold.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Predicted probabilities from the model
        threshold: Decision threshold for converting probabilities to predictions
        
    Returns:
        Tuple containing:
        - roc_auc: ROC AUC score
        - pr_auc: Precision-recall AUC score  
        - accuracy: Classification accuracy
        - confusion_mat: Confusion matrix (2x2 array)
        - sensitivity: Sensitivity/recall for positive class
        - specificity: Specificity (true negative rate)
        
    Example:
        >>> auc, pr_auc, acc, cm, sens, spec = compute_classification_metrics(
        ...     y_true, y_pred_proba, threshold=0.5
        ... )
        >>> print(f"ROC-AUC: {auc:.3f}, Sensitivity: {sens:.3f}")
    """
    # Input validation
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two different classes")
    
    # Compute probability-based metrics
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError as e:
        raise ValueError(f"Error computing AUC metrics: {e}")
    
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute threshold-based metrics
    accuracy = accuracy_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    if confusion_mat.shape == (2, 2):
        tn, fp, fn, tp = confusion_mat.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        # Handle edge case where only one class is present in predictions
        sensitivity, specificity = np.nan, np.nan
    
    return roc_auc, pr_auc, accuracy, confusion_mat, sensitivity, specificity


def compute_roc_curve_interpolated(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    n_points: int = 101
) -> np.ndarray:
    """
    Compute a standardized ROC curve with fixed interpolation points.
    
    Interpolates true positive rates (TPR) across evenly spaced false positive
    rates (FPR) from 0 to 1. This enables averaging ROC curves across multiple
    experiments.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities
        n_points: Number of interpolation points (default: 101 for 0.00 to 1.00)
        
    Returns:
        Interpolated TPR values at fixed FPR thresholds
        
    Example:
        >>> tpr_interp = compute_roc_curve_interpolated(y_true, y_pred_proba)
        >>> # tpr_interp contains TPR values at FPR = 0.00, 0.01, 0.02, ..., 1.00
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Create evenly spaced FPR points from 0 to 1
    fpr_grid = np.linspace(0, 1, n_points)
    
    # Interpolate TPR values at the grid points
    tpr_interpolated = np.interp(fpr_grid, fpr, tpr)
    
    # Ensure the curve starts at (0,0)
    tpr_interpolated[0] = 0.0
    
    return tpr_interpolated

def plot_roc_auc(save_dir: str, tprs: pd.DataFrame) -> None:
    """
    Plot ROC curves for each outcome using interpolated TPRs across folds.
    
    Creates a single ROC curve plot showing median curves for each outcome with 
    a diagonal reference line. Uses consistent styling with the rest of the codebase.
    
    Args:
        save_dir: Directory path to save the ROC curve plot
        tprs: DataFrame containing interpolated TPRs with columns:
              ['Outcome', 'Bootstrap', 'Fold'] + FPR values from 0.00 to 1.00
              
    Saves:
        'roc_curve.png' in the specified directory if it doesn't already exist
        
    Example:
        >>> plot_roc_auc('/path/to/output', pond_super_learner_tprs)
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if plot already exists
    plot_path = os.path.join(save_dir, 'roc_curve.png')
    if os.path.exists(plot_path):
        return
    
    # Set font and styling
    plt.rcParams["font.family"] = "Scala Sans Pro"
    plt.rcParams["font.size"] = 12
    
    # Define color palette
    palette = ['#8dbd61', '#6f459e', '#f66f8d', '#ffb643']
    
    # Create figure
    plt.figure(figsize=(3, 3))
    
    # Prepare data for plotting
    # Get FPR columns (all columns except Outcome, Bootstrap, Fold)
    fpr_cols = [col for col in tprs.columns if col not in ['Outcome', 'Bootstrap', 'Fold']]
    
    # Melt DataFrame for seaborn
    tprs_melted = pd.melt(
        tprs, 
        id_vars=['Outcome', 'Fold'], 
        value_vars=fpr_cols,
        var_name='FPR', 
        value_name='TPR'
    )
    
    # Convert FPR to numeric
    tprs_melted['FPR'] = pd.to_numeric(tprs_melted['FPR'])

    # Calculate median and IQR for each outcome and FPR point
    summary_stats = tprs_melted.groupby(['Outcome', 'FPR'])['TPR'].agg([
        ('median', 'median'),
        ('q1', lambda x: x.quantile(0.25)),
        ('q3', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    # Plot median curves with IQR bands
    for i, outcome in enumerate(tprs['Outcome'].unique()):
        outcome_data = summary_stats[summary_stats['Outcome'] == outcome]
        color = palette[i % len(palette)]
        
        # Plot median line
        plt.plot(outcome_data['FPR'], outcome_data['median'], 
                color=color, linewidth=1, label=outcome)
        
        # Plot IQR band
        plt.fill_between(outcome_data['FPR'], 
                       outcome_data['q1'], 
                       outcome_data['q3'], 
                       color=color, alpha=0.2)

    # Add diagonal reference line
    ax = plt.gca()
    sns.lineplot(
        x=[0, 1], 
        y=[0, 1], 
        ax=ax, 
        color='black', 
        linewidth=0.5, 
        linestyle='--'
    )
    
    # Set labels and formatting
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(fontsize=4, loc='lower right')
    
    # Set aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(0, 1), ylim=(0, 1), yticks=[0, 0.5, 1.0])
    
    # Save plot
    plt.savefig(plot_path, bbox_inches='tight', dpi=500)
    plt.close()  # Close figure to free memory    


def calculate_summary_statistics(
    data_group: pd.DataFrame,
    metrics_columns: Optional[List[str]] = None
) -> pd.Series:
    """
    Calculate median and IQR for numeric columns.
    
    Computes descriptive statistics including median and IQR.
    
    Args:
        data_group: DataFrame containing numeric columns to summarize
        metrics_columns: List of column names to analyze (if None, uses all numeric columns)
        
    Returns:
        Series containing median, Q1, and Q3 for each metric
        
    Example:
        >>> results_df = pd.DataFrame({
        ...     'ROC AUC': [0.75, 0.78, 0.72, 0.80],
        ...     'Accuracy': [0.70, 0.72, 0.68, 0.74]
        ... })
        >>> stats = calculate_summary_statistics_median(results_df)
        >>> print(f"ROC AUC: {stats['ROC AUC_median']:.3f} "
        ...       f"({stats['ROC AUC_q1']:.3f}-{stats['ROC AUC_q3']:.3f})")
    """
    if metrics_columns is None:
        # Use all numeric columns if not specified
        metrics_columns = data_group.select_dtypes(include=[np.number]).columns.tolist()
    
    summary_stats = {}
    
    for metric in metrics_columns:
        if metric in data_group.columns:
            values = data_group[metric].dropna()  # Remove NaN values
            
            if len(values) > 0:
                median_val = values.median()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                
                # Store median and IQRs
                summary_stats[f'{metric}_median'] = median_val
                summary_stats[f'{metric}_q1'] = q1
                summary_stats[f'{metric}_q3'] = q3
            else:
                # Handle empty data
                summary_stats[f'{metric}_median'] = np.nan
                summary_stats[f'{metric}_q1'] = np.nan
                summary_stats[f'{metric}_13'] = np.nan
    
    return pd.Series(summary_stats)

def create_results_dataframe(
    outcomes: List[str],
    n_subsamples: int,
    n_folds: int
) -> pd.DataFrame:
    """
    Create an empty results DataFrame with standard evaluation columns.
    
    Initializes a DataFrame with the proper structure for storing evaluation
    results across multiple outcomes, subsamples, and cross-validation folds.
    
    Args:
        outcomes: List of outcome variable names
        n_subsamples: Number of subsample iterations expected
        n_folds: Number of cross-validation folds expected
        
    Returns:
        Empty DataFrame with columns for storing evaluation results
        
    Example:
        >>> results_df = create_results_dataframe(['outcome1', 'outcome2'], 10, 5)
        >>> # DataFrame ready to store 10*5*2 = 100 rows of results
    """
    columns = [
        'Outcome', 'Subsample', 'Fold', 
        'ROC AUC', 'PR AUC', 'Accuracy', 
        'Confusion Matrix', 'Sensitivity', 'Specificity'
    ]
    
    # Return empty DataFrame - will use .loc[len(df)] to append rows
    return pd.DataFrame(columns=columns)


def format_confidence_interval(
    median: float, 
    q1: float, 
    q3: float, 
    decimals: int = 3
) -> str:
    """
    Format a confidence interval as a readable string.
    
    Args:
        median: Median value
        q1: Q1
        q3: Q3
        decimals: Number of decimal places to display
        
    Returns:
        Formatted string like "0.756 (0.721-0.791)"
        
    Example:
        >>> formatted = format_confidence_interval(0.756, 0.721, 0.791)
        >>> print(formatted)  # "0.756 (0.721-0.791)"
    """
    if pd.isna(median) or pd.isna(q1) or pd.isna(q3):
        return "N/A"
    
    format_str = f"{{:.{decimals}f}}"
    median_str = format_str.format(median)
    q1_str = format_str.format(q1)
    q3_str = format_str.format(q3)
    
    return f"{median_str} ({q1_str}-{q3_str})"


def validate_input_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_vars: List[str],
    continuous_features: List[str]
) -> None:
    """
    Validate input data for machine learning experiments.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        outcome_vars: List of target variable names
        continuous_features: List of continuous feature column names
        
    Raises:
        ValueError: If validation fails
    """
    # Check DataFrame is not empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check required columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    missing_outcomes = [col for col in outcome_vars if col not in df.columns]
    if missing_outcomes:
        raise ValueError(f"Missing outcome columns: {missing_outcomes}")
    
    missing_continuous = [col for col in continuous_features if col not in df.columns]
    if missing_continuous:
        raise ValueError(f"Missing continuous feature columns: {missing_continuous}")
    
    # Check continuous features are actually in feature list
    invalid_continuous = [col for col in continuous_features if col not in feature_cols]
    if invalid_continuous:
        raise ValueError(f"Continuous features not in feature list: {invalid_continuous}")
    
    # Check outcome variables are binary
    for outcome in outcome_vars:
        unique_values = df[outcome].dropna().unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"Outcome '{outcome}' must be binary (0/1), found: {unique_values}")
    
    print(f"Input validation passed: {len(df)} samples, {len(feature_cols)} features, {len(outcome_vars)} outcomes")

def calculate_super_learner_fairness(
    cohort_data_list: List[dict],
    outcome_vars: List[str],
    sensitive_cols: List[str],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculate DPR and EOR for Super Learner across specified cohorts, subsamples, and folds.
    
    Args:
        cohort_data_list: List of dictionaries, each containing:
            - 'name': cohort name (e.g., 'POND', 'PPP')
            - 'df': dataset DataFrame
            - 'feature_cols': feature column names
            - 'continuous_features': continuous feature names
            - 'model_dir': directory containing trained models
            - 'n_subsamples': number of subsamples
            - 'n_folds': number of folds
        outcome_vars: List of outcome variables
        sensitive_cols: List of sensitive attribute column names
        threshold: Decision threshold
        
    Returns:
        DataFrame with DPR and EOR for each outcome/sensitive attribute combination
    """
    results = []
    
    for outcome_var in outcome_vars:
        for sens_col in sensitive_cols:
            
            # Collect all predictions and labels across specified cohorts
            all_y_true = []
            all_y_pred = []
            all_sensitive = []
            
            for cohort_data in cohort_data_list:
                cohort_name = cohort_data['name']
                df = cohort_data['df']
                feature_cols = cohort_data['feature_cols']
                continuous_features = cohort_data['continuous_features']
                model_dir = cohort_data['model_dir']
                n_subsamples = cohort_data['n_subsamples']
                n_folds = cohort_data['n_folds']
                
                for subsample_idx in range(n_subsamples):
                    
                    # Load subsample data
                    subsample_file = os.path.join(
                        model_dir, outcome_var, 'indices', 
                        f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                    )
                    with open(subsample_file, 'rb') as f:
                        original_indices = pickle.load(f)
                    subsample_data = df.iloc[original_indices].reset_index(drop=True)
                    
                    for fold_idx in range(n_folds):
                        # Load validation indices
                        valid_file = os.path.join(
                            model_dir, outcome_var, 'indices',
                            f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
                        )
                        
                        with open(valid_file, 'rb') as f:
                            valid_indices = pickle.load(f)
                        
                        # Get validation data
                        X_valid = subsample_data[feature_cols].iloc[valid_indices]
                        y_valid = subsample_data[outcome_var].iloc[valid_indices]
                        sens_valid = subsample_data[sens_col].iloc[valid_indices]
                        
                        # Get Super Learner predictions
                        y_pred = _get_super_learner_predictions(
                            X_valid, continuous_features, subsample_idx, fold_idx, 
                            model_dir, outcome_var
                        )
                        
                        # Collect data
                        all_y_true.extend(y_valid.values)
                        all_y_pred.extend(y_pred)
                        all_sensitive.extend(sens_valid.values)
            
            # Convert to numpy arrays
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_sensitive = np.array(all_sensitive)
            
            # Remove any missing values
            valid_mask = ~(pd.isna(all_sensitive) | pd.isna(all_y_true) | pd.isna(all_y_pred))
            all_y_true = all_y_true[valid_mask]
            all_y_pred = all_y_pred[valid_mask]
            all_sensitive = all_sensitive[valid_mask]

            # Convert predictions to binary
            y_pred_binary = (all_y_pred >= threshold).astype(int)
            
            # Calculate fairness metrics using fairlearn
            dpr = demographic_parity_ratio(
                y_true=all_y_true,
                y_pred=y_pred_binary,
                sensitive_features=all_sensitive
            )
            eor = equalized_odds_ratio(
                y_true=all_y_true,
                y_pred=y_pred_binary,
                sensitive_features=all_sensitive
            )
            
            # Count group sizes and prediction rates
            unique_vals = np.unique(all_sensitive)
            group_counts = pd.Series(all_sensitive).value_counts().to_dict()
            
            # Calculate percentage predicted as positive for each group
            group_0_mask = all_sensitive == unique_vals[0]
            group_1_mask = all_sensitive == unique_vals[1]
            pct_pred_pos_group_0 = (y_pred_binary[group_0_mask].mean() * 100) if group_0_mask.any() else np.nan
            pct_pred_pos_group_1 = (y_pred_binary[group_1_mask].mean() * 100) if group_1_mask.any() else np.nan
            if len(unique_vals) > 2:
                group_2_mask = all_sensitive == unique_vals[2]
                pct_pred_pos_group_2 = (y_pred_binary[group_2_mask].mean() * 100) if group_2_mask.any() else np.nan
            else:
                pct_pred_pos_group_2 = np.nan

            results.append({
                'Outcome': outcome_var,
                'Sensitive_Attribute': sens_col,
                'DPR': dpr,
                'EOR': eor,
                'N_total': len(all_y_true),
                'N_group_0': group_counts.get(unique_vals[0], 0) if len(unique_vals) >= 1 else 0,
                'N_group_1': group_counts.get(unique_vals[1], 0) if len(unique_vals) >= 2 else 0,
                'N_group_2': group_counts.get(unique_vals[2], 0) if len(unique_vals) >= 3 else 0,
                'Pct_pred_pos_group_0': pct_pred_pos_group_0,
                'Pct_pred_pos_group_1': pct_pred_pos_group_1,
                'Pct_pred_pos_group_2': pct_pred_pos_group_2
            })
    
    return pd.DataFrame(results)

def _get_super_learner_predictions(X_valid, continuous_features, subsample_idx, fold_idx, 
                                  model_dir, outcome_var):
    """Get Super Learner predictions."""
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Get CatBoost predictions
    catboost_prefix = os.path.join(
        model_dir, outcome_var, 'catboost_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(f'{catboost_prefix}_model.pkl')
    
    with open(f'{catboost_prefix}_scaler.pkl', 'rb') as f:
        catboost_scaler = pickle.load(f)
    
    X_catboost = X_valid.copy()
    X_catboost[continuous_features] = catboost_scaler.transform(X_valid[continuous_features])
    catboost_preds = catboost_model.predict_proba(X_catboost)[:, 1]
    
    # Get Keras predictions
    keras_prefix = os.path.join(
        model_dir, outcome_var, 'keras_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    
    keras_model = load_model(f'{keras_prefix}_model.keras')
    
    with open(f'{keras_prefix}_scaler.pkl', 'rb') as f:
        keras_scaler = pickle.load(f)
    
    X_keras = X_valid.copy()
    X_keras[continuous_features] = keras_scaler.transform(X_valid[continuous_features])
    keras_preds = keras_model.predict(X_keras.to_numpy(dtype=np.float32), verbose=0).flatten()
    
    tf.keras.backend.clear_session()
    
    # Stack as meta-features
    meta_features = np.vstack([catboost_preds, keras_preds]).T
    
    # Load and apply meta-learner
    meta_learner_path = os.path.join(
        model_dir, outcome_var,
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_super_learner.pkl'
    )
    
    with open(meta_learner_path, 'rb') as f:
        meta_learner = pickle.load(f)
    
    return meta_learner.predict_proba(meta_features)[:, 1]

def calculate_super_learner_shap_values(
    df: pd.DataFrame,
    feature_cols: List[str],
    continuous_features: List[str],
    outcome_var: str,
    model_dir: str,
    n_subsamples: int,
    n_folds: int
) -> pd.DataFrame:
    """
    Calculate Super Learner SHAP values across all subsamples and folds.
    
    Args:
        df: Dataset containing features and targets
        feature_cols: List of feature column names
        continuous_features: List of continuous feature column names
        outcome_var: Target variable name
        model_dir: Directory containing trained models
        n_subsamples: Number of subsamples
        n_folds: Number of folds
        
    Returns:
        DataFrame with averaged SHAP values for each sample and feature
    """
    print(f"Computing Super Learner SHAP values for {outcome_var}")
    
    # Initialize storage for SHAP values
    max_index = len(df)
    n_features = len(feature_cols)
    
    # Store SHAP values and counts for averaging
    shap_accumulator = np.zeros((max_index, n_features))
    count_accumulator = np.zeros((max_index, 1))
    
    # Iterate over subsamples
    for subsample_idx in range(n_subsamples):
        print(f"Processing subsample {subsample_idx + 1}/{n_subsamples}")
        
        # Load subsample data
        subsample_file = os.path.join(
            model_dir, outcome_var, 'indices',
            f'subsample_{str(subsample_idx).zfill(3)}.pkl'
        )
        
        with open(subsample_file, 'rb') as f:
            original_indices = pickle.load(f)
        
        subsample_data = df.iloc[original_indices].reset_index(drop=True)
        
        # Iterate over folds
        for fold_idx in range(n_folds):
            print(f"  Processing fold {fold_idx + 1}/{n_folds}")
            
            # Load validation indices
            valid_file = os.path.join(
                model_dir, outcome_var, 'indices',
                f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
            )
            
            with open(valid_file, 'rb') as f:
                valid_indices = pickle.load(f)
            
            # Get validation data
            X_valid = subsample_data[feature_cols].iloc[valid_indices]
            
            # Get original dataset indices for this validation set
            validation_original_indices = [original_indices[i] for i in valid_indices]
            
            # Load meta-learner weights for this specific fold
            meta_learner_path = os.path.join(
                model_dir, outcome_var,
                f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_super_learner.pkl'
            )
            
            with open(meta_learner_path, 'rb') as f:
                meta_learner = pickle.load(f)
            
            catboost_weight, keras_weight = meta_learner.coef_[0]
            
            # Compute CatBoost SHAP values for validation set
            catboost_shap = _compute_catboost_shap(
                X_valid, continuous_features, subsample_idx, fold_idx, model_dir, outcome_var
            )
            
            # Compute Keras SHAP values for validation set
            keras_shap = _compute_keras_shap(
                X_valid, continuous_features, subsample_idx, fold_idx, model_dir, outcome_var
            )
            
            # Combine using this fold's meta-learner weights
            combined_shap = catboost_weight * catboost_shap + keras_weight * keras_shap
            
            # Store in accumulator using original dataset indices
            for i, orig_idx in enumerate(validation_original_indices):
                shap_accumulator[orig_idx] += combined_shap[i]
                count_accumulator[orig_idx] += 1
    
    # Average across subsamples/folds
    mask = count_accumulator.flatten() > 0
    averaged_shap = np.zeros_like(shap_accumulator)
    averaged_shap[mask] = shap_accumulator[mask] / count_accumulator[mask]
    
    # Create DataFrame
    shap_df = pd.DataFrame(averaged_shap, columns=feature_cols)
    shap_df['count'] = count_accumulator.flatten()
    
    print(f"SHAP calculation complete. {mask.sum()} samples have SHAP values.")
    
    return shap_df


def _compute_catboost_shap(
    X_valid: pd.DataFrame,
    continuous_features: List[str],
    subsample_idx: int,
    fold_idx: int,
    model_dir: str,
    outcome_var: str
) -> np.ndarray:
    """Compute CatBoost SHAP values for validation set."""
    # Check cache first
    cache_path = os.path.join(
        model_dir, outcome_var, 'catboost_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_catboost_shap.npy'
    )
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    # Compute SHAP values
    model_prefix = os.path.join(
        model_dir, outcome_var, 'catboost_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    
    model = CatBoostClassifier()
    model.load_model(f'{model_prefix}_model.pkl')
    
    with open(f'{model_prefix}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale data
    X_scaled = X_valid.copy()
    X_scaled[continuous_features] = scaler.transform(X_valid[continuous_features])
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Get positive class for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Cache results
    np.save(cache_path, shap_values)
    
    return shap_values


def _compute_keras_shap(
    X_valid: pd.DataFrame,
    continuous_features: List[str],
    subsample_idx: int,
    fold_idx: int,
    model_dir: str,
    outcome_var: str
) -> np.ndarray:
    """Compute Keras SHAP values for validation set."""
    # Check cache first
    cache_path = os.path.join(
        model_dir, outcome_var, 'keras_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_keras_shap.npy'
    )
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    # Compute SHAP values
    model_prefix = os.path.join(
        model_dir, outcome_var, 'keras_tuning',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'
    )
    
    model = load_model(f'{model_prefix}_model.keras')
    
    with open(f'{model_prefix}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale data
    X_scaled = X_valid.copy()
    X_scaled[continuous_features] = scaler.transform(X_valid[continuous_features])
    
    # Create prediction function
    def predict_fn(x):
        return model.predict(x.astype(np.float32), verbose=0).flatten()
    
    # Use background dataset for KernelExplainer
    background_size = min(50, len(X_scaled))
    background = X_scaled.sample(background_size, random_state=42).values
    
    # Calculate SHAP values
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_scaled.values)
    
    # Clean up TensorFlow session
    tf.keras.backend.clear_session()
    
    # Cache results
    np.save(cache_path, shap_values)
    
    return shap_values


def plot_shap_beeswarm(
    shap_df: pd.DataFrame,
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    save_dir: str,
    top_n: int = 10
) -> None:
    """
    Generate SHAP beeswarm plot from SHAP DataFrame.
    
    Args:
        shap_df: DataFrame with SHAP values (from calculate_super_learner_shap_values)
        df: Original dataset for feature values
        feature_cols: List of feature names
        outcome_var: Outcome variable name for file naming
        save_dir: Directory to save the plot
        top_n: Number of top features to display
    """
    # Set font styling
    plt.rcParams["font.family"] = "Scala Sans Pro"
    plt.rcParams["font.size"] = 12
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if plot already exists
    plot_path = os.path.join(save_dir, f'{outcome_var}_shap.png')
    plot_path_labels = os.path.join(save_dir, f'{outcome_var}_shap_labels.png')
    
    # Get samples that have SHAP values
    valid_mask = shap_df['count'] > 0
    shap_values = shap_df.loc[valid_mask, feature_cols].values
    feature_values = df.loc[valid_mask, feature_cols].values
    
    if len(shap_values) == 0:
        print(f"No SHAP values available for {outcome_var}")
        return
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=feature_values,
        feature_names=feature_cols
    )
    
    # Create beeswarm plot - labels
    plt.figure()
    shap.plots.beeswarm(
        shap_explanation,
        show=False,
        s=16,
        max_display=top_n,
        group_remaining_features=False,
        color='coolwarm'
    )
    a = plt.gcf().get_children()
    a[1].set_xlabel('', visible=False)
    a[1].set_xlim([-np.max(np.abs(a[1].get_xlim())), np.max(np.abs(a[1].get_xlim()))])
    plt.savefig(plot_path_labels, bbox_inches='tight', dpi=500)
    plt.close()

    # Create beeswarm plot - no labels
    plt.figure()
    shap_explanation.feature_names = ['' for feature in shap_explanation.feature_names]
    shap.plots.beeswarm(
        shap_explanation,
        show=False,
        s=1,
        max_display=top_n,
        group_remaining_features=False,
        color='coolwarm',
        plot_size=(1.5,2),
        color_bar=False
    )
    a = plt.gcf().get_children()
    a[1].set_ylabel('', visible=False)
    a[1].set_xlabel('', visible=False)
    a[1].set_xticklabels('')
    a[1].set_xlim([-np.max(np.abs(a[1].get_xlim())), np.max(np.abs(a[1].get_xlim()))])
    plt.savefig(plot_path, bbox_inches='tight', dpi=500)
    plt.close()
    
    print(f"SHAP beeswarm plot saved: {plot_path}")

