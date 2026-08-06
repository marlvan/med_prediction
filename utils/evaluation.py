"""
Model evaluation utilities for binary classification tasks.

This module provides functions for computing classification metrics,
ROC curve statistics, and summary statistics across multiple experiments,
along with the fairness and calibration analyses, which are computed from
predictions collected once from the cached models.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Sequence
import pickle
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
import shap
from catboost import CatBoostClassifier
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)

# Green, purple, pink, amber - used by all figures so the medication classes are
# coloured consistently across the ROC and calibration plots
PALETTE = ['#8dbd61', '#6f459e', '#f66f8d', '#ffb643']

OUTCOME_LABELS = {
    'stimulant_outcome': 'Stimulant',
    'antidepressant_outcome': 'Anti-depressant',
    'antipsychotic_outcome': 'Anti-psychotic',
    'nonstimulant_outcome': 'Non-stimulant',
}

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
    palette = PALETTE

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


def plot_calibration(
    save_dir: str,
    curves: pd.DataFrame,
    cohort: str,
    outcome_order: Optional[List[str]] = None
) -> None:
    """
    Plot calibration curves for each outcome in a single cohort.

    Plots the observed event rate against the mean predicted probability within
    each bin, with a diagonal reference line. Uses consistent styling with
    plot_roc_auc.

    Args:
        save_dir: Directory path to save the calibration plot
        curves: DataFrame of curve points from calculate_calibration, with columns
                ['Cohort', 'Outcome', 'Mean_Predicted', 'Observed', 'N']
        cohort: Cohort to plot
        outcome_order: Outcomes to plot, in order (default: the three primary
                       medication classes)

    Saves:
        'calibration.png' in the specified directory if it doesn't already exist

    Example:
        >>> plot_calibration(pond_output_dir, calibration_curves, 'POND')
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Check if plot already exists
    plot_path = os.path.join(save_dir, 'calibration.png')
    if os.path.exists(plot_path):
        return

    # Set font and styling
    plt.rcParams["font.family"] = "Scala Sans Pro"
    plt.rcParams["font.size"] = 12

    # Define color palette
    palette = PALETTE

    # Restrict to this cohort
    cohort_curves = curves[curves['Cohort'] == cohort]
    if cohort_curves.empty:
        print(f"No calibration curve data for {cohort}")
        return

    outcomes = outcome_order or [
        'stimulant_outcome', 'antidepressant_outcome', 'antipsychotic_outcome'
    ]

    # Create figure
    plt.figure(figsize=(3, 3))

    # Plot one curve per outcome
    for i, outcome in enumerate(outcomes):
        points = cohort_curves[cohort_curves['Outcome'] == outcome]
        if points.empty:
            continue
        points = points.sort_values('Mean_Predicted')
        plt.plot(points['Mean_Predicted'], points['Observed'],
                 color=palette[i % len(palette)], linewidth=1, marker='o',
                 markersize=3, label=OUTCOME_LABELS.get(outcome, outcome))

    # Add diagonal reference line
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], color='black', linewidth=0.5, linestyle='--')

    # Set labels and formatting
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed proportion')
    plt.legend(fontsize=4, loc='upper left')

    # Set aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(0, 1), ylim=(0, 1))

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

def _evaluation_positions(
    subsample_data: pd.DataFrame,
    indices_dir: str,
    outcome_var: str,
    subsample_idx: int,
    fold_idx: int,
    external: bool
) -> np.ndarray:
    """
    Get the row positions within a balanced subsample that a fold model is evaluated on.

    Internally validated cohorts use the fold's held-out validation rows. Externally
    validated cohorts use every row, matching SuperLearner.evaluate_external, which
    applies each fold's model to the entire balanced external sample and therefore
    writes no fold-level index files.

    Args:
        subsample_data: Balanced subsample for this cohort and outcome
        indices_dir: Directory containing this cohort's saved indices
        outcome_var: Target variable name
        subsample_idx: Subsample number
        fold_idx: Fold number
        external: Whether the models were trained on a different cohort

    Returns:
        Array of row positions within subsample_data
    """
    if external:
        return np.arange(len(subsample_data))

    valid_file = os.path.join(
        indices_dir, outcome_var, 'indices',
        f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}_valid.pkl'
    )

    with open(valid_file, 'rb') as f:
        return np.asarray(pickle.load(f))


def collect_super_learner_predictions(
    cohort_data_list: List[dict],
    outcome_vars: List[str],
    extra_cols: Sequence[str] = (),
    cache_path: Optional[str] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Collect Super Learner predicted probabilities for every cohort, outcome, subsample
    and fold.

    Loading the cached models is the expensive step, so predictions are collected once
    here and both the fairness and calibration analyses are computed from the result.

    Args:
        cohort_data_list: List of dictionaries, each containing:
            - 'name': cohort name (e.g., 'POND', 'PPP')
            - 'df': dataset DataFrame
            - 'feature_cols': feature column names
            - 'continuous_features': continuous feature names
            - 'model_dir': directory containing trained models (the training cohort's
              directory for externally validated cohorts)
            - 'indices_dir': directory containing this cohort's balanced-subsample and
              fold indices (optional; defaults to 'model_dir')
            - 'external': True if the models were trained on a different cohort
              (optional; default False)
            - 'n_subsamples': number of subsamples
            - 'n_folds': number of folds
        outcome_vars: List of outcome variables
        extra_cols: Additional columns to carry through (e.g., sensitive attributes)
        cache_path: If provided, load from or save to this pickle
        overwrite: Recompute even if the cache already exists

    Returns:
        DataFrame with columns ['Cohort', 'Outcome', 'Subsample', 'Fold', 'Index',
        'y_true', 'y_prob'] plus one column per entry in extra_cols

    Example:
        >>> predictions = collect_super_learner_predictions(
        ...     research_cohorts, outcome_vars, extra_cols=sensitive_cols
        ... )
    """
    if cache_path and os.path.exists(cache_path) and not overwrite:
        print(f"Loading cached predictions: {cache_path}")
        return pd.read_pickle(cache_path)

    records = []

    for outcome_var in outcome_vars:
        for cohort_data in cohort_data_list:
            name = cohort_data['name']
            df = cohort_data['df'].reset_index(drop=True)
            feature_cols = cohort_data['feature_cols']
            continuous_features = cohort_data['continuous_features']
            model_dir = cohort_data['model_dir']
            indices_dir = cohort_data.get('indices_dir', model_dir)
            external = cohort_data.get('external', False)
            n_subsamples = cohort_data['n_subsamples']
            n_folds = cohort_data['n_folds']

            print(f"Collecting predictions for {outcome_var}, {name} "
                  f"({'external' if external else 'internal'})")

            for subsample_idx in range(n_subsamples):
                # Load subsample data
                subsample_file = os.path.join(
                    indices_dir, outcome_var, 'indices',
                    f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                )

                with open(subsample_file, 'rb') as f:
                    original_indices = np.asarray(pickle.load(f))

                subsample_data = df.iloc[original_indices].reset_index(drop=True)

                for fold_idx in range(n_folds):
                    positions = _evaluation_positions(
                        subsample_data, indices_dir, outcome_var,
                        subsample_idx, fold_idx, external
                    )

                    X_valid = subsample_data[feature_cols].iloc[positions]
                    y_valid = subsample_data[outcome_var].iloc[positions].to_numpy()

                    y_scores = _get_super_learner_predictions(
                        X_valid, continuous_features, subsample_idx, fold_idx,
                        model_dir, outcome_var
                    )

                    fold_records = {
                        'Cohort': name,
                        'Outcome': outcome_var,
                        'Subsample': subsample_idx,
                        'Fold': fold_idx,
                        'Index': original_indices[positions],
                        'y_true': y_valid.astype(float),
                        'y_prob': np.asarray(y_scores, dtype=float)
                    }

                    for col in extra_cols:
                        fold_records[col] = (
                            subsample_data[col].iloc[positions].to_numpy()
                            if col in subsample_data.columns
                            else np.full(len(positions), np.nan)
                        )

                    records.append(pd.DataFrame(fold_records))

    predictions = pd.concat(records, ignore_index=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        predictions.to_pickle(cache_path)
        print(f"Cached predictions: {cache_path} ({len(predictions)} rows)")

    return predictions


def _all_model_predictions(
    X_valid: pd.DataFrame,
    continuous_features: List[str],
    subsample_idx: int,
    fold_idx: int,
    model_dir: str,
    outcome_var: str
) -> Dict[str, np.ndarray]:
    """
    Get predictions from both base learners and the Super Learner in a single load.

    _get_super_learner_predictions computes the base learner outputs internally but
    returns only the ensemble. The model comparison needs all three, and loading the
    Keras model is the expensive step, so they are produced together here.

    Args:
        X_valid: Validation features
        continuous_features: List of continuous feature column names
        subsample_idx: Subsample number
        fold_idx: Fold number
        model_dir: Directory containing trained models
        outcome_var: Target variable name

    Returns:
        Dictionary with 'catboost', 'keras' and 'super' predicted probabilities
    """
    tag = f'subsample_{str(subsample_idx).zfill(3)}_fold_{str(fold_idx).zfill(2)}'

    # Get CatBoost predictions
    catboost_prefix = os.path.join(model_dir, outcome_var, 'catboost_tuning', tag)

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(f'{catboost_prefix}_model.pkl')

    with open(f'{catboost_prefix}_scaler.pkl', 'rb') as f:
        catboost_scaler = pickle.load(f)

    X_catboost = X_valid.copy()
    X_catboost[continuous_features] = catboost_scaler.transform(X_valid[continuous_features])
    catboost_preds = catboost_model.predict_proba(X_catboost)[:, 1]

    # Get Keras predictions
    keras_prefix = os.path.join(model_dir, outcome_var, 'keras_tuning', tag)

    keras_model = load_model(f'{keras_prefix}_model.keras')

    with open(f'{keras_prefix}_scaler.pkl', 'rb') as f:
        keras_scaler = pickle.load(f)

    X_keras = X_valid.copy()
    X_keras[continuous_features] = keras_scaler.transform(X_valid[continuous_features])
    keras_preds = keras_model.predict(X_keras.to_numpy(dtype=np.float32), verbose=0).flatten()

    tf.keras.backend.clear_session()

    # Load and apply meta-learner
    meta_learner_path = os.path.join(model_dir, outcome_var, f'{tag}_super_learner.pkl')

    with open(meta_learner_path, 'rb') as f:
        meta_learner = pickle.load(f)

    meta_features = np.vstack([catboost_preds, keras_preds]).T
    super_preds = meta_learner.predict_proba(meta_features)[:, 1]

    return {'catboost': catboost_preds, 'keras': keras_preds, 'super': super_preds}


def collect_model_predictions(
    cohort_data_list: List[dict],
    outcome_vars: List[str],
    cache_path: Optional[str] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Collect CatBoost, Keras and Super Learner predictions for the model comparison.

    Args:
        cohort_data_list: List of dictionaries, using the same keys as
            collect_super_learner_predictions
        outcome_vars: List of outcome variables
        cache_path: If provided, load from or save to this pickle
        overwrite: Recompute even if the cache already exists

    Returns:
        DataFrame with columns ['Cohort', 'Outcome', 'Subsample', 'Fold', 'Index',
        'y_true', 'prob_catboost', 'prob_keras', 'prob_super']
    """
    if cache_path and os.path.exists(cache_path) and not overwrite:
        print(f"Loading cached model predictions: {cache_path}")
        return pd.read_pickle(cache_path)

    records = []

    for outcome_var in outcome_vars:
        for cohort_data in cohort_data_list:
            name = cohort_data['name']
            df = cohort_data['df'].reset_index(drop=True)
            feature_cols = cohort_data['feature_cols']
            continuous_features = cohort_data['continuous_features']
            model_dir = cohort_data['model_dir']
            indices_dir = cohort_data.get('indices_dir', model_dir)
            external = cohort_data.get('external', False)
            n_subsamples = cohort_data['n_subsamples']
            n_folds = cohort_data['n_folds']

            print(f"Collecting model predictions for {outcome_var}, {name} "
                  f"({'external' if external else 'internal'})")

            for subsample_idx in range(n_subsamples):
                subsample_file = os.path.join(
                    indices_dir, outcome_var, 'indices',
                    f'subsample_{str(subsample_idx).zfill(3)}.pkl'
                )

                with open(subsample_file, 'rb') as f:
                    original_indices = np.asarray(pickle.load(f))

                subsample_data = df.iloc[original_indices].reset_index(drop=True)

                for fold_idx in range(n_folds):
                    positions = _evaluation_positions(
                        subsample_data, indices_dir, outcome_var,
                        subsample_idx, fold_idx, external
                    )

                    X_valid = subsample_data[feature_cols].iloc[positions]
                    y_valid = subsample_data[outcome_var].iloc[positions].to_numpy()

                    preds = _all_model_predictions(
                        X_valid, continuous_features, subsample_idx, fold_idx,
                        model_dir, outcome_var
                    )

                    records.append(pd.DataFrame({
                        'Cohort': name,
                        'Outcome': outcome_var,
                        'Subsample': subsample_idx,
                        'Fold': fold_idx,
                        'Index': original_indices[positions],
                        'y_true': y_valid.astype(float),
                        'prob_catboost': preds['catboost'],
                        'prob_keras': preds['keras'],
                        'prob_super': preds['super']
                    }))

    predictions = pd.concat(records, ignore_index=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        predictions.to_pickle(cache_path)
        print(f"Cached model predictions: {cache_path} ({len(predictions)} rows)")

    return predictions


def average_fold_predictions(
    predictions: pd.DataFrame,
    extra_cols: Sequence[str] = ()
) -> pd.DataFrame:
    """
    Average predicted probabilities across folds so each participant contributes once
    per subsample.

    For internally validated cohorts this is a no-op, since a participant appears in
    exactly one fold's validation set. For externally validated cohorts it collapses
    the fold models into a single prediction per participant, which keeps each cohort's
    weight in a pooled analysis proportional to its balanced sample size rather than to
    sample size multiplied by the number of folds.

    Args:
        predictions: Output of collect_super_learner_predictions
        extra_cols: Additional columns to carry through

    Returns:
        DataFrame with one row per cohort, outcome, subsample and participant
    """
    group_cols = ['Cohort', 'Outcome', 'Subsample', 'Index']
    aggregations = {'y_true': 'first', 'y_prob': 'mean'}
    aggregations.update({col: 'first' for col in extra_cols})

    return predictions.groupby(group_cols, as_index=False, observed=True).agg(aggregations)


def _group_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate the rates underlying the fairness ratios within a single group.

    Args:
        y_true: Ground truth binary labels
        y_pred: Binary predictions
        mask: Boolean mask selecting the group

    Returns:
        Tuple of (percentage predicted positive, TPR, FPR, accuracy)
    """
    if not mask.any():
        return (np.nan, np.nan, np.nan, np.nan)

    group_true, group_pred = y_true[mask], y_pred[mask]

    pct_pos = group_pred.mean() * 100
    tpr = group_pred[group_true == 1].mean() * 100 if (group_true == 1).any() else np.nan
    fpr = group_pred[group_true == 0].mean() * 100 if (group_true == 0).any() else np.nan
    accuracy = (group_pred == group_true).mean() * 100

    return pct_pos, tpr, fpr, accuracy


def calculate_fairness_metrics(
    predictions: pd.DataFrame,
    sensitive_cols: List[str],
    outcome_vars: Optional[List[str]] = None,
    threshold: float = 0.5,
    baseline: bool = False
) -> pd.DataFrame:
    """
    Calculate DPR and EOR from collected predictions, pooled across cohorts.

    Args:
        predictions: Output of collect_super_learner_predictions, optionally passed
            through average_fold_predictions
        sensitive_cols: List of sensitive attribute column names
        outcome_vars: List of outcome variables (default: all present)
        threshold: Decision threshold
        baseline: If True, use the observed labels in place of the predictions, giving
            the baseline DPR of actual prescribing. EOR requires both predicted and
            actual labels and is returned as NaN in this case.

    Returns:
        DataFrame with DPR and EOR for each outcome/sensitive attribute combination,
        the per-group rates underlying them, and pairwise ratios for attributes with
        three groups

    Example:
        >>> model_fairness = calculate_fairness_metrics(
        ...     research_avg, sensitive_cols, outcome_vars
        ... )
    """
    results = []

    for outcome_var in (outcome_vars or sorted(predictions['Outcome'].unique())):
        subset = predictions[predictions['Outcome'] == outcome_var]

        if subset.empty:
            continue

        for sens_col in sensitive_cols:
            all_y_true = subset['y_true'].to_numpy()
            all_sensitive = subset[sens_col].to_numpy()

            # Use the observed labels as the predictions for the baseline
            if baseline:
                y_pred_binary = all_y_true.astype(int)
            else:
                y_pred_binary = (subset['y_prob'].to_numpy() >= threshold).astype(int)

            # Remove any missing values
            valid_mask = ~(pd.isna(all_sensitive) | pd.isna(all_y_true))
            all_y_true = all_y_true[valid_mask]
            y_pred_binary = y_pred_binary[valid_mask]
            all_sensitive = all_sensitive[valid_mask]

            if len(np.unique(all_sensitive)) < 2:
                continue

            # Calculate fairness metrics using fairlearn
            dpr = demographic_parity_ratio(
                y_true=all_y_true,
                y_pred=y_pred_binary,
                sensitive_features=all_sensitive
            )

            if baseline:
                eor = np.nan
            else:
                eor = equalized_odds_ratio(
                    y_true=all_y_true,
                    y_pred=y_pred_binary,
                    sensitive_features=all_sensitive
                )

            result = {
                'Outcome': outcome_var,
                'Sensitive_Attribute': sens_col,
                'DPR': dpr,
                'EOR': eor,
                'N_total': len(all_y_true),
            }

            # Count group sizes and prediction rates
            unique_vals = np.unique(all_sensitive)

            for i in range(3):
                if i < len(unique_vals):
                    mask = all_sensitive == unique_vals[i]
                    pct_pos, tpr, fpr, acc = _group_rates(all_y_true, y_pred_binary, mask)
                    result[f'Group_{i}_label'] = unique_vals[i]
                    result[f'N_group_{i}'] = int(mask.sum())
                    result[f'Pct_pred_pos_group_{i}'] = pct_pos
                    result[f'TPR_group_{i}'] = tpr
                    result[f'FPR_group_{i}'] = fpr
                    result[f'Accuracy_group_{i}'] = acc
                else:
                    result[f'Group_{i}_label'] = np.nan
                    result[f'N_group_{i}'] = 0
                    result[f'Pct_pred_pos_group_{i}'] = np.nan
                    result[f'TPR_group_{i}'] = np.nan
                    result[f'FPR_group_{i}'] = np.nan
                    result[f'Accuracy_group_{i}'] = np.nan

            # Pairwise DPR and EOR (only meaningful for 3+ groups)
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                if i < len(unique_vals) and j < len(unique_vals) and len(unique_vals) > 2:
                    pair_mask = ((all_sensitive == unique_vals[i]) |
                                 (all_sensitive == unique_vals[j]))
                    pair_sens = all_sensitive[pair_mask]
                    pair_yt = all_y_true[pair_mask]
                    pair_yp = y_pred_binary[pair_mask]
                    try:
                        pair_dpr = demographic_parity_ratio(
                            y_true=pair_yt, y_pred=pair_yp, sensitive_features=pair_sens
                        )
                    except Exception:
                        pair_dpr = np.nan
                    try:
                        pair_eor = np.nan if baseline else equalized_odds_ratio(
                            y_true=pair_yt, y_pred=pair_yp, sensitive_features=pair_sens
                        )
                    except Exception:
                        pair_eor = np.nan
                    result[f'DPR_pair_{i}_{j}'] = pair_dpr
                    result[f'EOR_pair_{i}_{j}'] = pair_eor
                else:
                    result[f'DPR_pair_{i}_{j}'] = np.nan
                    result[f'EOR_pair_{i}_{j}'] = np.nan

            results.append(result)

    return pd.DataFrame(results)


def _logit(p: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to the log-odds scale, clipped away from 0 and 1.

    Args:
        p: Predicted probabilities

    Returns:
        Log-odds of the clipped probabilities
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibration_in_the_large(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the calibration intercept with the slope fixed at one.

    Solves sum(y - sigmoid(logit(p) + a)) = 0 for a, which is monotone decreasing
    in a and so has a unique root.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities

    Returns:
        Calibration intercept (zero indicates that mean predicted risk matches
        mean observed risk)
    """
    log_odds = _logit(y_prob)

    def score(a):
        return float(np.sum(y_true - 1.0 / (1.0 + np.exp(-(log_odds + a)))))

    lower, upper = -30.0, 30.0

    if score(lower) * score(upper) > 0:
        return np.nan

    return float(brentq(score, lower, upper, xtol=1e-8))


def _calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate the calibration slope, intercept and Brier score.

    The slope is the coefficient of an unpenalized logistic regression of the outcome
    on the log-odds of the predicted probability. The intercept is calibration-in-the-
    large, estimated with the slope fixed at one. Perfect calibration corresponds to a
    slope of one and an intercept of zero.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary with 'Slope', 'Intercept', 'Brier' and 'N'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = float(np.mean((y_prob - y_true) ** 2))

    if len(np.unique(y_true)) < 2:
        return {'Slope': np.nan, 'Intercept': np.nan, 'Brier': brier, 'N': len(y_true)}

    log_odds = _logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(log_odds, y_true)

    return {
        'Slope': float(model.coef_[0][0]),
        'Intercept': _calibration_in_the_large(y_true, y_prob),
        'Brier': brier,
        'N': len(y_true)
    }


def _calibration_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate the observed event rate against the mean predicted probability.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of equal-sized bins

    Returns:
        DataFrame with one row per bin
    """
    data = pd.DataFrame({'y_true': np.asarray(y_true, dtype=float),
                         'y_prob': np.asarray(y_prob, dtype=float)})

    # Rank-based binning keeps the bins equally sized when there are many tied values
    data['Bin'] = pd.qcut(data['y_prob'].rank(method='first'), n_bins, labels=False)

    binned = data.groupby('Bin', observed=True).agg(
        Mean_Predicted=('y_prob', 'mean'),
        Observed=('y_true', 'mean'),
        N=('y_true', 'size')
    ).reset_index(drop=True)

    return binned


def calculate_calibration(
    predictions: pd.DataFrame,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate calibration metrics and curve points for each cohort and outcome.

    Calibration is assessed on the balanced evaluation samples, matching the design
    used for every other metric. Predicted probabilities are therefore calibrated to
    a 50% event rate, and Brier scores should be interpreted relative to the 0.25
    expected from an uninformative model on balanced data.

    Args:
        predictions: Output of collect_super_learner_predictions
        n_bins: Number of bins for the calibration curve

    Returns:
        Tuple of (metrics DataFrame, curve points DataFrame)

    Example:
        >>> calibration_metrics, calibration_curves = calculate_calibration(predictions)
    """
    metric_rows, curve_rows = [], []

    for (cohort, outcome), group in predictions.groupby(['Cohort', 'Outcome'],
                                                        observed=True):
        y_true = group['y_true'].to_numpy()
        y_prob = group['y_prob'].to_numpy()

        row = {'Cohort': cohort, 'Outcome': outcome}
        row.update(_calibration_metrics(y_true, y_prob))
        metric_rows.append(row)

        curve = _calibration_curve_points(y_true, y_prob, n_bins)
        curve['Cohort'], curve['Outcome'] = cohort, outcome
        curve_rows.append(curve)

    return pd.DataFrame(metric_rows), pd.concat(curve_rows, ignore_index=True)


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

