"""
Data sampling utilities for experiments.

This module provides functions for undersampling datasets to balance classes
and generating cross-validation indices with stratification.
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold


def create_balanced_sample(
    df: pd.DataFrame, 
    target_col: str, 
    random_state: int, 
    id_col: str
) -> Tuple[List[int], pd.DataFrame]:
    """
    Create a balanced dataset by undersampling the majority class.
    
    Randomly samples from the negative class to match the number of positive samples,
    creating a balanced dataset for training.
    
    Args:
        df: Input dataset containing features and target
        target_col: Name of the binary target column (0/1)
        random_state: Random seed for reproducibility
        id_col: Column name used to sort the final dataset
        
    Returns:
        original_indices: List of original row indices from the input dataframe
        balanced_data: Balanced dataset with equal positive and negative samples
        
    Example:
        >>> indices, balanced_df = create_balanced_sample(
        ...     df, 'outcome', random_state=42, id_col='subject_id'
        ... )
    """
    # Separate positive and negative samples
    positive_samples = df.loc[df[target_col] == 1]
    negative_samples = df.loc[df[target_col] == 0]
    
    # Undersample negative class to match positive class size
    n_positive = len(positive_samples)
    negative_undersampled = negative_samples.sample(
        n=n_positive, 
        random_state=random_state
    )
    
    # Combine and sort by ID column
    balanced_data = pd.concat([positive_samples, negative_undersampled], axis=0)
    balanced_data = balanced_data.sort_values(id_col)
    
    # Store original indices and reset index
    original_indices = list(balanced_data.index)
    balanced_data = balanced_data.reset_index(drop=True)
    
    return original_indices, balanced_data


def generate_stratified_cv_splits(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5, 
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified cross-validation splits for binary classification.
    
    Creates stratified k-fold cross-validation splits that maintain the same
    proportion of positive and negative samples in each fold.
    
    Args:
        X: Feature matrix
        y: Binary target vector
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of tuples containing (train_indices, validation_indices) for each fold
        
    Example:
        >>> cv_splits = generate_stratified_cv_splits(X, y, n_splits=5)
        >>> for fold, (train_idx, val_idx) in enumerate(cv_splits):
        ...     print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))


def save_indices_to_file(indices: List[int], filepath: str) -> None:
    """
    Save a list of indices to a pickle file.
    
    Args:
        indices: List of integer indices to save
        filepath: Path where the pickle file should be saved
    """
    with open(filepath, 'wb') as f:
        pickle.dump(indices, f)


def load_indices_from_file(filepath: str) -> List[int]:
    """
    Load a list of indices from a pickle file.
    
    Args:
        filepath: Path to the pickle file containing indices
        
    Returns:
        List of integer indices
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)