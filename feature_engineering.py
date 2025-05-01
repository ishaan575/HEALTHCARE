import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

def calculate_feature_importance(X, y, method='chi2'):
    """
    Calculate feature importance using statistical tests or model-based methods.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        method (str): Method for importance calculation ('chi2' or 'model')
        
    Returns:
        pandas.Series: Feature importances
    """
    if method == 'chi2':
        # Chi-squared test requires non-negative features
        # Scale features to [0, 1] range
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate chi-squared stats
        chi2_stats, _ = chi2(X_scaled, y)
        
        # Create Series with feature importances
        importances = pd.Series(chi2_stats, index=X.columns)
        
    elif method == 'model':
        # Use Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Create Series with feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort importances in descending order
    importances = importances.sort_values(ascending=False)
    
    return importances

def engineer_features(X_train, X_test, selected_features=None, 
                     create_interactions=False, max_interactions=5,
                     create_polynomial=False):
    """
    Engineer new features based on existing ones.
    
    Args:
        X_train (pandas.DataFrame): Training feature matrix
        X_test (pandas.DataFrame): Testing feature matrix
        selected_features (list): List of features to use (if None, use all)
        create_interactions (bool): Whether to create interaction features
        max_interactions (int): Maximum number of interaction features to create
        create_polynomial (bool): Whether to create polynomial features
        
    Returns:
        tuple: (X_train_new, X_test_new, feature_names)
    """
    # Make copies to avoid modifying originals
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # Use selected features if provided
    if selected_features is not None:
        X_train_new = X_train_new[selected_features]
        X_test_new = X_test_new[selected_features]
    
    # Create interaction features if requested
    if create_interactions and len(X_train_new.columns) >= 2:
        # Get pairs of features
        feature_pairs = list(combinations(X_train_new.columns, 2))
        
        # Limit to max_interactions if specified
        if max_interactions > 0 and max_interactions < len(feature_pairs):
            feature_pairs = feature_pairs[:max_interactions]
        
        # Create interaction features
        for feat1, feat2 in feature_pairs:
            interaction_name = f"{feat1}_x_{feat2}"
            X_train_new[interaction_name] = X_train_new[feat1] * X_train_new[feat2]
            X_test_new[interaction_name] = X_test_new[feat1] * X_test_new[feat2]
    
    # Create polynomial features (squared terms) if requested
    if create_polynomial:
        for feature in X_train_new.columns.tolist():
            # Skip interaction features for squaring
            if "_x_" not in feature:
                squared_name = f"{feature}_squared"
                X_train_new[squared_name] = X_train_new[feature] ** 2
                X_test_new[squared_name] = X_test_new[feature] ** 2
    
    # Return the engineered features and their names
    feature_names = X_train_new.columns.tolist()
    
    return X_train_new, X_test_new, feature_names

def select_best_features(X, y, k=10, method='chi2'):
    """
    Select the best k features based on statistical tests.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        k (int): Number of features to select
        method (str): Method for feature selection ('chi2')
        
    Returns:
        list: Names of the best k features
    """
    # Ensure k is not larger than the number of features
    k = min(k, X.shape[1])
    
    if method == 'chi2':
        # Chi-squared test requires non-negative features
        # Scale features to [0, 1] range
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select best k features
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_scaled, y)
        
        # Get the selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # Get the feature names
        selected_features = X.columns[selected_indices].tolist()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return selected_features

def create_binned_features(X_train, X_test, features_to_bin, num_bins=5):
    """
    Create binned versions of numerical features.
    
    Args:
        X_train (pandas.DataFrame): Training feature matrix
        X_test (pandas.DataFrame): Testing feature matrix
        features_to_bin (list): List of features to bin
        num_bins (int): Number of bins to create
        
    Returns:
        tuple: (X_train_new, X_test_new)
    """
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    for feature in features_to_bin:
        if feature in X_train.columns:
            # Compute the bin edges based on training data
            bin_edges = pd.qcut(X_train[feature], q=num_bins, retbins=True, duplicates='drop')[1]
            
            # Create binned version of the feature
            binned_feature_name = f"{feature}_binned"
            X_train_new[binned_feature_name] = pd.cut(X_train[feature], bins=bin_edges, labels=False, include_lowest=True)
            X_test_new[binned_feature_name] = pd.cut(X_test[feature], bins=bin_edges, labels=False, include_lowest=True)
            
            # Handle out-of-bounds values in test set
            X_test_new[binned_feature_name] = X_test_new[binned_feature_name].fillna(0)
    
    return X_train_new, X_test_new
