import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train, model_type='Logistic Regression', model_params=None):
    """
    Train a machine learning model on the given data.
    
    Args:
        X_train (pandas.DataFrame): Training feature matrix
        y_train (pandas.Series): Training target vector
        model_type (str): Type of model to train
        model_params (dict): Parameters for the model
        
    Returns:
        tuple: (trained_model, training_time)
    """
    # Set default parameters if none provided
    if model_params is None:
        model_params = {}
    
    # Initialize the appropriate model
    if model_type == 'Logistic Regression':
        model = LogisticRegression(**model_params)
    elif model_type == 'Decision Tree':
        model = DecisionTreeClassifier(**model_params)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'Support Vector Machine':
        model = SVC(**model_params)
    elif model_type == 'K-Nearest Neighbors':
        model = KNeighborsClassifier(**model_params)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model and measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

def cross_validate_model(X, y, model_type='Logistic Regression', model_params=None, cv=5):
    """
    Perform cross-validation for a machine learning model.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        model_type (str): Type of model to train
        model_params (dict): Parameters for the model
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    # Set default parameters if none provided
    if model_params is None:
        model_params = {}
    
    # Initialize the appropriate model
    if model_type == 'Logistic Regression':
        model = LogisticRegression(**model_params)
    elif model_type == 'Decision Tree':
        model = DecisionTreeClassifier(**model_params)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'Support Vector Machine':
        model = SVC(**model_params)
    elif model_type == 'K-Nearest Neighbors':
        model = KNeighborsClassifier(**model_params)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform cross-validation
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Prepare results
    results = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean(),
        'std_accuracy': cv_results['test_accuracy'].std(),
        'std_precision': cv_results['test_precision'].std(),
        'std_recall': cv_results['test_recall'].std(),
        'std_f1': cv_results['test_f1'].std(),
        'std_roc_auc': cv_results['test_roc_auc'].std(),
        'fit_time': cv_results['fit_time'].mean()
    }
    
    return results

def tune_hyperparameters(X, y, model_type='Logistic Regression', param_grid=None, cv=5):
    """
    Tune model hyperparameters using grid search.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        model_type (str): Type of model to train
        param_grid (dict): Grid of parameters to search
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Best parameters and results
    """
    from sklearn.model_selection import GridSearchCV
    
    # Set default parameter grid if none provided
    if param_grid is None:
        if model_type == 'Logistic Regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'none'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'Decision Tree':
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'Support Vector Machine':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        elif model_type == 'K-Nearest Neighbors':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        elif model_type == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize the appropriate model
    if model_type == 'Logistic Regression':
        model = LogisticRegression()
    elif model_type == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_type == 'Random Forest':
        model = RandomForestClassifier()
    elif model_type == 'Support Vector Machine':
        model = SVC()
    elif model_type == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X, y)
    
    # Prepare results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'mean_fit_time': grid_search.cv_results_['mean_fit_time'][grid_search.best_index_]
    }
    
    return results
