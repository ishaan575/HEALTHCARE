import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained ML model with predict and predict_proba methods
        X_test (pandas.DataFrame): Test feature matrix
        y_test (pandas.Series): Test target vector
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        # For models that don't have predict_proba (like SVM without probability=True)
        # Create a simple probability-like array based on the predictions
        y_prob = np.zeros((len(y_pred), 2))
        y_prob[np.arange(len(y_pred)), y_pred] = 1
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle case where there might be only one class in y_test or y_pred
    try:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    except:
        precision = 0
        recall = 0
        f1 = 0
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob[:, 1])
    
    # Assemble results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'precisions': precisions,
        'recalls': recalls,
        'pr_thresholds': pr_thresholds,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return results

def evaluate_at_threshold(y_prob, y_true, threshold=0.5):
    """
    Evaluate model predictions at a specific probability threshold.
    
    Args:
        y_prob (numpy.ndarray): Predicted probabilities
        y_true (pandas.Series): True target values
        threshold (float): Probability threshold for classification
        
    Returns:
        dict: Dictionary with evaluation metrics at the threshold
    """
    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_prob[:, 1] >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle case where there might be only one class in y_true or y_pred
    try:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    except:
        precision = 0
        recall = 0
        f1 = 0
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Assemble results
    results = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'y_pred': y_pred
    }
    
    return results

def get_classification_report(y_true, y_pred, labels=None):
    """
    Generate a classification report with key metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        labels (list): List of labels to include in the report
        
    Returns:
        pandas.DataFrame: Classification report as a DataFrame
    """
    from sklearn.metrics import classification_report
    
    # Get the report as a string
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    return report_df

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained ML model
        feature_names (list): List of feature names
        
    Returns:
        pandas.DataFrame: Feature importances
    """
    # Check if the model has feature importances
    if hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest and Gradient Boosting
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have accessible feature importances")
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate data for a learning curve plot to analyze model performance with varying training set sizes.
    
    Args:
        model: Estimator with fit/predict methods
        X (array-like): Training data
        y (array-like): Target values
        cv (int): Number of cross-validation folds
        train_sizes (array-like): Relative or absolute training set sizes
        
    Returns:
        dict: Learning curve data
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
    )
    
    # Calculate mean and std for train/test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    result = {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std
    }
    
    return result
