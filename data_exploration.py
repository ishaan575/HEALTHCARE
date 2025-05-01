import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

def get_basic_statistics(data):
    """
    Calculate basic statistical information about the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset to analyze
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    stats_dict = {}
    
    # Basic shape info
    stats_dict['num_rows'] = data.shape[0]
    stats_dict['num_cols'] = data.shape[1]
    
    # Target distribution
    if 'target' in data.columns:
        stats_dict['target_distribution'] = data['target'].value_counts().to_dict()
        stats_dict['target_percentage'] = (data['target'].value_counts(normalize=True) * 100).to_dict()
    
    # Missing values
    stats_dict['missing_values'] = data.isnull().sum().to_dict()
    stats_dict['total_missing'] = data.isnull().sum().sum()
    
    # Data types
    stats_dict['dtypes'] = data.dtypes.astype(str).to_dict()
    
    # Summary statistics for numerical columns
    stats_dict['numeric_summary'] = data.describe().to_dict()
    
    return stats_dict

def analyze_correlations(data):
    """
    Analyze correlations between features and with the target.
    
    Args:
        data (pandas.DataFrame): The dataset to analyze
        
    Returns:
        tuple: (correlation matrix, correlations with target)
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Get correlations with target
    if 'target' in data.columns:
        target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
    else:
        target_corr = pd.Series()
    
    return corr_matrix, target_corr

def create_feature_histograms(data, target_col='target'):
    """
    Create histogram data for numeric features, colored by target.
    
    Args:
        data (pandas.DataFrame): The dataset to visualize
        target_col (str): Name of the target column
        
    Returns:
        dict: Dictionary with histogram data for each numeric feature
    """
    histograms = {}
    
    # Select numeric columns excluding the target
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    for col in numeric_cols:
        histograms[col] = {
            'data': data[[col, target_col]],
            'col_name': col
        }
    
    return histograms

def create_categorical_barplots(data, target_col='target'):
    """
    Create barplot data for categorical features, showing target distribution.
    
    Args:
        data (pandas.DataFrame): The dataset to visualize
        target_col (str): Name of the target column
        
    Returns:
        dict: Dictionary with barplot data for each categorical feature
    """
    barplots = {}
    
    # Identify categorical columns (including binary)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also include columns with small number of unique values
    for col in data.columns:
        if col not in categorical_cols and col != target_col:
            if data[col].nunique() < 10:  # Threshold for considering a column categorical
                categorical_cols.append(col)
    
    for col in categorical_cols:
        # Calculate target distribution for each category
        grouped = data.groupby(col)[target_col].mean().reset_index()
        grouped.columns = [col, 'target_rate']
        
        # Calculate counts for each category
        counts = data[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        
        # Merge the data
        merged = pd.merge(grouped, counts, on=col)
        
        barplots[col] = {
            'data': merged,
            'col_name': col
        }
    
    return barplots

def identify_outliers(data, method='zscore', threshold=3):
    """
    Identify outliers in numerical features.
    
    Args:
        data (pandas.DataFrame): The dataset to analyze
        method (str): Method to use ('zscore' or 'iqr')
        threshold (float): Threshold for outlier detection
        
    Returns:
        dict: Dictionary with outlier information for each numerical feature
    """
    outliers = {}
    
    # Get numerical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            outlier_values = data[col].dropna().iloc[outlier_indices]
            
            outliers[col] = {
                'method': 'zscore',
                'threshold': threshold,
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(data[col].dropna()) * 100,
                'min_value': data[col].min(),
                'max_value': data[col].max(),
                'outlier_min': outlier_values.min() if len(outlier_values) > 0 else None,
                'outlier_max': outlier_values.max() if len(outlier_values) > 0 else None
            }
            
        elif method == 'iqr':
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            
            outliers[col] = {
                'method': 'iqr',
                'threshold': threshold,
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(data[col].dropna()) * 100,
                'min_value': data[col].min(),
                'max_value': data[col].max(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    return outliers

def get_feature_distributions(data):
    """
    Analyze distributions of numerical features.
    
    Args:
        data (pandas.DataFrame): The dataset to analyze
        
    Returns:
        dict: Dictionary with distribution statistics for each numerical feature
    """
    distributions = {}
    
    # Get numerical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # Basic statistics
        mean = data[col].mean()
        median = data[col].median()
        mode = data[col].mode()[0]
        std = data[col].std()
        skew = data[col].skew()
        kurt = data[col].kurtosis()
        
        # Test for normality (Shapiro-Wilk)
        if len(data[col].dropna()) <= 5000:  # Shapiro-Wilk limited to 5000 samples
            sample = data[col].dropna()
            if len(sample) > 10:  # Need at least a few samples for the test
                shapiro_test = stats.shapiro(sample)
                shapiro_p = shapiro_test.pvalue
            else:
                shapiro_p = None
        else:
            # For larger datasets, use a sample
            shapiro_test = stats.shapiro(data[col].dropna().sample(5000, random_state=42))
            shapiro_p = shapiro_test.pvalue
        
        distributions[col] = {
            'mean': mean,
            'median': median,
            'mode': mode,
            'std': std,
            'skewness': skew,
            'kurtosis': kurt,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05 if shapiro_p is not None else None
        }
    
    return distributions
