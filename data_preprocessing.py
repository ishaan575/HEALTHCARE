import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(data, scaling_method='StandardScaler', outlier_method='No Treatment', 
                   test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning.
    
    Args:
        data (pandas.DataFrame): The input dataset
        scaling_method (str): Method to use for feature scaling
        outlier_method (str): Method to handle outliers
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (preprocessed_data, X_train, X_test, y_train, y_test)
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Handle outliers if needed
    if outlier_method != 'No Treatment':
        df = handle_outliers(df, method=outlier_method)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply feature scaling if needed
    if scaling_method != 'No Scaling':
        X_train, X_test = scale_features(X_train, X_test, method=scaling_method)
    
    # Combine data back for returning
    preprocessed_data = pd.concat([X, y], axis=1)
    
    return preprocessed_data, X_train, X_test, y_train, y_test

def handle_outliers(data, method='Cap Outliers', z_threshold=3):
    """
    Handle outliers in the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset to process
        method (str): Method to handle outliers ('Remove Outliers' or 'Cap Outliers')
        z_threshold (float): Z-score threshold for identifying outliers
        
    Returns:
        pandas.DataFrame: Dataset with outliers handled
    """
    # Make a copy of the data
    df = data.copy()
    
    # Get numerical columns (excluding the target)
    if 'target' in df.columns:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('target', errors='ignore')
    else:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if method == 'Remove Outliers':
        # Calculate z-scores for each numeric column
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < z_threshold]
    
    elif method == 'Cap Outliers':
        # Cap outliers at z_threshold standard deviations
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std
            
            # Cap the outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df

def scale_features(X_train, X_test, method='StandardScaler'):
    """
    Apply feature scaling to the data.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        method (str): Scaling method to use ('StandardScaler' or 'MinMaxScaler')
        
    Returns:
        tuple: (scaled_X_train, scaled_X_test) as DataFrames with the same column names
    """
    # Get column names to preserve them after scaling
    feature_names = X_train.columns
    
    # Select the appropriate scaler
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit the scaler on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame with original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    return X_train_scaled_df, X_test_scaled_df

def handle_missing_values(data, strategy='median'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset with missing values
        strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
        
    Returns:
        pandas.DataFrame: Dataset with missing values handled
    """
    # Make a copy of the data
    df = data.copy()
    
    # Get numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Create imputers
    numeric_imputer = SimpleImputer(strategy=strategy)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Apply imputation
    if len(numeric_cols) > 0:
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    if len(categorical_cols) > 0:
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    return df

def encode_categorical_features(data, method='one-hot'):
    """
    Encode categorical features in the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset with categorical features
        method (str): Encoding method ('one-hot' or 'label')
        
    Returns:
        pandas.DataFrame: Dataset with encoded categorical features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) == 0:
        return df  # No categorical columns to encode
    
    # Apply encoding based on the selected method
    if method == 'one-hot':
        # Get dummies (one-hot encoding)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    elif method == 'label':
        # Label encoding (convert to integers)
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df
