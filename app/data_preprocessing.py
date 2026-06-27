import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(df, strategy=\'mean\'):
    """Handles missing values using a specified strategy."""
    if df is None:
        return None
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.number]:
                if strategy == \'mean\':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == \'median\':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == \'mode\':
                    df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def feature_engineering(df):
    """Performs basic feature engineering (example: interaction term)."""
    if df is None:
        return None
    # Example: create an interaction term if feature1 and feature2 exist
    if \'feature1\' in df.columns and \'feature2\' in df.columns:
        df['feature1_x_feature2'] = df['feature1'] * df['feature2']
    return df

def preprocess_data(df, target_column=\'target\'):
    """Combines all preprocessing steps."""
    if df is None:
        return None

    df = handle_missing_values(df)
    df = feature_engineering(df)

    if target_column not in df.columns:
        print(f"Target column \'{target_column}\' not found in data.")
        return None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=\'object\').columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, X.columns
