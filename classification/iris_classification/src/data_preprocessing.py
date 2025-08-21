import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub

def load_and_preprocess_data():
    # Download dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("himanshunakrani/iris-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Load data
    df = pd.read_csv(f"{path}/iris.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['species'].unique()}")
    print(f"Features: {list(df.columns[:-1])}")
    
    # Prepare features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler