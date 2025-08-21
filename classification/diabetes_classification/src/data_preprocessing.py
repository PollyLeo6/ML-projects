import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub

def load_and_preprocess_data():
    # Download dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("yasserhessein/multiclass-diabetes-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Load data
    df = pd.read_csv(f"{path}/diabetes_dataset.csv")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['class'].unique()}")
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables if any
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('class')  # Remove target variable
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Prepare features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le_target, scaler