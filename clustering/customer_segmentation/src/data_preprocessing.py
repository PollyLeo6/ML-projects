import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import kagglehub

def load_and_preprocess_data():
    # Download dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
    print(f"Dataset downloaded to: {path}")
    
    # Load data
    df = pd.read_csv(f"{path}/marketing_campaign.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values
    df = df.dropna()
    
    # Feature engineering
    df['Age'] = 2024 - df['Year_Birth']
    df['Total_Spent'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + 
                        df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['Total_Purchases'] = (df['NumDealsPurchases'] + df['NumWebPurchases'] + 
                           df['NumCatalogPurchases'] + df['NumStorePurchases'])
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    
    # Select features for clustering
    features = ['Age', 'Income', 'Total_Spent', 'Total_Purchases', 'Total_Children',
               'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
               'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases',
               'NumStorePurchases', 'NumWebVisitsMonth']
    
    # Handle categorical variables
    categorical_cols = ['Education', 'Marital_Status']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            features.append(f'{col}_encoded')
    
    X = df[features].copy()
    
    # Remove outliers using IQR method
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X_clean = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original features: {X.shape[1]}")
    print(f"After PCA: {X_pca.shape[1]} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_scaled, X_pca, X_clean, scaler, pca, df