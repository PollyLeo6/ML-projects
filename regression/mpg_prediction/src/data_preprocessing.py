import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Обработка пропущенных значений в horsepower
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
    
    # Удаление ненужных колонок
    df = df.drop('car name', axis=1)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test