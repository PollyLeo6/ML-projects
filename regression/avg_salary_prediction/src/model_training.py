from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os

def train_models(X_train, y_train):
    models = {}
    
    # Создаем папку models если её нет
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Линейная регрессия
    models['linear'] = LinearRegression()
    models['linear'].fit(X_train, y_train)
    joblib.dump(models['linear'], os.path.join(models_dir, 'linear_regression.pkl'))
    
    # Random Forest
    models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['random_forest'].fit(X_train, y_train)
    joblib.dump(models['random_forest'], os.path.join(models_dir, 'random_forest.pkl'))
    
    # Gradient Boosting
    models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    models['gradient_boosting'].fit(X_train, y_train)
    joblib.dump(models['gradient_boosting'], os.path.join(models_dir, 'gradient_boosting.pkl'))
    
    print(f"Модели сохранены в: {models_dir}")
    return models