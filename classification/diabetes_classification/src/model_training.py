from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

def train_models(X_train, y_train):
    models = {}
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Logistic Regression
    models['logistic'] = LogisticRegression(random_state=42, max_iter=1000)
    models['logistic'].fit(X_train, y_train)
    joblib.dump(models['logistic'], os.path.join(models_dir, 'logistic_regression.pkl'))
    
    # Random Forest
    models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['random_forest'].fit(X_train, y_train)
    joblib.dump(models['random_forest'], os.path.join(models_dir, 'random_forest.pkl'))
    
    # SVM
    models['svm'] = SVC(random_state=42, probability=True)
    models['svm'].fit(X_train, y_train)
    joblib.dump(models['svm'], os.path.join(models_dir, 'svm.pkl'))
    
    print(f"Models saved to: {models_dir}")
    return models