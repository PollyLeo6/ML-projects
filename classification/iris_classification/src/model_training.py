from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train_models(X_train, y_train):
    models = {}
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Logistic Regression
    models['logistic'] = LogisticRegression(random_state=42)
    models['logistic'].fit(X_train, y_train)
    joblib.dump(models['logistic'], os.path.join(models_dir, 'logistic.pkl'))
    
    # Random Forest
    models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['random_forest'].fit(X_train, y_train)
    joblib.dump(models['random_forest'], os.path.join(models_dir, 'random_forest.pkl'))
    
    # SVM
    models['svm'] = SVC(random_state=42, probability=True)
    models['svm'].fit(X_train, y_train)
    joblib.dump(models['svm'], os.path.join(models_dir, 'svm.pkl'))
    
    # KNN
    models['knn'] = KNeighborsClassifier(n_neighbors=3)
    models['knn'].fit(X_train, y_train)
    joblib.dump(models['knn'], os.path.join(models_dir, 'knn.pkl'))
    
    print(f"Models saved to: {models_dir}")
    return models