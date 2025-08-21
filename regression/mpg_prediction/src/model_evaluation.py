from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import os
from datetime import datetime

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results[name] = {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
        }
        
        print(f"{name.upper()}:")
        print(f"  R² Score: {results[name]['r2']:.4f}")
        print(f"  RMSE: {results[name]['rmse']:.4f}")
        print()
    
    # Сохранение результатов
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'test_size': len(y_test),
        'results': results
    }
    
    results_path = os.path.join(models_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в: {results_path}")
    return results