from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime

def evaluate_models(models, X_test, y_test, label_encoder):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"{name.upper()}:")
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        print()
    
    # Save results
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'test_size': len(y_test),
        'classes': label_encoder.classes_.tolist(),
        'results': results
    }
    
    results_path = os.path.join(models_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    return results