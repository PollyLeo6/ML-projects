import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_and_preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

def main():
    print("=== DIABETES CLASSIFICATION PROJECT ===")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data()
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models:")
    results = evaluate_models(models, X_test, y_test, label_encoder)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best model: {best_model[0].upper()}")
    print(f"Accuracy: {best_model[1]['accuracy']:.4f}")

if __name__ == "__main__":
    main()