import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_and_preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

def main():
    # Путь к данным
    data_path = os.path.join('..', '..', 'data', 'raw', 'auto-mpg.csv')
    
    print("=== MPG PREDICTION PROJECT ===")
    print("Загрузка и предобработка данных...")
    
    # Загрузка и предобработка данных
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Обучение моделей
    print("\nОбучение моделей...")
    models = train_models(X_train, y_train)
    
    # Оценка моделей
    print("\nОценка моделей:")
    results = evaluate_models(models, X_test, y_test)
    
    # Определение лучшей модели
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"Лучшая модель: {best_model[0].upper()}")
    print(f"R² Score: {best_model[1]['r2']:.4f}")

if __name__ == "__main__":
    main()