# Iris Classification Project

Classic machine learning project for iris flower species classification using multiple algorithms.

## Description

This project implements multiclass classification on the famous Iris dataset to predict flower species based on sepal and petal measurements.

## Project Structure

```
iris_classification/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # ML model training
│   └── model_evaluation.py       # Model evaluation
├── models/                       # Saved trained models
├── main.py                       # Main script
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## Machine Learning Models

- **Logistic Regression** - Linear classifier
- **Random Forest** - Ensemble method
- **Support Vector Machine (SVM)** - Non-linear classifier
- **K-Nearest Neighbors (KNN)** - Instance-based learning

## Dataset

- **Source**: Kaggle (himanshunakrani/iris-dataset)
- **Features**: Sepal length, sepal width, petal length, petal width
- **Target**: 3 iris species (setosa, versicolor, virginica)
- **Size**: 150 samples

## Installation and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project:**
   ```bash
   python main.py
   ```

## Expected Results

The project will:
- Download the Iris dataset automatically
- Train 4 different classification models
- Evaluate each model's performance
- Identify the best performing model
- Save all models and results

Example output:
```
=== IRIS CLASSIFICATION PROJECT ===
Dataset shape: (150, 5)
Classes: ['setosa' 'versicolor' 'virginica']

Training models...
Models saved to: ./models

Evaluating models:
LOGISTIC:
  Accuracy: 1.0000

Best model: LOGISTIC
Accuracy: 1.0000
```