# Diabetes Classification Project

Machine learning project for multiclass diabetes classification using Kaggle dataset.

## Description

This project analyzes a multiclass diabetes dataset and builds classification models to predict diabetes types based on various health indicators and symptoms.

## Project Structure

```
diabetes_classification/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # ML model training
│   └── model_evaluation.py       # Model evaluation
├── models/                       # Saved trained models
├── main.py                       # Main script to run the project
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Machine Learning Models

The project includes three classification models:
1. **Logistic Regression** - baseline linear classifier
2. **Random Forest** - ensemble method for better accuracy
3. **Support Vector Machine (SVM)** - powerful non-linear classifier

## Dataset

The project automatically downloads the "Multiclass Diabetes Dataset" from Kaggle using kagglehub:
- **Source**: Kaggle (yasserhessein/multiclass-diabetes-dataset)
- **Type**: Multiclass classification
- **Features**: Various health indicators and symptoms
- **Target**: Different types of diabetes

## Installation and Usage

### Requirements
- Python 3.7+
- pip
- Kaggle account (for dataset download)

### Setup Steps

1. **Navigate to project directory:**
   ```bash
   cd classification/diabetes_classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle credentials (if not already done):**
   - Go to Kaggle → Account → API → Create New API Token
   - Place kaggle.json in ~/.kaggle/ directory

4. **Run the project:**
   ```bash
   python main.py
   ```

## Features

- **Automatic dataset download** using kagglehub
- **Data preprocessing** with missing value handling and feature scaling
- **Multiple classification algorithms** for comparison
- **Comprehensive evaluation** with accuracy, classification reports, and confusion matrices
- **Model persistence** - trained models are saved for future use
- **Results logging** - evaluation results saved to JSON file

## Expected Results

After running, you will see:
- Dataset download progress and path
- Dataset shape and class information
- Training and test set sizes
- Model training progress
- Detailed evaluation metrics for each model
- Best performing model identification

Example output:
```
=== DIABETES CLASSIFICATION PROJECT ===
Downloading dataset...
Dataset downloaded to: /path/to/dataset
Dataset shape: (XXX, XX)
Classes: ['No Diabetes' 'Pre-diabetes' 'Type 1 diabetes' 'Type 2 diabetes']
Training set size: (XXX, XX)
Test set size: (XXX, XX)

Training models...
Models saved to: ./models

Evaluating models:
LOGISTIC:
  Accuracy: 0.XXXX
  Classification Report:
  ...

Best model: RANDOM_FOREST
Accuracy: 0.XXXX
```

## Model Outputs

The project saves:
- **Trained models**: `.pkl` files for each algorithm
- **Evaluation results**: JSON file with detailed metrics
- **Classification reports**: Precision, recall, F1-score for each class
- **Confusion matrices**: For detailed performance analysis

## Usage Notes

- First run will download the dataset (may take a few minutes)
- Subsequent runs will use cached dataset
- Models are automatically saved and can be loaded for predictions
- All results are timestamped for tracking experiments