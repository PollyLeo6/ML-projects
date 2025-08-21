# MPG Prediction Project

Machine learning project for predicting car fuel consumption based on technical characteristics.

## Description

This project analyzes the Auto MPG dataset and builds models to predict fuel consumption (MPG - Miles Per Gallon) based on the following characteristics:
- Number of cylinders
- Engine displacement
- Horsepower
- Vehicle weight
- Acceleration
- Model year
- Country of origin

## Project Structure

```
mpg_prediction/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # ML model training
│   └── model_evaluation.py       # Model quality evaluation
├── models/                       # Folder for saving trained models
├── main.py                       # Main script to run the project
├── requirements.txt              # Python dependencies list
├── regression_mpg_auto.ipynb     # Jupyter notebook with analysis
└── README.md                     # Project documentation
```

## Machine Learning Models

The project includes the following models:
1. **Linear Regression** - baseline model for comparison
2. **Polynomial Regression** - accounts for non-linear dependencies
3. **Random Forest** - ensemble method for improved accuracy

## Installation and Usage

### Requirements
- Python 3.7+
- pip

### Setup Steps

1. **Navigate to project directory:**
   ```bash
   cd regression/mpg_prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   python main.py
   ```

## File Descriptions

### `src/data_preprocessing.py`
- Loads data from CSV file
- Handles missing values in horsepower column
- Removes unnecessary columns (car name)
- Splits data into training and test sets

### `src/model_training.py`
- Creates and trains three machine learning models
- Linear regression for baseline comparison
- Polynomial regression (degree 2) for non-linear relationships
- Random Forest with 100 trees for ensemble approach
- Saves trained models to disk

### `src/model_evaluation.py`
- Evaluates model quality on test data
- Calculates R² (coefficient of determination) and RMSE (root mean square error) metrics
- Outputs results in readable format
- Saves evaluation results to JSON file

### `main.py`
- Coordinates the entire machine learning process
- Sequentially calls data loading, model training, and evaluation
- Identifies the best model based on R² metric

### `regression_mpg_auto.ipynb`
- Jupyter notebook with exploratory data analysis
- Visualization of data patterns and relationships
- Interactive model development and testing

## Expected Results

After running, you will see:
- Training and test set sizes
- Quality metrics for each model (R² and RMSE)
- Best performing model name

Example output:
```
=== MPG PREDICTION PROJECT ===
Loading and preprocessing data...
Training set size: (318, 7)
Test set size: (80, 7)

Training models...
Models saved in: ./models

Model evaluation:
LINEAR:
  R² Score: 0.8123
  RMSE: 3.2456

Best model: RANDOM_FOREST
R² Score: 0.8756
```

## Dataset

The project uses the Auto MPG dataset from UCI Machine Learning Repository, containing information about fuel consumption of various cars from the 1970s-1980s.