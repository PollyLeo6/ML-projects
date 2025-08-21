# Average Salary Prediction Project

Machine learning project for predicting average salaries based on HeadHunter job data from Tomsk region.

## Description

This project analyzes HeadHunter job postings data and builds models to predict average salary based on:
- Work experience level
- Schedule type (full-time, part-time, remote, shift)
- Employment type (full, part, contract)
- Number of required skills

## Project Structure

```
avg_salary_prediction/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # ML model training
│   └── model_evaluation.py       # Model evaluation
├── models/                       # Saved trained models
├── main.py                       # Main script to run the project
├── requirements.txt              # Python dependencies
├── hh_hard.csv                   # Dataset
└── README.md                     # Project documentation
```

## Machine Learning Models

The project includes three regression models:
1. **Linear Regression** - baseline model
2. **Random Forest** - ensemble method for better accuracy
3. **Gradient Boosting** - advanced ensemble technique

## Installation and Usage

### Requirements
- Python 3.7+
- pip

### Setup Steps

1. **Navigate to project directory:**
   ```bash
   cd regression/avg_salary_prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   python main.py
   ```

## Data Features

The model uses the following features:
- **Experience Years**: Converted from categorical (0, 1-3, 3-6, 6+) to numerical values
- **Schedule Type**: Encoded categorical variable (full_day, part, remote, shift)
- **Employment Type**: Encoded categorical variable (full, part, contract)
- **Skills Count**: Number of required skills extracted from job descriptions

## Expected Results

After running, you will see:
- Training and test set sizes
- Average salary in the dataset
- Model performance metrics (R², RMSE, MAE) for each model
- Best performing model identification

Example output:
```
=== SALARY PREDICTION PROJECT ===
Loading and preprocessing data...
Training set size: (XXX, 4)
Test set size: (XXX, 4)
Average salary in data: XXXXX rub.

Training models...

Model evaluation:
LINEAR:
  R² Score: 0.XXXX
  RMSE: XXXXX rub.
  MAE: XXXXX rub.

Best model: GRADIENT_BOOSTING
R² Score: 0.XXXX
RMSE: XXXXX rub.
```

## Dataset

The project uses HeadHunter job postings data from Tomsk region, containing information about job requirements, salaries, and employment conditions.