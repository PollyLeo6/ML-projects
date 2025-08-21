# Machine Learning Projects Portfolio

A comprehensive collection of machine learning projects covering classification, regression, and clustering algorithms with real-world datasets.

## 🚀 Repository Overview

This repository contains complete end-to-end machine learning projects, each with modular code structure, comprehensive documentation, and practical applications.

## 📁 Project Structure

```
ML-projects/
├── classification/           # Classification projects
│   ├── diabetes_classification/    # Multiclass diabetes prediction
│   └── iris_classification/        # Classic iris flower classification
├── regression/              # Regression projects
│   ├── mpg_prediction/            # Car fuel efficiency prediction
│   └── avg_salary_prediction/     # Salary prediction from job data
├── clustering/              # Unsupervised learning projects
│   └── customer_segmentation/     # Customer personality analysis
└── data/                    # Shared datasets
```

## 🎯 Featured Projects

### Classification

| Project | Dataset | Algorithms | Best Accuracy | Description |
|---------|---------|------------|---------------|-------------|
| **[Diabetes Classification](classification/diabetes_classification/)** | Kaggle Multiclass Diabetes | Logistic Regression, Random Forest, SVM | ~95% | Predict diabetes types from health indicators |
| **[Iris Classification](classification/iris_classification/)** | Kaggle Iris Dataset | Logistic Regression, Random Forest, SVM, KNN | 96.7% | Classic flower species classification |

### Regression

| Project | Dataset | Algorithms | Best R² Score | Description |
|---------|---------|------------|---------------|-------------|
| **[MPG Prediction](regression/mpg_prediction/)** | Auto MPG Dataset | Linear, Polynomial, Random Forest | ~0.87 | Predict car fuel efficiency from technical specs |
| **[Salary Prediction](regression/avg_salary_prediction/)** | HeadHunter Job Data | Linear, Random Forest, Gradient Boosting | ~0.75 | Predict average salary from job requirements |

### Clustering

| Project | Dataset | Algorithms | Best Silhouette Score | Description |
|---------|---------|------------|----------------------|-------------|
| **[Customer Segmentation](clustering/customer_segmentation/)** | Customer Personality Analysis | K-Means, GMM, DBSCAN, Hierarchical | ~0.65 | Segment customers for targeted marketing |

## 🛠️ Technologies Used

- **Core ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Data Sources**: Kaggle datasets via kagglehub
- **Model Persistence**: joblib
- **Development**: Jupyter notebooks, Python scripts

## 🔧 Installation & Usage

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PollyLeo6/ML-projects.git
   cd ML-projects
   ```

2. **Choose a project and install dependencies:**
   ```bash
   cd classification/iris_classification
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   python main.py
   ```

### Project Features

- **🔄 Automated Data Loading**: Projects automatically download datasets from Kaggle
- **📊 Comprehensive Evaluation**: Multiple metrics and visualizations
- **💾 Model Persistence**: Trained models saved for future use
- **📈 Results Tracking**: JSON files with timestamped results
- **🧹 Clean Code**: Modular structure with separate preprocessing, training, and evaluation

## 📊 Key Achievements

- **5 Complete Projects** with production-ready code
- **12+ ML Algorithms** implemented and compared
- **4 Different Problem Types** (binary/multiclass classification, regression, clustering)
- **Real-world Datasets** from various domains (healthcare, automotive, HR, marketing)
- **Comprehensive Documentation** for each project

## 🎓 Learning Outcomes

This portfolio demonstrates proficiency in:

- **Data Preprocessing**: Handling missing values, feature engineering, scaling
- **Model Selection**: Comparing multiple algorithms and selecting optimal models
- **Evaluation Metrics**: Using appropriate metrics for different problem types
- **Code Organization**: Modular, maintainable, and documented code
- **End-to-End Workflow**: From data loading to model deployment

## 📋 Requirements

- Python 3.7+
- Kaggle account (for dataset access)
- See individual project `requirements.txt` files for specific dependencies

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new projects or algorithms
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.