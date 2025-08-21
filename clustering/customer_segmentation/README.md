# Customer Segmentation Project

Machine learning project for customer personality analysis and segmentation using clustering algorithms.

## Description

This project analyzes customer personality data and performs customer segmentation using various clustering algorithms. The goal is to identify distinct customer groups based on their purchasing behavior, demographics, and engagement patterns.

## Project Structure

```
customer_segmentation/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── clustering_models.py      # Clustering algorithms implementation
│   └── model_evaluation.py       # Model evaluation and analysis
├── models/                       # Saved trained models
├── main.py                       # Main script to run the project
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Clustering Algorithms

The project implements and compares multiple clustering algorithms:
1. **K-Means** - Centroid-based clustering
2. **K-Means with PCA** - K-Means on reduced dimensions
3. **Gaussian Mixture Model (GMM)** - Probabilistic clustering
4. **Hierarchical Clustering** - Agglomerative clustering
5. **DBSCAN** - Density-based clustering

## Dataset

The project automatically downloads the "Customer Personality Analysis" dataset from Kaggle:
- **Source**: Kaggle (imakash3011/customer-personality-analysis)
- **Type**: Customer behavior and demographics data
- **Features**: Age, income, spending patterns, purchase channels, family structure
- **Purpose**: Customer segmentation for marketing strategies

## Features Used

The model analyzes customers based on:
- **Demographics**: Age, Education, Marital Status, Income
- **Spending Behavior**: Total spent on different product categories
- **Purchase Patterns**: Web, catalog, store purchases
- **Engagement**: Website visits, campaign responses
- **Family Structure**: Number of children and teenagers

## Installation and Usage

### Requirements
- Python 3.7+
- pip
- Kaggle account (for dataset download)

### Setup Steps

1. **Navigate to project directory:**
   ```bash
   cd clustering/customer_segmentation
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

## Key Features

- **Automatic dataset download** using kagglehub
- **Feature engineering** with derived metrics (Total_Spent, Total_Purchases, etc.)
- **Outlier removal** using IQR method
- **Dimensionality reduction** with PCA
- **Optimal cluster selection** using elbow method and silhouette analysis
- **Multiple clustering algorithms** for comparison
- **Comprehensive evaluation** with silhouette score and Calinski-Harabasz index
- **Cluster analysis** with detailed customer segment characteristics

## Evaluation Metrics

The project uses several metrics to evaluate clustering quality:
- **Silhouette Score**: Measures how similar objects are to their own cluster vs other clusters
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion
- **Number of Clusters**: Optimal number determined by silhouette analysis
- **Noise Points**: For DBSCAN, points that don't belong to any cluster

## Expected Results

After running, you will see:
- Dataset download progress and information
- Data preprocessing steps and feature engineering
- Optimal number of clusters determination
- Model training progress
- Detailed evaluation metrics for each algorithm
- Best performing model identification
- Customer segment analysis with characteristics

Example output:
```
=== CUSTOMER SEGMENTATION PROJECT ===
Downloading dataset...
Dataset downloaded to: /path/to/dataset
Dataset shape: (XXXX, XX)

Finding optimal number of clusters...
Optimal number of clusters: 4

Training clustering models with 4 clusters...
Models saved to: ./models

Evaluating clustering models:
KMEANS:
  Silhouette Score: 0.XXXX
  Calinski-Harabasz Score: XXXX.XX
  Number of Clusters: 4

Best model: KMEANS
Silhouette Score: 0.XXXX

Analyzing kmeans clusters:
Cluster 0 (n=XXX):
Top characteristics:
  Income: XXXXX.XX
  Total_Spent: XXXX.XX
  ...
```

## Business Applications

The customer segments can be used for:
- **Targeted Marketing**: Customize campaigns for different customer groups
- **Product Recommendations**: Suggest products based on segment preferences
- **Pricing Strategies**: Optimize pricing for different customer segments
- **Customer Retention**: Identify high-value customers and at-risk segments
- **Resource Allocation**: Focus marketing budget on most profitable segments

## Model Outputs

The project saves:
- **Trained clustering models**: `.pkl` files for each algorithm
- **Evaluation results**: JSON file with detailed metrics
- **Cluster assignments**: Customer segment labels
- **Feature importance**: Key characteristics of each segment