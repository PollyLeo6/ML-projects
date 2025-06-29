# Machine Learning Study Projects

Repository containing my learning projects and experiments in machine learning, organized by topic.

## Repository Structure
/projects
├── supervised_learning
│ ├── classification
│ └── regression
├── unsupervised_learning
│ ├── clustering
│ └── dimensionality_reduction
├── nlp
├── computer_vision
└── time_series
/notebooks
├── data_exploration
├── model_training
└── experiments
/data
├── raw
└── processed

## Key Features

- Complete ML workflow in Jupyter notebooks: EDA → Preprocessing → Modeling → Evaluation
- Hands-on experience with:
  - Core ML: Scikit-learn, XGBoost
  - Clustering: K-Means, DBSCAN, Hierarchical
  - Deep Learning: PyTorch, TensorFlow
  - NLP: Transformers, spaCy
  - Visualization: Matplotlib, Seaborn, Plotly

## Clustering Projects

| Project | Description | Techniques | Metrics |
|---------|-------------|------------|---------|
| [Customer Segmentation](projects/unsupervised_learning/clustering/customer_segmentation.ipynb) | Grouping customers by purchasing behavior | K-Means, PCA | Silhouette Score |
| [Anomaly Detection](projects/unsupervised_learning/clustering/fraud_detection.ipynb) | Identifying unusual patterns in transactions | DBSCAN | Cluster Stability |
| [Document Clustering](projects/nlp/document_clustering.ipynb) | Topic modeling using text embeddings | HDBSCAN, TF-IDF | Davies-Bouldin Index |

## Usage

1. Clone the repo:
```bash
git clone https://github.com/yourusername/ml-studying-projects.git
cd ml-studying-projects
conda create -n ml-projects python=3.9
conda activate ml-projects
pip install -r requirements.txt
