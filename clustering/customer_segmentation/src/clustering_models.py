from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    silhouette_scores = []
    
    from sklearn.metrics import silhouette_score
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Find elbow point
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
    
    return optimal_k, inertias, silhouette_scores

def train_clustering_models(X, X_pca, optimal_k=4):
    """Train multiple clustering models"""
    models = {}
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # K-Means
    models['kmeans'] = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    models['kmeans'].fit(X)
    joblib.dump(models['kmeans'], os.path.join(models_dir, 'kmeans.pkl'))
    
    # K-Means on PCA
    models['kmeans_pca'] = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    models['kmeans_pca'].fit(X_pca)
    joblib.dump(models['kmeans_pca'], os.path.join(models_dir, 'kmeans_pca.pkl'))
    
    # Gaussian Mixture Model
    models['gmm'] = GaussianMixture(n_components=optimal_k, random_state=42)
    models['gmm'].fit(X)
    joblib.dump(models['gmm'], os.path.join(models_dir, 'gmm.pkl'))
    
    # Hierarchical Clustering
    models['hierarchical'] = AgglomerativeClustering(n_clusters=optimal_k)
    models['hierarchical'].fit(X)
    joblib.dump(models['hierarchical'], os.path.join(models_dir, 'hierarchical.pkl'))
    
    # DBSCAN
    models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
    models['dbscan'].fit(X)
    joblib.dump(models['dbscan'], os.path.join(models_dir, 'dbscan.pkl'))
    
    print(f"Models saved to: {models_dir}")
    return models